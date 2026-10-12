"""On-demand preview encoding. One latest JPEG is shared by all viewers."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import math
import threading
import time

import cv2

from motion_person.domain.contracts import ConfigError, FrameAnalysis
from motion_person.domain.geometry import box_to_original


@dataclass(frozen=True)
class PreviewConfig:
    width: int = 640
    fps: float = 8.0
    jpeg_quality: int = 75
    mirror: bool = False
    overlay: bool = False

    def __post_init__(self):
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or self.width < 1
        ):
            raise ConfigError("preview width must be a positive integer")
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not math.isfinite(self.fps)
            or not 0 < self.fps <= 30
        ):
            raise ConfigError("preview fps must be within (0, 30]")
        if (
            isinstance(self.jpeg_quality, bool)
            or not isinstance(self.jpeg_quality, int)
            or not 1 <= self.jpeg_quality <= 100
        ):
            raise ConfigError("preview jpeg_quality must be within [1, 100]")
        if not isinstance(self.mirror, bool) or not isinstance(self.overlay, bool):
            raise ConfigError("preview mirror and overlay must be boolean")


@dataclass(frozen=True)
class PreviewImage:
    jpeg: bytes
    frame_index: int
    elapsed_sec: float
    width: int
    height: int


class FramePreview:
    """No inference, camera opening, file writes or unbounded frame queue."""

    def __init__(self, config: PreviewConfig | None = None):
        self.config = config or PreviewConfig()
        self._condition = threading.Condition()
        self._source = self._analysis = self._image = self._thread = None
        self._closed = False
        self._viewers = 0
        self._requested_until = 0.0
        self.encode_count = 0
        self.last_error: str | None = None

    def attach(self, source) -> None:
        with self._condition:
            self._source = source
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="motion-preview", daemon=True
                )
                self._thread.start()

    def publish(self, analysis: FrameAnalysis) -> None:
        with self._condition:
            self._analysis = analysis
            self._condition.notify_all()

    @contextmanager
    def subscribe(self):
        with self._condition:
            self._viewers += 1
            self._condition.notify_all()
        try:
            yield self
        finally:
            with self._condition:
                self._viewers -= 1

    def snapshot(self, *, after: int = -1, timeout: float = 1.0) -> PreviewImage | None:
        with self._condition:
            self._requested_until = time.monotonic() + 1.0
            self._condition.notify_all()
            self._condition.wait_for(
                lambda: (
                    self._closed
                    or (self._image is not None and self._image.frame_index > after)
                ),
                timeout=timeout,
            )
            return (
                self._image
                if self._image is not None and self._image.frame_index > after
                else None
            )

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with self._condition:
            self._source = self._analysis = None

    def _run(self) -> None:
        last_index = -1
        deadline = 0.0
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._closed
                        or self._viewers
                        or time.monotonic() < self._requested_until
                    )
                )
                if self._closed:
                    return
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                source, analysis = self._source, self._analysis
            deadline = time.monotonic() + 1 / self.config.fps
            try:
                item = None
                if not self.config.overlay and source is not None:
                    item = source.snapshot()
                if item is None and analysis is not None:
                    item = (
                        analysis.packet.frame_index,
                        analysis.packet.video_time_sec,
                        analysis.packet.raw_frame,
                    )
                if item is None or item[0] == last_index:
                    continue
                image = self._encode(item, analysis if self.config.overlay else None)
            except Exception as exc:
                # An optional UI adapter must not silently lose its worker.
                with self._condition:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._condition.notify_all()
                continue
            last_index = item[0]
            with self._condition:
                if self._closed:
                    return
                self._image = image
                self.encode_count += 1
                self.last_error = None
                self._condition.notify_all()

    def _encode(self, item, analysis) -> PreviewImage:
        index, elapsed, raw = item
        original_h, original_w = raw.shape[:2]
        width = min(original_w, self.config.width)
        height = max(1, round(original_h * width / original_w))
        frame = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
        if self.config.mirror:
            frame = cv2.flip(frame, 1)
        if analysis is not None:
            for obj in analysis.decision.objects:
                box = box_to_original(obj.box, analysis.packet)
                x1, x2 = (
                    round(box.x1 * width / original_w),
                    round(box.x2 * width / original_w),
                )
                y1, y2 = (
                    round(box.y1 * height / original_h),
                    round(box.y2 * height / original_h),
                )
                if self.config.mirror:
                    x1, x2 = width - x2, width - x1
                color = (0, 220, 255) if obj.qualifies else (0, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    obj.label,
                    (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
        ok, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        if not ok:
            raise ValueError("preview JPEG encoder returned false")
        return PreviewImage(jpeg.tobytes(), index, elapsed, width, height)
