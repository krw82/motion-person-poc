"""Latest-frame webcam input. Camera pixels remain in memory and are never saved."""
from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np

from .contracts import ContractError, VideoDecodeError, VideoOpenError


class WebcamSource:
    """One camera per instance; monotonic receipt time instead of frame_index/FPS.

    The reader replaces one buffered frame while inference is busy. read() returns
    each selected frame at most once. Capture indices may skip when processing
    cannot keep up. This controls application queueing, not camera/driver delay.
    """

    def __init__(self, camera_index: int = 0, timeout_sec: float = 3.0):
        if not isinstance(camera_index, int) or isinstance(camera_index, bool) or camera_index < 0:
            raise ContractError("camera index must be an integer >= 0")
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ContractError("camera timeout must be finite and > 0")
        self.camera_index = camera_index
        self.timeout_sec = timeout_sec
        self.frames_read = 0
        self.frames_acquired = 0
        self.frames_skipped = 0
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._capture = None
        self._latest = None
        self._last_index = -1
        self._error = None
        self.started_perf = None

    def open(self) -> tuple[int, int]:
        if self._capture is not None:
            raise ContractError("WebcamSource is already open")
        capture = cv2.VideoCapture(self.camera_index)
        try:
            if not capture.isOpened():
                raise VideoOpenError(
                    f"cannot open webcam {self.camera_index}; allow camera access and close other camera apps",
                    context={"camera_index": self.camera_index},
                )
            ok, frame = capture.read()
            if not ok or frame is None:
                raise VideoDecodeError("webcam opened but returned no frame")
            self.started_perf = time.perf_counter()
            self._latest = (0, 0.0, frame)
            self.frames_read = self.frames_skipped = 0
            self.frames_acquired = 1
            self._last_index = -1
            self._error = None
            self._stop.clear()
            self._capture = capture
            self._thread = threading.Thread(target=self._reader, name="webcam-latest-frame", daemon=True)
            self._thread.start()
            return int(frame.shape[1]), int(frame.shape[0])
        except Exception:
            capture.release()
            self._capture = None
            raise

    def _reader(self) -> None:
        capture = self._capture
        try:
            while not self._stop.is_set():
                ok, frame = capture.read()
                if self._stop.is_set():
                    break
                if not ok or frame is None:
                    raise VideoDecodeError("webcam frame read failed")
                now = time.perf_counter() - self.started_perf
                with self._condition:
                    index = self.frames_acquired
                    self.frames_acquired += 1
                    self._latest = (index, now, frame)
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc if isinstance(exc, VideoDecodeError) else VideoDecodeError(
                    "webcam reader failed", context={"reason": str(exc)})
                self._condition.notify_all()
        finally:
            capture.release()

    def read(self) -> tuple[int, float, np.ndarray]:
        if self._capture is None:
            raise ContractError("WebcamSource is not open")
        deadline = time.perf_counter() + self.timeout_sec
        with self._condition:
            while self._latest is None or self._latest[0] == self._last_index:
                if self._error is not None:
                    raise self._error
                if self._stop.is_set():
                    raise VideoDecodeError("webcam stopped")
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise VideoDecodeError("webcam stopped delivering new frames")
                self._condition.wait(remaining)
            if self._error is not None:
                raise self._error
            selected = self._latest
            self.frames_skipped += max(0, selected[0] - self._last_index - 1)
            self._last_index = selected[0]
            self.frames_read += 1
            return selected

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise VideoDecodeError("webcam reader did not stop; close the preview process")
        self._capture = self._thread = self._latest = None
