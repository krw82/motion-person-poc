"""Bounded, timestamp-based photo bundles shared by file and live inputs.

YOLO/MOG2 decide when to open or extend an episode. Once open, sampling also
keeps quiet/empty frames. No cloud requests or notification policy live here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import cv2

from .contracts import CaptureSaveError, ConfigError, ContractError, FrameDecision, FramePacket
from .frame_processor import box_to_original


@dataclass(frozen=True)
class EventConfig:
    pre_capture_sec: float = 3.0
    quiet_sec: float = 3.0
    max_duration_sec: float = 10.0
    sample_interval_sec: float = 0.5
    max_images: int = 6
    max_buffer_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for name in ("pre_capture_sec", "quiet_sec", "max_duration_sec", "sample_interval_sec"):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value)):
                raise ConfigError(f"{name} must be a finite number", context={"key": name})
        if not 0 < self.max_duration_sec <= 10:
            raise ConfigError("max_duration_sec must be within (0, 10]", context={"key": "max_duration_sec"})
        if not 0 <= self.pre_capture_sec < self.max_duration_sec:
            raise ConfigError("pre_capture_sec must be >= 0 and shorter than the bundle limit")
        if not 0 < self.quiet_sec <= self.max_duration_sec:
            raise ConfigError("quiet_sec must be positive and no longer than the bundle limit")
        if not 0 < self.sample_interval_sec <= self.max_duration_sec:
            raise ConfigError("sample_interval_sec must be positive and no longer than the bundle limit")
        if (not isinstance(self.max_images, int) or isinstance(self.max_images, bool)
                or not 2 <= self.max_images <= 6):
            raise ConfigError("max_images must be an integer within [2, 6]")
        if (not isinstance(self.max_buffer_bytes, int) or isinstance(self.max_buffer_bytes, bool)
                or self.max_buffer_bytes <= 0):
            raise ConfigError("max_buffer_bytes must be a positive integer")


@dataclass(frozen=True)
class EventObject:
    class_name: str
    track_id: int | None
    confidence: float
    box_xyxy: tuple[int, int, int, int]
    moving: bool
    motion_pixels: int
    motion_ratio: float


@dataclass(frozen=True)
class EventImage:
    path: Path
    frame_index: int
    timestamp_sec: float
    phase: str
    roles: tuple[str, ...]
    original_size: tuple[int, int]
    objects: tuple[EventObject, ...]

    def to_dict(self):
        result = asdict(self)
        result["path"] = str(self.path)
        return result


@dataclass(frozen=True)
class EventBundle:
    run_id: str
    event_id: str
    part_index: int
    camera_id: str
    start_sec: float
    end_sec: float
    trigger_sec: float
    last_motion_sec: float
    end_reason: str
    episode_end: bool
    context_complete: bool
    pre_context_complete: bool
    post_context_complete: bool
    buffer_dropped_frames: int
    sampling_gap_count: int
    sample_count: int
    images: tuple[EventImage, ...]
    analysis_roi: tuple[float, float, float, float] | None
    manifest_path: Path

    @property
    def duration_sec(self):
        return self.end_sec - self.start_sec

    @property
    def image_paths(self):
        return tuple(image.path for image in self.images)

    def to_dict(self):
        result = asdict(self)
        result["schema_version"] = 1
        result["duration_sec"] = self.duration_sec
        result["manifest_path"] = str(self.manifest_path)
        result["images"] = [image.to_dict() for image in self.images]
        return result


@dataclass(frozen=True)
class _Sample:
    frame_index: int
    timestamp: float
    jpeg: bytes
    size: tuple[int, int]
    objects: tuple[EventObject, ...]
    matched: bool

    @property
    def score(self):
        return max(((o.motion_pixels, o.confidence) for o in self.objects if o.moving), default=(0, 0.0))


@dataclass
class _Part:
    event_id: str
    index: int
    start: float
    trigger: float
    last_motion: float
    pre_complete: bool
    frames: list[_Sample] = field(default_factory=list)
    dropped: int = 0
    gaps: int = 0
    sample_count: int = 0


class EventCaptureEngine:
    """One camera/analysis region per instance. Track IDs never split an episode.

    process() returns only completed bundles, after directory publication.
    10 seconds limits the complete window, including pre/post context. A hard
    split keeps the event ID and increments part_index. flush() closes remaining
    observed data without fabricating future post-capture frames.
    """

    def __init__(self, output_dir: Path, run_id: str, camera_id: str, *,
                 config: EventConfig | None = None, jpeg_quality: int = 95,
                 analysis_roi=None):
        self.config = config or EventConfig()
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
            raise ContractError("run_id must be a nonempty filename component")
        if not isinstance(jpeg_quality, int) or isinstance(jpeg_quality, bool) or not 1 <= jpeg_quality <= 100:
            raise ConfigError("jpeg_quality must be an integer within [1, 100]")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.run_id, self.camera_id = run_id, str(camera_id)
        self.jpeg_quality, self.analysis_roi = jpeg_quality, analysis_roi
        self._ring: deque[_Sample] = deque()
        self._part: _Part | None = None
        self._continuation: tuple[str, int, float, float, float] | None = None
        self._last_index = -1
        self._last_time = -1.0
        self._last_sample_time = -math.inf
        self._event_sequence = 0
        self.bundle_count = self.image_count = 0
        self.peak_buffer_bytes = 0

    @property
    def active(self):
        return self._part is not None or self._continuation is not None

    @property
    def buffer_bytes(self):
        samples = list(self._ring) + (self._part.frames if self._part else [])
        return sum(len(s.jpeg) for s in {s.frame_index: s for s in samples}.values())

    def process(self, packet: FramePacket, decision: FrameDecision) -> tuple[EventBundle, ...]:
        try:
            return self._process(packet, decision)
        except Exception:
            # Failed encoding/publication must not leave partly ingested future
            # frames to be published by an error flush with an older end time.
            self.discard()
            raise

    def discard(self):
        """Release unfinished data after an input/capture failure; keep saved bundles."""
        self._part = self._continuation = None
        self._ring.clear()

    def _process(self, packet: FramePacket, decision: FrameDecision) -> tuple[EventBundle, ...]:
        now = packet.video_time_sec
        if (not isinstance(packet.frame_index, int) or isinstance(packet.frame_index, bool)
                or packet.frame_index <= self._last_index
                or not isinstance(now, (int, float)) or isinstance(now, bool)
                or not math.isfinite(now) or now < 0 or now <= self._last_time
                or decision.frame_index != packet.frame_index or decision.video_time_sec != now):
            raise ContractError("event frames and decisions must agree and increase in index/time")
        now = float(now)
        completed = []
        sample = None
        new_part = False
        # Resolve elapsed deadlines before this frame can prolong the old part.
        if self._part is not None:
            part = self._part
            quiet_end = part.last_motion + self.config.quiet_sec
            hard_end = part.start + self.config.max_duration_sec
            boundary = min(quiet_end, hard_end)
            if now + 1e-9 >= boundary:
                reason = "QUIET" if quiet_end <= hard_end + 1e-9 else "MAX_DURATION"
                if abs(now - boundary) <= 1e-9 and not decision.candidate:
                    sample = self._encode(packet, decision)
                    self._append(sample)
                completed.append(self._finish(boundary, reason,
                                              post_complete=reason == "QUIET" and abs(now-boundary) <= self.config.sample_interval_sec))
                if reason == "MAX_DURATION" and now < quiet_end - 1e-9:
                    self._continuation = (part.event_id, part.index + 1, boundary, part.trigger, part.last_motion)

        if self._part is None and self._continuation is not None:
            event_id, index, start, trigger, last_motion = self._continuation
            self._continuation = None
            if now < last_motion + self.config.quiet_sec - 1e-9:
                # No repeated pre-roll in continuation parts, so every window <= 10s.
                self._part = _Part(event_id, index, now if now-start >= self.config.max_duration_sec else start,
                                   trigger, last_motion, False)
                new_part = True
        if self._part is None and decision.candidate:
            self._event_sequence += 1
            event_id = f"{self.run_id}_e{self._event_sequence:06d}"
            wanted = max(0.0, now - self.config.pre_capture_sec)
            before = [s for s in self._ring if wanted - 1e-9 <= s.timestamp < now]
            start = before[0].timestamp if before else now
            pre_complete = (self.config.pre_capture_sec == 0
                            or (bool(before) and before[0].timestamp <= wanted + self.config.sample_interval_sec))
            self._part = _Part(event_id, 1, start, now, now, pre_complete,
                               frames=before.copy(), sample_count=len(before))
            self._part.gaps = sum(b.timestamp-a.timestamp > self.config.sample_interval_sec*2+1e-9
                                  for a,b in zip(before,before[1:]))
            new_part = True

        if self._part is not None and decision.candidate:
            self._part.last_motion = now
        first_motion = self._part is not None and not any(s.matched and s.timestamp >= self._part.trigger for s in self._part.frames)
        due = now - self._last_sample_time + 1e-9 >= self.config.sample_interval_sec
        if due or new_part or (decision.candidate and first_motion):
            sample = sample or self._encode(packet, decision)
        if sample is not None:
            self._last_sample_time = now
            self._ring.append(sample)
            if self._part is not None:
                self._append(sample)
        while self._ring and self._ring[0].timestamp < now - self.config.pre_capture_sec - 1e-9:
            self._ring.popleft()
        self._bound_memory()
        self._last_index, self._last_time = packet.frame_index, now
        return tuple(completed)

    def flush(self, reason="END_OF_VIDEO") -> tuple[EventBundle, ...]:
        if reason not in ("END_OF_VIDEO", "USER_STOP", "USER_INTERRUPT", "ERROR", "DURATION_LIMIT"):
            raise ContractError("unsupported event flush reason")
        completed = ()
        try:
            if self._part is not None and self._part.frames:
                completed = (self._finish(min(self._last_time, self._part.start + self.config.max_duration_sec),
                                          reason, post_complete=False),)
        finally:
            self.discard()
        return completed

    def _encode(self, packet, decision):
        try:
            ok, encoded = cv2.imencode(".jpg", packet.raw_frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise ValueError("JPEG encoder returned false")
            jpeg = encoded.tobytes()
        except Exception as exc:
            raise CaptureSaveError("event JPEG encoding failed", context={"reason": str(exc)}) from exc
        if len(jpeg) > self.config.max_buffer_bytes:
            raise CaptureSaveError("one JPEG exceeds max_buffer_bytes; increase the event buffer budget")
        objects = tuple(EventObject(e.class_name, int(e.track_id) if e.track_id is not None else None,
                                   float(e.confidence), tuple(box_to_original(e.box, packet).to_list()),
                                   bool(e.qualifies), int(e.motion_pixels), float(e.motion_ratio))
                        for e in decision.objects)
        return _Sample(packet.frame_index, float(packet.video_time_sec), jpeg, packet.original_size,
                       objects, bool(decision.candidate))

    def _append(self, sample):
        part = self._part
        if part.frames and part.frames[-1].frame_index == sample.frame_index:
            return
        if part.frames and sample.timestamp - part.frames[-1].timestamp > self.config.sample_interval_sec * 2 + 1e-9:
            part.gaps += 1
        part.frames.append(sample)
        part.sample_count += 1
        self._bound_memory()

    def _bound_memory(self):
        while self.buffer_bytes > self.config.max_buffer_bytes:
            if self._ring:
                self._ring.popleft()
                continue
            part = self._part
            if part is None or len(part.frames) <= 1:
                raise CaptureSaveError("event buffer budget cannot hold required evidence")
            first_motion = next((s for s in part.frames if s.matched), None)
            victims = [s for s in part.frames[:-1] if s is not first_motion]
            if not victims:
                raise CaptureSaveError("event buffer budget cannot hold trigger and latest photo")
            part.frames.remove(victims[len(victims)//2])
            part.dropped += 1
            part.pre_complete = False
        self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.buffer_bytes)

    def _select(self, part):
        frames = part.frames
        roles: dict[int, set[str]] = {}
        def choose(sample, role):
            if sample is not None:
                roles.setdefault(sample.frame_index, set()).add(role)
        # When only 2 images are requested, keep the temporal endpoints.
        choose(frames[0], "before" if frames[0].timestamp < part.trigger else "start")
        choose(frames[-1], "end")
        if self.config.max_images > 2:
            choose(next((s for s in frames if s.matched and s.timestamp >= max(part.trigger, part.start)), None), "start")
        if len(roles) < self.config.max_images:
            active = [s for s in frames if s.matched and s.timestamp >= max(part.trigger, part.start)]
            choose(max(active, key=lambda s: s.score) if active else None, "peak")
        after = next((s for s in frames if s.timestamp > part.last_motion and not s.matched), None)
        if len(roles) < self.config.max_images:
            choose(after, "after")
        while len(roles) < min(self.config.max_images, len(frames)):
            chosen_times = [s.timestamp for s in frames if s.frame_index in roles]
            remaining = [s for s in frames if s.frame_index not in roles]
            choose(max(remaining, key=lambda s: min(abs(s.timestamp-t) for t in chosen_times)), "context")
        return [(s, tuple(sorted(roles[s.frame_index]))) for s in frames if s.frame_index in roles]

    def _finish(self, end, reason, *, post_complete):
        part = self._part
        assert part is not None and part.frames
        selected = self._select(part)
        final_dir = self.output_dir / part.event_id / f"part_{part.index:04d}"
        images = []
        for i, (sample, roles) in enumerate(selected, 1):
            phase = ("before" if sample.timestamp < part.trigger else "after"
                     if sample.timestamp > part.last_motion else "start"
                     if "start" in roles else "during")
            images.append(EventImage(final_dir / f"{i:02d}_{phase}_f{sample.frame_index:08d}.jpg",
                                     sample.frame_index, sample.timestamp, phase, roles, sample.size, sample.objects))
        complete = bool(part.pre_complete and post_complete and part.dropped == 0 and part.gaps == 0)
        bundle = EventBundle(self.run_id, part.event_id, part.index, self.camera_id,
                             part.start, end, part.trigger, part.last_motion, reason,
                             reason != "MAX_DURATION", complete, bool(part.pre_complete),
                             bool(post_complete), part.dropped, part.gaps, part.sample_count,
                             tuple(images), self.analysis_roi, final_dir / "event.json")
        staging = final_dir.parent / f".{final_dir.name}.{uuid.uuid4().hex}.tmp"
        try:
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                raise FileExistsError(f"bundle directory already exists: {final_dir}")
            staging.mkdir()
            for image, (sample, _) in zip(images, selected):
                self._write(staging / image.path.name, sample.jpeg)
            self._write(staging / "event.json",
                        (json.dumps(bundle.to_dict(), ensure_ascii=False, allow_nan=False, indent=2)+"\n").encode("utf-8"))
            staging.rename(final_dir)
        except (OSError, ValueError, TypeError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise CaptureSaveError("event bundle publication failed",
                                   context={"path": str(final_dir), "reason": str(exc)}) from exc
        self.bundle_count += 1
        self.image_count += len(images)
        self._part = None
        return bundle

    @staticmethod
    def _write(path, data):
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
