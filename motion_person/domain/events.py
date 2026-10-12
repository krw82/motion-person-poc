"""Event settings and immutable evidence metadata, without storage."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from motion_person.domain.contracts import ConfigError


@dataclass(frozen=True)
class EventConfig:
    pre_capture_sec: float = 3.0
    quiet_sec: float = 3.0
    max_duration_sec: float = 10.0
    sample_interval_sec: float = 0.5
    max_images: int = 6
    max_buffer_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for name in (
            "pre_capture_sec",
            "quiet_sec",
            "max_duration_sec",
            "sample_interval_sec",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ConfigError(
                    f"{name} must be a finite number", context={"key": name}
                )
        if not 0 < self.max_duration_sec <= 10:
            raise ConfigError(
                "max_duration_sec must be within (0, 10]",
                context={"key": "max_duration_sec"},
            )
        if not 0 <= self.pre_capture_sec < self.max_duration_sec:
            raise ConfigError(
                "pre_capture_sec must be >= 0 and shorter than the bundle limit"
            )
        if not 0 < self.quiet_sec <= self.max_duration_sec:
            raise ConfigError(
                "quiet_sec must be positive and no longer than the bundle limit"
            )
        if not 0 < self.sample_interval_sec <= self.max_duration_sec:
            raise ConfigError(
                "sample_interval_sec must be positive and no longer than the bundle limit"
            )
        if (
            not isinstance(self.max_images, int)
            or isinstance(self.max_images, bool)
            or not 2 <= self.max_images <= 6
        ):
            raise ConfigError("max_images must be an integer within [2, 6]")
        if (
            not isinstance(self.max_buffer_bytes, int)
            or isinstance(self.max_buffer_bytes, bool)
            or self.max_buffer_bytes <= 0
        ):
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
    omitted_moving_objects: tuple[tuple[str, int | None], ...] = ()

    @property
    def duration_sec(self):
        return self.end_sec - self.start_sec

    @property
    def image_paths(self):
        return tuple(image.path for image in self.images)

    @property
    def evidence_complete(self) -> bool:
        return not self.omitted_moving_objects and any(
            obj.moving
            for image in self.images
            if image.timestamp_sec >= max(self.start_sec, self.trigger_sec)
            for obj in image.objects
        )

    def to_dict(self):
        result = asdict(self)
        result["schema_version"] = 1
        result["duration_sec"] = self.duration_sec
        result["evidence_complete"] = self.evidence_complete
        result["manifest_path"] = str(self.manifest_path)
        result["images"] = [image.to_dict() for image in self.images]
        result["omitted_moving_objects"] = [
            {"class_name": class_name, "track_id": track_id}
            for class_name, track_id in self.omitted_moving_objects
        ]
        return result
