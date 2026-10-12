"""Configuration values and validation, without CLI or filesystem writes."""

from __future__ import annotations
from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any
from motion_person.domain.contracts import ConfigError


@dataclass(frozen=True)
class Config:
    """실행 설정 (SPEC.md 9.1). 값 범위 검증은 validate_config 가 담당한다."""

    # 입력/출력 경로
    video_path: Path = Path("videos/demo.mp4")
    model_path: Path = Path("models/yolo11n.pt")
    capture_dir: Path = Path("captures")
    log_dir: Path = Path("logs")

    # 분석 크기와 면적 기준
    analysis_width: int = 960
    reference_width: int = 960
    reference_height: int = 540

    # YOLO
    device: str = "cpu"
    yolo_imgsz: int = 640
    person_confidence: float = 0.60
    yolo_iou: float = 0.50
    max_detections: int = 100
    target_classes: tuple[str, ...] = ("person",)
    tracking: bool = False
    tracker: str = "bytetrack"
    capture_scope: str = "frame"
    tracking_profile: str = "stable"
    track_buffer: int = 60
    max_track_states: int = 2048
    track_new_threshold: float = 0.4
    track_min_hits: int = 3
    track_reid: bool = False
    analysis_roi: tuple[float, float, float, float] | None = None

    # MOG2
    mog2_history: int = 500
    mog2_var_threshold: float = 16.0
    mog2_detect_shadows: bool = True
    mog2_learning_rate: float = -1.0

    # 판정
    warmup_sec: float = 2.0
    min_component_pixels_ref: int = 100
    min_person_motion_pixels_ref: int = 3000
    min_person_motion_ratio: float = 0.03

    # 저장
    capture_cooldown_sec: float = 3.0
    jpeg_quality: int = 95

    # 실행 형태
    display: bool = True
    show_mask: bool = False
    overlay_mode: str = "full"
    pace: str = "realtime"
    fallback_fps: float | None = None
    metrics_interval_sec: float = 1.0
    debug_decisions: bool = False

    # 모폴로지 (CLI 미노출, SPEC.md 22.5)
    open_kernel: int = 3
    close_kernel: int = 5
    morphology_iterations: int = 1


def _require_finite(config: Config, name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ConfigError(
            f"{name} must be finite", context={"key": name, "value": repr(value)}
        )


def validate_analysis_values(config: Config) -> None:
    """Validate analysis settings without requiring a video or creating output folders."""
    _validate_values(config)


def _validate_values(config: Config) -> None:
    if config.overlay_mode not in ("full", "objects", "none"):
        raise ConfigError(
            "overlay_mode must be full, objects or none",
            context={"key": "overlay_mode"},
        )
    if (
        not isinstance(config.target_classes, tuple)
        or not config.target_classes
        or any(
            not isinstance(c, str) or not c.strip() or c != c.strip().lower()
            for c in config.target_classes
        )
        or len(set(config.target_classes)) != len(config.target_classes)
    ):
        raise ConfigError(
            "target_classes must contain unique lowercase model class names",
            context={"key": "target_classes"},
        )
    if not isinstance(config.tracking, bool):
        raise ConfigError("tracking must be boolean", context={"key": "tracking"})
    if config.tracker not in ("bytetrack", "botsort"):
        raise ConfigError("unsupported tracker", context={"key": "tracker"})
    if config.tracking_profile not in ("stable", "baseline"):
        raise ConfigError(
            "unsupported tracking_profile", context={"key": "tracking_profile"}
        )
    if not isinstance(config.track_reid, bool):
        raise ConfigError("track_reid must be boolean", context={"key": "track_reid"})
    if config.track_reid and (not config.tracking or config.tracker != "botsort"):
        raise ConfigError(
            "ReID requires --track --tracker botsort", context={"key": "track_reid"}
        )
    if config.analysis_roi is not None:
        roi = config.analysis_roi
        if (
            not isinstance(roi, tuple)
            or len(roi) != 4
            or any(
                not isinstance(v, (int, float))
                or isinstance(v, bool)
                or not math.isfinite(v)
                for v in roi
            )
            or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)
        ):
            raise ConfigError(
                "roi must be normalized X1 Y1 X2 Y2 within [0, 1]",
                context={"key": "analysis_roi"},
            )
    if config.capture_scope not in ("frame", "object"):
        raise ConfigError("unsupported capture_scope", context={"key": "capture_scope"})
    if config.capture_scope == "object" and not config.tracking:
        raise ConfigError(
            "object capture scope requires --track", context={"key": "capture_scope"}
        )
    # 0 이상 1 이하 실수
    for name in ("person_confidence", "min_person_motion_ratio", "track_new_threshold"):
        value = getattr(config, name)
        _require_finite(config, name, value)
        if not 0.0 <= value <= 1.0:
            raise ConfigError(
                f"{name} must be within [0, 1]", context={"key": name, "value": value}
            )

    # yolo_iou 는 0 초과 1 이하
    _require_finite(config, "yolo_iou", config.yolo_iou)
    if not 0.0 < config.yolo_iou <= 1.0:
        raise ConfigError(
            "yolo_iou must be in (0, 1]",
            context={"key": "yolo_iou", "value": config.yolo_iou},
        )

    # 양의 정수
    for name in (
        "analysis_width",
        "reference_width",
        "reference_height",
        "yolo_imgsz",
        "mog2_history",
        "max_detections",
        "min_component_pixels_ref",
        "min_person_motion_pixels_ref",
        "track_buffer",
        "track_min_hits",
        "max_track_states",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(
                f"{name} must be a positive integer",
                context={"key": name, "value": repr(value)},
            )
    if config.max_track_states < config.max_detections:
        raise ConfigError(
            "max_track_states must cover max_detections",
            context={"key": "max_track_states"},
        )

    # 커널 크기는 1 이상의 홀수
    for name in ("open_kernel", "close_kernel"):
        value = getattr(config, name)
        if not isinstance(value, int) or value < 1 or value % 2 == 0:
            raise ConfigError(
                f"{name} must be an odd integer >= 1",
                context={"key": name, "value": repr(value)},
            )
    if (
        not isinstance(config.morphology_iterations, int)
        or config.morphology_iterations < 1
    ):
        raise ConfigError(
            "morphology_iterations must be an integer >= 1",
            context={
                "key": "morphology_iterations",
                "value": repr(config.morphology_iterations),
            },
        )

    # 0 이상 유한 실수
    for name in ("warmup_sec", "capture_cooldown_sec"):
        value = getattr(config, name)
        _require_finite(config, name, value)
        if value < 0:
            raise ConfigError(
                f"{name} must be >= 0", context={"key": name, "value": value}
            )
    _require_finite(config, "mog2_var_threshold", config.mog2_var_threshold)

    # learning_rate 는 -1 또는 0 이상 1 이하
    rate = config.mog2_learning_rate
    _require_finite(config, "mog2_learning_rate", rate)
    if not (rate == -1.0 or 0.0 <= rate <= 1.0):
        raise ConfigError(
            "mog2_learning_rate must be -1 or within [0, 1]",
            context={"key": "mog2_learning_rate", "value": rate},
        )

    # jpeg_quality 는 1 이상 100 이하 정수
    if not isinstance(config.jpeg_quality, int) or not 1 <= config.jpeg_quality <= 100:
        raise ConfigError(
            "jpeg_quality must be an integer in [1, 100]",
            context={"key": "jpeg_quality", "value": repr(config.jpeg_quality)},
        )

    # device 는 cpu 고정
    if config.device != "cpu":
        raise ConfigError(
            "device must be 'cpu'", context={"key": "device", "value": config.device}
        )

    # pace 는 realtime 또는 fast
    if config.pace not in ("realtime", "fast"):
        raise ConfigError(
            "pace must be 'realtime' or 'fast'",
            context={"key": "pace", "value": config.pace},
        )

    # fallback_fps 는 지정되면 양의 유한 수
    if config.fallback_fps is not None:
        _require_finite(config, "fallback_fps", config.fallback_fps)
        if config.fallback_fps <= 0:
            raise ConfigError(
                "fallback_fps must be > 0",
                context={"key": "fallback_fps", "value": config.fallback_fps},
            )

    # metrics_interval_sec 은 양의 유한 수
    _require_finite(config, "metrics_interval_sec", config.metrics_interval_sec)
    if config.metrics_interval_sec <= 0:
        raise ConfigError(
            "metrics_interval_sec must be > 0",
            context={
                "key": "metrics_interval_sec",
                "value": config.metrics_interval_sec,
            },
        )


def config_to_dict(config: Config) -> dict:
    """Path 등도 JSON 으로 기록 가능한 값으로 바꾼다 (SPEC.md 9.2)."""
    result: dict[str, Any] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, Path):
            result[field.name] = str(value)
        elif isinstance(value, tuple):
            result[field.name] = list(value)
        elif value is None or isinstance(value, (str, int, float, bool)):
            result[field.name] = value
        else:
            result[field.name] = repr(value)
    return result
