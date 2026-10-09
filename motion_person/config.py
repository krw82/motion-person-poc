"""기본 설정, CLI 값 병합과 검증 (SPEC.md 9절, 22.5절).

우선순위는 기본값 다음 CLI 다. 환경변수, GUI 설정, 원격 설정과 YAML
로더는 제공하지 않는다. run_config.json 에 남는 값은 실행 당시의 최종
유효 설정이다.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from .contracts import ConfigError, ModelNotFoundError, VideoNotFoundError

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motion_person_poc",
        description="YOLO person + MOG2 전경 결합 캡처 PoC (SPEC.md 1절)",
    )
    parser.add_argument("--video", type=Path, required=True,
                        help="입력 MP4 경로 (필수)")
    parser.add_argument("--model", type=Path, default=None,
                        help="YOLO Detection 가중치 경로 (기본 models/yolo11n.pt)")
    parser.add_argument("--analysis-width", type=int, default=None,
                        help="분석 프레임 최대 너비 px (기본 960)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="모델 추론 입력 크기 (기본 640)")
    parser.add_argument("--person-confidence", "--confidence", type=float, default=None,
                        help="선택한 객체 confidence 하한 (기본 0.60)")
    parser.add_argument("--min-person-motion-pixels", "--min-object-motion-pixels", type=int, default=None,
                        help="기준 960x540 해상도의 객체 내부 최소 전경 픽셀 수 (기본 3000)")
    parser.add_argument("--min-person-motion-ratio", "--min-object-motion-ratio", type=float, default=None,
                        help="객체 박스 내부 전경 비율 하한 (기본 0.03)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="모델의 영문 객체 종류, 예: person dog (기본 person)")
    parser.add_argument("--track", action="store_true", help="객체별 추적 ID 유지")
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"], default=None)
    parser.add_argument("--tracking-profile", choices=["stable", "baseline"], default="stable",
                        help="stable: 종류별 연결과 관측 확인, baseline: 이전 추적 방식 비교")
    parser.add_argument("--track-buffer", type=int, default=None,
                        help="놓친 추적 유지 업데이트 수 (stable 기본 60)")
    parser.add_argument("--track-new-threshold", type=float, default=None,
                        help="새 추적을 만드는 신뢰도 하한 (stable 기본 0.4)")
    parser.add_argument("--track-min-hits", type=int, default=None,
                        help="캡처 전에 필요한 신뢰도 통과 관측 횟수 (stable 기본 3)")
    parser.add_argument("--reid", action="store_true",
                        help="BoT-SORT 외형 특징 연결 (--track --tracker botsort 필수)")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="분석 영역, 원본 대비 0~1 좌표. 캡처는 원본 전체 유지")
    parser.add_argument("--capture-scope", choices=["frame", "object"], default=None,
                        help="frame: 전체 쿨다운, object: 추적 ID별 쿨다운 (--track 필수)")
    parser.add_argument("--warmup-sec", type=float, default=None,
                        help="초기 배경 학습 시간, 초 (기본 2.0)")
    parser.add_argument("--cooldown-sec", type=float, default=None,
                        help="성공 캡처 간 최소 영상 시간, 초 (기본 3.0)")
    parser.add_argument("--capture-dir", type=Path, default=None,
                        help="캡처 저장 폴더 (기본 captures)")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="로그 저장 폴더 (기본 logs)")
    parser.add_argument("--show-mask", action="store_true",
                        help="전경 마스크 창을 추가 표시")
    parser.add_argument("--overlay", dest="overlay_mode", choices=["full", "objects", "none"],
                        default=None, help="화면 표시: full 전체, objects 객체 박스만, none 박스·패널 숨김")
    parser.add_argument("--no-display", action="store_true",
                        help="OpenCV 창 없이 실행")
    parser.add_argument("--pace", choices=["realtime", "fast"], default=None,
                        help="realtime 은 원본 속도에 맞춰 대기, fast 는 대기 없음")
    parser.add_argument("--fallback-fps", type=float, default=None,
                        help="영상 FPS 가 유효하지 않을 때 사용할 값")
    parser.add_argument("--debug-decisions", action="store_true",
                        help="매 프레임 판정 로그 기록")
    return parser


def build_config(args: list[str] | None = None) -> Config:
    """기본값에 CLI 를 반영하고 검증한 설정을 반환한다 (SPEC.md 9.2)."""
    parsed = _build_parser().parse_args(args)

    overrides: dict[str, Any] = {
        "video_path": parsed.video,
        "tracking_profile": parsed.tracking_profile,
    }
    if parsed.tracking_profile == "baseline":
        overrides.update(track_buffer=30, track_new_threshold=0.25, track_min_hits=1)
    for key in ("track_buffer", "track_new_threshold", "track_min_hits"):
        value = getattr(parsed, key)
        if value is not None:
            overrides[key] = value
    if parsed.reid:
        overrides["track_reid"] = True
    if parsed.roi is not None:
        overrides["analysis_roi"] = tuple(parsed.roi)
    if parsed.model is not None:
        overrides["model_path"] = parsed.model
    if parsed.analysis_width is not None:
        overrides["analysis_width"] = parsed.analysis_width
    if parsed.imgsz is not None:
        overrides["yolo_imgsz"] = parsed.imgsz
    if parsed.person_confidence is not None:
        overrides["person_confidence"] = parsed.person_confidence
    if parsed.classes is not None:
        overrides["target_classes"] = tuple(dict.fromkeys(c.strip().lower() for c in parsed.classes))
    if parsed.track:
        overrides["tracking"] = True
    if parsed.tracker is not None:
        overrides["tracker"] = parsed.tracker
    if parsed.capture_scope is not None:
        overrides["capture_scope"] = parsed.capture_scope
    if parsed.min_person_motion_pixels is not None:
        overrides["min_person_motion_pixels_ref"] = parsed.min_person_motion_pixels
    if parsed.min_person_motion_ratio is not None:
        overrides["min_person_motion_ratio"] = parsed.min_person_motion_ratio
    if parsed.warmup_sec is not None:
        overrides["warmup_sec"] = parsed.warmup_sec
    if parsed.cooldown_sec is not None:
        overrides["capture_cooldown_sec"] = parsed.cooldown_sec
    if parsed.capture_dir is not None:
        overrides["capture_dir"] = parsed.capture_dir
    if parsed.log_dir is not None:
        overrides["log_dir"] = parsed.log_dir
    if parsed.show_mask:
        overrides["show_mask"] = True
    if parsed.overlay_mode is not None:
        overrides["overlay_mode"] = parsed.overlay_mode
    if parsed.no_display:
        overrides["display"] = False
    if parsed.pace is not None:
        overrides["pace"] = parsed.pace
    if parsed.fallback_fps is not None:
        overrides["fallback_fps"] = parsed.fallback_fps
    if parsed.debug_decisions:
        overrides["debug_decisions"] = True

    config = replace(Config(), **overrides)
    validate_config(config)
    return config


def _require_finite(config: Config, name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite", context={"key": name, "value": repr(value)})


def validate_config(config: Config) -> None:
    """값 범위 오류면 ConfigError, 파일 부재면 VideoNotFoundError/
    ModelNotFoundError 를 발생시킨다 (SPEC.md 9.3, 21절 — 종료 코드 2/3/4)."""
    _validate_values(config)
    _validate_files(config)


def validate_analysis_values(config: Config) -> None:
    """Validate analysis settings without requiring a video or creating output folders."""
    _validate_values(config)


def _validate_values(config: Config) -> None:
    if config.overlay_mode not in ("full", "objects", "none"):
        raise ConfigError("overlay_mode must be full, objects or none",
                          context={"key": "overlay_mode"})
    if (not isinstance(config.target_classes, tuple) or not config.target_classes
            or any(not isinstance(c, str) or not c.strip() or c != c.strip().lower()
                   for c in config.target_classes)
            or len(set(config.target_classes)) != len(config.target_classes)):
        raise ConfigError("target_classes must contain unique lowercase model class names",
                          context={"key": "target_classes"})
    if not isinstance(config.tracking, bool):
        raise ConfigError("tracking must be boolean", context={"key": "tracking"})
    if config.tracker not in ("bytetrack", "botsort"):
        raise ConfigError("unsupported tracker", context={"key": "tracker"})
    if config.tracking_profile not in ("stable", "baseline"):
        raise ConfigError("unsupported tracking_profile", context={"key": "tracking_profile"})
    if not isinstance(config.track_reid, bool):
        raise ConfigError("track_reid must be boolean", context={"key": "track_reid"})
    if config.track_reid and (not config.tracking or config.tracker != "botsort"):
        raise ConfigError("ReID requires --track --tracker botsort", context={"key": "track_reid"})
    if config.analysis_roi is not None:
        roi = config.analysis_roi
        if (not isinstance(roi, tuple) or len(roi) != 4
                or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                       or not math.isfinite(v) for v in roi)
                or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)):
            raise ConfigError("roi must be normalized X1 Y1 X2 Y2 within [0, 1]",
                              context={"key": "analysis_roi"})
    if config.capture_scope not in ("frame", "object"):
        raise ConfigError("unsupported capture_scope", context={"key": "capture_scope"})
    if config.capture_scope == "object" and not config.tracking:
        raise ConfigError("object capture scope requires --track",
                          context={"key": "capture_scope"})
    # 0 이상 1 이하 실수
    for name in ("person_confidence", "min_person_motion_ratio", "track_new_threshold"):
        value = getattr(config, name)
        _require_finite(config, name, value)
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"{name} must be within [0, 1]",
                              context={"key": name, "value": value})

    # yolo_iou 는 0 초과 1 이하
    _require_finite(config, "yolo_iou", config.yolo_iou)
    if not 0.0 < config.yolo_iou <= 1.0:
        raise ConfigError("yolo_iou must be in (0, 1]",
                          context={"key": "yolo_iou", "value": config.yolo_iou})

    # 양의 정수
    for name in ("analysis_width", "reference_width", "reference_height",
                 "yolo_imgsz", "mog2_history", "max_detections",
                 "min_component_pixels_ref", "min_person_motion_pixels_ref", "track_buffer", "track_min_hits"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{name} must be a positive integer",
                              context={"key": name, "value": repr(value)})

    # 커널 크기는 1 이상의 홀수
    for name in ("open_kernel", "close_kernel"):
        value = getattr(config, name)
        if not isinstance(value, int) or value < 1 or value % 2 == 0:
            raise ConfigError(f"{name} must be an odd integer >= 1",
                              context={"key": name, "value": repr(value)})
    if not isinstance(config.morphology_iterations, int) or config.morphology_iterations < 1:
        raise ConfigError("morphology_iterations must be an integer >= 1",
                          context={"key": "morphology_iterations",
                                   "value": repr(config.morphology_iterations)})

    # 0 이상 유한 실수
    for name in ("warmup_sec", "capture_cooldown_sec"):
        value = getattr(config, name)
        _require_finite(config, name, value)
        if value < 0:
            raise ConfigError(f"{name} must be >= 0", context={"key": name, "value": value})
    _require_finite(config, "mog2_var_threshold", config.mog2_var_threshold)

    # learning_rate 는 -1 또는 0 이상 1 이하
    rate = config.mog2_learning_rate
    _require_finite(config, "mog2_learning_rate", rate)
    if not (rate == -1.0 or 0.0 <= rate <= 1.0):
        raise ConfigError("mog2_learning_rate must be -1 or within [0, 1]",
                          context={"key": "mog2_learning_rate", "value": rate})

    # jpeg_quality 는 1 이상 100 이하 정수
    if not isinstance(config.jpeg_quality, int) or not 1 <= config.jpeg_quality <= 100:
        raise ConfigError("jpeg_quality must be an integer in [1, 100]",
                          context={"key": "jpeg_quality", "value": repr(config.jpeg_quality)})

    # device 는 cpu 고정
    if config.device != "cpu":
        raise ConfigError("device must be 'cpu'", context={"key": "device",
                                                           "value": config.device})

    # pace 는 realtime 또는 fast
    if config.pace not in ("realtime", "fast"):
        raise ConfigError("pace must be 'realtime' or 'fast'",
                          context={"key": "pace", "value": config.pace})

    # fallback_fps 는 지정되면 양의 유한 수
    if config.fallback_fps is not None:
        _require_finite(config, "fallback_fps", config.fallback_fps)
        if config.fallback_fps <= 0:
            raise ConfigError("fallback_fps must be > 0",
                              context={"key": "fallback_fps", "value": config.fallback_fps})

    # metrics_interval_sec 은 양의 유한 수
    _require_finite(config, "metrics_interval_sec", config.metrics_interval_sec)
    if config.metrics_interval_sec <= 0:
        raise ConfigError("metrics_interval_sec must be > 0",
                          context={"key": "metrics_interval_sec",
                                   "value": config.metrics_interval_sec})


def _validate_files(config: Config) -> None:
    # 파일 부재는 CONFIG_ERROR(2) 가 아니라 전용 오류로 종료한다
    # (SPEC.md 21절: VIDEO_NOT_FOUND=3, MODEL_NOT_FOUND=4).
    if not config.video_path.is_file():
        raise VideoNotFoundError(
            f"video file not found: {config.video_path}",
            context={"key": "video_path", "path": str(config.video_path)},
        )
    if not config.model_path.is_file():
        raise ModelNotFoundError(
            f"model file not found: {config.model_path}",
            context={"key": "model_path", "path": str(config.model_path)},
        )
    for label, path in (("capture", config.capture_dir), ("log", config.log_dir)):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise ConfigError(f"{label} dir is not writable: {path}",
                              context={"key": f"{label}_dir", "path": str(path),
                                       "reason": str(exc)}) from exc


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
