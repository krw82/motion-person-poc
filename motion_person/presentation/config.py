"""Legacy CLI configuration; runtime settings live in domain.settings."""

from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any
from motion_person.domain.settings import (
    Config,
    config_to_dict as config_to_dict,
    validate_analysis_values as validate_analysis_values,
    _validate_values,
    _require_finite as _require_finite,
)
from motion_person.domain.contracts import (
    ConfigError,
    ModelNotFoundError,
    VideoNotFoundError,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motion_person_poc",
        description="YOLO person + MOG2 전경 결합 캡처 PoC (SPEC.md 1절)",
    )
    parser.add_argument(
        "--video", type=Path, required=True, help="입력 MP4 경로 (필수)"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLO Detection 가중치 경로 (기본 models/yolo11n.pt)",
    )
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=None,
        help="분석 프레임 최대 너비 px (기본 960)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None, help="모델 추론 입력 크기 (기본 640)"
    )
    parser.add_argument(
        "--person-confidence",
        "--confidence",
        type=float,
        default=None,
        help="선택한 객체 confidence 하한 (기본 0.60)",
    )
    parser.add_argument(
        "--min-person-motion-pixels",
        "--min-object-motion-pixels",
        type=int,
        default=None,
        help="기준 960x540 해상도의 객체 내부 최소 전경 픽셀 수 (기본 3000)",
    )
    parser.add_argument(
        "--min-person-motion-ratio",
        "--min-object-motion-ratio",
        type=float,
        default=None,
        help="객체 박스 내부 전경 비율 하한 (기본 0.03)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="모델의 영문 객체 종류, 예: person dog (기본 person)",
    )
    parser.add_argument("--track", action="store_true", help="객체별 추적 ID 유지")
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"], default=None)
    parser.add_argument(
        "--tracking-profile",
        choices=["stable", "baseline"],
        default="stable",
        help="stable: 종류별 연결과 관측 확인, baseline: 이전 추적 방식 비교",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=None,
        help="놓친 추적 유지 업데이트 수 (stable 기본 60)",
    )
    parser.add_argument(
        "--track-new-threshold",
        type=float,
        default=None,
        help="새 추적을 만드는 신뢰도 하한 (stable 기본 0.4)",
    )
    parser.add_argument(
        "--track-min-hits",
        type=int,
        default=None,
        help="캡처 전에 필요한 신뢰도 통과 관측 횟수 (stable 기본 3)",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        help="BoT-SORT 외형 특징 연결 (--track --tracker botsort 필수)",
    )
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="분석 영역, 원본 대비 0~1 좌표. 캡처는 원본 전체 유지",
    )
    parser.add_argument(
        "--capture-scope",
        choices=["frame", "object"],
        default=None,
        help="frame: 전체 쿨다운, object: 추적 ID별 쿨다운 (--track 필수)",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=None,
        help="초기 배경 학습 시간, 초 (기본 2.0)",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=None,
        help="성공 캡처 간 최소 영상 시간, 초 (기본 3.0)",
    )
    parser.add_argument(
        "--capture-dir", type=Path, default=None, help="캡처 저장 폴더 (기본 captures)"
    )
    parser.add_argument(
        "--log-dir", type=Path, default=None, help="로그 저장 폴더 (기본 logs)"
    )
    parser.add_argument(
        "--show-mask", action="store_true", help="전경 마스크 창을 추가 표시"
    )
    parser.add_argument(
        "--overlay",
        dest="overlay_mode",
        choices=["full", "objects", "none"],
        default=None,
        help="화면 표시: full 전체, objects 객체 박스만, none 박스·패널 숨김",
    )
    parser.add_argument("--no-display", action="store_true", help="OpenCV 창 없이 실행")
    parser.add_argument(
        "--pace",
        choices=["realtime", "fast"],
        default=None,
        help="realtime 은 원본 속도에 맞춰 대기, fast 는 대기 없음",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=None,
        help="영상 FPS 가 유효하지 않을 때 사용할 값",
    )
    parser.add_argument(
        "--debug-decisions", action="store_true", help="매 프레임 판정 로그 기록"
    )
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
        overrides["target_classes"] = tuple(
            dict.fromkeys(c.strip().lower() for c in parsed.classes)
        )
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


def validate_config(config: Config) -> None:
    """값 범위 오류면 ConfigError, 파일 부재면 VideoNotFoundError/
    ModelNotFoundError 를 발생시킨다 (SPEC.md 9.3, 21절 — 종료 코드 2/3/4)."""
    _validate_values(config)
    _validate_files(config)


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
            raise ConfigError(
                f"{label} dir is not writable: {path}",
                context={"key": f"{label}_dir", "path": str(path), "reason": str(exc)},
            ) from exc
