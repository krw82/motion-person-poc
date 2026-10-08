"""config CLI 매핑과 검증 규칙 테스트 (SPEC.md 9.2, 9.3, 22.5).

커버리지 (SPEC ID):
- §22.5 필수 CLI 옵션 전체의 설정 키 매핑 (--min-person-motion-pixels ->
  min_person_motion_pixels_ref 등).
- §9.3 대표 검증 케이스: confidence 범위, yolo_iou (0,1], pace,
  jpeg [1,100], 커널 홀수, fallback_fps 양수, 파일 부재.
- tmp_path 로 실제 파일/폴더를 만들어 성공 케이스도 검증.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from config import Config, build_config, config_to_dict, validate_config
from contracts import ConfigError, ModelNotFoundError, VideoNotFoundError


def make_files(tmp_path: Path) -> tuple[Path, Path]:
    """더미 영상/모델 파일을 만든다 (Config 는 일반 파일 존재만 검사)."""
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake video bytes")
    model = tmp_path / "yolo11n.pt"
    model.write_bytes(b"fake model bytes")
    return video, model


def valid_config(tmp_path: Path, **overrides) -> Config:
    """검증을 통과하는 기본 Config. capture/log 폴더도 tmp_path 안에 둔다."""
    video, model = make_files(tmp_path)
    values: dict = dict(
        video_path=video,
        model_path=model,
        capture_dir=tmp_path / "cap",
        log_dir=tmp_path / "log",
    )
    values.update(overrides)
    return replace(Config(), **values)


def expect_invalid(config: Config) -> None:
    with pytest.raises(ConfigError):
        validate_config(config)


# ---------------------------------------------------------------------------
# §22.5 CLI 매핑
# ---------------------------------------------------------------------------


def test_cli_full_mapping(tmp_path: Path) -> None:
    """§22.5 전체 옵션이 올바른 설정 키로 매핑된다."""
    video, model = make_files(tmp_path)
    args = [
        "--video", str(video),
        "--model", str(model),
        "--analysis-width", "640",
        "--imgsz", "416",
        "--person-confidence", "0.5",
        "--min-person-motion-pixels", "2000",
        "--min-person-motion-ratio", "0.05",
        "--warmup-sec", "1.5",
        "--cooldown-sec", "5.0",
        "--capture-dir", str(tmp_path / "cli_cap"),
        "--log-dir", str(tmp_path / "cli_log"),
        "--show-mask",
        "--no-display",
        "--pace", "fast",
        "--fallback-fps", "24.0",
        "--debug-decisions",
    ]
    config = build_config(args)

    assert config.video_path == video
    assert config.model_path == model
    assert config.analysis_width == 640
    assert config.yolo_imgsz == 416
    assert config.person_confidence == 0.5
    assert config.min_person_motion_pixels_ref == 2000
    assert config.min_person_motion_ratio == 0.05
    assert config.warmup_sec == 1.5
    assert config.capture_cooldown_sec == 5.0
    assert config.capture_dir == tmp_path / "cli_cap"
    assert config.log_dir == tmp_path / "cli_log"
    assert config.show_mask is True
    assert config.display is False
    assert config.pace == "fast"
    assert config.fallback_fps == 24.0
    assert config.debug_decisions is True
    # capture/log 폴더가 실제로 만들어진다 (SPEC.md 9.3).
    assert config.capture_dir.is_dir()
    assert config.log_dir.is_dir()


def test_cli_defaults_when_options_omitted(tmp_path: Path) -> None:
    """§9.1/§22.5 기본값 확인 (플래그/선택 옵션 미지정)."""
    video, model = make_files(tmp_path)
    config = build_config([
        "--video", str(video),
        "--model", str(model),
        "--capture-dir", str(tmp_path / "cap"),
        "--log-dir", str(tmp_path / "log"),
    ])
    assert config.analysis_width == 960
    assert config.yolo_imgsz == 640
    assert config.person_confidence == 0.60
    assert config.min_person_motion_pixels_ref == 3000
    assert config.min_person_motion_ratio == 0.03
    assert config.warmup_sec == 2.0
    assert config.capture_cooldown_sec == 3.0
    assert config.pace == "realtime"
    assert config.display is True
    assert config.show_mask is False
    assert config.overlay_mode == "full"
    assert config.fallback_fps is None
    assert config.debug_decisions is False
    assert config.device == "cpu"
    assert config.jpeg_quality == 95


def test_cli_video_option_is_required(tmp_path: Path) -> None:
    """--video 없으면 argparse 가 종료된다 (§22.5 필수)."""
    with pytest.raises(SystemExit):
        build_config([])


def test_cli_invalid_pace_rejected_by_argparse(tmp_path: Path) -> None:
    """--pace 는 choices 제한으로 argparse 단계에서 거절된다 (§9.3)."""
    video, model = make_files(tmp_path)
    with pytest.raises(SystemExit):
        build_config([
            "--video", str(video), "--model", str(model),
            "--pace", "slow",
            "--capture-dir", str(tmp_path / "cap"),
            "--log-dir", str(tmp_path / "log"),
        ])


# ---------------------------------------------------------------------------
# §9.3 검증 규칙 (대표 케이스)
# ---------------------------------------------------------------------------


def test_validate_success_case(tmp_path: Path) -> None:
    """실제 파일/폴더로 검증을 통과하는 성공 케이스."""
    config = valid_config(tmp_path)
    validate_config(config)  # 예외 없음
    assert config.capture_dir.is_dir()
    assert config.log_dir.is_dir()


def test_confidence_and_ratio_range(tmp_path: Path) -> None:
    """confidence 와 비율은 0 이상 1 이하 (§9.3)."""
    validate_config(valid_config(tmp_path, person_confidence=0.0))
    validate_config(valid_config(tmp_path, person_confidence=1.0))
    expect_invalid(valid_config(tmp_path, person_confidence=1.5))
    expect_invalid(valid_config(tmp_path, person_confidence=-0.1))
    expect_invalid(valid_config(tmp_path, person_confidence=float("nan")))
    validate_config(valid_config(tmp_path, min_person_motion_ratio=1.0))
    expect_invalid(valid_config(tmp_path, min_person_motion_ratio=1.2))


def test_yolo_iou_excludes_zero(tmp_path: Path) -> None:
    """yolo_iou 는 0 초과 1 이하 (§9.3)."""
    validate_config(valid_config(tmp_path, yolo_iou=1.0))
    expect_invalid(valid_config(tmp_path, yolo_iou=0.0))
    expect_invalid(valid_config(tmp_path, yolo_iou=1.5))
    expect_invalid(valid_config(tmp_path, yolo_iou=float("inf")))


def test_positive_integer_fields(tmp_path: Path) -> None:
    """분석 너비/reference/imgsz/history/max_detections/면적 기준은 양의 정수."""
    for field in (
        "analysis_width", "reference_width", "reference_height", "yolo_imgsz",
        "mog2_history", "max_detections", "min_component_pixels_ref",
        "min_person_motion_pixels_ref",
    ):
        expect_invalid(valid_config(tmp_path, **{field: 0}))
        expect_invalid(valid_config(tmp_path, **{field: -10}))
    expect_invalid(valid_config(tmp_path, analysis_width=960.5))


def test_kernel_sizes_must_be_odd(tmp_path: Path) -> None:
    """커널 크기는 1 이상의 홀수 (§9.3)."""
    validate_config(valid_config(tmp_path, open_kernel=1, close_kernel=1))
    expect_invalid(valid_config(tmp_path, open_kernel=2))
    expect_invalid(valid_config(tmp_path, close_kernel=4))
    expect_invalid(valid_config(tmp_path, open_kernel=0))
    expect_invalid(valid_config(tmp_path, morphology_iterations=0))


def test_warmup_and_cooldown_non_negative(tmp_path: Path) -> None:
    """warmup 과 cooldown 은 0 이상의 유한한 수 (§9.3)."""
    validate_config(valid_config(tmp_path, warmup_sec=0.0))
    validate_config(valid_config(tmp_path, capture_cooldown_sec=0.0))
    expect_invalid(valid_config(tmp_path, warmup_sec=-1.0))
    expect_invalid(valid_config(tmp_path, capture_cooldown_sec=-0.1))
    expect_invalid(valid_config(tmp_path, warmup_sec=float("inf")))


def test_learning_rate_rule(tmp_path: Path) -> None:
    """learning_rate 는 -1 또는 0 이상 1 이하 (§9.3)."""
    validate_config(valid_config(tmp_path, mog2_learning_rate=-1.0))
    validate_config(valid_config(tmp_path, mog2_learning_rate=0.0))
    validate_config(valid_config(tmp_path, mog2_learning_rate=1.0))
    expect_invalid(valid_config(tmp_path, mog2_learning_rate=-0.5))
    expect_invalid(valid_config(tmp_path, mog2_learning_rate=1.5))


def test_jpeg_quality_range(tmp_path: Path) -> None:
    """jpeg_quality 는 1 이상 100 이하 정수 (§9.3)."""
    validate_config(valid_config(tmp_path, jpeg_quality=1))
    validate_config(valid_config(tmp_path, jpeg_quality=100))
    expect_invalid(valid_config(tmp_path, jpeg_quality=0))
    expect_invalid(valid_config(tmp_path, jpeg_quality=101))
    expect_invalid(valid_config(tmp_path, jpeg_quality=95.5))


def test_pace_and_device(tmp_path: Path) -> None:
    """pace 는 realtime/fast, device 는 cpu (§9.3)."""
    validate_config(valid_config(tmp_path, pace="fast"))
    expect_invalid(valid_config(tmp_path, pace="slow"))
    expect_invalid(valid_config(tmp_path, device="cuda"))


def test_fallback_fps_positive_finite(tmp_path: Path) -> None:
    """fallback_fps 는 지정되면 양의 유한 수 (§9.3)."""
    validate_config(valid_config(tmp_path, fallback_fps=30.0))
    expect_invalid(valid_config(tmp_path, fallback_fps=0.0))
    expect_invalid(valid_config(tmp_path, fallback_fps=-5.0))
    expect_invalid(valid_config(tmp_path, fallback_fps=float("inf")))
    expect_invalid(valid_config(tmp_path, fallback_fps=float("nan")))


def test_nan_infinity_rejected(tmp_path: Path) -> None:
    """config 에 NaN/infinity 를 허용하지 않는다 (§9.3)."""
    expect_invalid(valid_config(tmp_path, mog2_var_threshold=float("nan")))
    expect_invalid(valid_config(tmp_path, metrics_interval_sec=float("inf")))
    expect_invalid(valid_config(tmp_path, metrics_interval_sec=0.0))


def test_missing_video_file(tmp_path: Path) -> None:
    """영상 파일 부재 -> VIDEO_NOT_FOUND, 종료 코드 3 (§21)."""
    config = valid_config(tmp_path)
    missing = replace(config, video_path=tmp_path / "absent.mp4")
    with pytest.raises(VideoNotFoundError) as excinfo:
        validate_config(missing)
    assert excinfo.value.exit_code() == 3


def test_missing_model_file(tmp_path: Path) -> None:
    """모델 파일 부재 -> MODEL_NOT_FOUND, 종료 코드 4 (§21)."""
    config = valid_config(tmp_path)
    missing = replace(config, model_path=tmp_path / "absent.pt")
    with pytest.raises(ModelNotFoundError) as excinfo:
        validate_config(missing)
    assert excinfo.value.exit_code() == 4


def test_build_config_validates_missing_video(tmp_path: Path) -> None:
    """build_config 도 파일 검증까지 수행한다 (§9.2)."""
    _, model = make_files(tmp_path)
    with pytest.raises(VideoNotFoundError):
        build_config([
            "--video", str(tmp_path / "nope.mp4"),
            "--model", str(model),
            "--capture-dir", str(tmp_path / "cap"),
            "--log-dir", str(tmp_path / "log"),
        ])


# ---------------------------------------------------------------------------
# config_to_dict (SPEC.md 9.2)
# ---------------------------------------------------------------------------


def test_config_to_dict_json_safe(tmp_path: Path) -> None:
    """Path 는 문자열로 바뀌고 모든 값이 JSON 기록 가능 형태다."""
    video, model = make_files(tmp_path)
    config = valid_config(tmp_path)
    dumped = config_to_dict(config)

    assert dumped["video_path"] == str(video)
    assert dumped["model_path"] == str(model)
    assert dumped["capture_dir"] == str(tmp_path / "cap")
    assert dumped["fallback_fps"] is None
    for key, value in dumped.items():
        assert not isinstance(value, Path)
        if isinstance(value, float):
            assert math.isfinite(value)
