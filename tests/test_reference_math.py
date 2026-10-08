"""SPEC.md 31.6 검증 예시 전체와 참조 수학 함수 테스트.

커버리지 (SPEC ID):
- §31.6 검증 예시 전부 (person_mask_evidence / qualifies_person /
  effective_threshold / cooldown_state / to_original_box).
- U12: effective_threshold 960x540 -> 3000, 640x360 -> 1334 (SPEC.md 9.4).
- U13: to_original_box 원본 1920x1080 / 분석 960x540 좌표 정확히 2배
  (SPEC.md 11.2 예시 포함).
- U14/U15: cooldown_state 7.99 억제 / 8.0 허용 (SPEC.md 31.4, 16.2).
- make_packet 축소 시 scale 계산·원본 미수정, 동일 크기면 배열 공유
  (SPEC.md 11.1, 11.3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from capture_manager import cooldown_state
from contracts import Box
from event_detector import person_mask_evidence, qualifies_person
from frame_processor import (
    box_to_original,
    effective_threshold,
    make_packet,
    to_original_box,
)

# ---------------------------------------------------------------------------
# SPEC.md 31.6 검증 예시 (통합 재현)
# ---------------------------------------------------------------------------


def test_spec_31_6_reference_example() -> None:
    """SPEC.md 31.6 예시 블록을 그대로 재현한다."""
    mask = np.zeros((540, 960), dtype=np.uint8)
    box = Box(100, 100, 300, 300)
    mask[110:140, 110:210] = 255  # 30 x 100 = 3000 pixels
    pixels, ratio = person_mask_evidence(mask, box)
    assert pixels == 3000
    assert ratio == 0.075
    assert qualifies_person(0.90, pixels, ratio, False)[0]
    assert not qualifies_person(0.90, pixels, ratio, True)[0]

    outside = np.zeros((540, 960), dtype=np.uint8)
    outside[350:450, 500:700] = 255
    assert person_mask_evidence(outside, box) == (0, 0.0)

    assert effective_threshold(3000, 960, 540) == 3000
    assert effective_threshold(3000, 640, 360) == 1334
    assert cooldown_state(7.99, 5.0, 3.0)[0] is False
    assert cooldown_state(8.0, 5.0, 3.0)[0] is True
    assert cooldown_state(0.0, None, 3.0)[0] is True
    assert to_original_box(box, 960, 540, 1920, 1080) == Box(200, 200, 600, 600)


# ---------------------------------------------------------------------------
# person_mask_evidence / qualifies_person (SPEC.md 31.2, 31.3)
# ---------------------------------------------------------------------------


def test_person_mask_evidence_counts_only_inside_box() -> None:
    """박스 내부 255 픽셀만 세고 영역 밖 255 는 무시한다 (SPEC.md 31.2)."""
    mask = np.zeros((240, 320), dtype=np.uint8)
    box = Box(10, 10, 110, 110)  # area 10000
    mask[20:30, 20:70] = 255  # 10 x 50 = 500
    mask[200:240, 200:320] = 255  # 박스 밖 대형 전경
    pixels, ratio = person_mask_evidence(mask, box)
    assert pixels == 500
    assert ratio == 500 / 10000


def test_person_mask_evidence_rejects_invalid_inputs() -> None:
    """비정상 마스크/박스는 ValueError (SPEC.md 31.2)."""
    mask = np.zeros((240, 320), dtype=np.uint8)
    with pytest.raises(ValueError):
        person_mask_evidence(np.zeros((240, 320), dtype=np.int32), Box(0, 0, 10, 10))
    with pytest.raises(ValueError):
        person_mask_evidence(np.zeros((240, 320, 3), dtype=np.uint8), Box(0, 0, 10, 10))
    # x2 가 마스크 폭을 넘는다
    with pytest.raises(ValueError):
        person_mask_evidence(mask, Box(0, 0, 999, 10))


def test_qualifies_person_boundary_and_reasons() -> None:
    """경계 포함 판정과 거부 사유 누적 (SPEC.md 31.3, 14.4)."""
    # 모든 조건 충족 (경계값 P=3000, R=0.03 포함)
    ok, reasons = qualifies_person(0.60, 3000, 0.03, False)
    assert ok is True
    assert reasons == ()
    # warmup 만 위반
    ok, reasons = qualifies_person(0.90, 6000, 0.15, True)
    assert ok is False
    assert reasons == ("WARMUP",)
    # 여러 사유 동시 기록
    ok, reasons = qualifies_person(0.59, 2000, 0.02, True)
    assert ok is False
    assert set(reasons) == {
        "WARMUP",
        "LOW_CONFIDENCE",
        "MOTION_PIXELS_BELOW_MIN",
        "MOTION_RATIO_BELOW_MIN",
    }
    # 잘못된 입력 값
    with pytest.raises(ValueError):
        qualifies_person(1.5, 100, 0.1, False)
    with pytest.raises(ValueError):
        qualifies_person(0.9, -1, 0.1, False)
    with pytest.raises(ValueError):
        qualifies_person(0.9, 100, 1.2, False)
    with pytest.raises(ValueError):
        qualifies_person(float("nan"), 100, 0.1, False)


# ---------------------------------------------------------------------------
# U12: 해상도 보정 (SPEC.md 9.4, 31.1)
# ---------------------------------------------------------------------------


def test_u12_effective_threshold_resolution_scaling() -> None:
    """U12: 960x540 -> 3000, 640x360 -> 1334 등 §9.4 표 재현."""
    assert effective_threshold(3000, 960, 540) == 3000          # 계수 1.0
    assert effective_threshold(3000, 640, 360) == 1334          # 약 0.4444
    assert effective_threshold(3000, 480, 270) == 750           # 0.25
    assert effective_threshold(3000, 960, 720) == 4000          # 약 1.3333
    # 최소값 1 보장
    assert effective_threshold(1, 10, 10) == 1
    # 잘못된 인수
    with pytest.raises(ValueError):
        effective_threshold(0, 960, 540)
    with pytest.raises(ValueError):
        effective_threshold(100, 0, 540)
    with pytest.raises(ValueError):
        effective_threshold(100, 960, 0)


# ---------------------------------------------------------------------------
# U13: 원본 좌표 변환 (SPEC.md 11.2, 31.5)
# ---------------------------------------------------------------------------


def test_u13_to_original_box_exact_double_scale() -> None:
    """U13: 원본 1920x1080 / 분석 960x540 에서 좌표가 정확히 2배."""
    assert to_original_box(
        Box(100, 50, 200, 250), 960, 540, 1920, 1080
    ) == Box(200, 100, 400, 500)
    assert to_original_box(
        Box(100, 100, 300, 300), 960, 540, 1920, 1080
    ) == Box(200, 200, 600, 600)
    # 경계: 분석 박스가 프레임 끝에 닿으면 원본 경계로 clip
    assert to_original_box(
        Box(0, 0, 960, 540), 960, 540, 1920, 1080
    ) == Box(0, 0, 1920, 1080)


def test_u13_to_original_box_floor_ceil_and_validation() -> None:
    """좌변 floor / 우변 ceil 변환과 입력 검증 (SPEC.md 31.5)."""
    # 1920/959 처럼 정수 배가 아닌 축척: x1 은 floor, x2 는 ceil.
    scaled = to_original_box(Box(100, 100, 301, 301), 959, 539, 1920, 1080)
    assert scaled.x1 == math.floor(100 * 1920 / 959)
    assert scaled.x2 == math.ceil(301 * 1920 / 959)
    assert scaled.y1 == math.floor(100 * 1080 / 539)
    assert scaled.y2 == math.ceil(301 * 1080 / 539)
    # 범위 밖 박스는 ValueError
    with pytest.raises(ValueError):
        to_original_box(Box(-1, 0, 10, 10), 960, 540, 1920, 1080)
    with pytest.raises(ValueError):
        to_original_box(Box(0, 0, 961, 10), 960, 540, 1920, 1080)
    with pytest.raises(ValueError):
        to_original_box(Box(0, 0, 10, 10), 0, 540, 1920, 1080)
    # 0면적 분석 박스
    with pytest.raises(ValueError):
        to_original_box(Box(5, 5, 5, 10), 960, 540, 1920, 1080)


# ---------------------------------------------------------------------------
# U14/U15: 쿨다운 경계 (SPEC.md 31.4, 16.2)
# ---------------------------------------------------------------------------


def test_u14_u15_cooldown_boundary() -> None:
    """U14: 5.0 저장 후 7.99 억제. U15: 8.0 허용. 경계는 >= 다."""
    # U14 (SPEC.md 27.1)
    allowed, remaining = cooldown_state(7.99, 5.0, 3.0)
    assert allowed is False
    assert remaining == pytest.approx(0.01, abs=1e-9)
    # U15 (SPEC.md 27.1)
    allowed, remaining = cooldown_state(8.0, 5.0, 3.0)
    assert allowed is True
    assert remaining == 0.0
    # cooldown 0 이면 항상 허용 (SPEC.md 16.2)
    assert cooldown_state(5.0, 5.0, 0.0) == (True, 0.0)
    # 이전 성공 없으면 허용
    assert cooldown_state(0.0, None, 3.0) == (True, 0.0)


def test_cooldown_state_rejects_backwards_and_invalid() -> None:
    """시간 역행·비유한 수는 ValueError (SPEC.md 31.4)."""
    with pytest.raises(ValueError):
        cooldown_state(4.99, 5.0, 3.0)
    with pytest.raises(ValueError):
        cooldown_state(-1.0, None, 3.0)
    with pytest.raises(ValueError):
        cooldown_state(1.0, None, -3.0)
    with pytest.raises(ValueError):
        cooldown_state(1.0, -2.0, 3.0)
    with pytest.raises(ValueError):
        cooldown_state(float("nan"), None, 3.0)


@pytest.mark.parametrize("fps", [24, 25, 30, 60])
def test_cfr_cooldown_exact_frame_boundary(fps: int) -> None:
    for first in range(2 * fps, 20 * fps):
        assert cooldown_state((first + 3 * fps) / fps, first / fps, 3.0) == (True, 0.0)
        assert cooldown_state((first + 3 * fps - 1) / fps, first / fps, 3.0)[0] is False


def test_custom_reference_resolution() -> None:
    assert effective_threshold(3000, 1280, 720,
        reference_width=1280, reference_height=720) == 3000
    assert effective_threshold(3000, 640, 360,
        reference_width=1280, reference_height=720) == 750


# ---------------------------------------------------------------------------
# make_packet (SPEC.md 11.1, 11.3)
# ---------------------------------------------------------------------------


def test_make_packet_downscale_scale_and_no_mutation() -> None:
    """축소 시 분석 크기/scale 계산과 원본 배열 미수정 (SPEC.md 11.1, 8.1)."""
    raw = np.full((1080, 1920, 3), 137, dtype=np.uint8)
    raw_before = raw.copy()
    packet = make_packet("run", 7, 3.5, raw, 960)

    assert packet.analysis_frame.shape == (540, 960, 3)
    assert packet.analysis_frame.dtype == np.uint8
    assert packet.analysis_frame is not packet.raw_frame
    assert packet.scale_x == 1920 / 960
    assert packet.scale_y == 1080 / 540
    assert packet.frame_index == 7
    assert packet.video_time_sec == 3.5
    assert packet.run_id == "run"
    # 호출자가 넘긴 원본 프레임은 수정되지 않는다 (SPEC.md 8.1).
    np.testing.assert_array_equal(raw, raw_before)

    # packet 을 통한 좌표 변환은 2배 축척을 따른다 (U13 연계).
    assert box_to_original(Box(100, 50, 200, 250), packet) == Box(200, 100, 400, 500)


def test_make_packet_same_size_shares_array() -> None:
    """분석 크기가 원본과 같으면 analysis_frame 이 raw_frame 을 공유한다 (SPEC.md 11.3)."""
    raw = np.full((360, 640, 3), 90, dtype=np.uint8)
    packet = make_packet("run", 0, 0.0, raw, 960)  # 작은 원본은 확대하지 않는다
    assert packet.analysis_frame is raw
    assert packet.scale_x == 1.0
    assert packet.scale_y == 1.0
    assert packet.original_size == (640, 360)
    assert packet.analysis_size == (640, 360)


def test_make_packet_odd_ratio_keeps_separate_scales() -> None:
    """960 으로 나누어떨어지지 않는 원본은 scale_x/scale_y 가 다를 수 있다 (SPEC.md 11.1)."""
    raw = np.zeros((333, 1000, 3), dtype=np.uint8)
    packet = make_packet("run", 0, 0.0, raw, 960)
    assert packet.analysis_frame.shape == (320, 960, 3)  # round(333*960/1000)
    assert packet.scale_x == 1000 / 960
    assert packet.scale_y == 333 / 320
    assert packet.scale_x != packet.scale_y


def test_make_packet_rejects_invalid_raw_frame() -> None:
    """raw_frame 계약 위반은 ContractError (SPEC.md 8.1)."""
    from contracts import ContractError

    with pytest.raises(ContractError):
        make_packet("run", 0, 0.0, np.zeros((100, 200), dtype=np.uint8), 960)
    with pytest.raises(ContractError):
        make_packet("run", 0, 0.0, np.zeros((100, 200, 4), dtype=np.uint8), 960)
    with pytest.raises(ContractError):
        make_packet("run", 0, 0.0, np.zeros((100, 200, 3), dtype=np.float32), 960)
    with pytest.raises(ContractError):
        make_packet("run", 0, 0.0, np.zeros((100, 200, 3), dtype=np.uint8), 0)
