"""EventDetector 결합 판정 테스트 (SPEC.md 14절, 15.1, 27.1 U01~U11, U19).

커버리지 (SPEC ID):
- U01~U04, U07~U11 상태·후보·계약 예외.
- U05/U06 경계값 (픽셀 수 == 최소, 비율 == 최소).
- U19: warmup 경계 t=2.0 -> warming_up False (t < warmup_sec 규칙).
- §14.3 수치 예시 5행 재현 (분석 960x540, 최소 3000, 비율 0.03).
- §14.5: 사람 바깥 전경 합산 금지·두 사람 독립 계산.
- SPEC.md 8.1: evaluate 는 입력 배열/마스크를 수정하지 않는다.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config import Config
from contracts import (
    Box,
    ContractError,
    FramePacket,
    MotionResult,
    PersonDetection,
)
from event_detector import EventDetector
from frame_processor import make_packet

ANALYSIS_W, ANALYSIS_H = 960, 540


def make_config(**overrides) -> Config:
    """EventDetector 판정에 쓰는 설정. 기본값은 §9.1 과 같다."""
    return replace(Config(), **overrides)


def make_packet_at(video_time_sec: float, frame_index: int = 0) -> FramePacket:
    """960x540 분석 프레임 packet. t 는 warmup(2.0) 이후로 가정한다."""
    raw = np.zeros((ANALYSIS_H, ANALYSIS_W, 3), dtype=np.uint8)
    return make_packet("run", frame_index, video_time_sec, raw, ANALYSIS_W)


def motion_of(mask: np.ndarray, warming_up: bool = False) -> MotionResult:
    """마스크로 MotionResult 를 만든다. 전경 수치는 마스크에서 계산한다."""
    foreground_pixels = int(np.count_nonzero(mask))
    height, width = mask.shape
    return MotionResult(
        valid_mask=mask,
        regions=(),
        foreground_pixels=foreground_pixels,
        frame_foreground_ratio=foreground_pixels / float(width * height),
        warming_up=warming_up,
    )


def person(idx: int, box: Box, confidence: float = 0.90) -> PersonDetection:
    return PersonDetection(detection_index=idx, box=box, confidence=confidence, class_id=0)


def mask_with_foreground(shape: tuple[int, int], box: Box, pixels: int) -> np.ndarray:
    """box 내부 위쪽부터 정확히 pixels 개의 255 픽셀을 채운 마스크를 만든다."""
    mask = np.zeros(shape, dtype=np.uint8)
    width = box.x2 - box.x1
    full_rows, remainder = divmod(pixels, width)
    mask[box.y1:box.y1 + full_rows, box.x1:box.x2] = 255
    if remainder:
        mask[box.y1 + full_rows, box.x1:box.x1 + remainder] = 255
    return mask


@pytest.fixture
def detector() -> EventDetector:
    return EventDetector(make_config())


# ---------------------------------------------------------------------------
# U01~U03: 기본 상태
# ---------------------------------------------------------------------------


def test_u01_no_person_no_motion_is_idle(detector: EventDetector) -> None:
    """U01: 사람 결과 없음 + 빈 마스크 -> IDLE, 후보 false."""
    packet = make_packet_at(5.0)
    decision = detector.evaluate(packet, (), motion_of(np.zeros((540, 960), np.uint8)))
    assert decision.status == "IDLE"
    assert decision.candidate is False
    assert decision.persons == ()
    assert decision.frame_index == packet.frame_index
    assert decision.video_time_sec == 5.0


def test_config_reference_resolution_reaches_person_evaluation() -> None:
    packet = make_packet_at(5.0)
    box = Box(100, 100, 300, 300)
    mask = mask_with_foreground((540, 960), box, 2000)
    persons = (person(0, box),)
    assert not EventDetector(Config()).evaluate(packet, persons, motion_of(mask)).candidate
    custom = make_config(reference_width=1280, reference_height=720)
    assert EventDetector(custom).evaluate(packet, persons, motion_of(mask)).candidate


def test_u02_person_without_motion_is_person_only(detector: EventDetector) -> None:
    """U02: 사람 있음 + 빈 마스크 -> PERSON_ONLY, 후보 false."""
    packet = make_packet_at(5.0)
    box = Box(100, 100, 300, 300)
    decision = detector.evaluate(packet, (person(0, box),), motion_of(np.zeros((540, 960), np.uint8)))
    assert decision.status == "PERSON_ONLY"
    assert decision.candidate is False
    evidence = decision.persons[0]
    assert evidence.motion_pixels == 0
    assert evidence.motion_ratio == 0.0
    assert evidence.qualifies is False
    assert set(evidence.rejection_reasons) == {
        "MOTION_PIXELS_BELOW_MIN", "MOTION_RATIO_BELOW_MIN",
    }


def test_u03_motion_without_person_is_motion_only(detector: EventDetector) -> None:
    """U03: 사람 없음 + 전경 있음 -> MOTION_ONLY, 후보 false."""
    packet = make_packet_at(5.0)
    mask = np.zeros((540, 960), np.uint8)
    mask[400:500, 400:600] = 255
    decision = detector.evaluate(packet, (), motion_of(mask))
    assert decision.status == "MOTION_ONLY"
    assert decision.candidate is False


# ---------------------------------------------------------------------------
# U04 / §14.5: 사람 바깥 전경
# ---------------------------------------------------------------------------


def test_u04_foreground_outside_person_box_not_candidate(detector: EventDetector) -> None:
    """U04: 사람 박스 바깥에만 큰 전경 -> 후보 false (§14.5 합산 금지)."""
    packet = make_packet_at(5.0)
    person_box = Box(50, 50, 250, 250)
    mask = np.zeros((540, 960), np.uint8)
    mask[300:500, 600:900] = 255  # 200 x 300 = 60000 px 전부 사람 밖
    decision = detector.evaluate(packet, (person(0, person_box),), motion_of(mask))
    assert decision.candidate is False
    assert decision.status == "PERSON_AND_MOTION_UNMATCHED"
    assert decision.persons[0].motion_pixels == 0
    assert decision.persons[0].motion_ratio == 0.0


def test_14_5_two_persons_are_independent(detector: EventDetector) -> None:
    """§14.5: 두 사람은 독립 계산. 한쪽 전경을 다른 사람에게 합산하지 않는다."""
    packet = make_packet_at(5.0)
    left = Box(0, 100, 200, 300)        # area 40000
    right = Box(700, 100, 900, 300)     # area 40000
    mask = np.zeros((540, 960), np.uint8)
    mask[100:130, 700:900] = 255        # 오른쪽 사람 안에만 30 x 200 = 6000 px
    mask[400:480, 0:640] = 255          # 왼쪽 사람 밖 대형 전경 (80 x 640)
    decision = detector.evaluate(
        packet, (person(0, left), person(1, right)), motion_of(mask)
    )

    assert decision.candidate is True
    assert decision.status == "MOVING_PERSON"
    by_index = {e.detection_index: e for e in decision.persons}
    # 왼쪽 사람: 박스 안 전경 0 -> 면적/비율 미달
    assert by_index[0].motion_pixels == 0
    assert by_index[0].qualifies is False
    assert set(by_index[0].rejection_reasons) == {
        "MOTION_PIXELS_BELOW_MIN", "MOTION_RATIO_BELOW_MIN",
    }
    # 오른쪽 사람: 6000 px, ratio 0.15 -> 적격
    assert by_index[1].motion_pixels == 6000
    assert by_index[1].motion_ratio == 6000 / 40000
    assert by_index[1].qualifies is True
    assert by_index[1].rejection_reasons == ()
    # 화면 전체 전경이 왼쪽 사람 근거에 합산되지 않았다.
    assert by_index[0].motion_pixels < decision.persons[1].motion_pixels


# ---------------------------------------------------------------------------
# U05/U06 경계와 §14.3 수치 예시
# ---------------------------------------------------------------------------


def test_u05_pixels_equal_to_min_qualifies(detector: EventDetector) -> None:
    """U05: 픽셀 수가 최소값(3000)과 같고 나머지 조건 충족 -> true."""
    packet = make_packet_at(5.0)
    box = Box(100, 100, 300, 300)  # area 40000
    mask = mask_with_foreground((540, 960), box, 3000)
    decision = detector.evaluate(packet, (person(0, box),), motion_of(mask))
    assert decision.persons[0].motion_pixels == 3000
    assert decision.candidate is True
    assert decision.status == "MOVING_PERSON"


def test_u06_ratio_equal_to_min_qualifies(detector: EventDetector) -> None:
    """U06: 비율이 최소값(0.03)과 같고 나머지 조건 충족 -> true."""
    packet = make_packet_at(5.0)
    box = Box(0, 0, 400, 250)  # area 100000
    mask = mask_with_foreground((540, 960), box, 3000)
    decision = detector.evaluate(packet, (person(0, box),), motion_of(mask))
    assert decision.persons[0].motion_ratio == pytest.approx(0.03)
    assert decision.candidate is True


def test_14_3_numeric_example_rows(detector: EventDetector) -> None:
    """§14.3 수치 예시 5행을 그대로 재현한다 (960x540, min 3000, ratio 0.03)."""
    rows = [
        # (박스, 박스 면적, 내부 전경 픽셀, 비율, 후보)
        (Box(100, 100, 300, 300), 40000, 6000, 0.15, True),    # 면적/비율 만족
        (Box(100, 100, 300, 300), 40000, 2000, 0.05, False),   # 면적 불만족
        (Box(0, 0, 500, 400), 200000, 3500, 0.0175, False),    # 비율 불만족
        (Box(0, 0, 50, 100), 5000, 1000, 0.20, False),         # 면적 불만족
        (Box(0, 0, 400, 250), 100000, 3000, 0.03, True),       # 경계 포함
    ]
    for i, (box, area, pixels, ratio, expect_candidate) in enumerate(rows):
        assert box.area == area
        packet = make_packet_at(5.0, frame_index=i)
        mask = mask_with_foreground((540, 960), box, pixels)
        decision = detector.evaluate(packet, (person(i, box),), motion_of(mask))
        evidence = decision.persons[0]
        assert evidence.motion_pixels == pixels
        assert evidence.motion_ratio == pytest.approx(ratio)
        assert evidence.qualifies is expect_candidate
        assert decision.candidate is expect_candidate


def test_14_3_low_confidence_or_warmup_blocks_candidate(detector: EventDetector) -> None:
    """§14.3 각주: confidence 0.59 또는 warmup 중이면 조건 충족해도 후보 아니다."""
    box = Box(100, 100, 300, 300)
    mask = mask_with_foreground((540, 960), box, 6000)

    # confidence 0.59 vs 기준 0.60 (U07 과 같은 맥락)
    decision = detector.evaluate(
        make_packet_at(5.0, 0), (person(0, box, confidence=0.59),), motion_of(mask)
    )
    assert decision.candidate is False
    assert decision.persons[0].rejection_reasons == ("LOW_CONFIDENCE",)

    # warmup 중
    decision = detector.evaluate(
        make_packet_at(1.0, 1), (person(0, box),), motion_of(mask, warming_up=True)
    )
    assert decision.candidate is False
    assert decision.persons[0].rejection_reasons == ("WARMUP",)


# ---------------------------------------------------------------------------
# U07/U08/U19: confidence, warmup
# ---------------------------------------------------------------------------


def test_u07_confidence_059_below_threshold(detector: EventDetector) -> None:
    """U07: confidence 0.59, 기준 0.60 -> 후보 false."""
    packet = make_packet_at(5.0)
    box = Box(100, 100, 300, 300)
    mask = mask_with_foreground((540, 960), box, 6000)
    decision = detector.evaluate(
        packet, (person(0, box, confidence=0.59),), motion_of(mask)
    )
    assert decision.candidate is False
    assert decision.persons[0].confidence == 0.59
    assert "LOW_CONFIDENCE" in decision.persons[0].rejection_reasons


def test_u08_warmup_suppresses_candidate(detector: EventDetector) -> None:
    """U08: warmup 중 충분한 전경 -> WARMUP, 후보 false."""
    packet = make_packet_at(1.0)
    box = Box(100, 100, 300, 300)
    mask = mask_with_foreground((540, 960), box, 6000)
    decision = detector.evaluate(
        packet, (person(0, box),), motion_of(mask, warming_up=True)
    )
    assert decision.status == "WARMUP"
    assert decision.candidate is False
    # warmup 중에도 면적/비율 자체는 계산해 기록한다 (§13.7).
    assert decision.persons[0].motion_pixels == 6000


def test_u19_warmup_boundary_at_2_seconds(detector: EventDetector) -> None:
    """U19: warmup 경계 t=2.0 -> warming_up False (t < warmup_sec)."""
    config = make_config()  # warmup_sec 기본 2.0
    # 경계 규칙: warming_up = t < warmup_sec
    assert (2.0 < config.warmup_sec) is False

    # t = 2.0: warming_up False -> 조건 충족 시 후보 가능
    packet = make_packet_at(2.0)
    box = Box(100, 100, 300, 300)
    mask = mask_with_foreground((540, 960), box, 6000)
    decision = detector.evaluate(
        packet, (person(0, box),), motion_of(mask, warming_up=False)
    )
    assert decision.candidate is True
    assert decision.status == "MOVING_PERSON"

    # t = 1.999...: warming_up True -> WARMUP 억제
    packet = make_packet_at(1.9999, frame_index=1)
    decision = detector.evaluate(
        packet, (person(0, box),), motion_of(mask, warming_up=True)
    )
    assert decision.status == "WARMUP"
    assert decision.candidate is False


# ---------------------------------------------------------------------------
# U09: 여러 사람 중 일부 적격
# ---------------------------------------------------------------------------


def test_u09_two_boxes_only_one_qualifies(detector: EventDetector) -> None:
    """U09: 유효 박스 두 개 중 하나만 적격 -> 후보 true, 적격 수 1."""
    packet = make_packet_at(5.0)
    qualified = Box(100, 100, 300, 300)       # 6000 px 내부
    unqualified = Box(600, 100, 800, 300)     # 내부 전경 200 px
    mask = np.zeros((540, 960), np.uint8)
    mask[100:130, 100:300] = 255              # qualified 안 6000 px
    mask[100:101, 600:800] = 255              # unqualified 안 200 px
    decision = detector.evaluate(
        packet, (person(0, qualified), person(1, unqualified)), motion_of(mask)
    )
    assert decision.candidate is True
    assert decision.status == "MOVING_PERSON"
    qualified_evidences = [e for e in decision.persons if e.qualifies]
    assert len(qualified_evidences) == 1
    assert qualified_evidences[0].detection_index == 0
    assert decision.persons[1].qualifies is False
    # 200 px / 40000 면적 -> 비율 0.005 로 픽셀 수·비율 모두 미달
    assert decision.persons[1].motion_pixels == 200
    assert set(decision.persons[1].rejection_reasons) == {
        "MOTION_PIXELS_BELOW_MIN", "MOTION_RATIO_BELOW_MIN",
    }


# ---------------------------------------------------------------------------
# U10/U11: 계약 예외
# ---------------------------------------------------------------------------


def test_u10_mask_shape_mismatch_raises(detector: EventDetector) -> None:
    """U10: 마스크 크기가 analysis_frame 과 다름 -> ContractError."""
    packet = make_packet_at(5.0)
    wrong_mask = np.zeros((480, 960), dtype=np.uint8)
    with pytest.raises(ContractError):
        detector.evaluate(packet, (), motion_of(wrong_mask))


def test_u10_mask_dtype_mismatch_raises(detector: EventDetector) -> None:
    """U10 변형: 마스크가 uint8 2차원이 아니면 ContractError."""
    packet = make_packet_at(5.0)
    bad = np.zeros((540, 960), dtype=np.int32)
    with pytest.raises(ContractError):
        detector.evaluate(packet, (), motion_of(bad))


def test_u11_zero_area_and_out_of_range_boxes_raise(detector: EventDetector) -> None:
    """U11: 0면적이나 범위 밖 박스 -> ContractError (IDLE 로 처리하지 않음)."""
    packet = make_packet_at(5.0)
    mask = np.zeros((540, 960), dtype=np.uint8)
    zero_area = Box(100, 100, 100, 200)          # x1 == x2
    out_of_range = Box(0, 0, 2000, 200)          # x2 > 분석 폭 960
    out_of_range_y = Box(0, 0, 100, 9999)        # y2 > 분석 높이 540
    for i, bad_box in enumerate((zero_area, out_of_range, out_of_range_y)):
        with pytest.raises(ContractError):
            detector.evaluate(packet, (person(0, bad_box),), motion_of(mask))


# ---------------------------------------------------------------------------
# SPEC.md 8.1: 입력 배열 비수정
# ---------------------------------------------------------------------------


def test_evaluate_does_not_mutate_inputs(detector: EventDetector) -> None:
    """evaluate 는 호출자가 넘긴 프레임/마스크에 박스나 글자를 그리지 않는다."""
    packet = make_packet_at(5.0)
    raw_before = packet.raw_frame.copy()
    analysis_before = packet.analysis_frame.copy()
    box = Box(100, 100, 300, 300)
    mask = mask_with_foreground((540, 960), box, 6000)
    mask_before = mask.copy()

    detector.evaluate(packet, (person(0, box),), motion_of(mask))

    np.testing.assert_array_equal(packet.raw_frame, raw_before)
    np.testing.assert_array_equal(packet.analysis_frame, analysis_before)
    np.testing.assert_array_equal(mask, mask_before)


# ---------------------------------------------------------------------------
# SPEC 27.2: 같은 사람 박스 안의 분리된 작은 전경 성분 합산
# ---------------------------------------------------------------------------


def test_27_2_separated_small_components_sum_in_same_person_box(
    detector: EventDetector,
) -> None:
    """§27.2/§13.6: 각 성분이 최소 픽셀(3000) 미만이어도 같은 사람 박스 안이면 합산한다.

    person_mask_evidence 가 최대 연결 성분만 세도록 회귀하면 이 테스트가
    실패해야 한다 (2000 + 2310 = 4310).
    """
    mask = np.zeros((ANALYSIS_H, ANALYSIS_W), dtype=np.uint8)
    mask[110:130, 110:210] = 255   # 20 x 100 = 2000 px (< 3000)
    mask[250:271, 190:300] = 255   # 21 x 110 = 2310 px (< 3000), 첫 성분과 분리
    box = Box(100, 100, 300, 300)  # area = 40000

    packet = make_packet_at(5.0)
    decision = detector.evaluate(packet, (person(0, box),), motion_of(mask))

    evidence = decision.persons[0]
    assert evidence.motion_pixels == 2000 + 2310
    assert evidence.motion_ratio == pytest.approx(4310 / 40000)
    assert evidence.qualifies is True
    assert decision.candidate is True
    assert decision.status == "MOVING_PERSON"
