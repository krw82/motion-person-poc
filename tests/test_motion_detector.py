"""MotionDetector(MOG2) 테스트 (SPEC.md 13절, 27.2).

커버리지 (SPEC ID):
- §27.2: 동일 배경 반복 후 유효 전경 감소.
- §27.2: 큰 사각형 등장 시 전경 영역 발생.
- §27.2: clean_mask 에 127 그림자 값 직접 입력 -> 제거 확인.
- §27.2: min_component_pixels 미만 점 직접 입력 -> 제거 확인.
- §13.2: detect 중 shape 변경 -> FrameShapeError.
- §13.3/FR08: valid_mask 값은 {0, 255} 만 존재.
- §13.7: warming_up 플래그 (t < warmup_sec).
- FR06: detect 프레임당 subtractor.apply 정확히 1회.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from config import Config
from contracts import FrameShapeError
from motion_detector import MotionDetector, clean_mask

FRAME_W, FRAME_H = 160, 120
BACKGROUND_VALUE = 100
RECT_X1, RECT_Y1, RECT_X2, RECT_Y2 = 50, 30, 110, 90  # 60 x 80 사각형


def make_config(**overrides) -> Config:
    return replace(Config(), **overrides)


def background_frame() -> np.ndarray:
    return np.full((FRAME_H, FRAME_W, 3), BACKGROUND_VALUE, dtype=np.uint8)


def rectangle_frame() -> np.ndarray:
    frame = background_frame()
    frame[RECT_Y1:RECT_Y2, RECT_X1:RECT_X2] = 220
    return frame


def settle_background(detector: MotionDetector, frames: int = 10) -> None:
    """동일 배경 프레임을 frames 장 넣어 배경 모델을 안정화한다."""
    bg = background_frame()
    for i in range(frames):
        detector.detect(bg, video_time_sec=0.1 * i + 0.05)


# ---------------------------------------------------------------------------
# §27.2: 동일 배경 반복 후 유효 전경 감소
# ---------------------------------------------------------------------------


def test_repeated_background_reduces_valid_foreground() -> None:
    """§27.2: 반복된 동일 장면은 배경에 흡수되어 유효 전경이 줄어든다."""
    detector = MotionDetector(make_config())
    settle_background(detector, frames=10)

    # 사각형이 등장한 첫 프레임: 큰 전경 발생
    peak = detector.detect(rectangle_frame(), video_time_sec=2.0)
    assert peak.foreground_pixels > 1000
    assert peak.foreground_pixels == int(np.count_nonzero(peak.valid_mask))

    # 같은 장면(사각형이 멈춰 있는 프레임)을 계속 넣으면 전경이 감소한다.
    scene = rectangle_frame()
    last = peak
    for i in range(30):
        last = detector.detect(scene, video_time_sec=2.1 + 0.1 * i)
    assert last.foreground_pixels < peak.foreground_pixels
    assert last.foreground_pixels <= peak.foreground_pixels // 2
    assert last.frame_foreground_ratio == pytest.approx(
        last.foreground_pixels / float(FRAME_W * FRAME_H)
    )


# ---------------------------------------------------------------------------
# §27.2: 큰 사각형 등장 시 전경 영역
# ---------------------------------------------------------------------------


def test_large_rectangle_produces_foreground_region() -> None:
    """§27.2: 안정된 배경에 큰 사각형이 나타나면 전경 영역이 생긴다."""
    detector = MotionDetector(make_config())
    settle_background(detector, frames=10)

    settled = detector.detect(background_frame(), video_time_sec=2.0)
    result = detector.detect(rectangle_frame(), video_time_sec=2.1)

    assert result.foreground_pixels > settled.foreground_pixels
    assert result.regions, "전경 영역 박스가 최소 1개 있어야 한다"
    # 영역 박스가 사각형 위치와 겹친다 (SPEC.md 13.6 boundingRect).
    overlapping = [
        region for region in result.regions
        if region.box.x1 < RECT_X2 and region.box.x2 > RECT_X1
        and region.box.y1 < RECT_Y2 and region.box.y2 > RECT_Y1
    ]
    assert overlapping
    for region in overlapping:
        assert region.contour_area > 0.0
    # 유효 마스크의 255 픽셀은 사각형 내부에 집중한다 (모폴로지 여유 ±3px).
    ys, xs = np.nonzero(result.valid_mask)
    tol = 3
    assert xs.min() >= RECT_X1 - tol
    assert xs.max() < RECT_X2 + tol
    assert ys.min() >= RECT_Y1 - tol
    assert ys.max() < RECT_Y2 + tol


def test_detect_does_not_mutate_input_frame() -> None:
    """detect 는 호출자가 넘긴 프레임을 수정하지 않는다 (SPEC.md 8.1)."""
    detector = MotionDetector(make_config())
    frame = rectangle_frame()
    before = frame.copy()
    detector.detect(frame, video_time_sec=0.5)
    np.testing.assert_array_equal(frame, before)


# ---------------------------------------------------------------------------
# §27.2: clean_mask 직접 입력 검증
# ---------------------------------------------------------------------------


def test_clean_mask_removes_shadow_value_127() -> None:
    """§27.2: 127 그림자 값이 마스크 정리에서 제거되는지 직접 입력 확인."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255   # 실제 전경 성분 (10000 px)
    mask[50:150, 170:195] = 127  # MOG2 그림자 표식 (100 x 25)
    before = mask.copy()

    cleaned = clean_mask(mask, open_kernel=3, close_kernel=5, iterations=1,
                         min_component_pixels=100)

    # 입력 마스크는 수정되지 않는다.
    np.testing.assert_array_equal(mask, before)
    # 127 영역은 전경에서 제외된다 (SPEC.md 13.3: == 255 만 전경).
    assert np.all(cleaned[50:150, 170:195] == 0)
    # 255 성분은 유지된다.
    assert np.count_nonzero(cleaned[50:150, 50:150]) > 0
    assert cleaned.dtype == np.uint8


def test_clean_mask_removes_small_components() -> None:
    """§27.2: min_component_pixels 미만 점이 제거되는지 직접 입력 확인."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[10:110, 10:110] = 255    # 큰 성분 (10000 px >= 100)
    mask[150:155, 150:155] = 255  # 작은 점 (25 px < 100)

    cleaned = clean_mask(mask, open_kernel=3, close_kernel=5, iterations=1,
                         min_component_pixels=100)

    assert np.all(cleaned[140:170, 140:170] == 0), "작은 점은 제거되어야 한다"
    assert np.count_nonzero(cleaned[10:110, 10:110]) > 0, "큰 성분은 유지된다"


def test_clean_mask_output_is_binary() -> None:
    """clean_mask 출력 값은 0 과 255 만 있다 (FR08)."""
    rng = np.random.default_rng(7)
    mask = (rng.random((120, 160)) * 256).astype(np.uint8)  # 잡동사니 값 포함
    mask[40:80, 40:120] = 255
    cleaned = clean_mask(mask, open_kernel=3, close_kernel=5, iterations=1,
                         min_component_pixels=50)
    assert set(np.unique(cleaned)) <= {0, 255}


def test_clean_mask_rejects_invalid_input() -> None:
    """clean_mask 입력 계약 위반은 MotionError."""
    from contracts import MotionError

    with pytest.raises(MotionError):
        clean_mask(np.zeros((100, 100), dtype=np.int32), 3, 5, 1, 10)
    with pytest.raises(MotionError):
        clean_mask(np.zeros((100, 100, 3), dtype=np.uint8), 3, 5, 1, 10)


# ---------------------------------------------------------------------------
# §13.2: shape 변경 / §13.7: warming_up / FR06: apply 1회
# ---------------------------------------------------------------------------


def test_detect_shape_change_raises_frame_shape_error() -> None:
    """§13.2: 실행 중 프레임 shape 가 바뀌면 FrameShapeError."""
    detector = MotionDetector(make_config())
    detector.detect(np.full((48, 64, 3), 80, np.uint8), video_time_sec=0.1)
    with pytest.raises(FrameShapeError):
        detector.detect(np.full((24, 32, 3), 80, np.uint8), video_time_sec=0.2)


def test_valid_mask_values_are_only_0_and_255() -> None:
    """FR08: 어떤 입력에서도 valid_mask 값은 {0, 255} 만 존재한다."""
    detector = MotionDetector(make_config())  # detectShadows=True 기본
    settle_background(detector, frames=10)
    result = detector.detect(rectangle_frame(), video_time_sec=2.0)
    assert set(np.unique(result.valid_mask)) <= {0, 255}
    assert result.valid_mask.dtype == np.uint8
    assert result.valid_mask.shape == (FRAME_H, FRAME_W)


def test_warming_up_flag_follows_warmup_boundary() -> None:
    """§13.7: warming_up = t < warmup_sec. t=1.0 True, t=2.0 False."""
    detector = MotionDetector(make_config())  # warmup_sec 기본 2.0
    frame = background_frame()
    warming_1 = detector.detect(frame, video_time_sec=1.0)
    assert warming_1.warming_up is True
    warming_2 = detector.detect(frame, video_time_sec=2.0)
    assert warming_2.warming_up is False


class _ApplyCountingSubtractor:
    """subtractor.apply 호출 수를 세는 래퍼 (FR06 검증용)."""

    def __init__(self, inner: cv2.BackgroundSubtractorMOG2) -> None:
        self._inner = inner
        self.apply_calls = 0

    def apply(self, image: np.ndarray, learningRate: float | None = None) -> np.ndarray:
        self.apply_calls += 1
        return self._inner.apply(image, learningRate=learningRate)


def test_detect_calls_subtractor_apply_exactly_once() -> None:
    """FR06: detect 한 번에 MOG2 apply 는 정확히 1회다."""
    detector = MotionDetector(make_config())
    wrapper = _ApplyCountingSubtractor(detector._subtractor)
    detector._subtractor = wrapper

    frame = background_frame()
    detector.detect(frame, video_time_sec=0.5)
    detector.detect(frame, video_time_sec=0.6)

    assert wrapper.apply_calls == 2  # 프레임 2회 * 정확히 1회
