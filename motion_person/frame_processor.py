"""분석 리사이즈와 원본 좌표 변환 (SPEC.md 11절, 31.1, 31.5).

이 모듈은 상태 없는 순수 함수만 제공한다. 원본 프레임을 비율 유지
축소로 분석 프레임을 만들고, 분석 좌표계의 박스를 원본 좌표계로
되돌린다. 배열 입력은 읽기만 하며 박스나 글자를 그리지 않는다
(SPEC.md 8.1, 11.3).

분석 크기 규칙 (SPEC.md 11.1):
- analysis_w = min(original_w, configured_analysis_width)
  작은 원본을 960까지 확대하지 않는다.
- analysis_h = max(1, round(original_h * analysis_w / original_w))
  960으로 나누어떨어지지 않는 원본에서는 높이 반올림 때문에
  scale_x 와 scale_y 가 약간 달라질 수 있어 하나로 통합하지 않는다.
- 축소가 필요할 때만 cv2.resize(INTER_AREA) 를 호출하고, 분석 크기가
  원본과 같으면 analysis_frame 이 raw_frame 배열을 공유한다 (11.3).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .contracts import Box, ContractError, FramePacket, FrameShapeError

def effective_threshold(
    reference_pixels: int, width: int, height: int, *,
    reference_width: int = 960, reference_height: int = 540,
) -> int:
    """설정한 기준 해상도(기본 960 x 540)의 면적값을 분석 크기로 보정한다.

    실제 분석 면적을 기준 면적으로 나눈 계수를 적용한다. 정수 연산으로
    ceil을 계산하며 최소 1을 보장한다.
    """
    if min(reference_pixels, width, height, reference_width, reference_height) <= 0:
        raise ValueError("positive area and dimensions required")
    numerator = reference_pixels * width * height
    denominator = reference_width * reference_height
    return max(1, (numerator + denominator - 1) // denominator)


def to_original_box(
    box: Box,
    analysis_w: int,
    analysis_h: int,
    original_w: int,
    original_h: int,
) -> Box:
    """분석 좌표계 박스를 원본 좌표계 박스로 변환한다 (SPEC.md 31.5).

    왼쪽과 위는 floor, 오른쪽과 아래는 ceil 로 변환한 뒤 원본 이미지
    범위로 clip한다. 예: 원본 1920 x 1080, 분석 960 x 540 에서 분석 박스
    [100, 50, 200, 250] 은 원본 [200, 100, 400, 500] 이다 (SPEC.md 11.2).
    """
    if min(analysis_w, analysis_h, original_w, original_h) <= 0:
        raise ValueError("invalid image dimensions")
    if not (0 <= box.x1 < box.x2 <= analysis_w):
        raise ValueError("invalid analysis x coordinates")
    if not (0 <= box.y1 < box.y2 <= analysis_h):
        raise ValueError("invalid analysis y coordinates")
    sx, sy = original_w / analysis_w, original_h / analysis_h
    return Box(
        max(0, min(original_w, math.floor(box.x1 * sx))),
        max(0, min(original_h, math.floor(box.y1 * sy))),
        max(0, min(original_w, math.ceil(box.x2 * sx))),
        max(0, min(original_h, math.ceil(box.y2 * sy))),
    )


def _validate_raw_frame(raw_frame: np.ndarray) -> tuple[int, int]:
    """raw_frame 계약을 검사하고 (height, width) 를 반환한다 (SPEC.md 8.1).

    raw_frame 은 H x W x 3, uint8, BGR 순서이며 h >= 1, w >= 1 이어야
    한다. 위반하면 ContractError 를 발생시킨다.
    """
    if not isinstance(raw_frame, np.ndarray):
        raise ContractError(
            "raw_frame must be a numpy ndarray",
            context={"type": type(raw_frame).__name__},
        )
    if raw_frame.ndim != 3:
        raise ContractError(
            "raw_frame must be a 3-dimensional H x W x 3 array",
            context={"ndim": int(raw_frame.ndim)},
        )
    if raw_frame.dtype != np.uint8:
        raise ContractError(
            "raw_frame dtype must be uint8",
            context={"dtype": str(raw_frame.dtype)},
        )
    height, width, channels = raw_frame.shape
    if height < 1 or width < 1:
        raise ContractError(
            "raw_frame height and width must be >= 1",
            context={"height": int(height), "width": int(width)},
        )
    if channels != 3:
        raise ContractError(
            "raw_frame must have 3 BGR color channels",
            context={"channels": int(channels)},
        )
    return int(height), int(width)


def make_packet(
    run_id: str,
    frame_index: int,
    video_time_sec: float,
    raw_frame: np.ndarray,
    analysis_width: int,
    analysis_roi: tuple[float, float, float, float] | None = None,
) -> FramePacket:
    """원본 프레임으로 FramePacket 을 만든다 (SPEC.md 11.1).

    analysis_w = min(original_w, analysis_width), analysis_h =
    max(1, round(original_h * analysis_w / original_w)) 로 분석 크기를
    정한다. 축소 시에만 cv2.resize(INTER_AREA) 로 새 배열을 만들고 크기가
    같으면 raw_frame 배열을 공유한다 (SPEC.md 11.3). scale_x 와 scale_y
    는 각각 계산하며 하나의 값으로 대체하지 않는다.

    입력 배열은 수정하지 않고 raw_frame 을 그대로 packet 에 담는다.
    계약 위반(배열 형태, dtype, 크기, analysis_width)은 ContractError 다.
    """
    original_h, original_w = _validate_raw_frame(raw_frame)

    if (
        not isinstance(analysis_width, (int, np.integer))
        or isinstance(analysis_width, bool)
    ):
        raise ContractError(
            "analysis_width must be an integer",
            context={"analysis_width": repr(analysis_width)},
        )
    if analysis_width < 1:
        raise ContractError(
            "analysis_width must be >= 1",
            context={"analysis_width": int(analysis_width)},
        )

    roi, analysis_w, analysis_h = analysis_geometry(original_w, original_h, int(analysis_width), analysis_roi)
    x1, y1, x2, y2 = roi
    input_frame = raw_frame[y1:y2, x1:x2]
    input_w, input_h = x2 - x1, y2 - y1

    if analysis_w == input_w and analysis_h == input_h:
        # 분석 크기가 원본과 같으면 배열을 공유한다 (SPEC.md 11.3).
        # 분석 함수는 입력을 수정하지 않는다.
        analysis_frame = input_frame if analysis_roi is not None else raw_frame
    else:
        try:
            analysis_frame = cv2.resize(
                input_frame,
                (analysis_w, analysis_h),
                interpolation=cv2.INTER_AREA,
            )
        except Exception as exc:
            # 리사이즈 실패는 프레임 형태 문제로 종료한다(SPEC.md 21절).
            raise FrameShapeError(
                "cannot build analysis frame (resize failed)",
                context={
                    "original_size": [original_w, original_h],
                    "analysis_size": [analysis_w, analysis_h],
                    "reason": repr(exc),
                },
            ) from exc

    scale_x = input_w / analysis_w
    scale_y = input_h / analysis_h

    return FramePacket(
        run_id=run_id,
        frame_index=frame_index,
        video_time_sec=video_time_sec,
        raw_frame=raw_frame,
        analysis_frame=analysis_frame,
        scale_x=scale_x,
        scale_y=scale_y,
        roi_xyxy=roi if analysis_roi is not None else None,
    )


def box_to_original(box: Box, packet: FramePacket) -> Box:
    """packet 의 크기와 축척 정보로 박스를 원본 좌표계로 변환한다 (SPEC.md 11.2).

    to_original_box 에 packet.analysis_size 와 packet.original_size 를
    전달해 위임한다.
    """
    analysis_w, analysis_h = packet.analysis_size
    original_w, original_h = packet.original_size
    if packet.roi_xyxy is not None:
        x1, y1, x2, y2 = packet.roi_xyxy
        local = to_original_box(box, analysis_w, analysis_h, x2 - x1, y2 - y1)
        return Box(local.x1 + x1, local.y1 + y1, local.x2 + x1, local.y2 + y1)
    return to_original_box(box, analysis_w, analysis_h, original_w, original_h)


def analysis_geometry(original_w, original_h, analysis_width, roi=None):
    """Return pixel ROI and aspect-preserving analysis size for every consumer."""
    if min(original_w, original_h, analysis_width) <= 0:
        raise ContractError("positive image dimensions required")
    if roi is None:
        pixels = (0, 0, original_w, original_h)
    else:
        if (len(roi) != 4 or any(not math.isfinite(v) for v in roi)
                or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)):
            raise ContractError("invalid normalized analysis ROI")
        pixels = (math.floor(roi[0] * original_w), math.floor(roi[1] * original_h),
                  math.ceil(roi[2] * original_w), math.ceil(roi[3] * original_h))
    x1, y1, x2, y2 = pixels
    width = min(x2 - x1, analysis_width)
    height = max(1, round((y2 - y1) * width / (x2 - x1)))
    return pixels, width, height
