"""Pure image geometry and area thresholds."""

from __future__ import annotations
import math
from motion_person.domain.contracts import Box, ContractError, FramePacket


def effective_threshold(
    reference_pixels: int,
    width: int,
    height: int,
    *,
    reference_width: int = 960,
    reference_height: int = 540,
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
        if (
            len(roi) != 4
            or any(not math.isfinite(v) for v in roi)
            or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)
        ):
            raise ContractError("invalid normalized analysis ROI")
        pixels = (
            math.floor(roi[0] * original_w),
            math.floor(roi[1] * original_h),
            math.ceil(roi[2] * original_w),
            math.ceil(roi[3] * original_h),
        )
    x1, y1, x2, y2 = pixels
    width = min(x2 - x1, analysis_width)
    height = max(1, round((y2 - y1) * width / (x2 - x1)))
    return pixels, width, height
