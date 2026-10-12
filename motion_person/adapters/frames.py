"""Frame validation and OpenCV resize; pure geometry is in domain.geometry."""

from __future__ import annotations
import cv2
import numpy as np
from motion_person.domain.contracts import ContractError, FramePacket, FrameShapeError
from motion_person.domain.geometry import (
    effective_threshold as effective_threshold,
    to_original_box as to_original_box,
    box_to_original as box_to_original,
    analysis_geometry,
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

    if not isinstance(analysis_width, (int, np.integer)) or isinstance(
        analysis_width, bool
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

    roi, analysis_w, analysis_h = analysis_geometry(
        original_w, original_h, int(analysis_width), analysis_roi
    )
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
