"""MOG2 전경 탐지 모듈 (SPEC.md 13절).

배경 차분(MOG2)으로 분석 프레임의 전경 마스크를 만들고, 모폴로지 정리와
작은 성분 제거를 거쳐 MotionResult 를 반환한다. 이전 프레임들의 배경
모델을 보관하는 상태 있는 모듈이다. 입력 프레임과 입력 마스크는 수정하지
않는다 (SPEC.md 8.1, 11.3).
"""

from __future__ import annotations

import cv2
import numpy as np

from config import Config
from contracts import (
    Box,
    FrameShapeError,
    MotionError,
    MotionRegion,
    MotionResult,
)
from frame_processor import effective_threshold


def _require_mask(binary_mask: np.ndarray) -> None:
    """clean_mask 입력 계약 검사. 2차원 uint8 마스크여야 한다."""
    if (
        not isinstance(binary_mask, np.ndarray)
        or binary_mask.ndim != 2
        or binary_mask.dtype != np.uint8
    ):
        raise MotionError(
            "binary_mask must be a 2D uint8 array",
            context={
                "ndim": int(getattr(binary_mask, "ndim", -1)),
                "dtype": str(getattr(binary_mask, "dtype", type(binary_mask).__name__)),
            },
        )


def clean_mask(
    binary_mask: np.ndarray,
    open_kernel: int,
    close_kernel: int,
    iterations: int,
    min_component_pixels: int,
) -> np.ndarray:
    """전경 마스크에서 노이즈를 제거한다 (SPEC.md 13.4).

    처리 순서는 다음과 같다.

    1. open_kernel 크기 타원 커널로 opening 을 iterations 회 적용해
       작은 점 노이즈를 지운다.
    2. close_kernel 크기 타원 커널로 closing 을 iterations 회 적용해
       내부 구멍과 끊긴 영역을 붙인다.
    3. connectedComponentsWithStats(connectivity=8) 로 성분별 실제 픽셀
       수를 구한다.
    4. CC_STAT_AREA 가 min_component_pixels 보다 작은 성분을 제거하고
       남은 성분만 0/255 uint8 마스크로 반환한다. index 0 은 배경
       성분이므로 유지하지 않는다.

    입력 값 중 255 만 전경으로 본다. 127 그림자 표식 등 다른 값은 배경과
    같이 버린다 (SPEC.md 13.3). 윤곽선 채우기로 마스크를 재생성하지 않는다.
    입력 배열은 수정하지 않는다.
    """
    _require_mask(binary_mask)
    try:
        binary = np.where(binary_mask == 255, 255, 0).astype(np.uint8)
        open_element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        close_element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        opened = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, open_element, iterations=iterations
        )
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, close_element, iterations=iterations
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )
        keep = np.zeros(count, dtype=np.uint8)
        keep[1:] = (
            stats[1:, cv2.CC_STAT_AREA] >= min_component_pixels
        ).astype(np.uint8)
        return (keep[labels] * np.uint8(255)).astype(np.uint8)
    except cv2.error as exc:
        raise MotionError(
            "mask cleanup failed",
            context={"reason": str(exc)},
        ) from exc


def _require_analysis_frame(analysis_frame: np.ndarray) -> None:
    """detect 입력 계약 검사. H x W x 3 uint8 프레임이어야 한다."""
    if (
        not isinstance(analysis_frame, np.ndarray)
        or analysis_frame.ndim != 3
        or analysis_frame.dtype != np.uint8
    ):
        raise MotionError(
            "analysis_frame must be an HxWx3 uint8 array",
            context={
                "ndim": int(getattr(analysis_frame, "ndim", -1)),
                "dtype": str(
                    getattr(analysis_frame, "dtype", type(analysis_frame).__name__)
                ),
            },
        )


class MotionDetector:
    """상태 있는 MOG2 전경 탐지기 (SPEC.md 13.1).

    YOLO confidence 나 캡처 폴더 등 다른 모듈의 설정은 알 필요 없다.
    detect 는 프레임당 subtractor.apply 를 정확히 1회 호출한다.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._first_shape: tuple[int, int, int] | None = None
        self._subtractor = self._create_subtractor()

    def _create_subtractor(self) -> cv2.BackgroundSubtractorMOG2:
        """설정값으로 MOG2 subtractor 를 만든다 (SPEC.md 13.2)."""
        return cv2.createBackgroundSubtractorMOG2(
            history=self._config.mog2_history,
            varThreshold=self._config.mog2_var_threshold,
            detectShadows=self._config.mog2_detect_shadows,
        )

    def reset(self) -> None:
        """배경 모델을 새로 만들고 첫 프레임 형태 캐시를 지운다."""
        self._subtractor = self._create_subtractor()
        self._first_shape = None

    def detect(
        self, analysis_frame: np.ndarray, video_time_sec: float
    ) -> MotionResult:
        """한 분석 프레임의 전경 분석 결과를 반환한다 (SPEC.md 13.3~13.7).

        - raw 마스크에서 값이 255 인 픽셀만 전경으로 쓰고 127 그림자는
          버린다 (SPEC.md 13.3).
        - 노이즈 제거 임계값은 해상도 보정된 min_component_pixels 를
          쓴다 (SPEC.md 9.4).
        - 기본 판정 마스크에 dilate 를 추가하지 않는다 (SPEC.md 13.5).
        - warming_up 은 video_time_sec < warmup_sec 다 (SPEC.md 13.7).
        - cv2 처리 오류는 MotionError 로 변환한다 (SPEC.md 21절).
        - 입력 프레임은 수정하지 않는다.
        """
        _require_analysis_frame(analysis_frame)

        if self._first_shape is None:
            self._first_shape = analysis_frame.shape
        elif analysis_frame.shape != self._first_shape:
            raise FrameShapeError(
                "analysis frame shape changed",
                context={
                    "first_shape": tuple(int(v) for v in self._first_shape),
                    "current_shape": tuple(int(v) for v in analysis_frame.shape),
                },
            )

        try:
            raw_mask = self._subtractor.apply(
                analysis_frame, learningRate=self._config.mog2_learning_rate
            )
            binary_mask = np.where(raw_mask == 255, 255, 0).astype(np.uint8)

            height, width = analysis_frame.shape[:2]
            valid_mask = clean_mask(
                binary_mask,
                open_kernel=self._config.open_kernel,
                close_kernel=self._config.close_kernel,
                iterations=self._config.morphology_iterations,
                min_component_pixels=effective_threshold(
                    self._config.min_component_pixels_ref, int(width), int(height),
                    reference_width=self._config.reference_width,
                    reference_height=self._config.reference_height,
                ),
            )
            regions = self._extract_regions(valid_mask)
        except cv2.error as exc:
            raise MotionError(
                "MOG2 foreground extraction failed",
                context={"reason": str(exc)},
            ) from exc

        foreground_pixels = int(np.count_nonzero(valid_mask))
        frame_foreground_ratio = foreground_pixels / float(width * height)
        return MotionResult(
            valid_mask=valid_mask,
            regions=tuple(regions),
            foreground_pixels=foreground_pixels,
            frame_foreground_ratio=frame_foreground_ratio,
            warming_up=video_time_sec < self._config.warmup_sec,
        )

    def _extract_regions(self, valid_mask: np.ndarray) -> list[MotionRegion]:
        """정리된 마스크에서 전경 영역 박스 목록을 만든다 (SPEC.md 13.6).

        RETR_EXTERNAL 과 CHAIN_APPROX_SIMPLE 로 contour 를 찾는다. 각
        contour 에 최소 면적 조건을 두지 않는다. 손, 다리처럼 분리된 작은
        영역도 같은 사람 박스 안에서 합쳐 유효할 수 있기 때문이다.
        """
        contours, _ = cv2.findContours(
            valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions: list[MotionRegion] = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            regions.append(
                MotionRegion(
                    box=Box(
                        int(x), int(y), int(x + box_width), int(y + box_height)
                    ),
                    contour_area=float(cv2.contourArea(contour)),
                )
            )
        return regions
