"""사람 박스와 MOG2 유효 전경 마스크의 결합 판정 (SPEC.md 14절, 15.1, 31.2, 31.3).

상태 없는 판정 함수다. 입력 박스와 마스크만으로 같은 결과를 반환하며
파일 저장, 시간 대기, 화면 호출을 하지 않는다 (SPEC.md 14.1). 호출자가
넘긴 프레임 배열과 마스크에 박스나 글자를 그리지 않는다 (SPEC.md 8.1).

판정 수식 (SPEC.md 14.2):

    A_i = 사람 박스 너비 x 높이
    P_i = 박스 내부에서 valid_mask 가 255 인 픽셀 수
    R_i = P_i / A_i
    qualifies_i = (warming_up 아님) AND (confidence >= 기준)
                  AND (P_i >= 보정된 최소 픽셀) AND (R_i >= 최소 비율)

각 사람을 독립적으로 계산하고 사람 바깥 전경을 합산하지 않는다
(SPEC.md 14.5). INVALID_BOX 와 MASK_SHAPE_MISMATCH 는 정상 판정 원인이
아니라 계약 오류므로 예외를 발생시킨다 (SPEC.md 14.4).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .config import Config
from .contracts import (
    Box,
    ContractError,
    FrameDecision,
    FramePacket,
    MotionResult,
    PersonDetection,
    PersonMotionEvidence,
)

# 해상도 보정 함수의 정식 구현은 frame_processor 이 제공한다 (SPEC.md 31.1).
# 임계값 로직이 두 곳에 존재하면 drift 위험이 있으므로 중복 정의를 두지 않는다.
from .frame_processor import effective_threshold


def person_mask_evidence(mask: np.ndarray, box: Box) -> tuple[int, float]:
    """사람 박스 내부의 유효 전경 픽셀 수와 박스 면적 대비 비율 (SPEC.md 31.2).

    mask 는 MotionDetector 가 반환한 valid_mask (h x w, 값은 0 또는 255)를
    전달한다. count_nonzero(mask) 대신 == 255 비교를 쓰지만 임의의 비이진
    마스크를 정상 입력으로 허용하는 뜻은 아니다.

    Returns:
        (pixels, ratio): pixels 는 박스 내부 255 픽셀 수, ratio 는
        pixels / box.area 이다.

    Raises:
        ValueError: mask 가 2D uint8 이 아니거나 박스 좌표가 마스크
            크기에 대해 유효하지 않을 때.
    """
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("mask must be 2D uint8")
    height, width = mask.shape
    if not (0 <= box.x1 < box.x2 <= width):
        raise ValueError("invalid x coordinates")
    if not (0 <= box.y1 < box.y2 <= height):
        raise ValueError("invalid y coordinates")
    roi = mask[box.y1:box.y2, box.x1:box.x2]
    pixels = int(np.count_nonzero(roi == 255))
    return pixels, pixels / box.area


def qualifies_person(
    confidence: float,
    pixels: int,
    ratio: float,
    warming_up: bool,
    min_confidence: float = 0.60,
    min_pixels: int = 3000,
    min_ratio: float = 0.03,
) -> tuple[bool, tuple[str, ...]]:
    """사람 하나의 후보 조건 판정 (SPEC.md 31.3, 14.2).

    기준값 자체의 범위 검증은 Config 가 완료한다. 이 함수는 입력
    confidence, 실제 픽셀 수와 비율만 검증한다. 경계값은 포함이다
    (P_i == min_pixels, R_i == min_ratio 도 충족).

    Returns:
        (qualifies, reasons): reasons 는 해당하는 거부 사유 전부이며
        WARMUP, LOW_CONFIDENCE, MOTION_PIXELS_BELOW_MIN,
        MOTION_RATIO_BELOW_MIN 순서로 담는다. qualifies 는
        reasons 가 비었을 때만 true 다.

    Raises:
        ValueError: confidence 나 ratio 가 유한한 [0, 1] 값이 아니거나
            pixels 가 음수일 때.
    """
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("invalid confidence")
    if not isinstance(pixels, int) or pixels < 0:
        raise ValueError("invalid pixel count")
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("invalid motion ratio")
    reasons = []
    if warming_up:
        reasons.append("WARMUP")
    if confidence < min_confidence:
        reasons.append("LOW_CONFIDENCE")
    if pixels < min_pixels:
        reasons.append("MOTION_PIXELS_BELOW_MIN")
    if ratio < min_ratio:
        reasons.append("MOTION_RATIO_BELOW_MIN")
    return not reasons, tuple(reasons)


class EventDetector:
    """사람별 면적과 비율로 프레임 판정을 만드는 상태 없는 검출기 (SPEC.md 14.1).

    EventDetector 는 설정값만 보관하고 프레임 사이 상태를 갖지 않는다.
    같은 packet, persons, motion 입력에 대해 항상 같은 FrameDecision 을
    반환한다.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def evaluate(
        self,
        packet: FramePacket,
        persons: tuple[PersonDetection, ...],
        motion: MotionResult,
    ) -> FrameDecision:
        """한 프레임의 분석 상태와 캡처 후보 여부를 판정한다.

        절차 (SPEC.md 14절):
        1. valid_mask 가 2D uint8 이고 analysis_frame 의 (h, w) 와 일치하는지
           검사한다. 위반하면 ContractError (MASK_SHAPE_MISMATCH, 테스트 U10).
        2. 모든 사람 박스가 마스크 크기에 대해 유효한지 검사한다. 0면적이나
           범위 밖 박스는 ContractError (INVALID_BOX, 테스트 U11). 유효하지
           않은 입력을 IDLE 로 처리하지 않는다 (SPEC.md 14.4).
        3. 최소 픽셀 기준을 분석 해상도로 보정한다 (SPEC.md 9.4).
        4. 사람별로 P_i, R_i 를 독립 계산하고 confidence 재검증을 포함해
           qualifies 를 판정한다 (SPEC.md 14.6). 거부 사유는 해당 항목 전부를
           기록한다 (SPEC.md 14.4).
        5. candidate 와 상태를 §15.1 우선순위로 정한다.

        Args:
            packet: 분석 프레임 패킷. 배열은 읽기만 한다.
            persons: PersonDetector 가 반환한 같은 프레임의 사람 검출 목록.
            motion: MotionDetector 가 반환한 같은 프레임의 전경 결과.

        Returns:
            FrameDecision: 상태, 후보 여부와 사람별 근거.

        Raises:
            ContractError: 마스크 모양 불일치 (U10) 또는 무효 박스 (U11).
        """
        mask = motion.valid_mask
        analysis_height, analysis_width = packet.analysis_frame.shape[:2]

        # 1) 마스크 계약 검사 (SPEC.md 14.4, 테스트 U10)
        if mask.ndim != 2 or mask.dtype != np.uint8:
            raise ContractError(
                "MASK_SHAPE_MISMATCH: valid_mask must be a 2D uint8 array",
                context={
                    "check": "U10",
                    "frame_index": packet.frame_index,
                    "mask_ndim": int(mask.ndim),
                    "mask_dtype": str(mask.dtype),
                    "analysis_size": [analysis_width, analysis_height],
                },
            )
        if mask.shape != (analysis_height, analysis_width):
            raise ContractError(
                "MASK_SHAPE_MISMATCH: valid_mask shape must match analysis_frame",
                context={
                    "check": "U10",
                    "frame_index": packet.frame_index,
                    "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
                    "analysis_size": [analysis_width, analysis_height],
                },
            )

        # 2) 박스 계약 검사 (SPEC.md 14.4, 테스트 U11). 0면적 박스도
        #    x1 < x2, y1 < y2 위반으로 같은 계약 오류다.
        for person in persons:
            if not person.box.is_valid_for(analysis_width, analysis_height):
                raise ContractError(
                    "INVALID_BOX: person box is outside the analysis frame "
                    "or has zero area",
                    context={
                        "check": "U11",
                        "frame_index": packet.frame_index,
                        "detection_index": person.detection_index,
                        "box_xyxy": person.box.to_list(),
                        "analysis_size": [analysis_width, analysis_height],
                    },
                )

        # 3) 해상도 보정 최소 픽셀 기준 (SPEC.md 9.4)
        effective_min_pixels = effective_threshold(
            self._config.min_person_motion_pixels_ref,
            analysis_width,
            analysis_height,
            reference_width=self._config.reference_width,
            reference_height=self._config.reference_height,
        )

        # 4) 사람별 독립 판정 (SPEC.md 14.2, 14.5, 14.6)
        evidences: list[PersonMotionEvidence] = []
        for person in persons:
            pixels, ratio = person_mask_evidence(mask, person.box)
            qualifies, reasons = qualifies_person(
                confidence=person.confidence,
                pixels=pixels,
                ratio=ratio,
                warming_up=motion.warming_up,
                min_confidence=self._config.person_confidence,
                min_pixels=effective_min_pixels,
                min_ratio=self._config.min_person_motion_ratio,
            )
            if self._config.tracking and not person.confirmed:
                qualifies, reasons = False, (*reasons, "TRACK_PENDING")
            evidences.append(
                PersonMotionEvidence(
                    detection_index=person.detection_index,
                    box=person.box,
                    confidence=person.confidence,
                    motion_pixels=pixels,
                    motion_ratio=ratio,
                    qualifies=qualifies,
                    rejection_reasons=reasons,
                    class_id=person.class_id,
                    class_name=person.class_name,
                    track_id=person.track_id,
                    tracker_id=person.tracker_id,
                    observation_hits=person.observation_hits,
                    confirmed=person.confirmed,
                )
            )

        candidate = any(evidence.qualifies for evidence in evidences)

        # 5) 상태 우선순위 (SPEC.md 15.1)
        status: Literal[
            "WARMUP", "IDLE", "PERSON_ONLY", "MOTION_ONLY",
            "PERSON_AND_MOTION_UNMATCHED", "MOVING_PERSON",
        ]
        has_persons = len(persons) > 0
        has_foreground = motion.foreground_pixels > 0
        if motion.warming_up:
            status = "WARMUP"
        elif candidate:
            status = "MOVING_PERSON"
        elif has_persons and has_foreground:
            status = "PERSON_AND_MOTION_UNMATCHED"
        elif has_persons:
            status = "PERSON_ONLY"
        elif has_foreground:
            status = "MOTION_ONLY"
        else:
            status = "IDLE"

        if self._config.target_classes != ("person",):
            status = {
                "MOVING_PERSON": "MOVING_OBJECT",
                "PERSON_ONLY": "OBJECT_ONLY",
                "PERSON_AND_MOTION_UNMATCHED": "OBJECT_AND_MOTION_UNMATCHED",
            }.get(status, status)

        return FrameDecision(
            frame_index=packet.frame_index,
            video_time_sec=packet.video_time_sec,
            status=status,
            candidate=candidate,
            persons=tuple(evidences),
        )
