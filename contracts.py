"""공통 데이터 타입과 좌표 규칙 (SPEC.md 8절, 21절).

이 모듈은 순수 데이터 계약과 오류 계약만 담는다. 파일 입출력, 알고리즘,
화면 표시를 하지 않는다.

좌표 규칙 (SPEC.md 8.1):
- 이미지 배열은 uint8, BGR 채널 순서다.
- raw_frame 은 H x W x 3, analysis_frame 은 h x w x 3, 마스크는 h x w (0/255).
- 좌표 원점은 왼쪽 위. x 는 오른쪽, y 는 아래 방향으로 증가한다.
- 박스는 정수 xyxy. x1, y1 은 포함, x2, y2 는 제외.
- 유효 박스는 0 <= x1 < x2 <= w, 0 <= y1 < y2 <= h.

FramePacket 의 배열은 논리적으로 읽기 전용이다. frozen dataclass 가
NumPy 배열 자체의 수정까지 막지는 않으므로 각 소비 모듈이 수정 금지
계약을 지켜야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# 오류 계약 (SPEC.md 21절). 각 예외는 오류 코드와 main 이 사용할 종료 코드를
# 가진다. 예외를 광범위하게 무시하거나 모델 오류를 사람 없음으로 처리하지
# 않는다.
# ---------------------------------------------------------------------------

# 오류 코드 -> 프로세스 종료 코드 (SPEC.md 21절 표).
EXIT_CODES: dict[str, int] = {
    "CONFIG_ERROR": 2,
    "VIDEO_NOT_FOUND": 3,
    "VIDEO_OPEN_ERROR": 3,
    "VIDEO_DECODE_ERROR": 3,
    "INVALID_FPS": 3,
    "FRAME_SHAPE_ERROR": 3,
    "MODEL_NOT_FOUND": 4,
    "MODEL_LOAD_ERROR": 4,
    "MODEL_CLASS_ERROR": 4,
    "INFERENCE_ERROR": 4,
    "MOTION_ERROR": 4,
    "CONTRACT_ERROR": 4,
    "CAPTURE_SAVE_ERROR": 5,
    "LOG_WRITE_ERROR": 5,
    "DISPLAY_ERROR": 6,
}


class MotionPersonError(Exception):
    """모듈 공통 예외 기반 클래스. code 는 SPEC.md 21절의 오류 코드다."""

    code: str = "CONTRACT_ERROR"

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict = context or {}

    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 4)

    def to_payload(self) -> dict:
        """JSONL ERROR 이벤트에 기록할 수 있는 형태로 변환한다."""
        payload = {"error_code": self.code, "message": self.message}
        if self.context:
            payload["context"] = {
                key: value
                for key, value in self.context.items()
                if _is_json_safe(value)
            }
        return payload


def _is_json_safe(value: object) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_safe(item) for key, item in value.items())
    return False


class ConfigError(MotionPersonError):
    code = "CONFIG_ERROR"


class VideoNotFoundError(MotionPersonError):
    code = "VIDEO_NOT_FOUND"


class VideoOpenError(MotionPersonError):
    code = "VIDEO_OPEN_ERROR"


class VideoDecodeError(MotionPersonError):
    code = "VIDEO_DECODE_ERROR"


class InvalidFpsError(MotionPersonError):
    code = "INVALID_FPS"


class FrameShapeError(MotionPersonError):
    code = "FRAME_SHAPE_ERROR"


class ModelNotFoundError(MotionPersonError):
    code = "MODEL_NOT_FOUND"


class ModelLoadError(MotionPersonError):
    code = "MODEL_LOAD_ERROR"


class ModelClassError(MotionPersonError):
    code = "MODEL_CLASS_ERROR"


class InferenceError(MotionPersonError):
    code = "INFERENCE_ERROR"


class MotionError(MotionPersonError):
    code = "MOTION_ERROR"


class ContractError(MotionPersonError):
    code = "CONTRACT_ERROR"


class CaptureSaveError(MotionPersonError):
    code = "CAPTURE_SAVE_ERROR"


class LogWriteError(MotionPersonError):
    code = "LOG_WRITE_ERROR"


class DisplayError(MotionPersonError):
    code = "DISPLAY_ERROR"


# ---------------------------------------------------------------------------
# 데이터 계약 (SPEC.md 8.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """정수 xyxy 박스. x1, y1 은 포함, x2, y2 는 제외한다."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def is_valid_for(self, width: int, height: int) -> bool:
        """유효 박스는 0 <= x1 < x2 <= w, 0 <= y1 < y2 <= h (SPEC.md 8.1)."""
        return 0 <= self.x1 < self.x2 <= width and 0 <= self.y1 < self.y2 <= height

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class VideoInfo:
    """영상 메타데이터. timing_trusted 가 false 면 fallback FPS 사용 의미."""

    path: Path
    fps: float
    reported_frame_count: int | None
    width: int
    height: int
    timeline_source: str
    timing_trusted: bool

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "fps": self.fps,
            "reported_frame_count": self.reported_frame_count,
            "width": self.width,
            "height": self.height,
            "timeline_source": self.timeline_source,
            "timing_trusted": self.timing_trusted,
        }


@dataclass(frozen=True)
class FramePacket:
    """한 프레임의 원본/분석 배열과 좌표 변환 정보.

    raw_frame 과 analysis_frame 은 논리적으로 읽기 전용이다. 배열에 박스나
    글자를 그리지 않는다. 화면 표시용 복사는 overlay_renderer 가 만든다.
    """

    run_id: str
    frame_index: int
    video_time_sec: float
    raw_frame: np.ndarray
    analysis_frame: np.ndarray
    scale_x: float
    scale_y: float
    roi_xyxy: tuple[int, int, int, int] | None = None

    @property
    def analysis_size(self) -> tuple[int, int]:
        height, width = self.analysis_frame.shape[:2]
        return width, height

    @property
    def original_size(self) -> tuple[int, int]:
        height, width = self.raw_frame.shape[:2]
        return width, height


@dataclass(frozen=True)
class PersonDetection:
    """객체 검출 결과. detection_index는 프레임 내부, track_id는 실행 내 추적 ID.

    기존 PersonDetection 이름은 호출자 호환을 위해 유지한다.
    """

    detection_index: int
    box: Box
    confidence: float
    class_id: int
    class_name: str = "person"
    track_id: int | None = None
    tracker_id: int | None = None
    observation_hits: int = 1
    confirmed: bool = True

    @property
    def label(self) -> str:
        return f"{self.class_name} #{self.track_id}" if self.track_id is not None else self.class_name


@dataclass(frozen=True)
class MotionRegion:
    """MOG2 전경 영역 박스. UI와 디버깅용 진단 값이다."""

    box: Box
    contour_area: float


@dataclass(frozen=True)
class MotionResult:
    """MOG2 한 프레임 결과. valid_mask 값은 0 또는 255만 있다."""

    valid_mask: np.ndarray
    regions: tuple[MotionRegion, ...]
    foreground_pixels: int
    frame_foreground_ratio: float
    warming_up: bool


@dataclass(frozen=True)
class PersonMotionEvidence:
    """사람별 결합 판정 근거 (SPEC.md 14절).

    rejection_reasons 는 WARMUP, LOW_CONFIDENCE,
    MOTION_PIXELS_BELOW_MIN, MOTION_RATIO_BELOW_MIN 중 해당 항목 전부다.
    """

    detection_index: int
    box: Box
    confidence: float
    motion_pixels: int
    motion_ratio: float
    qualifies: bool
    rejection_reasons: tuple[str, ...]
    class_id: int = 0
    class_name: str = "person"
    track_id: int | None = None
    tracker_id: int | None = None
    observation_hits: int = 1
    confirmed: bool = True

    @property
    def label(self) -> str:
        return f"{self.class_name} #{self.track_id}" if self.track_id is not None else self.class_name


@dataclass(frozen=True)
class FrameDecision:
    """한 프레임의 분석 상태와 후보 판정 (SPEC.md 15.1 우선순위)."""

    frame_index: int
    video_time_sec: float
    status: Literal[
        "WARMUP", "IDLE", "PERSON_ONLY", "MOTION_ONLY",
        "PERSON_AND_MOTION_UNMATCHED", "MOVING_PERSON",
        "OBJECT_ONLY", "OBJECT_AND_MOTION_UNMATCHED", "MOVING_OBJECT",
    ]
    candidate: bool
    persons: tuple[PersonMotionEvidence, ...]

    @property
    def objects(self) -> tuple[PersonMotionEvidence, ...]:
        return self.persons


@dataclass(frozen=True)
class CaptureResult:
    """한 프레임의 저장 시도 결과 (SPEC.md 16절).

    저장에 실패하면 SAVED 를 반환하지 않고 CaptureSaveError 를 발생시킨다.
    """

    status: Literal["NOT_ELIGIBLE", "COOLDOWN", "SAVED", "DISABLED"]
    path: Path | None
    video_time_sec: float
    cooldown_remaining_sec: float
    sequence: int | None
    saved_track_ids: tuple[int, ...] = ()


# Generic names for new consumers; legacy constructors remain compatible.
ObjectDetection = PersonDetection
ObjectMotionEvidence = PersonMotionEvidence
