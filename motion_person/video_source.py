"""MP4 영상 입력 소스 (SPEC.md 10절).

영상 읽기, 입력 검증, 프레임 번호와 영상 시간 계산만 담당한다 (SPEC.md 10.1).
YOLO, MOG2, 캡처 조건은 알지 않는다.

열기 절차 (SPEC.md 10.2):
1. 영상 경로를 절대 경로로 변환한다.
2. 존재하는 일반 파일인지 확인한다. 아니면 VideoNotFoundError.
3. cv2.VideoCapture 로 연다. isOpened 가 false 면 VideoOpenError.
4. FPS 와 보고된 프레임 수를 읽고 유효성을 확인한다. 프레임 수가
   유효하지 않으면 None 로 둔다.
5. 첫 프레임을 디코드해 실제 크기를 확인하고 버퍼에 보관한다.
   실패하면 VideoDecodeError. width/height 는 프레임 배열의 shape 가
   최종 기준이다 (CAP_PROP_FRAME_WIDTH/HEIGHT 는 진단용일 뿐).
6. 첫 read() 는 버퍼한 첫 프레임을 프레임 번호 0 으로 반환해 준비
   과정에서 첫 프레임이 빠지지 않게 한다.

영상 시간 (SPEC.md 10.3):
- video_time_sec = frame_index / fps
- FPS 가 유한하고 0 보다 크면 timeline_source = frame_index_over_fps,
  timing_trusted = True.
- FPS 가 유효하지 않으면 fallback_fps 가 있을 때만 그 값으로 진행하고
  timeline_source = fallback_fps_constant, timing_trusted = False.
  fallback_fps 도 없으면 InvalidFpsError.

EOF 와 디코딩 오류 판별 (SPEC.md 10.4):
- 시작부터 실패하면 VideoDecodeError.
- 보고된 프레임 수가 유효하고 (보고된 수 - 실제 읽은 수) 가 2 를
  초과하면 EARLY_READ_FAILURE 로 VideoDecodeError.
- 프레임 수를 알 수 없거나 오차가 2 프레임 이내면 None 을 반환하되
  end_reason 속성에 END_OF_STREAM_OR_DECODE_FAILURE 를 남겨 판별
  한계를 기록한다. 실패를 정상 EOF 로 위장하지 않는다.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .contracts import (
    ContractError,
    InvalidFpsError,
    VideoDecodeError,
    VideoInfo,
    VideoNotFoundError,
    VideoOpenError,
)

# timeline_source 값 (SPEC.md 10.3)
TIMELINE_FRAME_INDEX_OVER_FPS = "frame_index_over_fps"
TIMELINE_FALLBACK_FPS_CONSTANT = "fallback_fps_constant"

# read() 가 None 을 반환했을 때의 종료 사유 (SPEC.md 10.4)
END_OF_STREAM_OR_DECODE_FAILURE = "END_OF_STREAM_OR_DECODE_FAILURE"
END_OF_STREAM = "END_OF_STREAM"

# 읽기 실패 context 의 reason 값 (SPEC.md 10.4)
REASON_FIRST_FRAME_READ_FAILURE = "FIRST_FRAME_READ_FAILURE"
REASON_EARLY_READ_FAILURE = "EARLY_READ_FAILURE"


def _read_reported_frame_count(capture: cv2.VideoCapture) -> int | None:
    """CAP_PROP_FRAME_COUNT 를 읽는다. 유효하지 않으면 None (SPEC.md 10.2)."""
    try:
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception as exc:  # 백엔드 조회 예외도 열기 실패로 종료한다(21절)
        raise VideoOpenError(
            "cannot read video metadata (frame count)",
            context={"reason": repr(exc)},
        ) from exc
    # NaN, 무한대, 1 미만 값은 메타데이터를 신뢰할 수 없다는 뜻이다.
    if math.isfinite(raw_count) and raw_count >= 1.0:
        return int(round(raw_count))
    return None


class VideoSource:
    """로컬 MP4 에서 원본 프레임을 순서대로 반환하는 리더 (SPEC.md 10.1)."""

    def __init__(self, video_path: Path, fallback_fps: float | None = None) -> None:
        self.video_path = Path(video_path)
        self.fallback_fps = fallback_fps

        self._capture: cv2.VideoCapture | None = None
        self._first_frame: np.ndarray | None = None
        self._absolute_path: Path = self.video_path
        self._fps: float = 0.0
        self._reported_frame_count: int | None = None
        self._timeline_source: str = TIMELINE_FRAME_INDEX_OVER_FPS
        self._timing_trusted: bool = False

        # 성공적으로 읽어 반환한 프레임 수 (SPEC.md 20.5 frames_read 정의).
        self.frames_read: int = 0
        # read() 가 None 을 반환한 뒤의 종료 사유. EOF 인지 디코드 실패인지
        # 확정할 수 없는 경우에도 판별 한계를 그대로 남긴다 (SPEC.md 10.4).
        self.end_reason: str | None = None

    # ------------------------------------------------------------------
    # 공개 인터페이스 (SPEC.md 10.1)
    # ------------------------------------------------------------------

    def open(self) -> VideoInfo:
        """영상을 열고 메타데이터를 반환한다 (SPEC.md 10.2).

        VideoInfo 의 width/height 는 첫 프레임 배열의 shape 를 기준으로
        한다. 첫 프레임은 버퍼에 보관되어 첫 read() 로 나간다. 실패 시
        이미 연 VideoCapture 는 해제한 뒤 예외를 다시 발생시킨다.
        """
        if self._capture is not None:
            raise ContractError(
                "VideoSource is already open; call close() first",
                context={
                    "path": str(self._absolute_path),
                    "frames_read": self.frames_read,
                },
            )

        # 1. 절대 경로 변환. 2. 일반 파일 확인.
        absolute_path = self.video_path.resolve()
        self._absolute_path = absolute_path
        if not absolute_path.is_file():
            raise VideoNotFoundError(
                f"video file not found: {absolute_path}",
                context={"path": str(absolute_path)},
            )

        # 3. cv2.VideoCapture 로 열기. 4. isOpened 확인.
        try:
            capture = cv2.VideoCapture(str(absolute_path))
        except Exception as exc:  # 백엔드 초기화 예외 (cv2.error 등)
            raise VideoOpenError(
                f"cannot open video: {absolute_path}",
                context={"path": str(absolute_path), "reason": repr(exc)},
            ) from exc
        if not capture.isOpened():
            capture.release()
            raise VideoOpenError(
                f"cannot open video: {absolute_path}",
                context={"path": str(absolute_path)},
            )

        # 5. FPS/보고 프레임 수 확인. 6. 첫 프레임 디코드 후 버퍼.
        self._capture = capture
        try:
            self._fps, self._timeline_source, self._timing_trusted = (
                self._resolve_fps(capture)
            )
            self._reported_frame_count = _read_reported_frame_count(capture)
            self._first_frame = self._read_first_frame(capture)
        except Exception:
            self.close()
            raise

        height, width = self._first_frame.shape[:2]
        return VideoInfo(
            path=absolute_path,
            fps=self._fps,
            reported_frame_count=self._reported_frame_count,
            width=int(width),
            height=int(height),
            timeline_source=self._timeline_source,
            timing_trusted=self._timing_trusted,
        )

    def read(self) -> tuple[int, float, np.ndarray] | None:
        """다음 프레임을 (frame_index, video_time_sec, raw_frame) 로 반환한다.

        첫 호출은 open() 이 버퍼한 첫 프레임을 번호 0 으로 반환한다.
        더 읽을 프레임이 없으면 None 을 반환하고 end_reason 에 판별
        결과를 남긴다. 반환 배열은 논리적으로 읽기 전용이다.
        """
        if self._capture is None:
            raise ContractError(
                "VideoSource is not open; call open() before read()",
                context={
                    "path": str(self._absolute_path),
                    "frames_read": self.frames_read,
                },
            )

        if self._first_frame is not None:
            # 준비 단계에서 디코드한 첫 프레임을 빠뜨리지 않는다.
            frame = self._first_frame
            self._first_frame = None
        else:
            decode_error: str | None = None
            try:
                ok, frame = self._capture.read()
            except Exception as exc:  # 디코더 예외도 읽기 실패로 취급
                ok, frame = False, None
                decode_error = repr(exc)
            if not ok or frame is None or frame.size == 0:
                self._handle_read_failure(decode_error)
                return None

        frame_index = self.frames_read
        self.frames_read += 1
        self.end_reason = None
        video_time_sec = frame_index / self._fps
        return frame_index, video_time_sec, frame

    def close(self) -> None:
        """VideoCapture 를 해제한다. 중복 호출도 안전하다 (SPEC.md 10.5).

        frames_read 나 end_reason 등 진단 값은 유지해 종료 요약에 쓸 수
        있게 한다.
        """
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._first_frame = None

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _resolve_fps(
        self, capture: cv2.VideoCapture
    ) -> tuple[float, str, bool]:
        """FPS 를 검증하고 (fps, timeline_source, timing_trusted) 를 반환한다.

        FPS 가 유한하고 0 보다 크면 정상 시간선으로 판단한다. 그렇지 않으면
        fallback_fps 가 있을 때만 대체 진행하고 timing_trusted 를 false 로
        기록한다 (SPEC.md 10.3).
        """
        try:
            raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        except Exception as exc:  # 백엔드 조회 예외도 열기 실패로 종료한다(21절)
            raise VideoOpenError(
                "cannot read video metadata (fps)",
                context={"reason": repr(exc)},
            ) from exc
        if math.isfinite(raw_fps) and raw_fps > 0.0:
            return raw_fps, TIMELINE_FRAME_INDEX_OVER_FPS, True

        # NaN/inf 는 JSON 로그에 쓸 수 없어 문자열 표현으로 남긴다.
        raw_fps_value: float | str = (
            raw_fps if math.isfinite(raw_fps) else repr(raw_fps)
        )
        if self.fallback_fps is not None:
            fallback = float(self.fallback_fps)
            if not math.isfinite(fallback) or fallback <= 0.0:
                raise InvalidFpsError(
                    "fallback_fps must be a positive finite number",
                    context={
                        "path": str(self._absolute_path),
                        "fallback_fps": repr(self.fallback_fps),
                    },
                )
            return fallback, TIMELINE_FALLBACK_FPS_CONSTANT, False

        raise InvalidFpsError(
            "video FPS is invalid and no fallback_fps is provided",
            context={"path": str(self._absolute_path), "raw_fps": raw_fps_value},
        )

    def _read_first_frame(self, capture: cv2.VideoCapture) -> np.ndarray:
        """첫 프레임을 디코드한다. 실패하면 VideoDecodeError (SPEC.md 10.2)."""
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise VideoDecodeError(
                "failed to decode the first frame",
                context={
                    "path": str(self._absolute_path),
                    "reason": REASON_FIRST_FRAME_READ_FAILURE,
                },
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            # raw_frame 계약은 H x W x 3 BGR 이다 (SPEC.md 8.1).
            raise VideoDecodeError(
                "first frame is not an HxWx3 BGR array",
                context={
                    "path": str(self._absolute_path),
                    "reason": "UNSUPPORTED_FRAME_LAYOUT",
                    "shape": [int(dim) for dim in frame.shape],
                },
            )
        return frame

    def _handle_read_failure(self, decode_error: str | None) -> None:
        """읽기 실패를 SPEC.md 10.4 규칙으로 판별한다.

        시작부터 실패, 보고 프레임 수와 2 프레임을 초과하는 차이는
        VideoDecodeError 로 올리고, 그 외에는 end_reason 에 판별 한계를
        남겨 호출자가 None 을 받도록 한다.
        """
        context: dict = {
            "path": str(self._absolute_path),
            "read": self.frames_read,
        }
        if decode_error is not None:
            context["decode_error"] = decode_error

        if self.frames_read == 0:
            # 시작부터 실패는 디코딩 오류이다 (SPEC.md 10.4).
            context["reason"] = REASON_FIRST_FRAME_READ_FAILURE
            raise VideoDecodeError("failed to decode any frame", context=context)

        if decode_error is not None:
            raise VideoDecodeError("video decoder raised an exception", context=context)

        reported = self._reported_frame_count
        if reported is not None and reported - self.frames_read > 2:
            context["reason"] = REASON_EARLY_READ_FAILURE
            context["reported"] = reported
            raise VideoDecodeError(
                "video ended earlier than the reported frame count",
                context=context,
            )

        # 보고된 프레임 수가 없거나 오차가 2 프레임 이하면 스트림 종료로
        # 본다. EOF 확정이 아니라는 판별 한계를 end_reason 에 남긴다.
        self.end_reason = (
            END_OF_STREAM if reported == self.frames_read
            else END_OF_STREAM_OR_DECODE_FAILURE
        )
