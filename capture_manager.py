"""쿨다운, JPG 인코딩과 파일 저장 (SPEC.md 16절, 31.4절, 21절).

CaptureManager 는 성공한 저장의 영상 시간과 저장 시퀀스를 관리하는 유일한
모듈이다. 검출기나 UI가 JPG를 직접 저장하지 않는다(SPEC.md 16.1).

주요 규칙:
- 쿨다운은 영상 타임라인으로 계산하고 경계는 >= 다(SPEC.md 3.4, 16.2).
- 모든 제출 프레임은 frame_index 와 video_time_sec 가 직전 제출보다
  각각 업격히 커야 한다. 후보 여부와 무관하며 위반은 CONTRACT_ERROR 다
  (SPEC.md 27.1 U20).
- 저장은 원본 프레임 전체를 인코딩하며 overlay, crop, 확대를 하지
  않는다(SPEC.md 16.5). 호출자가 넘긴 프레임 배열은 수정하지 않는다.
- 임시 파일을 같은 폴더에 xb 로 쓰고 fsync 뒤 rename 하며, 충돌과
  쓰기 실패는 CAPTURE_SAVE_ERROR 로 처리한다(SPEC.md 16.6).
- successful_capture_count, sequence, last_success_video_time 은 최종
  저장이 성공한 뒤에만 갱신한다(SPEC.md 16.6, 21절 FR15).
"""

from __future__ import annotations

import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

from contracts import (
    CaptureResult,
    CaptureSaveError,
    ContractError,
    FrameDecision,
    FramePacket,
)


def cooldown_state(
    now_video_sec: float,
    last_success_video_sec: float | None,
    cooldown_sec: float,
) -> tuple[bool, float]:
    """전역 쿨다운 판정 (SPEC.md 31.4, 경계 오차 1 ns 허용).

    반환값은 (저장 허용 여부, 남은 쿨다운 초). 경계는 >= 이므로
    elapsed 가 cooldown_sec 와 정확히 같으면 허용한다. cooldown_sec 가
    0이면 last_success 가 있어도 항상 허용한다(SPEC.md 16.2).

    인수 값 자체가 유효하지 않거나(비유한 수, 음수) 영상 시간이 이전
    성공 저장 시각보다 거꾸로 흐르면 ValueError 를 발생시킨다.
    """
    if not math.isfinite(now_video_sec) or now_video_sec < 0:
        raise ValueError("invalid video time")
    if not math.isfinite(cooldown_sec) or cooldown_sec < 0:
        raise ValueError("invalid cooldown")
    if last_success_video_sec is None:
        return True, 0.0
    if not math.isfinite(last_success_video_sec) or last_success_video_sec < 0:
        raise ValueError("invalid previous capture time")
    if now_video_sec < last_success_video_sec:
        raise ValueError("video time moved backwards")
    elapsed = now_video_sec - last_success_video_sec
    # CFR 영상 시간의 뺄셈에서 발생하는 1 ns 이내의 절대 오차만 허용한다.
    allowed = elapsed >= cooldown_sec or math.isclose(
        elapsed, cooldown_sec, rel_tol=0.0, abs_tol=1e-9
    )
    remaining = 0.0 if allowed else max(0.0, cooldown_sec - elapsed)
    return allowed, remaining


def _remove_file_quietly(path: Path) -> None:
    """임시 파일 정리. 정리 자체가 실패해도 오류를 숨기고 진행한다."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


class CaptureManager:
    """쿨다운 판정과 원본 프레임 JPG 저장을 담당한다 (SPEC.md 16.1).

    maybe_save 는 프레임당 한 번 호출된다. 사람이 여럿이어도 같은
    프레임의 JPG 는 한 개만 만든다(SPEC.md 21절 FR14).
    """

    def __init__(self, output_dir: Path, cooldown_sec: float, jpeg_quality: int,
                 *, scope: str = "frame") -> None:
        self._output_dir = Path(output_dir)
        if scope not in ("frame", "object"):
            raise ContractError("capture scope must be frame or object")
        self._scope = scope
        self._last_track_capture_time: dict[int, float] = {}

        if not isinstance(cooldown_sec, (int, float)) or isinstance(cooldown_sec, bool):
            raise ContractError(
                "cooldown_sec must be a finite number >= 0",
                context={"key": "cooldown_sec", "value": repr(cooldown_sec)},
            )
        cooldown_sec = float(cooldown_sec)
        if not math.isfinite(cooldown_sec) or cooldown_sec < 0:
            raise ContractError(
                "cooldown_sec must be a finite number >= 0",
                context={"key": "cooldown_sec", "value": cooldown_sec},
            )
        if (
            not isinstance(jpeg_quality, int)
            or isinstance(jpeg_quality, bool)
            or not 1 <= jpeg_quality <= 100
        ):
            raise ContractError(
                "jpeg_quality must be an integer in [1, 100]",
                context={"key": "jpeg_quality", "value": repr(jpeg_quality)},
            )

        self._cooldown_sec = cooldown_sec
        self._jpeg_quality = jpeg_quality

        # 실행 상태 (SPEC.md 16.2, 16.6)
        self._run_dir: Path | None = None
        self._last_success_video_time: float | None = None
        self._successful_capture_count = 0

        # 단조 증가 확인용 직전 제출 값 (U20). 후보 여부와 무관하게
        # 모든 maybe_save 제출을 대상으로 한다.
        self._last_submitted_frame_index: int | None = None
        self._last_submitted_video_time: float | None = None

    # ------------------------------------------------------------------
    # 노출 상태
    # ------------------------------------------------------------------

    @property
    def successful_capture_count(self) -> int:
        """실제 저장된 JPG 수. 최종 저장 성공 뒤에만 증가한다."""
        return self._successful_capture_count

    @property
    def last_success_video_time(self) -> float | None:
        """마지막으로 성공한 저장의 영상 시간. 없으면 None."""
        return self._last_success_video_time

    # ------------------------------------------------------------------
    # 실행 준비
    # ------------------------------------------------------------------

    def prepare(self, run_id: str) -> None:
        """output_dir/run_id 실행 폴더를 만든다 (SPEC.md 16.4).

        폴더가 이미 존재하면 기존 파일을 덮어쓰지 않도록
        CaptureSaveError 를 발생시킨다.
        """
        run_dir = self._output_dir / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise CaptureSaveError(
                f"capture run directory already exists: {run_dir}",
                context={"run_id": run_id, "path": str(run_dir)},
            ) from exc
        except OSError as exc:
            raise CaptureSaveError(
                f"cannot create capture run directory: {run_dir}",
                context={"run_id": run_id, "path": str(run_dir), "reason": str(exc)},
            ) from exc
        self._run_dir = run_dir

    # ------------------------------------------------------------------
    # 프레임 저장 판정
    # ------------------------------------------------------------------

    def maybe_save(
        self, packet: FramePacket, decision: FrameDecision
    ) -> CaptureResult:
        """한 프레임의 저장 시도를 판정하고 실행한다 (SPEC.md 16.1).

        절차:
        0. 제출 단조 증가 검사(후보 여부와 무관, 위반은 ContractError).
        1. candidate 가 아니면 NOT_ELIGIBLE. 남은 쿨다운은 표시용으로 계산.
        2. cooldown_state 로 판정해 대기 중이면 COOLDOWN. 상태를 갱신하지 않음.
        3. 원본 프레임 전체를 JPG 로 저장(§16.5, §16.6). 실패는 CaptureSaveError.
        4. 최종 저장 성공 뒤에만 시퀀스/카운터/마지막 성공 시각을 갱신.
        """
        self._check_and_record_monotonic(packet)

        if self._run_dir is None:
            raise ContractError(
                "prepare() must be called before maybe_save()",
                context={"frame_index": packet.frame_index},
            )

        allowed, remaining = cooldown_state(
            packet.video_time_sec,
            self._last_success_video_time,
            self._cooldown_sec,
        )
        eligible_ids = tuple(sorted({e.track_id for e in decision.objects
                                     if e.qualifies and e.track_id is not None}))
        saved_ids = eligible_ids
        if self._scope == "object":
            states = {
                track_id: cooldown_state(packet.video_time_sec,
                                         self._last_track_capture_time.get(track_id),
                                         self._cooldown_sec)
                for track_id in eligible_ids
            }
            saved_ids = tuple(i for i, (due, _) in states.items() if due)
            allowed = bool(saved_ids)
            remaining = min((left for _, left in states.values()), default=0.0)

        # 1) 후보가 아니면 저장하지 않는다 (SPEC.md 16.2).
        if not decision.candidate or (self._scope == "object" and not eligible_ids):
            return CaptureResult(
                status="NOT_ELIGIBLE",
                path=None,
                video_time_sec=packet.video_time_sec,
                cooldown_remaining_sec=remaining,
                sequence=None,
            )

        # 2) 쿨다운 대기 중이면 어떤 상태도 갱신하지 않는다 (SPEC.md 16.2).
        if not allowed:
            return CaptureResult(
                status="COOLDOWN",
                path=None,
                video_time_sec=packet.video_time_sec,
                cooldown_remaining_sec=remaining,
                sequence=None,
            )

        # 3) 저장. 시퀀스는 성공 수 + 1 (SPEC.md 16.4).
        sequence = self._successful_capture_count + 1
        final_path = self._save_frame(packet, sequence)

        # 4) 최종 저장이 성공한 뒤에만 상태를 갱신한다 (SPEC.md 16.6).
        self._successful_capture_count = sequence
        self._last_success_video_time = packet.video_time_sec
        for track_id in saved_ids:
            self._last_track_capture_time[track_id] = packet.video_time_sec
        return CaptureResult(
            status="SAVED",
            path=final_path,
            video_time_sec=packet.video_time_sec,
            cooldown_remaining_sec=remaining,
            sequence=sequence,
            saved_track_ids=saved_ids,
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _check_and_record_monotonic(self, packet: FramePacket) -> None:
        """제출된 프레임 번호와 영상 시간의 단조 증가를 검사한다 (U20).

        같은 프레임 재제출과 시간 역행을 모두 거절한다. 검사를 통과한
        제출 값은 후보 여부와 무관하게 직전 제출로 기록한다.
        """
        frame_index = packet.frame_index
        video_time_sec = packet.video_time_sec

        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            raise ContractError(
                "frame_index must be a non-negative integer",
                context={"frame_index": repr(frame_index)},
            )
        if not isinstance(video_time_sec, (int, float)) or isinstance(video_time_sec, bool):
            raise ContractError(
                "video_time_sec must be a finite number >= 0",
                context={"video_time_sec": repr(video_time_sec)},
            )
        video_time_sec = float(video_time_sec)
        if not math.isfinite(video_time_sec) or video_time_sec < 0:
            raise ContractError(
                "video_time_sec must be a finite number >= 0",
                context={"video_time_sec": video_time_sec},
            )

        if self._last_submitted_frame_index is not None:
            if frame_index <= self._last_submitted_frame_index:
                raise ContractError(
                    "frame_index must be strictly increasing across submissions",
                    context={
                        "frame_index": frame_index,
                        "previous_frame_index": self._last_submitted_frame_index,
                    },
                )
            if video_time_sec <= self._last_submitted_video_time:  # type: ignore[operator]
                raise ContractError(
                    "video_time_sec must be strictly increasing across submissions",
                    context={
                        "video_time_sec": video_time_sec,
                        "previous_video_time_sec": self._last_submitted_video_time,
                    },
                )

        self._last_submitted_frame_index = frame_index
        self._last_submitted_video_time = video_time_sec

    def _save_frame(self, packet: FramePacket, sequence: int) -> Path:
        """원본 프레임 전체를 JPG 로 저장하고 최종 경로를 반환한다 (§16.5, §16.6).

        인코딩 실패, 임시 쓰기 실패, 최종 경로 충돌, rename 실패는 모두
        CaptureSaveError 로 처리한다. 임시 파일이 남으면 정리한다.
        호출자의 프레임 배열은 수정하지 않는다.
        """
        run_dir = self._run_dir
        assert run_dir is not None  # maybe_save 에서 보장한다.

        # 파일명의 시각은 저장 시각(실제 UTC)이다 (SPEC.md 16.4).
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        filename = (
            f"event_{timestamp}_f{packet.frame_index:08d}_c{sequence:06d}.jpg"
        )
        final_path = run_dir / filename

        # 한글 경로 대응을 위해 imencode 와 파일 쓰기를 분리한다 (NFR04).
        ok, encoded = cv2.imencode(
            ".jpg",
            packet.raw_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not ok:
            raise CaptureSaveError(
                "JPEG encoding failed",
                context={
                    "run_dir": str(run_dir),
                    "frame_index": packet.frame_index,
                    "jpeg_quality": self._jpeg_quality,
                },
            )

        # 임시 파일은 최종 JPG 와 같은 폴더에 고유한 이름으로 만든다.
        temp_path = run_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
        context = {
            "run_dir": str(run_dir),
            "frame_index": packet.frame_index,
            "path": str(final_path),
        }
        try:
            with temp_path.open("xb") as stream:
                stream.write(encoded.tobytes())
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            _remove_file_quietly(temp_path)
            raise CaptureSaveError(
                "capture temporary file write failed",
                context={**context, "reason": str(exc)},
            ) from exc

        # 같은 경로가 예상 밖으로 존재하면 충돌 오류로 처리한다 (SPEC.md 16.4).
        if final_path.exists():
            _remove_file_quietly(temp_path)
            raise CaptureSaveError(
                "capture filename collision",
                context={**context, "temporary_path": str(temp_path)},
            )

        try:
            temp_path.rename(final_path)
        except OSError as exc:
            _remove_file_quietly(temp_path)
            raise CaptureSaveError(
                "capture file rename failed",
                context={**context, "reason": str(exc)},
            ) from exc

        return final_path
