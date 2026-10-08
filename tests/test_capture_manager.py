"""CaptureManager 쿨다운/저장 계약 테스트 (SPEC.md 16절, 27.1 U16, U17, U20).

커버리지 (SPEC ID):
- U16: imencode/쓰기 실패 -> 성공 카운터·last_success_video_time 불변.
- U17: 사람 두 명이 같은 프레임에서 적격 -> JPG 한 개.
- U20: video_time 역행·같은 frame_index 재제출 -> ContractError,
  후보 아닌 프레임에서도 역행 검사 적용.
- §16.3 연속 후보 시퀀스 재현 (5.0 SAVED -> 5.5/7.9 COOLDOWN ->
  8.0 SAVED -> 불충족 NOT_ELIGIBLE).
- 저장 JPG 재디코딩 해상도 == raw_frame shape, 파일명 패턴
  event_*_f%08d_c%06d.jpg, prepare 재호출 충돌 (SPEC.md 16.4~16.6).
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

import capture_manager
from capture_manager import CaptureManager
from contracts import (
    Box,
    CaptureSaveError,
    ContractError,
    FrameDecision,
    FramePacket,
    PersonMotionEvidence,
)
from frame_processor import make_packet

RAW_W, RAW_H = 320, 240

# 파일명 패턴: event_<UTC시각>_f%08d_c%06d.jpg (SPEC.md 16.4)
FILENAME_PATTERN = re.compile(r"^event_\d{8}T\d{12}Z_f(\d{8})_c(\d{6})\.jpg$")


def packet_at(frame_index: int, video_time_sec: float, fill: int = 120) -> FramePacket:
    """320x240 원본 프레임 packet. analysis_width 960 >= 폭이라 배열 공유."""
    raw = np.full((RAW_H, RAW_W, 3), fill, dtype=np.uint8)
    return make_packet("run", frame_index, video_time_sec, raw, 960)


def decision_for(packet: FramePacket, candidate: bool) -> FrameDecision:
    """candidate 여부만 다른 FrameDecision. U17 에서는 사람 2명 근거를 넣는다."""
    return FrameDecision(
        frame_index=packet.frame_index,
        video_time_sec=packet.video_time_sec,
        status="MOVING_PERSON" if candidate else "IDLE",
        candidate=candidate,
        persons=(),
    )


def two_person_candidate_decision(packet: FramePacket) -> FrameDecision:
    """적격 사람 2명이 있는 후보 프레임 판정 (U17, SPEC.md 14.5)."""
    box = Box(0, 0, 10, 10)
    evidences = tuple(
        PersonMotionEvidence(
            detection_index=i,
            box=box,
            confidence=0.9,
            motion_pixels=6000,
            motion_ratio=0.15,
            qualifies=True,
            rejection_reasons=(),
        )
        for i in range(2)
    )
    return FrameDecision(
        frame_index=packet.frame_index,
        video_time_sec=packet.video_time_sec,
        status="MOVING_PERSON",
        candidate=True,
        persons=evidences,
    )


def jpg_files(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.iterdir() if p.suffix == ".jpg")


# ---------------------------------------------------------------------------
# §16.3 연속 후보 시퀀스
# ---------------------------------------------------------------------------


def test_16_3_consecutive_candidate_sequence(tmp_path: Path) -> None:
    """§16.3: 5.0 SAVED -> 5.5 COOLDOWN -> 7.9 COOLDOWN -> 8.0 SAVED -> 8.2 NOT_ELIGIBLE."""
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    run_id = "seq_run"
    manager.prepare(run_id)
    run_dir = tmp_path / run_id

    # 5.0초 후보 -> SAVED
    pkt_5_0 = packet_at(50, 5.0)
    result = manager.maybe_save(pkt_5_0, decision_for(pkt_5_0, True))
    assert result.status == "SAVED"
    assert result.sequence == 1
    assert result.path is not None and result.path.is_file()
    assert manager.successful_capture_count == 1
    assert manager.last_success_video_time == 5.0

    # 5.5초 후보 -> COOLDOWN (남은 2.5s)
    result = manager.maybe_save(packet_at(55, 5.5), decision_for(packet_at(55, 5.5), True))
    assert result.status == "COOLDOWN"
    assert result.sequence is None
    assert result.path is None
    assert result.cooldown_remaining_sec == pytest.approx(2.5, abs=1e-9)
    assert manager.successful_capture_count == 1

    # 7.9초 후보 -> COOLDOWN (남은 0.1s)
    result = manager.maybe_save(packet_at(79, 7.9), decision_for(packet_at(79, 7.9), True))
    assert result.status == "COOLDOWN"
    assert result.cooldown_remaining_sec == pytest.approx(0.1, abs=1e-9)

    # 8.0초 후보 -> SAVED (경계 >=)
    result = manager.maybe_save(packet_at(80, 8.0), decision_for(packet_at(80, 8.0), True))
    assert result.status == "SAVED"
    assert result.sequence == 2
    assert manager.successful_capture_count == 2
    assert manager.last_success_video_time == 8.0

    # 8.2초 조건 불충족 -> NOT_ELIGIBLE
    result = manager.maybe_save(packet_at(82, 8.2), decision_for(packet_at(82, 8.2), False))
    assert result.status == "NOT_ELIGIBLE"
    assert result.path is None
    assert manager.successful_capture_count == 2

    # 저장 파일은 정확히 2개
    assert len(jpg_files(run_dir)) == 2


# ---------------------------------------------------------------------------
# U16: 저장 실패 시 상태 불변
# ---------------------------------------------------------------------------


def test_u16_imencode_failure_keeps_counters(tmp_path: Path, monkeypatch) -> None:
    """U16: imencode 실패 -> CaptureSaveError, 카운터·마지막 저장 시간 불변 후 회복."""
    monkeypatch.setattr(
        capture_manager.cv2, "imencode", lambda *args, **kwargs: (False, None)
    )
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("enc_fail")
    pkt = packet_at(100, 5.0)

    with pytest.raises(CaptureSaveError):
        manager.maybe_save(pkt, decision_for(pkt, True))
    assert manager.successful_capture_count == 0
    assert manager.last_success_video_time is None
    assert jpg_files(tmp_path / "enc_fail") == []

    # 회복: imencode 복구 후 같은 조건에서 저장 성공 (U16 '불변' 확인의 대조군)
    monkeypatch.undo()
    pkt2 = packet_at(101, 6.0)
    result = manager.maybe_save(pkt2, decision_for(pkt2, True))
    assert result.status == "SAVED"
    assert manager.successful_capture_count == 1
    assert manager.last_success_video_time == 6.0


def test_u16_write_failure_keeps_counters(tmp_path: Path, monkeypatch) -> None:
    """U16: 임시 파일 쓰기 실패 -> CaptureSaveError, 상태 불변, 잔여 tmp 없음."""
    original_open = Path.open

    def failing_open(self: Path, mode: str = "r", *args, **kwargs):
        if "x" in str(mode):  # "xb" 배타적 생성 시도를 실패시킨다
            raise OSError("simulated disk full")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("write_fail")
    pkt = packet_at(100, 5.0)

    with pytest.raises(CaptureSaveError):
        manager.maybe_save(pkt, decision_for(pkt, True))
    assert manager.successful_capture_count == 0
    assert manager.last_success_video_time is None
    run_dir = tmp_path / "write_fail"
    assert jpg_files(run_dir) == []
    # 실패한 임시 파일도 남기지 않는다 (SPEC.md 16.6).
    assert [p for p in run_dir.iterdir() if p.name.endswith(".tmp")] == []


# ---------------------------------------------------------------------------
# U17: 프레임당 저장 1회
# ---------------------------------------------------------------------------


def test_u17_two_qualified_persons_save_single_jpg(tmp_path: Path) -> None:
    """U17: 적격 사람 2명 -> JPG 한 개, 시퀀스 1 (SPEC.md 14.5, FR14)."""
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("two_persons")
    pkt = packet_at(10, 4.0)

    decision = two_person_candidate_decision(pkt)
    assert len(decision.persons) == 2
    assert decision.candidate is True

    result = manager.maybe_save(pkt, decision)
    assert result.status == "SAVED"
    assert result.sequence == 1
    assert len(jpg_files(tmp_path / "two_persons")) == 1


# ---------------------------------------------------------------------------
# U20: 시간 역행 / 같은 프레임 재제출
# ---------------------------------------------------------------------------


def test_u20_video_time_backwards_raises(tmp_path: Path) -> None:
    """U20: video_time 역행 -> ContractError."""
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("backwards")
    pkt = packet_at(10, 1.0)
    manager.maybe_save(pkt, decision_for(pkt, False))

    later_frame_earlier_time = packet_at(11, 0.9)
    with pytest.raises(ContractError):
        manager.maybe_save(later_frame_earlier_time, decision_for(later_frame_earlier_time, False))


def test_u20_same_frame_index_resubmission_raises(tmp_path: Path) -> None:
    """U20: 같은 frame_index 재제출 -> ContractError."""
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("resubmit")
    pkt = packet_at(10, 1.0)
    manager.maybe_save(pkt, decision_for(pkt, False))

    duplicate = packet_at(10, 2.0)  # frame_index 만 같고 시간은 증가
    with pytest.raises(ContractError):
        manager.maybe_save(duplicate, decision_for(duplicate, False))


def test_u20_backwards_check_applies_to_non_candidate_frames(tmp_path: Path) -> None:
    """U20: 후보가 아닌 프레임에서도 역행 검사를 적용한다 (SPEC.md 27.1)."""
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("non_candidate")
    # 후보 아닌 프레임으로 시작해도 제출 값은 기록된다.
    pkt = packet_at(20, 6.0)
    manager.maybe_save(pkt, decision_for(pkt, False))

    non_candidate_backwards = packet_at(21, 5.9)
    with pytest.raises(ContractError):
        manager.maybe_save(
            non_candidate_backwards, decision_for(non_candidate_backwards, False)
        )


def test_u20_equal_time_is_rejected(tmp_path: Path) -> None:
    """U20 변형: 시간이 전혀 늘지 않은 재제출도 거절한다."""
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("equal_time")
    pkt = packet_at(30, 9.0)
    manager.maybe_save(pkt, decision_for(pkt, False))
    same_time = packet_at(31, 9.0)
    with pytest.raises(ContractError):
        manager.maybe_save(same_time, decision_for(same_time, False))


# ---------------------------------------------------------------------------
# 파일 규격 (SPEC.md 16.4, 16.5, 27.2)
# ---------------------------------------------------------------------------


def test_saved_jpg_redecodes_to_raw_resolution(tmp_path: Path) -> None:
    """저장 JPG 재디코딩 해상도 == raw_frame shape (SPEC.md 16.5, 27.2)."""
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("redecode")
    pkt = packet_at(10, 4.0, fill=200)
    result = manager.maybe_save(pkt, decision_for(pkt, True))

    assert result.path is not None
    decoded = cv2.imread(str(result.path), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == pkt.raw_frame.shape  # (240, 320, 3)
    assert decoded.dtype == np.uint8
    # 회색 프레임 내용이 보존되었는지 (JPEG 손실 감안 완만 검증).
    assert abs(float(decoded.mean()) - 200.0) < 10.0


def test_saved_jpg_has_no_overlay_artifacts(tmp_path: Path) -> None:
    """원본 프레임 그대로 인코딩: 박스/글자가 그려지지 않은 균일 프레임 확인."""
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("no_overlay")
    pkt = packet_at(10, 4.0, fill=128)
    result = manager.maybe_save(pkt, decision_for(pkt, True))

    decoded = cv2.imread(str(result.path), cv2.IMREAD_GRAYSCALE)
    assert decoded is not None
    # 균일 프레임에 UI 를 그렸다면 픽셀 값 분산이 커진다.
    assert float(decoded.std()) < 5.0
    # 원본 배열도 여전히 수정되지 않았다 (SPEC.md 8.1).
    np.testing.assert_array_equal(pkt.raw_frame, np.full((RAW_H, RAW_W, 3), 128, np.uint8))


def test_filename_pattern_event_frame_sequence(tmp_path: Path) -> None:
    """파일명 패턴 event_*_f%08d_c%06d.jpg (SPEC.md 16.4)."""
    manager = CaptureManager(tmp_path, cooldown_sec=0.0, jpeg_quality=95)
    manager.prepare("naming")
    for seq, (frame_index, t) in enumerate(((5, 1.0), (42, 1.5), (12345678, 2.0)), start=1):
        pkt = packet_at(frame_index, t)
        result = manager.maybe_save(pkt, decision_for(pkt, True))
        assert result.status == "SAVED"
        match = FILENAME_PATTERN.match(result.path.name)  # type: ignore[union-attr]
        assert match is not None, result.path.name
        assert match.group(1) == f"{frame_index:08d}"
        assert match.group(2) == f"{seq:06d}"


def test_prepare_conflict_and_maybe_save_before_prepare(tmp_path: Path) -> None:
    """prepare 재호출 충돌과 미호출 maybe_save (SPEC.md 16.4, 16.6)."""
    manager = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("dup_run")
    with pytest.raises(CaptureSaveError):
        manager.prepare("dup_run")  # 같은 run_id 폴더 재생성 불가

    # prepare 없이 maybe_save -> ContractError
    fresh = CaptureManager(tmp_path, cooldown_sec=3.0, jpeg_quality=95)
    pkt = packet_at(1, 0.5)
    with pytest.raises(ContractError):
        fresh.maybe_save(pkt, decision_for(pkt, True))


# ---------------------------------------------------------------------------
# 한글 경로 저장 (SPEC.md 27.2, NFR04)
# ---------------------------------------------------------------------------


def test_27_2_korean_output_path_save(tmp_path: Path) -> None:
    """한글 저장 폴더 경로에서도 원본 해상도 JPG 가 저장된다 (NFR04).

    디코딩은 np.fromfile + cv2.imdecode 로 한다. cv2.imread 는 Windows 의
    비 ASCII 경로에서 실패할 수 있어 검증 경로로 쓰지 않는다 (§16.6).
    """
    korean_root = tmp_path / "캡처_결과"
    manager = CaptureManager(korean_root, cooldown_sec=3.0, jpeg_quality=95)
    manager.prepare("run_한글경로")

    pkt = packet_at(30, 5.0, fill=180)
    result = manager.maybe_save(pkt, decision_for(pkt, True))

    assert result.status == "SAVED"
    assert result.path is not None
    assert result.path.exists()
    encoded = np.fromfile(str(result.path), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == pkt.raw_frame.shape
    assert abs(float(decoded.mean()) - 180.0) < 10.0
