"""VideoSource 영상 입력 계약 테스트 (SPEC.md 10절).

커버리지 (SPEC ID):
- 합성 mp4(64x48, 10fps, 30프레임) 전체 읽기: 30프레임, 첫 프레임 index 0
  누락 없음, video_time == i/10, close 멱등 (SPEC.md 10.2, 10.3, 10.5).
- 없는 파일 -> VideoNotFoundError (SPEC.md 21절 VIDEO_NOT_FOUND).
- monkeypatch 스텁으로 FPS=0 -> InvalidFpsError (SPEC.md 10.3, 21절).
- fallback 지정 시 timing_trusted False (SPEC.md 10.3).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import video_source
from contracts import InvalidFpsError, VideoNotFoundError
from video_source import (
    END_OF_STREAM,
    END_OF_STREAM_OR_DECODE_FAILURE,
    TIMELINE_FALLBACK_FPS_CONSTANT,
    TIMELINE_FRAME_INDEX_OVER_FPS,
    VideoSource,
)

VIDEO_W, VIDEO_H, VIDEO_FPS, VIDEO_FRAMES = 64, 48, 10.0, 30


def make_synthetic_video(directory: Path) -> Path:
    """64x48, 10fps, 30프레임 mp4 를 만든다. 프레임 i 는 밝기 i*8."""
    path = directory / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (VIDEO_W, VIDEO_H)
    )
    if not writer.isOpened():
        # 백엔드가 mp4 를 못 쓰는 환경 대비 avi 폴백 (SPEC.md 10절 계약 검증 동일).
        path = directory / "synthetic.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), VIDEO_FPS, (VIDEO_W, VIDEO_H)
        )
        assert writer.isOpened()
    try:
        for i in range(VIDEO_FRAMES):
            brightness = min(255, i * 8)
            writer.write(np.full((VIDEO_H, VIDEO_W, 3), brightness, dtype=np.uint8))
    finally:
        writer.release()
    return path


class StubCapture:
    """FPS=0 파일을 흉내내는 cv2.VideoCapture 스텁 (SPEC.md 10.3 검증용)."""

    def __init__(self, path: str, fps: float, frame_count: int = 8) -> None:
        self._fps = fps
        self._frame_count = frame_count
        self._frame = np.full((VIDEO_H, VIDEO_W, 3), 128, dtype=np.uint8)
        self._reads = 0

    def isOpened(self) -> bool:  # noqa: N802 (OpenCV API 이름)
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._reads >= self._frame_count:
            return False, None
        self._reads += 1
        return True, self._frame.copy()

    def release(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 정상 읽기 계약 (SPEC.md 10.2, 10.3)
# ---------------------------------------------------------------------------


def test_reads_all_frames_with_index_and_time(tmp_path: Path) -> None:
    """전체 30프레임, 첫 프레임 index 0 누락 없음, video_time == i/10."""
    video = make_synthetic_video(tmp_path)
    source = VideoSource(video)
    info = source.open()

    assert info.width == VIDEO_W
    assert info.height == VIDEO_H
    assert info.fps == pytest.approx(VIDEO_FPS, abs=1e-9)
    assert info.reported_frame_count == VIDEO_FRAMES
    assert info.timeline_source == TIMELINE_FRAME_INDEX_OVER_FPS
    assert info.timing_trusted is True
    assert info.path == video.resolve()

    records: list[tuple[int, float, np.ndarray]] = []
    while True:
        record = source.read()
        if record is None:
            break
        records.append(record)

    # 전체 프레임 수와 index 0 부터의 연속 번호 (첫 프레임 누락 없음).
    assert source.frames_read == VIDEO_FRAMES
    assert [r[0] for r in records] == list(range(VIDEO_FRAMES))
    # 영상 시간은 frame_index / fps (SPEC.md 10.3).
    for i, (_, video_time, frame) in enumerate(records):
        assert video_time == pytest.approx(i / VIDEO_FPS, abs=1e-9)
        assert frame.shape == (VIDEO_H, VIDEO_W, 3)
        assert frame.dtype == np.uint8

    # 첫 read 가 원본 첫 프레임(어두운 프레임 0)이고 마지막이 밝은 프레임 29.
    assert float(records[0][2].mean()) < 30.0
    assert float(records[-1][2].mean()) > 150.0

    # EOF 후 진단 값 (SPEC.md 10.4: 판별 한계를 end_reason 으로 남긴다).
    assert source.end_reason == END_OF_STREAM
    source.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    """close 중복 호출은 안전하다 (SPEC.md 10.5)."""
    from contracts import ContractError

    video = make_synthetic_video(tmp_path)
    source = VideoSource(video)
    source.open()
    assert source.read() is not None
    source.close()
    source.close()  # 중복 호출도 예외 없음
    # 닫힌 뒤 read 는 계약 오류 (open 없이 read 금지).
    with pytest.raises(ContractError):
        source.read()


# ---------------------------------------------------------------------------
# 입력 오류 (SPEC.md 21절)
# ---------------------------------------------------------------------------


def test_missing_file_raises_video_not_found(tmp_path: Path) -> None:
    """없는 파일 -> VideoNotFoundError."""
    source = VideoSource(tmp_path / "no_such_file.mp4")
    with pytest.raises(VideoNotFoundError):
        source.open()


# ---------------------------------------------------------------------------
# FPS 계약 (SPEC.md 10.3)
# ---------------------------------------------------------------------------


def test_zero_fps_without_fallback_raises_invalid_fps(tmp_path: Path, monkeypatch) -> None:
    """monkeypatch 스텁 FPS=0 -> InvalidFpsError."""
    dummy = tmp_path / "fps_zero.mp4"
    dummy.write_bytes(b"not a real video")
    monkeypatch.setattr(
        video_source.cv2, "VideoCapture", lambda path: StubCapture(path, 0.0)
    )
    source = VideoSource(dummy)
    with pytest.raises(InvalidFpsError):
        source.open()


def test_zero_fps_with_fallback_marks_timing_untrusted(
    tmp_path: Path, monkeypatch
) -> None:
    """fallback 지정 시 timing_trusted False, 시간은 fallback FPS 기준 (SPEC.md 10.3)."""
    dummy = tmp_path / "fps_zero_fallback.mp4"
    dummy.write_bytes(b"not a real video")
    monkeypatch.setattr(
        video_source.cv2, "VideoCapture", lambda path: StubCapture(path, 0.0)
    )
    source = VideoSource(dummy, fallback_fps=30.0)
    info = source.open()

    assert info.timing_trusted is False
    assert info.timeline_source == TIMELINE_FALLBACK_FPS_CONSTANT
    assert info.fps == 30.0

    first = source.read()
    assert first is not None
    index, video_time, frame = first
    assert index == 0
    assert video_time == pytest.approx(0.0, abs=1e-12)
    second = source.read()
    assert second is not None
    assert second[0] == 1
    assert second[1] == pytest.approx(1 / 30.0, abs=1e-12)
    source.close()


def test_negative_fallback_fps_rejected(tmp_path: Path, monkeypatch) -> None:
    """fallback_fps 자체가 무효하면 InvalidFpsError (SPEC.md 10.3)."""
    dummy = tmp_path / "fps_bad_fallback.mp4"
    dummy.write_bytes(b"not a real video")
    monkeypatch.setattr(
        video_source.cv2, "VideoCapture", lambda path: StubCapture(path, float("nan"))
    )
    source = VideoSource(dummy, fallback_fps=-5.0)
    with pytest.raises(InvalidFpsError):
        source.open()


def test_double_open_raises_contract_error(tmp_path: Path) -> None:
    """이미 열린 상태에서 재호출은 ContractError (close 먼저)."""
    from contracts import ContractError

    video = make_synthetic_video(tmp_path)
    source = VideoSource(video)
    source.open()
    with pytest.raises(ContractError):
        source.open()
    source.close()


def test_unknown_frame_count_does_not_claim_verified_eof(tmp_path: Path, monkeypatch) -> None:
    dummy = tmp_path / "unknown.mp4"
    dummy.touch()
    capture = StubCapture(str(dummy), fps=25.0, frame_count=8)
    original_get = capture.get
    capture.get = lambda prop: 0.0 if prop == cv2.CAP_PROP_FRAME_COUNT else original_get(prop)
    monkeypatch.setattr(video_source.cv2, "VideoCapture", lambda _: capture)
    source = VideoSource(dummy)
    source.open()
    while source.read() is not None:
        pass
    assert source.end_reason == END_OF_STREAM_OR_DECODE_FAILURE
    source.close()


def test_explicit_decoder_exception_is_error_even_near_eof(tmp_path: Path) -> None:
    from contracts import VideoDecodeError
    source = VideoSource(tmp_path / "unused.mp4")
    source.frames_read = 8
    source._reported_frame_count = 8
    with pytest.raises(VideoDecodeError):
        source._handle_read_failure("decoder crashed")
