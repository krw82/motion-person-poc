"""전체 orchestration 진입점 (SPEC.md 18~22절).

main 은 설정 검증, 실행 준비, 프레임 반복, 로그 기록, 재생 대기와 종료
정리를 지휘한다. 프레임 반복의 핵심 계약 (SPEC.md 18.3):

- 한 프레임에 YOLO 추론 1회, MOG2 apply 1회, EventDetector.evaluate 1회,
  CaptureManager.maybe_save 1회만 호출한다.
- 모델 load, 해시 계산, 실행 폴더 생성은 반복문 밖에서 한 번만 한다.
- 화면에 박스를 그린 display 프레임을 MOG2 입력으로 재사용하지 않는다.
- 영상 시간(video_time_sec)은 VideoSource 반환값만 사용한다 (SPEC.md 3.4).

시작 순서는 SPEC.md 18.1 을 따른다: 설정 검증 -> run_id/로그 준비(RUN_START)
-> 영상 open(VIDEO_OPENED, 비신뢰 타이밍이면 WARNING 별도 이벤트) -> 모델
load/warmup(MODEL_READY) -> MOG2/EventDetector/CaptureManager 준비 ->
(display 면) 시연 창 -> 프레임 반복.

종료 사유는 END_OF_VIDEO, USER_STOP, USER_INTERRUPT, ERROR 네 가지다.
정리(영상 close, 창 close, RUN_END + summary.json, logger close)는 각각
개별 try 로 감싸 하나가 실패해도 나머지를 시도한다 (SPEC.md 18.2).
"""

from __future__ import annotations

import math
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from capture_manager import CaptureManager
from console_messages import error_message
from config import Config, build_config, config_to_dict
from contracts import (
    EXIT_CODES,
    CaptureResult,
    CaptureSaveError,
    ContractError,
    DisplayError,
    FrameDecision,
    FramePacket,
    LogWriteError,
    MotionPersonError,
    MotionResult,
    VideoInfo,
)
from event_detector import EventDetector
from frame_processor import analysis_geometry, box_to_original, effective_threshold, make_packet
from motion_detector import MotionDetector
from object_detector import PersonDetector
from object_state import ObjectStateStore
from overlay_renderer import DisplayWindow, render_mask_view, render_overlay
from run_logger import RunLogger
from video_source import END_OF_STREAM, VideoSource

__all__ = ["main"]

# 종료 사유 (SPEC.md 18.2, 21절). EOF/ESC/q 는 0, Ctrl+C 는 130.
END_OF_VIDEO = "END_OF_VIDEO"
USER_STOP = "USER_STOP"
USER_INTERRUPT = "USER_INTERRUPT"
ERROR = "ERROR"

# 단계별 성능 지표 이름 (SPEC.md 19.3).
_STAGE_NAMES: tuple[str, ...] = (
    "read_ms",
    "resize_ms",
    "yolo_ms",
    "mog2_ms",
    "decision_ms",
    "capture_ms",
    "render_ms",
    "pacing_ms",
)

# 카운터 이름 (SPEC.md 20.5).
_COUNTER_NAMES: tuple[str, ...] = (
    "frames_read",
    "frames_processed",
    "warmup_frames",
    "person_present_frames",
    "object_present_frames",
    "candidate_frames",
    "cooldown_suppressed_frames",
    "successful_capture_count",
    "failed_capture_count",
)


# ---------------------------------------------------------------------------
# 소형 헬퍼
# ---------------------------------------------------------------------------


def _make_run_id() -> str:
    """UTC 시각 + 짧은 uuid 로 실행 ID 를 만든다 (SPEC.md 16.4)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _elapsed_ms(started: float) -> float:
    """perf_counter 시작점부터의 경과 밀리초."""
    return (time.perf_counter() - started) * 1000.0


def _print_error(exc: MotionPersonError, *, friendly: bool = False) -> None:
    """오류 코드, 원인, 관련 키/경로를 stderr 로 출력한다 (SPEC.md 21절)."""
    if friendly:
        print(error_message(exc), file=sys.stderr)
        return
    print(f"[{exc.code}] {exc.message}", file=sys.stderr)
    key = exc.context.get("key")
    if key is not None:
        print(f"  key: {key}", file=sys.stderr)
    path = exc.context.get("path")
    if path is not None:
        print(f"  path: {path}", file=sys.stderr)
    if isinstance(exc, DisplayError):
        print("  hint: retry with --no-display for headless runs", file=sys.stderr)


def _fps_or_none(numerator: float, denominator_sec: float | None) -> float | None:
    """분모가 0 이하이면 null 을 반환하는 fps 계산 (SPEC.md 26.3 참고)."""
    if denominator_sec is None or denominator_sec <= 0.0:
        return None
    if not math.isfinite(denominator_sec):
        return None
    value = numerator / denominator_sec
    return value if math.isfinite(value) else None


def _round_fps(value: float | None) -> float | None:
    """로그 기록용 fps 반올림. None 은 그대로 둔다."""
    return None if value is None else round(value, 3)


def _analysis_size(video_info: VideoInfo, config: Config) -> tuple[int, int]:
    """warmup 더미 프레임에 쓸 분석 크기 (SPEC.md 11.1 공식과 동일)."""
    _, width, height = analysis_geometry(video_info.width, video_info.height,
                                         config.analysis_width, config.analysis_roi)
    return width, height


def _current_fps(
    counters: dict[str, int],
    stage_totals: dict[str, float],
    run_stats: dict[str, Any],
    now: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    """현재 시점의 (processing, playback, yolo_capacity) fps (SPEC.md 19.3).

    processing_fps 는 반복문 시작 후 경과에서 재생 대기(pacing) 합계를
    뺀 처리시간을 분모로 쓰고, playback_fps 는 전체 실제 경과시간을,
    yolo_capacity_fps 는 YOLO 호출 횟수 / YOLO 시간 합을 사용한다.
    """
    loop_start = run_stats.get("loop_started_perf")
    if loop_start is None:
        return None, None, None
    now = time.perf_counter() if now is None else now
    playback_sec = now - float(loop_start)
    processing_sec = playback_sec - stage_totals.get("pacing_ms", 0.0) / 1000.0
    yolo_sec = stage_totals.get("yolo_ms", 0.0) / 1000.0
    processed = counters["frames_processed"]
    return (
        _fps_or_none(processed, processing_sec),
        _fps_or_none(processed, playback_sec),
        _fps_or_none(run_stats.get("yolo_calls", 0), yolo_sec),
    )


# ---------------------------------------------------------------------------
# 로그 payload 빌더
# ---------------------------------------------------------------------------


def _capture_saved_payload(
    packet: FramePacket,
    decision: FrameDecision,
    capture: CaptureResult,
    config: Config,
) -> dict:
    """CAPTURE_SAVED 이벤트 payload (SPEC.md 20.4 예시 구조 그대로)."""
    analysis_w, analysis_h = packet.analysis_size
    original_w, original_h = packet.original_size
    matched_persons: list[dict[str, Any]] = []
    for evidence in decision.persons:
        if not evidence.qualifies:
            continue
        matched_persons.append(
            {
                "detection_index": evidence.detection_index,
                "class_id": evidence.class_id,
                "class_name": evidence.class_name,
                "track_id": evidence.track_id,
                "tracker_id": evidence.tracker_id,
                "label": evidence.label,
                "observation_hits": evidence.observation_hits,
                "confirmed": evidence.confirmed,
                "confidence": round(float(evidence.confidence), 4),
                "analysis_box_xyxy": evidence.box.to_list(),
                "original_box_xyxy": box_to_original(evidence.box, packet).to_list(),
                "person_box_area": evidence.box.area,
                "object_box_area": evidence.box.area,
                "motion_pixels": evidence.motion_pixels,
                "motion_ratio": round(float(evidence.motion_ratio), 6),
            }
        )
    return {
        "frame_index": packet.frame_index,
        "video_time_sec": packet.video_time_sec,
        "status": decision.status,
        "capture_sequence": capture.sequence,
        "capture_path": str(capture.path) if capture.path is not None else None,
        "original_size": [int(original_w), int(original_h)],
        "analysis_size": [int(analysis_w), int(analysis_h)],
        "analysis_roi_xyxy": packet.roi_xyxy,
        "effective_min_person_motion_pixels": effective_threshold(
            config.min_person_motion_pixels_ref, int(analysis_w), int(analysis_h),
            reference_width=config.reference_width,
            reference_height=config.reference_height,
        ),
        "min_person_motion_ratio": config.min_person_motion_ratio,
        "matched_objects": matched_persons,
        "matched_persons": [e for e in matched_persons if e["class_name"] == "person"],
        "capture_scope": config.capture_scope,
        "saved_track_ids": list(capture.saved_track_ids),
    }


def _frame_decision_payload(
    packet: FramePacket,
    decision: FrameDecision,
    motion: MotionResult,
    capture: CaptureResult,
) -> dict:
    """FRAME_DECISION(debug_decisions) 이벤트 payload (SPEC.md 20.3)."""
    payload = {
        "frame_index": packet.frame_index,
        "video_time_sec": packet.video_time_sec,
        "status": decision.status,
        "candidate": decision.candidate,
        "warming_up": bool(motion.warming_up),
        "foreground_pixels": int(motion.foreground_pixels),
        "frame_foreground_ratio": round(float(motion.frame_foreground_ratio), 6),
        "foreground_regions": [
            {
                "box_xyxy": region.box.to_list(),
                "contour_area": round(float(region.contour_area), 3),
            }
            for region in motion.regions
        ],
        "persons": [
            {
                "detection_index": evidence.detection_index,
                "class_id": evidence.class_id,
                "class_name": evidence.class_name,
                "track_id": evidence.track_id,
                "tracker_id": evidence.tracker_id,
                "label": evidence.label,
                "observation_hits": evidence.observation_hits,
                "confirmed": evidence.confirmed,
                "confidence": round(float(evidence.confidence), 4),
                "box_xyxy": evidence.box.to_list(),
                "person_box_area": evidence.box.area,
                "object_box_area": evidence.box.area,
                "motion_pixels": evidence.motion_pixels,
                "motion_ratio": round(float(evidence.motion_ratio), 6),
                "qualifies": evidence.qualifies,
                "rejection_reasons": list(evidence.rejection_reasons),
            }
            for evidence in decision.persons
        ],
        "capture_status": capture.status,
        "saved_track_ids": list(capture.saved_track_ids),
    }
    payload["objects"] = payload["persons"]
    payload["persons"] = [o for o in payload["objects"] if o["class_name"] == "person"]
    return payload


def _metrics_payload(
    packet: FramePacket,
    counters: dict[str, int],
    stage_totals: dict[str, float],
    run_stats: dict[str, Any],
) -> dict:
    """주기적 METRICS payload: 카운터 전부 + fps 3종 + 단계별 누적 시간."""
    processing_fps, playback_fps, yolo_capacity_fps = _current_fps(
        counters, stage_totals, run_stats
    )
    payload: dict[str, Any] = {
        "frame_index": packet.frame_index,
        "video_time_sec": packet.video_time_sec,
    }
    payload.update({name: counters[name] for name in _COUNTER_NAMES})
    payload.update(
        {
            "processing_fps": _round_fps(processing_fps),
            "playback_fps": _round_fps(playback_fps),
            "yolo_capacity_fps": _round_fps(yolo_capacity_fps),
            "stage_totals_ms": {
                name: round(stage_totals[name], 3) for name in _STAGE_NAMES
            },
        }
    )
    return payload


def _final_report(
    counters: dict[str, int],
    stage_totals: dict[str, float],
    run_stats: dict[str, Any],
    video_end_reason: str | None,
    model_init_ms: float | None,
) -> dict:
    """RUN_END 이벤트와 summary.json 이 공유하는 종료 요약 (SPEC.md 20.5).

    분모가 0인 fps 지표는 null 이다 (SPEC.md 26.3).
    """
    loop_start = run_stats.get("loop_started_perf")
    loop_end = run_stats.get("loop_ended_perf")
    processing_fps: float | None = None
    playback_fps: float | None = None
    yolo_capacity_fps: float | None = None
    if loop_start is not None and loop_end is not None and loop_end > loop_start:
        processing_fps, playback_fps, yolo_capacity_fps = _current_fps(
            counters, stage_totals, run_stats, now=float(loop_end)
        )
    report: dict[str, Any] = {name: counters[name] for name in _COUNTER_NAMES}
    report.update(
        {
            "processing_fps": _round_fps(processing_fps),
            "playback_fps": _round_fps(playback_fps),
            "yolo_capacity_fps": _round_fps(yolo_capacity_fps),
            "video_end_reason": video_end_reason,
            "model_init_ms": (
                round(model_init_ms, 3) if model_init_ms is not None else None
            ),
            "stage_totals_ms": {
                name: round(stage_totals[name], 3) for name in _STAGE_NAMES
            },
        }
    )
    object_store = run_stats.get("object_store")
    if object_store is not None:
        report["objects"] = object_store.snapshot()
        report["tracked_object_count"] = len(object_store.objects)
        report["confirmed_object_count"] = sum(o.get("confirmed", True) for o in object_store.objects.values())
        report["class_identity_splits"] = run_stats.get("class_identity_splits", 0)
    return report


# ---------------------------------------------------------------------------
# 재생 대기 (SPEC.md 19.2)
# ---------------------------------------------------------------------------


def _wait_until_deadline_display(
    window: DisplayWindow, deadline: float
) -> str | None:
    """창이 있을 때의 재생 대기: waitKey 폴링으로 키 처리를 병행한다.

    누적 마감시각까지 대기하되 매 루프에서 waitKey(1)(DisplayWindow.poll_key)
    로 키와 창 이벤트를 처리한다. 종료 키("esc"/"quit")나 창 닫힘을 감지하면
    사유 문자열을 반환한다. 남은 시간이 많으면 중간에 짧은 sleep 만 쉰다.
    """
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return None
        key = window.poll_key()
        if key is not None:
            return key
        if window.is_closed():
            return "closed"
        remaining = deadline - time.perf_counter()
        if remaining > 0.002:
            time.sleep(min(remaining - 0.002, 0.005))


def _wait_until_deadline_headless(deadline: float) -> None:
    """창이 없을 때의 재생 대기: 남은 시간만 sleep 한다 (SPEC.md 19.2)."""
    remaining = deadline - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)


# ---------------------------------------------------------------------------
# 프레임 반복 (SPEC.md 18.2, 18.3)
# ---------------------------------------------------------------------------


def _run_frame_loop(
    config: Config,
    run_id: str,
    video: VideoSource,
    video_info: VideoInfo,
    person_detector: PersonDetector,
    motion_detector: MotionDetector,
    event_detector: EventDetector,
    capture_manager: CaptureManager,
    window: DisplayWindow | None,
    logger: RunLogger,
    counters: dict[str, int],
    stage_totals: dict[str, float],
    run_stats: dict[str, Any],
) -> str:
    """프레임 반복을 실행하고 종료 사유(END_OF_VIDEO/USER_STOP)를 반환한다.

    KeyboardInterrupt 와 MotionPersonError 는 호출자(main)의 예외 처리로
    전파한다. video_time 은 VideoSource.read 반환값만 사용하고(SPEC.md
    3.4), 한 프레임당 각 단계를 정확히 1회씩 호출한다(SPEC.md 18.3).
    """
    frame_interval = 1.0 / float(video_info.fps)
    run_stats["loop_started_perf"] = time.perf_counter()
    run_stats["yolo_calls"] = 0
    next_deadline = float(run_stats["loop_started_perf"]) + frame_interval
    last_metrics_at = float(run_stats["loop_started_perf"])
    last_status: str | None = None
    object_store = ObjectStateStore() if config.tracking else None
    run_stats["object_store"] = object_store

    try:
        while True:
            # 1) 프레임 읽기 (read_ms). None 이면 EOF 다.
            started = time.perf_counter()
            read_result = video.read()
            stage_totals["read_ms"] += _elapsed_ms(started)
            if read_result is None:
                if video.end_reason != END_OF_STREAM:
                    logger.write("WARNING", {
                        "reason": "INPUT_COMPLETION_UNVERIFIED",
                        "video_end_reason": video.end_reason,
                        "message": "Stream ended; full input completion could not be verified.",
                    })
                return END_OF_VIDEO
            frame_index, video_time_sec, raw_frame = read_result
            counters["frames_read"] += 1
            run_stats["frame_index"] = frame_index
            run_stats["video_time_sec"] = video_time_sec

            # 2) 분석 패킷 생성 (resize_ms).
            started = time.perf_counter()
            packet = make_packet(
                run_id, frame_index, video_time_sec, raw_frame, config.analysis_width, config.analysis_roi
            )
            stage_totals["resize_ms"] += _elapsed_ms(started)

            # 3) YOLO 사람 탐지 (yolo_ms). 프레임당 추론 1회.
            started = time.perf_counter()
            persons = person_detector.detect(packet.analysis_frame)
            run_stats["class_identity_splits"] = person_detector.class_identity_splits
            stage_totals["yolo_ms"] += _elapsed_ms(started)
            run_stats["yolo_calls"] = int(run_stats["yolo_calls"]) + 1

            # 4) MOG2 전경 탐지 (mog2_ms). 프레임당 apply 1회.
            started = time.perf_counter()
            motion = motion_detector.detect(
                packet.analysis_frame, packet.video_time_sec
            )
            stage_totals["mog2_ms"] += _elapsed_ms(started)

            # 5) 결합 판정 (decision_ms).
            started = time.perf_counter()
            decision = event_detector.evaluate(packet, persons, motion)
            stage_totals["decision_ms"] += _elapsed_ms(started)
            counters["frames_processed"] += 1
            if decision.status == "WARMUP":
                counters["warmup_frames"] += 1
            if any(o.class_name == "person" for o in persons):
                counters["person_present_frames"] += 1
            if persons:
                counters["object_present_frames"] += 1
            if decision.candidate:
                counters["candidate_frames"] += 1

            # 6) 저장 시도 (capture_ms). 프레임당 maybe_save 1회.
            started = time.perf_counter()
            capture = capture_manager.maybe_save(packet, decision)
            stage_totals["capture_ms"] += _elapsed_ms(started)
            if capture.status == "COOLDOWN":
                # 쿨다운 프레임마다 콘솔 출력을 하지 않고 카운터만 올린다.
                counters["cooldown_suppressed_frames"] += 1
            elif capture.status == "SAVED":
                counters["successful_capture_count"] += 1
                logger.write(
                    "CAPTURE_SAVED",
                    _capture_saved_payload(packet, decision, capture, config),
                )

            if object_store is not None:
                for first in object_store.update(decision, capture):
                    logger.write("OBJECT_FIRST_SEEN", {
                        "frame_index": frame_index, "video_time_sec": video_time_sec, **first,
                    })

            # 7) 상태 변화/디버그 로그.
            if decision.status != last_status:
                logger.write(
                    "STATUS_CHANGED",
                    {
                        "frame_index": frame_index,
                        "video_time_sec": video_time_sec,
                        "previous_status": last_status,
                        "status": decision.status,
                        "candidate": decision.candidate,
                    },
                )
                last_status = decision.status
            if config.debug_decisions:
                logger.write(
                    "FRAME_DECISION",
                    _frame_decision_payload(packet, decision, motion, capture),
                )

            # 8) 화면 표시 (render_ms). analysis_frame copy 위에만 그린다.
            if window is not None:
                started = time.perf_counter()
                processing_fps, playback_fps, _capacity = _current_fps(
                    counters, stage_totals, run_stats
                )
                display_metrics = {
                    "processing_fps": processing_fps,
                    "playback_fps": playback_fps,
                    "successful_capture_count": counters["successful_capture_count"],
                    "tracking": config.tracking,
                }
                display_frame = render_overlay(
                    packet, persons, motion, decision, capture, display_metrics,
                    overlay_mode=config.overlay_mode,
                )
                mask_view = (
                    render_mask_view(motion) if window.show_mask else None
                )
                window.show(display_frame, mask_view)
                stage_totals["render_ms"] += _elapsed_ms(started)
                key = window.poll_key()
                if key is not None or window.is_closed():
                    return USER_STOP

            # 9) 재생 대기 (pacing_ms). fast 는 대기 없이 진행한다.
            started = time.perf_counter()
            pace_stop: str | None = None
            if config.pace == "realtime":
                if window is not None:
                    pace_stop = _wait_until_deadline_display(window, next_deadline)
                else:
                    _wait_until_deadline_headless(next_deadline)
            stage_totals["pacing_ms"] += _elapsed_ms(started)
            next_deadline += frame_interval
            if pace_stop is not None:
                return USER_STOP

            # 10) 주기 성능 로그(실제 시간 기준, SPEC.md 20.5).
            now = time.perf_counter()
            if (
                now - last_metrics_at >= config.metrics_interval_sec
                and counters["frames_processed"] > 0
            ):
                logger.write(
                    "METRICS",
                    _metrics_payload(packet, counters, stage_totals, run_stats),
                )
                last_metrics_at = now
    finally:
        run_stats["loop_ended_perf"] = time.perf_counter()


# ---------------------------------------------------------------------------
# 진입점 (SPEC.md 18.1, 21절, 22절)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, friendly: bool = False) -> int:
    """명령행 인자를 받아 실행하고 프로세스 종료 코드를 반환한다."""
    try:
        config = build_config(argv)
    except MotionPersonError as exc:
        # ConfigError(2) 뿐 아니라 파일 부재의 VideoNotFoundError(3) /
        # ModelNotFoundError(4) 도 같은 경로로 종료한다 (SPEC.md 21절).
        _print_error(exc, friendly=friendly)
        return exc.exit_code()
    config_dict = config_to_dict(config)
    run_id = _make_run_id()

    counters: dict[str, int] = {name: 0 for name in _COUNTER_NAMES}
    stage_totals: dict[str, float] = {name: 0.0 for name in _STAGE_NAMES}
    run_stats: dict[str, Any] = {}

    video: VideoSource | None = None
    logger: RunLogger | None = None
    logger_started = False
    window: DisplayWindow | None = None
    model_init_ms: float | None = None
    end_reason: str | None = None
    error_exc: MotionPersonError | None = None
    # 정리 단계에서 로그 쓰기가 실패하면 종료 코드 5 로 보고한다
    # (SPEC.md 21절 LOG_WRITE_ERROR, 20.6).
    log_write_failed = False

    def _record_error(exc: MotionPersonError) -> None:
        """알려진 오류와 계약 밖 예외가 같은 경로로 종료되게 한다."""
        nonlocal end_reason, error_exc
        end_reason = ERROR
        error_exc = exc
        if isinstance(exc, CaptureSaveError):
            counters["failed_capture_count"] += 1
        if logger_started and logger is not None:
            # ERROR 이벤트 기록 자체가 실패해도 원본 오류 처리를 우선한다.
            try:
                logger.write(
                    "ERROR",
                    {
                        **exc.to_payload(),
                        "frame_index": run_stats.get("frame_index"),
                        "video_time_sec": run_stats.get("video_time_sec"),
                    },
                )
            except Exception as log_exc:
                print(f"[CLEANUP] ERROR event write failed: {log_exc}",
                      file=sys.stderr)
        _print_error(exc, friendly=friendly)

    try:
        # 2~3단계: run_id, 캡처 폴더와 로그 폴더를 먼저 만든다(SPEC.md 18.1).
        # 캡처 폴더 생성 실패를 영상/모델 열기 전에 발견한다.
        logger = RunLogger(config.log_dir, run_id)
        logger.start(config_dict, {})
        logger_started = True
        print(f"실행 ID: {run_id}" if friendly else f"run_id: {run_id}")
        print(f"로그 폴더: {logger.run_dir}" if friendly else f"log dir: {logger.run_dir}")

        capture_manager = CaptureManager(
            config.capture_dir,
            cooldown_sec=config.capture_cooldown_sec,
            jpeg_quality=config.jpeg_quality,
            scope=config.capture_scope,
        )
        capture_manager.prepare(run_id)
        print(f"캡처 폴더: {config.capture_dir / run_id}" if friendly else f"capture dir: {config.capture_dir / run_id}")

        # 4단계: 영상을 열고 메타데이터를 확인한다. RUN_START 시점에는
        # 영상을 아직 열지 않았으므로 메타데이터는 VIDEO_OPENED 에 실어
        # 보낸다.
        video = VideoSource(config.video_path, fallback_fps=config.fallback_fps)
        video_info = video.open()
        analysis_w, analysis_h = _analysis_size(video_info, config)
        logger.write(
            "VIDEO_OPENED",
            {
                **video_info.to_dict(),
                # §22.4: 입력 기준값과 실제 분석 크기 보정값을 함께 기록한다.
                # 캡처 0건 실행에서도 실제 적용된 보정 임계값이 남게 한다.
                "analysis_size": [analysis_w, analysis_h],
                "analysis_roi": config.analysis_roi,
                "analysis_roi_xyxy": analysis_geometry(video_info.width, video_info.height,
                                                        config.analysis_width, config.analysis_roi)[0],
                "min_person_motion_pixels_ref": config.min_person_motion_pixels_ref,
                "effective_min_person_motion_pixels": effective_threshold(
                    config.min_person_motion_pixels_ref, analysis_w, analysis_h,
                    reference_width=config.reference_width,
                    reference_height=config.reference_height,
                ),
            },
        )
        if not video_info.timing_trusted:
            logger.write(
                "WARNING",
                {
                    "reason": "TIMING_NOT_TRUSTED",
                    "message": (
                        "video FPS metadata is invalid; fallback_fps is in use"
                    ),
                    "timeline_source": video_info.timeline_source,
                    "fps": video_info.fps,
                    "fallback_fps": config.fallback_fps,
                },
            )

        person_detector = PersonDetector(config)
        warmup_w, warmup_h = analysis_w, analysis_h
        model_started = time.perf_counter()
        person_detector.load()
        tracker_settings = (person_detector.prepare_tracking(logger.run_dir / "tracker.yaml")
                            if config.tracking else None)
        person_detector.warmup(warmup_w, warmup_h)
        model_init_ms = _elapsed_ms(model_started)
        logger.write(
            "MODEL_READY",
            {
                "model_path": str(config.model_path),
                "sha256": person_detector.model_sha256,
                "person_class_id": person_detector.person_class_id,
                "target_classes": {str(i): person_detector.class_names[i]
                                   for i in person_detector.target_class_ids},
                "tracking": config.tracking,
                "tracker": config.tracker if config.tracking else None,
                "tracker_settings": tracker_settings,
                "tracking_profile": config.tracking_profile if config.tracking else None,
                "track_min_hits": config.track_min_hits if config.tracking else None,
                "model_init_ms": round(model_init_ms, 3),
            },
        )

        motion_detector = MotionDetector(config)
        event_detector = EventDetector(config)

        if config.display:
            window = DisplayWindow(show_mask=config.show_mask)

        end_reason = _run_frame_loop(
            config,
            run_id,
            video,
            video_info,
            person_detector,
            motion_detector,
            event_detector,
            capture_manager,
            window,
            logger,
            counters,
            stage_totals,
            run_stats,
        )
    except KeyboardInterrupt:
        end_reason = USER_INTERRUPT
        print(f"[USER_INTERRUPT] interrupted by Ctrl+C (exit 130)", file=sys.stderr)
    except MotionPersonError as exc:
        _record_error(exc)
    except Exception as exc:  # 계약 밖 예외(cv2.error 등)도 오류로 종료한다
        _record_error(
            ContractError(
                f"unexpected error: {exc!r}",
                context={"type": type(exc).__name__},
            )
        )
    finally:
        # 각 정리를 개별 try 로 감싸 하나 실패해도 나머지를 시도한다(18.2).
        video_end_reason = video.end_reason if video is not None else None
        final_reason = end_reason if end_reason is not None else ERROR

        if video is not None:
            try:
                video.close()
            except Exception as exc:
                print(f"[CLEANUP] video close failed: {exc}", file=sys.stderr)
        if window is not None:
            try:
                window.close()
            except Exception as exc:
                print(f"[CLEANUP] window close failed: {exc}", file=sys.stderr)

        report = _final_report(
            counters, stage_totals, run_stats, video_end_reason, model_init_ms
        )
        report["end_reason"] = final_reason
        report["input_completion_verified"] = (
            final_reason == END_OF_VIDEO and video_end_reason == END_OF_STREAM
        )
        if logger_started and logger is not None:
            # start() 가 실패한 실행은 스트림이 없으므로 정리 쓰기를 시도하지
            # 않는다(SPEC.md 18.2 — 생성하지 못한 자원은 정리 대상 제외).
            try:
                logger.write("RUN_END", dict(report))
            except LogWriteError as exc:
                # 정리 단계 로그 실패도 재현성 손실이다(SPEC.md 20.6, 21절).
                log_write_failed = True
                print(f"[CLEANUP] RUN_END write failed: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"[CLEANUP] RUN_END write failed: {exc}", file=sys.stderr)
            try:
                logger.write_summary(
                    {"schema_version": 1, "run_id": run_id, **report}
                )
            except LogWriteError as exc:
                log_write_failed = True
                print(f"[CLEANUP] summary write failed: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"[CLEANUP] summary write failed: {exc}", file=sys.stderr)
            try:
                logger.close()
            except LogWriteError as exc:
                log_write_failed = True
                print(f"[CLEANUP] logger close failed: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"[CLEANUP] logger close failed: {exc}", file=sys.stderr)

        if friendly:
            reason = {END_OF_VIDEO: '영상 끝', USER_STOP: '사용자 종료',
                      USER_INTERRUPT: '사용자 중단', ERROR: '오류'}.get(final_reason, final_reason)
            print(f"완료: {reason} · {counters['frames_processed']}프레임 처리 · 캡처 {counters['successful_capture_count']}장")
        else:
            print(f"end_reason={final_reason} frames={counters['frames_processed']} "
                  f"captures={counters['successful_capture_count']}")

    if error_exc is not None:
        return error_exc.exit_code()
    if end_reason == USER_INTERRUPT:
        return 130
    if log_write_failed:
        # 정상 종료처럼 보여도 로그를 못 남긴 실행은 오류로 보고한다
        # (SPEC.md 21절 LOG_WRITE_ERROR=5, 20.6).
        return EXIT_CODES["LOG_WRITE_ERROR"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
