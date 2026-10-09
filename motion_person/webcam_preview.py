"""Webcam visual test: YOLO + tracking + MOG2, without image/video/log saving.

Only the effective tracker YAML uses a temporary directory, removed on exit.
This entry point does not import CaptureManager or RunLogger.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
import time

import cv2

from .config import Config, validate_analysis_values
from .console_messages import error_message
from .contracts import CaptureResult, ContractError, MotionPersonError
from .event_detector import EventDetector
from .frame_processor import make_packet
from .motion_detector import MotionDetector
from .object_detector import ObjectDetector
from .overlay_renderer import DisplayWindow, render_mask_view, render_overlay
from .webcam_source import WebcamSource


from .processor import FrameProcessor as PreviewProcessor


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="웹캠 미리보기 · 사진과 영상 저장 없음")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--classes", nargs="+", default=["person"])
    parser.add_argument("--confidence", type=float, default=.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--analysis-width", type=int, default=960)
    parser.add_argument("--min-object-motion-pixels", type=int, default=1000)
    parser.add_argument("--min-object-motion-ratio", type=float, default=.03)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--show-mask", action="store_true")
    parser.add_argument("--overlay", choices=["full", "objects", "none"], default="full",
                        help="화면 표시: full 전체, objects 객체 박스만, none 박스·패널 숨김")
    parser.add_argument("--no-mirror", action="store_true", help="disable mirror view")
    parser.add_argument("--duration-sec", type=float, help="stop after this many live seconds")
    parser.add_argument("--no-display", action="store_true", help="console-only diagnostic; needs --duration-sec")
    parsed = parser.parse_args(args)
    if parsed.camera < 0:
        parser.error("--camera must be >= 0")
    if parsed.duration_sec is not None and (not math.isfinite(parsed.duration_sec) or parsed.duration_sec <= 0):
        parser.error("--duration-sec must be finite and > 0")
    if parsed.no_display and parsed.duration_sec is None:
        parser.error("--no-display needs --duration-sec")
    if parsed.no_display and parsed.show_mask:
        parser.error("--show-mask needs display")
    return parsed


def main(args=None) -> int:
    options = parse_args(args)
    camera = window = None
    processed = detected = moving = 0
    started = None
    exit_code = 0
    try:
        config = replace(Config(), model_path=options.model,
                         target_classes=tuple(dict.fromkeys(c.strip().lower() for c in options.classes)),
                         tracking=True, person_confidence=options.confidence,
                         yolo_imgsz=options.imgsz, analysis_width=options.analysis_width,
                         min_person_motion_pixels_ref=options.min_object_motion_pixels,
                         min_person_motion_ratio=options.min_object_motion_ratio,
                         warmup_sec=options.warmup_sec, overlay_mode=options.overlay)
        processor = PreviewProcessor(config)
        with tempfile.TemporaryDirectory(prefix="motion-person-tracker-") as temporary:
            print("웹캠 미리보기를 준비해요. 사진·영상·로그는 저장하지 않아요.", flush=True)
            processor.load(Path(temporary) / "tracker.yaml")
            camera = WebcamSource(options.camera)
            width, height = camera.open()
            started = time.perf_counter()
            colors = {"full": "녹색: 탐지됨 · 노란색: 움직임 조건 충족 · 회색: 추적 확인 중",
                      "objects": "녹색: 탐지됨 · 회색: 추적 확인 중 · 노란 강조 숨김",
                      "none": "박스·상태 패널을 숨기고 카메라 화면을 표시해요."}[config.overlay_mode]
            print(f"웹캠 {options.camera}: {width}×{height}\n{colors}\n"
                  f"처음 {options.warmup_sec:g}초는 배경 학습이에요. ESC 또는 q로 종료해요.", flush=True)
            if not options.no_display:
                window = DisplayWindow(show_mask=options.show_mask)
            last_metrics = started
            while True:
                index, elapsed, raw = camera.read()
                if options.duration_sec is not None and elapsed >= options.duration_sec:
                    break
                if not options.no_mirror:
                    raw = cv2.flip(raw, 1)
                packet, objects, motion, decision, capture = processor.process(raw, index, elapsed)
                processed += 1
                detected += bool(objects)
                moving += decision.candidate
                now = time.perf_counter()
                fps = processed / max(now - started, 1e-9)
                metrics = {"preview": True, "tracking": True, "processing_fps": fps,
                           "frame_age_ms": max(0, (now - camera.started_perf - elapsed) * 1000)}
                if window is not None:
                    display = render_overlay(packet, objects, motion, decision, capture, metrics,
                                             overlay_mode=config.overlay_mode)
                    window.show(display, render_mask_view(motion) if options.show_mask else None)
                    if window.poll_key() is not None or window.is_closed():
                        break
                if now - last_metrics >= 2:
                    print(f"현재 탐지 {len(objects)}개 · 움직임 {sum(o.qualifies for o in decision.objects)}개 · "
                          f"처리 {fps:.1f} FPS · 프레임 경과 {metrics['frame_age_ms']:.0f}ms", flush=True)
                    last_metrics = now
    except KeyboardInterrupt:
        exit_code = 130
    except MotionPersonError as exc:
        print(error_message(exc), file=sys.stderr, flush=True)
        exit_code = exc.exit_code()
    finally:
        for resource in (camera, window):
            if resource is not None:
                try:
                    resource.close()
                except MotionPersonError as exc:
                    print(f"[CLEANUP] {exc.message}", file=sys.stderr, flush=True)
                    if not exit_code:
                        exit_code = exc.exit_code()
        if started is not None:
            print(f"미리보기 종료: {processed}프레임 처리 · 대상 관측 {detected}프레임 · 움직임 {moving}프레임 · 저장 0장", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
