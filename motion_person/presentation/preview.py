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


from motion_person.presentation.config import Config
from motion_person.presentation.messages import error_message
from motion_person.domain.contracts import MotionPersonError


from motion_person.application.processor import FrameProcessor

# Compatibility constructor; the CLI itself uses MotionEngine.
PreviewProcessor = FrameProcessor


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="웹캠 미리보기 · 사진과 영상 저장 없음"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="camera index (default 0)"
    )
    parser.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--classes", nargs="+", default=["person"])
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--analysis-width", type=int, default=960)
    parser.add_argument("--min-object-motion-pixels", type=int, default=1000)
    parser.add_argument("--min-object-motion-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--show-mask", action="store_true")
    parser.add_argument(
        "--overlay",
        choices=["full", "objects", "none"],
        default="full",
        help="화면 표시: full 전체, objects 객체 박스만, none 박스·패널 숨김",
    )
    parser.add_argument("--no-mirror", action="store_true", help="disable mirror view")
    parser.add_argument(
        "--duration-sec", type=float, help="stop after this many live seconds"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="console-only diagnostic; needs --duration-sec",
    )
    parsed = parser.parse_args(args)
    if parsed.camera < 0:
        parser.error("--camera must be >= 0")
    if parsed.duration_sec is not None and (
        not math.isfinite(parsed.duration_sec) or parsed.duration_sec <= 0
    ):
        parser.error("--duration-sec must be finite and > 0")
    if parsed.no_display and parsed.duration_sec is None:
        parser.error("--no-display needs --duration-sec")
    if parsed.no_display and parsed.show_mask:
        parser.error("--show-mask needs display")
    return parsed


def main(args=None) -> int:
    from motion_person.application.engine import MotionEngine

    options = parse_args(args)
    try:
        config = replace(
            Config(),
            model_path=options.model,
            target_classes=tuple(
                dict.fromkeys(c.strip().lower() for c in options.classes)
            ),
            tracking=True,
            person_confidence=options.confidence,
            yolo_imgsz=options.imgsz,
            analysis_width=options.analysis_width,
            min_person_motion_pixels_ref=options.min_object_motion_pixels,
            min_person_motion_ratio=options.min_object_motion_ratio,
            warmup_sec=options.warmup_sec,
            show_mask=options.show_mask,
        )
        print("웹캠 미리보기를 준비해요. 사진·영상·로그는 저장하지 않아요.", flush=True)
        engine = MotionEngine.webcam(
            options.camera,
            config=config,
            capture=False,
            display=not options.no_display,
            overlay=options.overlay,
            mirror=not options.no_mirror,
        )
        summary = engine.start(duration_sec=options.duration_sec)
        print(
            f"미리보기 종료: {summary.frames_processed}프레임 처리 · 움직임 {summary.candidate_frames}프레임 · 저장 0장",
            flush=True,
        )
        return 130 if summary.end_reason == "USER_INTERRUPT" else 0
    except KeyboardInterrupt:
        return 130
    except MotionPersonError as exc:
        print(error_message(exc), file=sys.stderr, flush=True)
        return exc.exit_code()


if __name__ == "__main__":
    sys.exit(main())
