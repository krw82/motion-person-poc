"""Shared one-inference-per-frame analysis for files, webcams and preview."""

from __future__ import annotations
import math
import time
from typing import Callable
from pathlib import Path
from motion_person.domain.settings import Config, validate_analysis_values
from motion_person.domain.contracts import ContractError, FrameAnalysis
from motion_person.application.ports import Detector, MotionAnalyzer, MotionRules
from motion_person.composition import analysis_components


class FrameProcessor:
    """Process frames without opening a camera, showing a window, or saving pixels."""

    def __init__(
        self,
        config: Config,
        run_id: str = "webcam-preview",
        *,
        detector: Detector | None = None,
        motion: MotionAnalyzer | None = None,
        events: MotionRules | None = None,
        prepare_detector: Callable[[Path], None] | None = None,
        frame_builder=None,
    ):
        self.run_id = run_id
        validate_analysis_values(config)
        self.config = config
        (
            self.detector,
            self.motion,
            self.events,
            self._frame_builder,
            self._prepare_detector,
        ) = analysis_components(
            config, detector, motion, events, frame_builder, prepare_detector
        )
        self._last_index = -1
        self._last_time = -1.0

    def load(self, tracker_path: Path) -> None:
        if self._prepare_detector is not None:
            self._prepare_detector(tracker_path)

    def process(
        self,
        raw_frame,
        frame_index: int,
        elapsed_sec: float,
        *,
        stage_totals: dict[str, float] | None = None,
    ) -> FrameAnalysis:
        if (
            not isinstance(frame_index, int)
            or isinstance(frame_index, bool)
            or frame_index <= self._last_index
            or not math.isfinite(elapsed_sec)
            or elapsed_sec < 0
            or elapsed_sec < self._last_time
        ):
            raise ContractError(
                "preview indices must increase and elapsed time must not go backwards"
            )

        def measure(name, function, *args):
            if stage_totals is None:
                return function(*args)
            started = time.perf_counter()
            value = function(*args)
            stage_totals[name] += (time.perf_counter() - started) * 1000
            return value

        packet = measure(
            "resize_ms",
            self._frame_builder,
            self.run_id,
            frame_index,
            elapsed_sec,
            raw_frame,
            self.config.analysis_width,
            self.config.analysis_roi,
        )
        objects = measure("yolo_ms", self.detector.detect, packet.analysis_frame)
        motion = measure(
            "mog2_ms", self.motion.detect, packet.analysis_frame, elapsed_sec
        )
        decision = measure("decision_ms", self.events.evaluate, packet, objects, motion)
        self._last_index, self._last_time = frame_index, elapsed_sec
        return FrameAnalysis(
            packet=packet, objects=objects, motion=motion, decision=decision
        )
