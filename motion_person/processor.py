"""Shared one-inference-per-frame analysis for files, webcams and preview."""
from __future__ import annotations
import math
from pathlib import Path
from .config import Config, validate_analysis_values
from .contracts import CaptureResult, ContractError
from .event_detector import EventDetector
from .frame_processor import make_packet
from .motion_detector import MotionDetector
from .object_detector import ObjectDetector


class FrameProcessor:
    """Process frames without opening a camera, showing a window, or saving pixels."""

    def __init__(self, config: Config, run_id: str = "webcam-preview"):
        self.run_id = run_id
        validate_analysis_values(config)
        self.config = config
        self.detector = ObjectDetector(config)
        self.motion = MotionDetector(config)
        self.events = EventDetector(config)
        self._last_index = -1
        self._last_time = -1.0

    def load(self, tracker_path: Path) -> None:
        self.detector.load()
        if self.config.tracking:
            self.detector.prepare_tracking(tracker_path)
        # Dummy predict only; no camera frames, background or tracker advancement.
        self.detector.warmup(640, 480)

    def process(self, raw_frame, frame_index: int, elapsed_sec: float):
        if (not isinstance(frame_index, int) or isinstance(frame_index, bool)
                or frame_index <= self._last_index or not math.isfinite(elapsed_sec)
                or elapsed_sec < 0 or elapsed_sec < self._last_time):
            raise ContractError("preview indices must increase and elapsed time must not go backwards")
        packet = make_packet(self.run_id, frame_index, elapsed_sec, raw_frame,
                             self.config.analysis_width, self.config.analysis_roi)
        objects = self.detector.detect(packet.analysis_frame)
        motion = self.motion.detect(packet.analysis_frame, elapsed_sec)
        decision = self.events.evaluate(packet, objects, motion)
        disabled = CaptureResult("DISABLED", None, elapsed_sec, 0.0, None)
        self._last_index, self._last_time = frame_index, elapsed_sec
        return packet, objects, motion, decision, disabled
