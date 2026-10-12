"""Small consumer contracts; no concrete detector, storage or UI imports."""

from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, Sequence
from motion_person.domain.contracts import (
    FrameAnalysis,
    FrameDecision,
    FramePacket,
    MotionResult,
    ObjectDetection,
)
from motion_person.domain.events import EventBundle

if TYPE_CHECKING:
    import numpy as np


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> tuple[ObjectDetection, ...]: ...


class MotionAnalyzer(Protocol):
    def detect(self, frame: np.ndarray, elapsed_sec: float) -> MotionResult: ...


class MotionRules(Protocol):
    def evaluate(
        self,
        packet: FramePacket,
        objects: tuple[ObjectDetection, ...],
        motion: MotionResult,
    ) -> FrameDecision: ...


class BundleStore(Protocol):
    def publish(self, bundle: EventBundle, images: Sequence[bytes]) -> None: ...


class PreviewSink(Protocol):
    def attach(self, source) -> None: ...
    def publish(self, analysis: FrameAnalysis) -> None: ...
    def close(self) -> None: ...
