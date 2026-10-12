"""Convenience API composition. Concrete implementations are chosen here."""

from __future__ import annotations


def analysis_components(
    config, detector, motion, events, frame_builder, prepare_detector
):
    if detector is None:
        from motion_person.adapters.yolo import ObjectDetector

        detector = ObjectDetector(config)

        def prepare(path):
            detector.load()
            if config.tracking:
                detector.prepare_tracking(path)
            detector.warmup(640, 480)

        prepare_detector = prepare_detector or prepare
    if motion is None:
        from motion_person.adapters.motion import MotionDetector

        motion = MotionDetector(config)
    if events is None:
        from motion_person.domain.motion_rules import EventDetector

        events = EventDetector(config)
    if frame_builder is None:
        from motion_person.adapters.frames import make_packet

        frame_builder = make_packet
    return detector, motion, events, frame_builder, prepare_detector


def event_storage(store, encoder):
    if store is None:
        from motion_person.adapters.bundle_store import FileBundleStore

        store = FileBundleStore()
    if encoder is None:
        from motion_person.adapters.jpeg import encode_jpeg

        encoder = encode_jpeg
    return store, encoder
