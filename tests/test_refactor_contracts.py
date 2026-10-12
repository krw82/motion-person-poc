from dataclasses import replace

import cv2
import numpy as np
import pytest

from motion_person import EventConfig, MotionEngine
from motion_person.contracts import LogWriteError
from motion_person.event_capture import EventCaptureEngine
from motion_person.photo_selection import select_photos
from tests.test_engine import inputs as inputs
from tests.test_event_capture import frame, feed
from tests.test_object_tracking import detector


def test_default_capture_preserves_files_and_callbacks_without_journals(
    inputs, tmp_path
):
    video, _, _, _ = inputs
    received = []
    engine = MotionEngine.video(video, output_dir=tmp_path / "out", pace="fast")
    assert not engine.config.display and not engine.diagnostics
    summary = engine.start(on_event=received.append)
    assert summary.bundle_count == summary.episode_count == len(received) == 1
    assert received[0].evidence_complete and received[0].manifest_path.is_file()
    assert not any(
        (summary.run_dir / name).exists()
        for name in ("run_config.json", "events.jsonl", "summary.json", "tracker.yaml")
    )


@pytest.mark.parametrize("duration", [None, 4])
def test_terminal_progress_includes_flushed_bundle_and_clears_active_capture(
    inputs, tmp_path, duration
):
    video, _, _, _ = inputs
    engine = MotionEngine.video(
        video, events=EventConfig(quiet_sec=10), output_dir=tmp_path, pace="fast"
    )
    summary = engine.start(duration_sec=duration)
    state = engine.progress()
    assert (
        state["status"] == "stopped"
        and state["bundle_count"] == summary.bundle_count == 1
    )
    assert not state["event_active"]


@pytest.mark.parametrize("where", ["start", "EVENT_COMPLETED", "summary", "close"])
def test_diagnostic_failure_cannot_block_event_delivery(
    inputs, tmp_path, monkeypatch, where
):
    video, resources, _, module = inputs

    def fail(*args):
        raise LogWriteError("diagnostic disk failure")

    if where == "EVENT_COMPLETED":
        original = module.RunLogger.write

        def write(self, kind, payload):
            if kind == where:
                fail()
            return original(self, kind, payload)

        monkeypatch.setattr(module.RunLogger, "write", write)
    else:
        method = {"start": "start", "summary": "write_summary", "close": "close"}[where]
        monkeypatch.setattr(module.RunLogger, method, fail)
    received = []
    summary = MotionEngine.video(
        video, output_dir=tmp_path / "out", diagnostics=True, pace="fast"
    ).start(on_event=received.append)
    assert len(received) == 1 and received[0].manifest_path.is_file()
    assert summary.end_reason == "END_OF_VIDEO" and summary.diagnostic_errors >= 1
    assert all(source.closed for source in resources)


def test_interrupt_recovers_published_part_and_drops_unfinished_input(
    inputs, tmp_path, monkeypatch
):
    video, resources, _, module = inputs
    original_source = module.VideoSource

    class LongSource(original_source):
        def read(self):
            index, elapsed = self.i, self.i / 2
            self.i += 1
            return index, elapsed, frame(index, elapsed)[0].raw_frame

    original = module.FrameProcessor.process

    def process(self, raw, index, elapsed):
        analysis = original(self, raw, index, elapsed)
        _, decision = frame(index, elapsed, elapsed >= 3)
        if elapsed == 10:

            def interrupt(*args):
                raise KeyboardInterrupt()

            monkeypatch.setattr(cv2, "imencode", interrupt)
        return replace(analysis, decision=decision)

    monkeypatch.setattr(module, "VideoSource", LongSource)
    monkeypatch.setattr(module.FrameProcessor, "process", process)
    received = []
    summary = MotionEngine.video(video, output_dir=tmp_path / "out", pace="fast").start(
        on_event=received.append
    )
    assert summary.end_reason == "USER_INTERRUPT" and len(received) == 1
    assert (
        received[0].end_reason == "MAX_DURATION" and received[0].manifest_path.is_file()
    )
    assert all(source.closed for source in resources)


def test_two_photo_budget_contains_motion_and_end(tmp_path):
    engine = EventCaptureEngine(
        tmp_path, "run", "camera", config=EventConfig(max_images=2)
    )
    (bundle,) = feed(engine, np.arange(0, 7, 0.5), lambda t: 3 <= t <= 3.5)
    assert len(bundle.images) == 2 and bundle.images[-1].timestamp_sec == 6.5
    assert bundle.evidence_complete and any(
        o.moving for image in bundle.images for o in image.objects
    )


@pytest.mark.parametrize("trigger", [0.5, 1, 2])
def test_requested_pre_context_is_incomplete_near_start(tmp_path, trigger):
    engine = EventCaptureEngine(tmp_path, "run", "camera")
    (bundle,) = feed(engine, np.arange(0, trigger + 3.1, 0.5), lambda t: t == trigger)
    assert not bundle.pre_context_complete and not bundle.context_complete


def test_missing_post_observations_are_not_marked_complete(tmp_path):
    engine = EventCaptureEngine(tmp_path, "run", "camera")
    (bundle,) = feed(
        engine, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 6.5], lambda t: 3 <= t <= 3.5
    )
    assert bundle.sampling_gap_count == 1 and not bundle.post_context_complete


def test_original_callback_error_contains_cleanup_failure(
    inputs, tmp_path, monkeypatch
):
    video, _, _, module = inputs
    original = module.VideoSource.close

    def close(self):
        original(self)
        raise RuntimeError("source cleanup failed")

    monkeypatch.setattr(module.VideoSource, "close", close)

    def completed(bundle):
        raise RuntimeError("consumer failure")

    with pytest.raises(RuntimeError, match="consumer failure") as raised:
        MotionEngine.video(video, output_dir=tmp_path / "out", pace="fast").start(
            on_event=completed
        )
    assert any("source cleanup failed" in note for note in raised.value.__notes__)


def test_tracking_state_is_bounded_and_evicted_ids_are_never_reused():
    obj = detector()
    obj.config = replace(obj.config, track_buffer=4, max_track_states=100)
    coordinates, confidence, classes = (
        np.array([[1, 1, 20, 20]]),
        np.array([0.9]),
        np.array([0]),
    )
    ids = []
    for native_id in range(1, 2001):
        items = obj._cleanup_detections(
            coordinates, confidence, classes, 100, 60, np.array([native_id])
        )
        ids.append(items[0].track_id)
        assert obj.retained_track_count <= 5
    assert len(set(ids)) == len(ids) and ids == sorted(ids)
    new = obj._cleanup_detections(
        coordinates, confidence, classes, 100, 60, np.array([1])
    )[0]
    assert new.track_id > ids[-1] and new.observation_hits == 1


def test_tracking_confirmation_survives_brief_gap_but_expires_on_empty_input():
    obj = detector()
    obj.config = replace(obj.config, track_buffer=2)
    raw = np.zeros((60, 100, 3), np.uint8)
    first = obj.detect(raw)[0]
    obj._model.ids = None
    assert obj.detect(raw) == ()
    obj._model.ids = (8, 3)
    again = obj.detect(raw)[0]
    assert again.track_id == first.track_id and again.observation_hits == 2
    obj._model.ids = None
    for _ in range(3):
        obj.detect(raw)
    assert obj.retained_track_count == 0
    obj._model.ids = (8, 3)
    assert obj.detect(raw)[0].track_id != first.track_id


def test_photo_selection_rejects_empty_input_without_private_sample_dependency():
    with pytest.raises(ValueError, match="at least one"):
        select_photos([], start=0, trigger=0, last_motion=0, max_images=6)
