from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from motion_person import EventConfig
from motion_person.contracts import Box, CaptureSaveError, ConfigError, ContractError, FrameDecision, ObjectMotionEvidence
from motion_person.event_capture import EventCaptureEngine
from motion_person.frame_processor import make_packet


def frame(index, timestamp, moving=False, *, track=1, class_name="person", empty=False):
    raw = np.full((72, 128, 3), int(timestamp*19) % 255, np.uint8)
    raw[:, :13] = (15, 40, 200)  # asymmetric content: check raw captures/mirroring
    packet = make_packet("run", index, timestamp, raw, 128, None)
    objects = () if empty else (ObjectMotionEvidence(0, Box(15, 10, 85, 65), .9,
                    1200 if moving else 0, .4 if moving else 0, moving, (),
                    0 if class_name == "person" else 16, class_name, track),)
    decision = FrameDecision(index, timestamp, "MOVING_OBJECT" if moving else "IDLE", moving, objects)
    return packet, decision


def feed(engine, times, match, **kwargs):
    bundles = []
    for i, t in enumerate(times):
        moving = match(t)
        bundles.extend(engine.process(*frame(i, t, moving, **kwargs)))
    return bundles


def test_quiet_closes_once_with_before_action_and_empty_after_photos(tmp_path):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    bundles = []
    for i, t in enumerate(np.arange(0, 7.5, .5)):
        packet, decision = frame(i, float(t), 3 <= t <= 3.5, empty=t > 3.5)
        emitted = engine.process(packet, decision)
        if t < 6.5:
            assert not emitted
        bundles.extend(emitted)
    assert len(bundles) == 1 and engine.flush() == ()
    bundle = bundles[0]
    assert bundle.end_reason == "QUIET" and bundle.episode_end and bundle.context_complete
    assert (bundle.start_sec, bundle.end_sec, bundle.trigger_sec, bundle.last_motion_sec) == (0, 6.5, 3, 3.5)
    assert 2 <= len(bundle.images) <= 6
    assert {image.phase for image in bundle.images} >= {"before", "start", "after"}
    assert any(image.timestamp_sec > 3.5 and not image.objects for image in bundle.images)
    assert len(list(bundle.manifest_path.parent.glob("*.jpg"))) == len(bundle.images)
    document = json.loads(bundle.manifest_path.read_text())
    assert document["schema_version"] == 1 and document["context_complete"]
    assert all(Path(image["path"]).is_file() for image in document["images"])
    for image in bundle.images:
        original, _ = frame(image.frame_index, image.timestamp_sec)
        _, expected = cv2.imencode(".jpg", original.raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert image.path.read_bytes() == expected.tobytes()
        assert cv2.imdecode(np.frombuffer(image.path.read_bytes(), np.uint8), cv2.IMREAD_COLOR).shape == (72,128,3)


def test_long_motion_has_ten_second_total_windows_and_same_episode_id(tmp_path):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    bundles = feed(engine, np.arange(0, 31.5, .5), lambda t: t >= 3)
    bundles.extend(engine.flush())
    assert len(bundles) == 4
    assert len({b.event_id for b in bundles}) == 1
    assert [b.part_index for b in bundles] == [1,2,3,4]
    assert [b.duration_sec for b in bundles] == [10,10,10,1]
    assert all(b.end_reason == "MAX_DURATION" and not b.episode_end and not b.post_context_complete for b in bundles[:3])
    assert bundles[-1].end_reason == "END_OF_VIDEO" and bundles[-1].episode_end
    for b in bundles:
        assert b.duration_sec <= 10 and 1 <= len(b.images) <= 6
        times = [im.timestamp_sec for im in b.images]
        assert times == sorted(set(times))
        assert all(b.start_sec <= t <= b.end_sec for t in times)
    # Start is 3 seconds before trigger, so the first hard close is t=10, not t=13.
    assert bundles[0].start_sec == 0 and bundles[0].trigger_sec == 3


def test_exact_quiet_and_duration_tie_closes_episode_without_continuation(tmp_path):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    bundles = feed(engine, np.arange(0,10.5,.5), lambda t: 3 <= t <= 7)
    assert len(bundles) == 1 and bundles[0].end_reason == "QUIET"
    assert bundles[0].duration_sec == 10 and engine.flush() == ()


def test_resume_before_quiet_and_track_class_changes_do_not_split_episode(tmp_path):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    bundles = []
    for i, t in enumerate(np.arange(0,9,.5)):
        bundles.extend(engine.process(*frame(i, float(t), t in (3,5.5),
                       track=1 if t <= 3 else 98, class_name="person" if t <= 3 else "dog")))
    assert len(bundles) == 1 and bundles[0].last_motion_sec == 5.5
    assert bundles[0].end_reason == "QUIET"
    assert {o.class_name for image in bundles[0].images for o in image.objects} >= {"person", "dog"}


def test_restart_after_quiet_uses_new_episode_and_forces_non_grid_trigger_photo(tmp_path):
    config = EventConfig(quiet_sec=1)
    engine = EventCaptureEngine(tmp_path, "run", "camera:0", config=config)
    times = [0,.5,1,1.5,2,2.1,2.5,3,3.1]
    bundles = feed(engine, times, lambda t: t in (1,2.1))
    bundles.extend(engine.flush())
    assert len(bundles) == 2 and bundles[0].event_id != bundles[1].event_id
    assert bundles[1].trigger_sec == 2.1
    assert any(im.timestamp_sec == 2.1 and "start" in im.roles for im in bundles[1].images)


@pytest.mark.parametrize("reason", ["END_OF_VIDEO","USER_STOP","USER_INTERRUPT","ERROR","DURATION_LIMIT"])
def test_flush_persists_only_observed_tail_and_is_idempotent(tmp_path, reason):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    feed(engine, [0,.5,1,1.5,2,2.5,3,3.5,4], lambda t: t >= 3)
    bundle, = engine.flush(reason)
    assert bundle.end_reason == reason and bundle.end_sec == 4
    assert not bundle.post_context_complete and not bundle.context_complete
    assert engine.flush(reason) == () and not engine.active and engine.buffer_bytes == 0


def test_sparse_receipt_times_use_elapsed_time_and_flag_sampling_gaps(tmp_path):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    bundles = feed(engine, [0,.5,1,1.5,2,2.5,3,3.5,5.5,6,6.5,7,7.5,8,8.5], lambda t: t in (3,3.5,5.5))
    bundle, = bundles
    assert bundle.last_motion_sec == 5.5 and bundle.end_sec == 8.5
    assert bundle.sampling_gap_count > 0 and not bundle.context_complete


def test_jpeg_buffer_is_bounded_and_eviction_is_reported(tmp_path):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0", config=EventConfig(max_buffer_bytes=3600))
    bundles = feed(engine, np.arange(0,11.5,.5), lambda t: t >= 3)
    bundles.extend(engine.flush())
    assert engine.peak_buffer_bytes <= 3600
    assert any(b.buffer_dropped_frames > 0 and not b.context_complete for b in bundles)
    assert all(len(b.images) <= 6 for b in bundles)


def test_empty_scene_saves_nothing_and_discarding_buffer_creates_no_folder(tmp_path):
    output = tmp_path / "nothing"
    engine = EventCaptureEngine(output, "run", "camera:0")
    assert feed(engine, np.arange(0,15,.5), lambda t: False, empty=True) == []
    assert engine.flush() == () and not output.exists()


def test_failed_bundle_write_has_no_published_bundle_or_success_counter(tmp_path, monkeypatch):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    feed(engine, [0,.5,1,1.5,2,2.5,3], lambda t: t == 3)
    def fail(path, data):
        raise OSError("disk full")
    monkeypatch.setattr(engine, "_write", fail)
    with pytest.raises(CaptureSaveError, match="publication"):
        engine.flush()
    assert engine.bundle_count == engine.image_count == 0
    assert not list(tmp_path.rglob("*.jpg")) and not list(tmp_path.rglob("event.json"))
    assert not list(tmp_path.rglob("*.tmp"))
    assert not engine.active and engine.flush("ERROR") == ()


def test_failed_encoder_does_not_modify_input_or_report_saved(tmp_path, monkeypatch):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    packet, decision = frame(0,0,True)
    original = packet.raw_frame.copy()
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False,None))
    with pytest.raises(CaptureSaveError, match="encoding"):
        engine.process(packet, decision)
    assert engine.bundle_count == 0 and np.array_equal(packet.raw_frame, original)


def test_oversized_single_jpeg_fails_explicitly(tmp_path):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0", config=EventConfig(max_buffer_bytes=1))
    with pytest.raises(CaptureSaveError, match="one JPEG"):
        engine.process(*frame(0,0,True))


def test_gaps_inside_preroll_also_mark_context_incomplete(tmp_path):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    bundle, = feed(engine, [0,.5,2.5,3,3.5,4,4.5,5,5.5,6], lambda t: t == 3)
    assert bundle.pre_context_complete and bundle.post_context_complete
    assert bundle.sampling_gap_count == 1 and not bundle.context_complete


def test_encoder_failure_after_prior_samples_cannot_flush_future_or_false_evidence(tmp_path, monkeypatch):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    feed(engine, [0,.5,1,1.5,2,2.5,3], lambda t: t == 3)
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False,None))
    with pytest.raises(CaptureSaveError):
        engine.process(*frame(8,4,True))
    assert not engine.active and engine.buffer_bytes == 0
    assert engine.flush("ERROR") == () and not list(tmp_path.rglob("event.json"))


@pytest.mark.parametrize("changes", [{"max_duration_sec":11},{"max_duration_sec":0},
    {"max_duration_sec":True},{"pre_capture_sec":10},{"quiet_sec":0},
    {"sample_interval_sec":0},{"sample_interval_sec":float("nan")},
    {"max_images":7},{"max_images":True},{"max_buffer_bytes":0}])
def test_invalid_event_configuration(changes):
    with pytest.raises(ConfigError):
        EventConfig(**changes)


def test_mismatched_repeated_or_backwards_frames_fail_before_saving(tmp_path):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0")
    engine.process(*frame(2,1))
    for i, t in [(2,2),(1,2),(3,.9),(3,1),(3,float("nan"))]:
        if t != t:
            packet, decision = frame(3,2)
            packet = replace(packet, video_time_sec=t)
            decision = replace(decision, video_time_sec=t)
        else:
            packet, decision = frame(i,t)
        with pytest.raises(ContractError):
            engine.process(packet, decision)
    packet, decision = frame(3,2)
    with pytest.raises(ContractError):
        engine.process(packet, replace(decision, frame_index=8))


@pytest.mark.parametrize("max_images", [2,3,4,5,6])
def test_representatives_keep_temporal_endpoints_with_configured_photo_limit(tmp_path, max_images):
    engine = EventCaptureEngine(tmp_path, "run", "camera:0", config=EventConfig(max_images=max_images))
    bundle, = feed(engine, np.arange(0,7,.5), lambda t: 3 <= t <= 3.5)
    assert len(bundle.images) <= max_images
    assert bundle.images[0].timestamp_sec == 0 and bundle.images[-1].timestamp_sec == 6.5
