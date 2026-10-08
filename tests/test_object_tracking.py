from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from capture_manager import CaptureManager
from config import Config, build_config, config_to_dict
from contracts import (Box, CaptureResult, ConfigError, FrameDecision, FramePacket,
                       ModelClassError, MotionResult, ObjectDetection, ObjectMotionEvidence)
from event_detector import EventDetector
from main import _capture_saved_payload, _frame_decision_payload
from object_detector import ObjectDetector, resolve_target_classes
from object_state import ObjectStateStore


def packet(index, time):
    frame = np.zeros((60, 100, 3), np.uint8)
    return FramePacket("test", index, time, frame, frame, 1, 1)


def evidence(track_id, *, name="person", qualifies=True):
    return ObjectMotionEvidence(0, Box(10, 10, 40, 50), .9, 1000, .8,
                                qualifies, () if qualifies else ("MOTION_PIXELS_BELOW_MIN",),
                                class_id=0 if name == "person" else 16,
                                class_name=name, track_id=track_id)


def decision(pkt, entries):
    return FrameDecision(pkt.frame_index, pkt.video_time_sec, "MOVING_OBJECT",
                         any(e.qualifies for e in entries), tuple(entries))


def test_new_class_can_capture_during_other_objects_cooldown(tmp_path):
    manager = CaptureManager(tmp_path, .5, 95, scope="object")
    manager.prepare("ids")
    sequence = [(3, [evidence(1)], (1,)),
                (3.2, [evidence(1), evidence(2, name="dog")], (2,)),
                (3.4, [evidence(1), evidence(2, name="dog")], ()),
                (3.5, [evidence(1), evidence(2, name="dog")], (1,)),
                (3.7, [evidence(1), evidence(2, name="dog")], (2,))]
    for index, (time, entries, expected) in enumerate(sequence):
        pkt = packet(index, time)
        result = manager.maybe_save(pkt, decision(pkt, entries))
        assert result.saved_track_ids == expected
        assert result.status == ("SAVED" if expected else "COOLDOWN")
    assert len(list((tmp_path / "ids").glob("*.jpg"))) == 4


def test_multiple_due_ids_share_one_original_frame(tmp_path):
    manager = CaptureManager(tmp_path, .5, 95, scope="object")
    manager.prepare("shared")
    pkt = packet(0, 3)
    result = manager.maybe_save(pkt, decision(pkt, [evidence(1), evidence(2)]))
    assert result.saved_track_ids == (1, 2)
    assert len(list((tmp_path / "shared").glob("*.jpg"))) == 1


def test_untracked_and_stationary_objects_cannot_trigger_object_capture(tmp_path):
    manager = CaptureManager(tmp_path, .5, 95, scope="object")
    manager.prepare("ineligible")
    for index, entries in enumerate([[evidence(None)], [evidence(1, qualifies=False)]]):
        pkt = packet(index, 3 + index)
        assert manager.maybe_save(pkt, decision(pkt, entries)).status == "NOT_ELIGIBLE"
    assert manager.successful_capture_count == 0


def test_save_failure_does_not_advance_object_cooldown(tmp_path, monkeypatch):
    from contracts import CaptureSaveError
    manager = CaptureManager(tmp_path, .5, 95, scope="object")
    manager.prepare("retry")
    monkeypatch.setattr("capture_manager.cv2.imencode", lambda *a, **k: (False, None))
    pkt = packet(0, 3)
    with pytest.raises(CaptureSaveError):
        manager.maybe_save(pkt, decision(pkt, [evidence(1)]))
    monkeypatch.undo()
    pkt = packet(1, 3.1)
    assert manager.maybe_save(pkt, decision(pkt, [evidence(1)])).saved_track_ids == (1,)


def test_motion_is_independent_for_person_and_dog():
    pkt = packet(0, 3)
    mask = np.zeros((60, 100), np.uint8)
    mask[10:50, 10:40] = 255
    objects = (ObjectDetection(0, Box(10, 10, 40, 50), .9, 0, "person", 1),
               ObjectDetection(1, Box(60, 10, 90, 50), .9, 16, "dog", 2))
    motion = MotionResult(mask, (), 1200, .2, False)
    config = replace(Config(), target_classes=("person", "dog"))
    result = EventDetector(config).evaluate(pkt, objects, motion)
    assert result.status == "MOVING_OBJECT"
    assert [o.qualifies for o in result.objects] == [True, False]
    assert [o.track_id for o in result.objects] == [1, 2]


def test_state_preserves_identity_when_detection_order_changes():
    store = ObjectStateStore()
    pkt = packet(0, 3)
    saved = CaptureResult("SAVED", None, 3, 0, 1, (2,))
    store.update(decision(pkt, [evidence(1), evidence(2, name="dog")]), saved)
    pkt = packet(1, 3.1)
    store.update(decision(pkt, [evidence(2, name="dog"), evidence(1)]),
                 CaptureResult("COOLDOWN", None, 3.1, .4, None))
    assert store.objects[2]["class_name"] == "dog"
    assert store.objects[2]["capture_count"] == 1
    assert store.objects[1]["capture_count"] == 0
    pkt = packet(2, 3.2)
    store.update(decision(pkt, []), CaptureResult("NOT_ELIGIBLE", None, 3.2, 0, None))
    assert not store.objects[2]["visible"]
    assert not store.objects[2]["motion_qualified"]
    assert store.snapshot()["2"]["last_seen_sec"] == 3.1


def test_multiclass_logs_keep_dogs_out_of_legacy_person_fields():
    pkt = packet(0, 3)
    result = decision(pkt, [evidence(1), evidence(2, name="dog")])
    capture = CaptureResult("SAVED", None, 3, 0, 1, (2,))
    saved = _capture_saved_payload(pkt, result, capture, Config())
    assert [o["class_name"] for o in saved["matched_objects"]] == ["person", "dog"]
    assert [o["track_id"] for o in saved["matched_persons"]] == [1]
    assert saved["saved_track_ids"] == [2]
    motion = MotionResult(np.zeros((60, 100), np.uint8), (), 0, 0, False)
    logged = _frame_decision_payload(pkt, result, motion, capture)
    assert len(logged["objects"]) == 2
    assert len(logged["persons"]) == 1


def test_class_resolution_uses_model_names_instead_of_hardcoded_ids():
    mapping, ids = resolve_target_classes({7: "person", 23: "dog"}, ("dog", "person"))
    assert ids == (23, 7)
    assert mapping[23] == "dog"
    with pytest.raises(ModelClassError):
        resolve_target_classes({0: "person"}, ("dog",))
    with pytest.raises(ModelClassError):
        resolve_target_classes({0: "person", 1: "person"}, ("person",))


def test_cli_tracking_and_generic_threshold_aliases(tmp_path):
    video = tmp_path / "video.mp4"; video.touch()
    model = tmp_path / "model.pt"; model.touch()
    args = ["--video", str(video), "--model", str(model),
            "--capture-dir", str(tmp_path / "captures"), "--log-dir", str(tmp_path / "logs"),
            "--classes", "person", "dog", "person", "--track", "--capture-scope", "object",
            "--confidence", ".4", "--min-object-motion-pixels", "1000"]
    config = build_config(args)
    assert config.tracking and config.capture_scope == "object"
    assert config.target_classes == ("person", "dog")
    assert config.person_confidence == .4
    assert config.min_person_motion_pixels_ref == 1000
    assert config_to_dict(config)["target_classes"] == ["person", "dog"]
    with pytest.raises(ConfigError):
        build_config([a for a in args if a != "--track"])


class Tensor:
    def __init__(self, array): self.array = np.asarray(array)
    def cpu(self): return self
    def numpy(self): return self.array


class Boxes:
    def __init__(self, ids):
        self.xyxy = Tensor([[10, 10, 40, 50], [60, 10, 90, 50]])
        self.conf = Tensor([.9, .8]); self.cls = Tensor([0, 16])
        self.id = None if ids is None else Tensor(ids)
    def __len__(self): return 2


class Model:
    def __init__(self, ids=(8, 3)): self.calls = []; self.ids = ids
    def predict(self, **kwargs):
        self.calls.append(("predict", kwargs)); return [SimpleNamespace(boxes=Boxes(self.ids))]
    def track(self, **kwargs):
        self.calls.append(("track", kwargs)); return [SimpleNamespace(boxes=Boxes(self.ids))]


def detector(tracking=True, ids=(8, 3)):
    result = ObjectDetector(replace(Config(), tracking=tracking, target_classes=("person", "dog")))
    result._loaded = True; result._model = Model(ids)
    result.class_names = {0: "person", 16: "dog"}; result.target_class_ids = (0, 16)
    return result


def test_ids_come_from_tracker_and_warmup_does_not_advance_tracker():
    obj = detector()
    obj.warmup(100, 60)
    detections = obj.detect(np.zeros((60, 100, 3), np.uint8))
    assert [d.track_id for d in detections] == [1, 2]
    assert [d.tracker_id for d in detections] == [8, 3]
    assert [d.class_name for d in detections] == ["person", "dog"]
    assert [mode for mode, _ in obj._model.calls] == ["predict", "track"]
    kwargs = obj._model.calls[-1][1]
    assert kwargs["persist"] is True and kwargs["tracker"] == "bytetrack.yaml"
    assert kwargs["classes"] == [0, 16]
    assert kwargs["conf"] == .1


def test_tracker_association_confidence_does_not_lower_capture_confidence():
    obj = detector()
    boxes = Boxes((8, 3)); boxes.conf = Tensor([.9, .2])
    obj._model.track = lambda **kwargs: [SimpleNamespace(boxes=boxes)]
    items = obj.detect(np.zeros((60, 100, 3), np.uint8))
    assert [d.class_name for d in items] == ["person"]


def test_tracking_does_not_invent_ids_for_unconfirmed_detections():
    obj = detector(ids=None)
    assert obj.detect(np.zeros((60, 100, 3), np.uint8)) == ()
    assert obj.untracked_detection_count == 2


def test_tracking_rejects_non_finite_ids():
    obj = detector(ids=(float("nan"), 3))
    items = obj.detect(np.zeros((60, 100, 3), np.uint8))
    assert [d.track_id for d in items] == [1]
    assert [d.tracker_id for d in items] == [3]


def test_detection_without_tracking_keeps_ids_unset():
    obj = detector(tracking=False, ids=None)
    assert [d.track_id for d in obj.detect(np.zeros((60, 100, 3), np.uint8))] == [None, None]
    assert obj._model.calls[-1][0] == "predict"


def test_one_tracker_id_cannot_change_an_objects_class():
    obj = detector()
    person_id = obj._identity_id(0, 8)
    dog_id = obj._identity_id(16, 8)
    assert person_id != dog_id
    assert obj._identity_id(0, 8) == person_id
    assert obj._identity_id(16, 8) == dog_id
    assert obj.class_identity_splits == 1


def test_duplicate_tracker_ids_are_an_error():
    from contracts import InferenceError
    obj = detector(ids=(8, 8))
    with pytest.raises(InferenceError):
        obj.detect(np.zeros((60, 100, 3), np.uint8))


def test_tracking_panel_does_not_cover_the_source_frame():
    from overlay_renderer import render_overlay
    pkt = packet(0, 3)
    before = pkt.raw_frame.copy()
    result = decision(pkt, [evidence(1)])
    motion = MotionResult(np.zeros((60, 100), np.uint8), (), 0, 0, False)
    image = render_overlay(pkt, (), motion, result,
                           CaptureResult("NOT_ELIGIBLE", None, 3, 0, None), {"tracking": True})
    assert image.shape[1] == pkt.analysis_size[0] + 380
    assert np.array_equal(before, pkt.raw_frame)
    assert np.any(image[:60, :100] != 0)
