from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from config import Config, build_config, validate_config
from contracts import Box, ConfigError, ObjectDetection, MotionResult
from event_detector import EventDetector
from frame_processor import make_packet, box_to_original
from tracking_adapter import ClassSeparatedTracker
from tests.test_object_tracking import detector


def arguments(tmp_path, *extra):
    video = tmp_path / 'input.mp4'; video.touch()
    model = tmp_path / 'model.pt'; model.touch()
    return ['--video', str(video), '--model', str(model), '--capture-dir', str(tmp_path / 'caps'),
            '--log-dir', str(tmp_path / 'logs'), *extra]


def real_tracker(buffer=3):
    args = SimpleNamespace(track_high_thresh=.25, track_low_thresh=.1, new_track_thresh=.4,
                           track_buffer=buffer, match_thresh=.8, fuse_score=True)
    return ClassSeparatedTracker(BYTETracker, args, (0, 16))


def boxes(rows):
    return Boxes(torch.tensor(rows, dtype=torch.float32).reshape(-1, 6), (240, 320)).cpu().numpy()


def test_cross_class_high_and_low_confidence_cannot_steal_a_track():
    tracker = real_tracker(10)
    first = tracker.update(boxes([[50, 50, 100, 100, .9, 16]]))
    dog_id = first[0, 4]
    # Same geometry would be matched by a shared ByteTrack. Low-score foreign
    # class must also be blocked in its second association, not just the first.
    assert len(tracker.update(boxes([[50, 50, 100, 100, .15, 0]]))) == 0
    tracker.update(boxes([[50, 50, 100, 100, .9, 0]]))
    person = tracker.update(boxes([[50, 50, 100, 100, .9, 0]]))
    assert person[0, 4] != dog_id and person[0, 6] == 0
    dog = tracker.update(boxes([[50, 50, 100, 100, .9, 16]]))
    assert dog[0, 4] == dog_id and dog[0, 6] == 16


def test_class_subset_indices_are_restored_and_weak_same_class_keeps_id():
    tracker = real_tracker()
    first = tracker.update(boxes([[150, 50, 200, 100, .9, 16], [50, 50, 100, 100, .9, 0]]))
    assert first[:, -1].tolist() == [0, 1]
    second = tracker.update(boxes([[150, 50, 200, 100, .15, 16], [50, 50, 100, 100, .9, 0]]))
    assert second[:, 4].tolist() == first[:, 4].tolist()
    assert second[:, -1].tolist() == [0, 1]


def test_empty_frames_age_every_class_and_expire_lost_track():
    tracker = real_tracker(3)
    old = tracker.update(boxes([[50, 50, 100, 100, .9, 16]]))[0, 4]
    for _ in range(5):
        assert len(tracker.update(boxes([]))) == 0
    assert {t.frame_id for t in tracker.trackers.values()} == {6}
    assert not tracker.trackers[16].lost_stracks
    tracker.update(boxes([[50, 50, 100, 100, .9, 16]]))
    new = tracker.update(boxes([[50, 50, 100, 100, .9, 16]]))[0, 4]
    assert new != old


def test_native_reid_features_follow_the_same_class_subset():
    class Child:
        def __init__(self, args): self.calls = []
        def update(self, results, img, feats=None, **kw):
            self.calls.append((results.cls.tolist(), feats)); return np.empty((0, 8))
        def reset(self): self.calls.clear()
    tracker = ClassSeparatedTracker(Child, None, (0, 16))
    features = [torch.tensor([1.]), torch.tensor([2.]), torch.tensor([3.])]
    tracker.update(boxes([[1,1,5,5,.9,16], [10,10,15,15,.9,0], [20,20,25,25,.9,16]]), feats=features)
    assert tracker.trackers[0].calls[0][1].tolist() == [[2.]]
    assert tracker.trackers[16].calls[0][1].tolist() == [[1.], [3.]]


def test_real_botsort_native_features_survive_both_levels_of_subset_indexing():
    from ultralytics.trackers.bot_sort import BOTSORT
    args = SimpleNamespace(track_high_thresh=.25, track_low_thresh=.1, new_track_thresh=.4,
                           track_buffer=60, match_thresh=.8, fuse_score=True, gmc_method='none',
                           with_reid=True, model='auto', proximity_thresh=.5, appearance_thresh=.8)
    tracker = ClassSeparatedTracker(BOTSORT, args, (0, 16))
    inputs = boxes([[150,50,200,100,.9,16], [50,50,100,100,.9,0]])
    features = torch.tensor([[1.,0.], [0.,1.]])
    first = tracker.update(inputs, np.zeros((240,320,3),np.uint8), feats=features)
    second = tracker.update(inputs, np.zeros((240,320,3),np.uint8), feats=features)
    assert first[:,4].tolist() == second[:,4].tolist()
    assert tracker.trackers[0].tracked_stracks[0].smooth_feat.tolist() == [0.,1.]
    assert tracker.trackers[16].tracked_stracks[0].smooth_feat.tolist() == [1.,0.]


def test_confirmation_uses_only_observed_high_confidence_boxes():
    obj = detector()
    frame = np.zeros((60, 100, 3), np.uint8)
    for hits in (1, 2, 3):
        items = obj.detect(frame)
        assert all(o.observation_hits == hits for o in items)
        assert all(o.confirmed == (hits >= 3) for o in items)
    obj._model.ids = None
    assert obj.detect(frame) == ()
    obj._model.ids = (8, 3)
    assert all(o.observation_hits == 4 and o.confirmed for o in obj.detect(frame))


def test_pending_track_cannot_trigger_capture_even_with_large_motion():
    cfg = replace(Config(), tracking=True, person_confidence=.4)
    frame = np.zeros((60, 100, 3), np.uint8)
    packet = make_packet('test', 0, 3, frame, 100)
    pending = ObjectDetection(0, Box(10, 10, 90, 50), .9, 16, 'dog', 1, 1, 2, False)
    motion = MotionResult(np.full((60,100), 255, np.uint8), (), 6000, 1., False)
    result = EventDetector(cfg).evaluate(packet, (pending,), motion)
    assert not result.candidate
    assert 'TRACK_PENDING' in result.objects[0].rejection_reasons
    assert EventDetector(cfg).evaluate(packet, (replace(pending, confirmed=True, observation_hits=3),), motion).candidate


def test_roi_projection_and_original_capture_are_preserved(tmp_path):
    from capture_manager import CaptureManager
    from tests.test_object_tracking import decision, evidence
    raw = np.zeros((120, 200, 3), np.uint8); raw[30:90, 50:150] = [20, 80, 200]
    before = raw.copy()
    packet = make_packet('roi', 0, 3, raw, 50, (.25, .25, .75, .75))
    assert packet.analysis_size == (50, 30)
    assert box_to_original(Box(5, 5, 45, 25), packet) == Box(60, 40, 140, 80)
    manager = CaptureManager(tmp_path, .5, 95, scope='object'); manager.prepare('roi')
    saved = manager.maybe_save(packet, decision(packet, [evidence(1)]))
    _, encoded = cv2.imencode('.jpg', raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert saved.path.read_bytes() == encoded.tobytes()
    assert np.array_equal(raw, before)


def test_cli_profiles_roi_and_reid_validation(tmp_path):
    stable = build_config(arguments(tmp_path, '--track', '--roi', '0', '.3', '1', '.7'))
    assert stable.track_buffer == 60 and stable.track_min_hits == 3
    assert stable.analysis_roi == (0, .3, 1, .7)
    baseline = build_config(arguments(tmp_path, '--track', '--tracking-profile', 'baseline'))
    assert (baseline.track_buffer, baseline.track_min_hits, baseline.track_new_threshold) == (30,1,.25)
    reid = build_config(arguments(tmp_path, '--track', '--tracker', 'botsort', '--reid', '--track-buffer', '90'))
    assert reid.track_reid and reid.track_buffer == 90
    for options in [('--reid',), ('--track', '--reid'), ('--track-buffer','0'), ('--track-min-hits','0'),
                    ('--roi','0','.7','1','.3'), ('--roi','0','nan','1','1')]:
        with pytest.raises(ConfigError): build_config(arguments(tmp_path, *options))


def test_effective_yaml_records_fixed_camera_reid(tmp_path):
    from ultralytics.utils import YAML
    obj = detector()
    obj.config = replace(obj.config, tracker='botsort', track_reid=True, track_buffer=90)
    settings = obj.prepare_tracking(tmp_path / 'tracker.yaml')
    assert settings['track_buffer'] == 90 and settings['with_reid']
    assert settings['gmc_method'] == 'none' and settings['model'] == 'auto'
    assert YAML.load(tmp_path / 'tracker.yaml') == settings


def test_official_callback_handles_empty_and_interleaved_class_results():
    from ultralytics.engine.results import Results
    from ultralytics.trackers.track import on_predict_postprocess_end
    tracker = real_tracker()
    frame = np.zeros((240, 320, 3), np.uint8)
    predictor = SimpleNamespace(args=SimpleNamespace(mode='track', task='detect'),
                                dataset=SimpleNamespace(mode='image'), trackers=[tracker], vid_path=['x'])
    for rows in [[], [[150,50,200,100,.9,16], [50,50,100,100,.9,0]],
                    [[150,50,200,100,.9,16], [50,50,100,100,.9,0]]]:
        predictor.results = [Results(frame, path='x', names={0:'person',16:'dog'},
                                     boxes=torch.tensor(rows, dtype=torch.float32).reshape(-1,6))]
        on_predict_postprocess_end(predictor, persist=True)
    assert predictor.results[0].boxes.cls.tolist() == [16, 0]
    assert predictor.results[0].boxes.xyxy[:,0].tolist() == [150, 50]
    assert len(set(predictor.results[0].boxes.id.tolist())) == 2
