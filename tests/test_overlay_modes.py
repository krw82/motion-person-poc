"""Display choices must preserve analysis, raw pixels and actual capture behavior."""
from dataclasses import replace

import cv2
import numpy as np
import pytest

from capture_manager import CaptureManager
from config import Config, validate_analysis_values
from contracts import (Box, CaptureResult, ConfigError, ContractError,
                       MotionRegion, MotionResult, ObjectDetection)
from event_detector import EventDetector
from frame_processor import make_packet
from overlay_renderer import COLOR_FG, COLOR_MATCH, COLOR_PERSON, render_overlay


@pytest.fixture
def moving_object():
    config = replace(Config(), analysis_width=640, tracking=True, capture_scope="object",
                     warmup_sec=0, person_confidence=.4, min_person_motion_pixels_ref=1000)
    packet = make_packet("overlay-test", 0, 3., np.full((480, 1280, 3), 60, np.uint8), 640)
    objects = (ObjectDetection(0, Box(160, 70, 290, 190), .9, 0, "person", 1,
                               observation_hits=3, confirmed=True),)
    mask = np.zeros((240, 640), np.uint8)
    mask[20:60, 5:80] = 255
    mask[100:155, 180:260] = 255
    motion = MotionResult(mask, (MotionRegion(Box(5, 20, 80, 60), 3000),
                                MotionRegion(Box(180, 100, 260, 155), 4400)),
                          7400, 7400 / mask.size, False)
    decision = EventDetector(config).evaluate(packet, objects, motion)
    assert decision.candidate and decision.objects[0].qualifies
    return config, packet, objects, motion, decision


def color_count(image, color):
    return np.count_nonzero(np.all(image == color, axis=2))


def test_objects_mode_hides_motion_colors_and_none_is_unannotated_copy(moving_object):
    config, packet, objects, motion, decision = moving_object
    capture = CaptureResult("COOLDOWN", None, 3., .2, None)
    metrics = {"tracking": True, "preview": True}
    raw_before, mask_before = packet.raw_frame.copy(), motion.valid_mask.copy()
    full = render_overlay(packet, objects, motion, decision, capture, metrics)
    simple = render_overlay(packet, objects, motion, decision, capture, metrics, overlay_mode="objects")
    clean = render_overlay(packet, objects, motion, decision, capture, metrics, overlay_mode="none")

    # Match highlighting appears during cooldown too, not only on a saved frame.
    assert color_count(full, COLOR_MATCH) > 0 and color_count(full, COLOR_FG) > 0
    assert color_count(simple, COLOR_PERSON) > 0
    assert color_count(simple, COLOR_MATCH) == color_count(simple, COLOR_FG) == 0
    assert full.shape == simple.shape == (240, 1020, 3)
    assert np.array_equal(clean, packet.analysis_frame) and clean.shape == (240, 640, 3)
    assert not np.shares_memory(clean, packet.analysis_frame)
    assert np.array_equal(packet.raw_frame, raw_before)
    assert np.array_equal(motion.valid_mask, mask_before)
    assert EventDetector(config).evaluate(packet, objects, motion) == decision


@pytest.mark.parametrize("mode", ["full", "objects", "none"])
def test_display_choice_keeps_capture_as_original_full_size_jpg(moving_object, mode, tmp_path):
    _, packet, objects, motion, decision = moving_object
    render_overlay(packet, objects, motion, decision,
                   CaptureResult("NOT_ELIGIBLE", None, 3., 0., None),
                   {"tracking": True}, overlay_mode=mode)
    writer = CaptureManager(tmp_path, .5, 95, scope="object")
    writer.prepare(mode)
    saved = writer.maybe_save(packet, decision)
    assert saved.status == "SAVED" and saved.saved_track_ids == (1,)
    ok, encoded = cv2.imencode(".jpg", packet.raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok and saved.path.read_bytes() == encoded.tobytes()
    assert cv2.imdecode(np.frombuffer(saved.path.read_bytes(), np.uint8), cv2.IMREAD_COLOR).shape == (480, 1280, 3)


def test_invalid_display_choice_is_rejected_in_python_api(moving_object):
    config, packet, objects, motion, decision = moving_object
    with pytest.raises(ConfigError, match="overlay_mode"):
        validate_analysis_values(replace(config, overlay_mode="invalid"))
    with pytest.raises(ContractError, match="overlay_mode"):
        render_overlay(packet, objects, motion, decision,
                       CaptureResult("DISABLED", None, 3., 0., None), {}, overlay_mode="invalid")
