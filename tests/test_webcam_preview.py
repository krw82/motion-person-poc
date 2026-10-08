from dataclasses import replace
from pathlib import Path
import queue
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from config import Config
from contracts import Box, ContractError, ObjectDetection, VideoDecodeError, VideoOpenError
from overlay_renderer import _status_lines, render_overlay
from webcam_preview import PreviewProcessor, main, parse_args
from webcam_source import WebcamSource


class Camera:
    def __init__(self, opened=True, first=True):
        self.opened = opened
        self.items = queue.Queue()
        self.items.put((first, np.zeros((60, 100, 3), np.uint8) if first else None))
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.items.get(timeout=2)

    def release(self):
        self.released = True

    def frame(self, value):
        self.items.put((True, np.full((60, 100, 3), value, np.uint8)))


def test_latest_frame_replaces_old_frames_and_uses_receipt_time(monkeypatch):
    camera = Camera()
    monkeypatch.setattr('webcam_source.cv2.VideoCapture', lambda index: camera)
    source = WebcamSource()
    assert source.open() == (100, 60)
    try:
        index, elapsed, frame = source.read()
        assert index == 0 and elapsed == 0 and frame.sum() == 0
        for value in (1, 2, 3):
            camera.frame(value)
        with source._condition:
            assert source._condition.wait_for(lambda: source.frames_acquired == 4, timeout=1)
        index, elapsed, frame = source.read()
        assert index == 3 and elapsed > 0 and np.all(frame == 3)
        assert source.frames_read == 2 and source.frames_skipped == 2
    finally:
        camera.frame(4)
        source.close()
    assert camera.released
    source.close()


def test_no_new_camera_frame_is_timeout_not_duplicate(monkeypatch):
    camera = Camera()
    monkeypatch.setattr('webcam_source.cv2.VideoCapture', lambda index: camera)
    source = WebcamSource(timeout_sec=.02)
    source.open()
    try:
        source.read()
        with pytest.raises(VideoDecodeError, match='new frames'):
            source.read()
    finally:
        camera.frame(1)
        source.close()


def test_camera_read_failure_is_reported_and_reader_releases(monkeypatch):
    camera = Camera()
    monkeypatch.setattr('webcam_source.cv2.VideoCapture', lambda index: camera)
    source = WebcamSource()
    source.open()
    try:
        camera.items.put((False, None))
        with source._condition:
            assert source._condition.wait_for(lambda: source._error is not None, timeout=1)
        with pytest.raises(VideoDecodeError, match='read failed'):
            source.read()
    finally:
        source.close()
    assert camera.released


@pytest.mark.parametrize('opened,first,error', [(False, True, VideoOpenError), (True, False, VideoDecodeError)])
def test_failed_camera_open_releases_handle(monkeypatch, opened, first, error):
    camera = Camera(opened, first)
    monkeypatch.setattr('webcam_source.cv2.VideoCapture', lambda index: camera)
    source = WebcamSource()
    with pytest.raises(error):
        source.open()
    assert camera.released
    source.close()


def test_invalid_camera_contracts_and_preview_cli():
    for index in (-1, True, '0'):
        with pytest.raises(ContractError):
            WebcamSource(index)
    with pytest.raises(ContractError):
        WebcamSource().read()
    for args in (['--camera', '-1'], ['--no-display'], ['--duration-sec', 'nan'],
                 ['--no-display', '--duration-sec', '1', '--show-mask']):
        with pytest.raises(SystemExit):
            parse_args(args)


def test_preview_motion_and_detection_never_save_pixels_or_create_output_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = replace(Config(), warmup_sec=0, target_classes=('person', 'dog'), tracking=True)
    processor = PreviewProcessor(config)
    obj = ObjectDetection(0, Box(10, 10, 90, 50), .9, 0, 'person', 1)
    calls = []
    processor.detector = SimpleNamespace(detect=lambda frame: calls.append(frame.shape) or (obj,))
    raw = np.full((60, 100, 3), 160, np.uint8)
    original = raw.copy()
    packet, objects, motion, decision, disabled = processor.process(raw, 0, 0)
    assert calls == [(60, 100, 3)]
    assert objects == (obj,) and not decision.candidate
    assert disabled.status == 'DISABLED' and disabled.path is None and not disabled.saved_track_ids
    assert np.array_equal(original, raw)
    assert list(tmp_path.iterdir()) == []
    metrics = {'preview': True, 'tracking': True, 'processing_fps': 20, 'frame_age_ms': 40}
    rendered = render_overlay(packet, objects, motion, decision, disabled, metrics)
    assert rendered.shape[1] > raw.shape[1]
    assert np.array_equal(original, raw)
    lines = _status_lines(packet, decision, disabled, metrics)
    assert 'NO IMAGE SAVING' in lines[0]
    assert lines[1] == 'DETECTED 1 MOVING 0'
    # Establish background, then introduce motion inside the known object box.
    for i in range(1, 25):
        packet, objects, motion, decision, disabled = processor.process(raw, i, i / 10)
    moving_frame = raw.copy()
    moving_frame[10:50, 10:90] = 255
    packet, objects, motion, decision, disabled = processor.process(moving_frame, 25, 2.5)
    assert decision.candidate
    assert _status_lines(packet, decision, disabled, metrics)[1] == 'DETECTED 1 MOVING 1'
    assert disabled.status == 'DISABLED' and disabled.path is None
    # After the new posture settles the person remains detected without motion.
    for i in range(26, 80):
        packet, objects, motion, decision, disabled = processor.process(moving_frame, i, i / 10)
    assert len(objects) == 1 and not decision.candidate
    assert _status_lines(packet, decision, disabled, metrics)[1] == 'DETECTED 1 MOVING 0'
    assert list(tmp_path.iterdir()) == []


def test_preview_rejects_replayed_indices_and_backwards_time():
    processor = PreviewProcessor(Config())
    processor.detector = SimpleNamespace(detect=lambda frame: ())
    frame = np.zeros((60, 100, 3), np.uint8)
    processor.process(frame, 3, 1.)
    for index, timestamp in [(3, 2.), (2, 2.), (4, .9), (4, float('nan'))]:
        with pytest.raises(ContractError):
            processor.process(frame, index, timestamp)
    assert processor.process(frame, 9, 2.)[-1].status == 'DISABLED'


def test_model_failure_does_not_open_camera_or_create_capture_log_folders(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    def should_not_open(index):
        raise AssertionError('camera should stay closed until model initialization succeeds')
    monkeypatch.setattr('webcam_source.cv2.VideoCapture', should_not_open)
    assert main(['--model', str(tmp_path/'missing.pt'), '--duration-sec', '1']) == 4
    assert list(tmp_path.iterdir()) == []
