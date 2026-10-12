import threading

import cv2
import numpy as np
import pytest

from motion_person import FramePreview, MotionEngine, PreviewConfig
from motion_person.contracts import FrameAnalysis, MotionResult
from motion_person.video_source import VideoSource
from motion_person.webcam_source import WebcamSource
from tests.test_engine import inputs as inputs
from tests.test_event_capture import frame


def test_many_viewers_share_one_encoded_frame_without_consuming_input():
    packet, _ = frame(7, 3.5)

    class Source:
        def snapshot(self):
            return 7, 3.5, packet.raw_frame

        def read(self):
            pytest.fail("a preview reader must never consume analysis input")

    preview = FramePreview()
    preview.attach(Source())
    try:
        assert preview.encode_count == 0
        with preview.subscribe(), preview.subscribe():
            first = preview.snapshot()
            second = preview.snapshot()
            assert first is second and preview.encode_count == 1
            assert first.frame_index == 7 and first.elapsed_sec == 3.5
            assert cv2.imdecode(np.frombuffer(first.jpeg, np.uint8), 1).shape == (
                72,
                128,
                3,
            )
        assert preview._viewers == 0
    finally:
        preview.close()
    assert not preview._thread.is_alive() and preview._source is None


def test_camera_snapshot_does_not_mark_frame_consumed():
    source = WebcamSource()
    source._capture = object()
    source._latest = (12, 1.2, frame(12, 1.2)[0].raw_frame)
    assert source.snapshot() is source.snapshot()
    assert source.frames_read == 0 and source._last_index == -1
    assert source.read() is source.snapshot()
    assert source.frames_read == 1 and source._last_index == 12
    source._capture = None


def test_video_snapshot_never_decodes_an_extra_frame(tmp_path):
    packet, _ = frame(0, 0)

    class Capture:
        calls = 0

        def read(self):
            self.calls += 1
            return True, packet.raw_frame

        def release(self):
            pass

    source = VideoSource(tmp_path / "fake.mp4")
    source._capture, source._fps = Capture(), 30
    source._first_frame = packet.raw_frame
    first = source.read()
    assert source.snapshot() is first and source.snapshot() is first
    assert source.frames_read == 1 and source._capture.calls == 0
    second = source.read()
    assert source.snapshot() is second and source._capture.calls == 1
    source.close()


def test_mirrored_overlay_uses_original_coordinates_without_changing_evidence(
    monkeypatch,
):
    packet, decision = frame(5, 2.5, True)
    analysis = FrameAnalysis(
        packet,
        (),
        MotionResult(np.zeros((72, 128), np.uint8), (), 0, 0, False),
        decision,
    )
    original = packet.raw_frame.copy()
    rectangles = []

    def rectangle(image, start, end, *args):
        rectangles.append((start, end))
        return image

    monkeypatch.setattr(cv2, "rectangle", rectangle)
    preview = FramePreview(PreviewConfig(width=64, mirror=True, overlay=True))
    image = preview._encode((5, 2.5, packet.raw_frame), analysis)
    assert (image.width, image.height, image.frame_index) == (64, 36, 5)
    assert rectangles == [((22, 5), (56, 32))]
    np.testing.assert_array_equal(original, packet.raw_frame)


def test_unexpected_source_preview_error_is_reported_and_worker_can_recover():
    attempted = threading.Event()
    packet, _ = frame(0, 0)

    class Source:
        fail = True

        def snapshot(self):
            attempted.set()
            if self.fail:
                raise RuntimeError("preview source failed")
            return 0, 0.0, packet.raw_frame

    source = Source()
    preview = FramePreview(PreviewConfig(fps=30))
    preview.attach(source)
    try:
        with preview.subscribe():
            assert attempted.wait(1)
            with preview._condition:
                assert preview._condition.wait_for(
                    lambda: preview.last_error is not None, timeout=1
                )
            assert "preview source failed" in preview.last_error
            source.fail = False
            assert preview.snapshot() is not None
            assert preview.last_error is None
    finally:
        preview.close()


@pytest.mark.parametrize("failure", ["attach", "publish", "close"])
def test_optional_preview_failure_preserves_capture_and_callback(
    inputs, tmp_path, failure
):
    video, resources, _, _ = inputs

    class BrokenPreview:
        closed = False

        def attach(self, source):
            if failure == "attach":
                raise RuntimeError("preview failure")

        def publish(self, analysis):
            if failure == "publish":
                raise RuntimeError("preview failure")

        def close(self):
            self.closed = True
            if failure == "close":
                raise RuntimeError("preview failure")

    received = []
    preview = BrokenPreview()
    engine = MotionEngine.video(video, output_dir=tmp_path, pace="fast")
    summary = engine.start(preview=preview, on_event=received.append)
    assert len(received) == 1 and received[0].manifest_path.is_file()
    assert summary.end_reason == "END_OF_VIDEO" and summary.preview_errors == 1
    assert "preview failure" in engine.progress()["preview_error"]
    assert preview.closed and all(source.closed for source in resources)
