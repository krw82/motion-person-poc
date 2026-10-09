from dataclasses import replace
import json
from pathlib import Path
import time
import subprocess
import sys

import cv2
import numpy as np
import pytest

from motion_person import Config, EventConfig, MotionEngine
from motion_person.contracts import Box, CaptureResult, ConfigError, MotionResult, ObjectDetection, VideoDecodeError, VideoInfo
from motion_person.engine import _mirror_view
from tests.test_event_capture import frame


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    import motion_person.engine as module
    video = tmp_path / "input.mp4"
    video.touch()
    resources = []
    observations = []
    class Processor:
        def __init__(self, config, run_id):
            self.run_id = run_id
        def load(self, path):
            path.write_text("fake tracker config")
        def process(self, raw, index, elapsed):
            packet, decision = frame(index, elapsed, 3 <= elapsed <= 3.5, empty=elapsed > 3.5)
            packet = replace(packet, run_id=self.run_id, raw_frame=raw, analysis_frame=raw)
            observations.append(elapsed)
            motion = MotionResult(np.zeros(raw.shape[:2],np.uint8), (), 0,0,False)
            return packet, (), motion, decision, CaptureResult("DISABLED",None,elapsed,0,None)
    class Source:
        end_reason = None
        frames_skipped = 0
        def __init__(self, *args):
            self.i = 0
            self.closed = False
            self.started_perf = time.perf_counter()
        def open(self):
            resources.append(self)
            return VideoInfo(video,2,15,128,72,"frame_index_over_fps",True)
        def read(self):
            if self.i == 15:
                self.end_reason = "END_OF_STREAM"
                return None
            index, t = self.i, self.i / 2
            self.i += 1
            return index,t,frame(index,t)[0].raw_frame
        def close(self):
            self.closed = True
    class Webcam(Source):
        def open(self):
            resources.append(self)
            return 128,72
        def read(self):
            item = super().read()
            if item is None:
                raise VideoDecodeError("camera disconnected")
            index,t,raw = item
            self.frames_skipped += 2
            return index*3,t,raw
    monkeypatch.setattr(module,"FrameProcessor",Processor)
    monkeypatch.setattr(module,"VideoSource",Source)
    monkeypatch.setattr(module,"WebcamSource",Webcam)
    return video,resources,observations,module


def test_file_and_webcam_use_same_completed_bundle_flow(inputs,tmp_path):
    video,resources,observations,_ = inputs
    results = []
    for engine in (MotionEngine.video(video,output_dir=tmp_path/"file",display=False,pace="fast"),
                   MotionEngine.webcam(capture=True,output_dir=tmp_path/"camera",display=False)):
        received=[]
        def completed(bundle):
            assert observations[-1] == 6.5
            assert bundle.manifest_path.is_file()
            assert all(path.is_file() for path in bundle.image_paths)
            received.append(bundle)
        summary = engine.start(on_event=completed,duration_sec=6.6)
        assert summary.end_reason == "DURATION_LIMIT" and summary.frames_processed == 14
        assert summary.event_count == 1 and summary.image_count == 6
        assert all(resource.closed for resource in resources)
        assert json.loads((summary.run_dir/"summary.json").read_text())["event_count"] == 1
        assert received[0].context_complete and received[0].duration_sec == 6.5
        results.append((summary,received[0]))
    assert [i.timestamp_sec for i in results[0][1].images] == [i.timestamp_sec for i in results[1][1].images]
    assert [p.read_bytes() for p in results[0][1].image_paths] == [p.read_bytes() for p in results[1][1].image_paths]
    assert results[1][0].frames_skipped > 0


def test_preview_does_not_create_output_and_callback_requires_capture(inputs,tmp_path):
    _,resources,_,_ = inputs
    output = tmp_path / "no-output"
    engine = MotionEngine.webcam(output_dir=output,display=False)
    with pytest.raises(ConfigError,match="capture=True"):
        engine.start(on_event=lambda b: None)
    summary = engine.start(duration_sec=2)
    assert summary.run_dir is None and summary.image_count == summary.event_count == 0
    assert not output.exists() and all(source.closed for source in resources)


def test_stop_from_callback_cleans_up_and_engine_can_be_reused(inputs,tmp_path):
    video,resources,_,_ = inputs
    engine = MotionEngine.video(video,output_dir=tmp_path/"out",display=False,pace="fast")
    summaries=[]
    for _ in range(2):
        summary = engine.start(on_event=lambda bundle: engine.stop())
        summaries.append(summary)
        assert summary.end_reason == "USER_STOP" and summary.event_count == 1
        assert all(source.closed for source in resources)
    assert summaries[0].run_id != summaries[1].run_id


def test_callback_error_preserves_saved_bundle_and_releases_input(inputs,tmp_path):
    video,resources,_,_ = inputs
    output = tmp_path/"out"
    calls=[]
    def fail(bundle):
        calls.append(bundle)
        raise RuntimeError("consumer failed")
    engine = MotionEngine.video(video,output_dir=output,display=False,pace="fast")
    with pytest.raises(RuntimeError,match="consumer failed"):
        engine.start(on_event=fail)
    assert len(calls) == 1 and calls[0].manifest_path.is_file()
    assert all(source.closed for source in resources) and not engine._running
    summary, = output.glob("*/summary.json")
    assert json.loads(summary.read_text())["end_reason"] == "ERROR"


def test_eof_flush_and_completion_verified_are_recorded(inputs,tmp_path):
    video,resources,_,_ = inputs
    engine = MotionEngine.video(video,events=EventConfig(quiet_sec=10),output_dir=tmp_path/"out",display=False,pace="fast")
    received=[]
    summary=engine.start(on_event=received.append)
    assert summary.end_reason == "END_OF_VIDEO" and summary.input_completion_verified
    assert received[0].end_reason == "END_OF_VIDEO" and not received[0].post_context_complete
    assert all(source.closed for source in resources)


def test_webcam_disconnect_closes_observed_tail_as_error(inputs,tmp_path):
    _,resources,_,_ = inputs
    output = tmp_path/"out"
    engine = MotionEngine.webcam(capture=True,events=EventConfig(quiet_sec=10),output_dir=output,display=False)
    received=[]
    with pytest.raises(VideoDecodeError,match="disconnected"):
        engine.start(on_event=received.append)
    # Error tails are saved for recovery but never sent to the external callback.
    assert not received
    manifest, = output.glob("*/events/*/*/event.json")
    assert json.loads(manifest.read_text())["end_reason"] == "ERROR"
    assert all(source.closed for source in resources)


def test_model_failure_before_camera_open_or_output_creation(tmp_path):
    output = tmp_path/"out"
    engine=MotionEngine.webcam(model_path=tmp_path/"missing.pt",output_dir=output,capture=True,display=False)
    from motion_person.contracts import ModelNotFoundError
    with pytest.raises(ModelNotFoundError):
        engine.start()
    assert not output.exists() and not engine._running


@pytest.mark.parametrize("duration", [0,-1,float("nan"),True])
def test_invalid_duration_fails_before_open(inputs,tmp_path,duration):
    video,resources,_,_ = inputs
    engine=MotionEngine.video(video,display=False)
    with pytest.raises(ConfigError):
        engine.start(duration_sec=duration)
    assert not resources


def test_webcam_factory_alias_and_selected_classes():
    engine=MotionEngine.webCam(objects=("person","dog"),capture=True,display=False)
    assert engine.config.target_classes == ("person","dog") and engine.capture


def test_mirror_only_changes_display_pixels_boxes_and_preserves_original_packet():
    packet,decision=frame(0,0,True)
    detection=ObjectDetection(0,Box(15,10,85,65),.9,0)
    motion=MotionResult(np.zeros((72,128),np.uint8),(),0,0,False)
    before=packet.raw_frame.copy()
    view,objects,_,mirrored=_mirror_view(packet,(detection,),motion,decision)
    assert np.array_equal(packet.raw_frame,before)
    assert np.array_equal(view.analysis_frame,cv2.flip(before,1))
    assert objects[0].box == Box(43,10,113,65) and mirrored.objects[0].box == objects[0].box


def test_cli_dispatches_capture_and_event_modes_without_changing_preview(tmp_path,monkeypatch):
    import cli
    model=tmp_path/"model.pt"
    model.touch()
    video=tmp_path/"video.mp4"
    video.touch()
    calls=[]
    monkeypatch.setattr(cli,"launch_bundles",lambda options,targets,path: calls.append((options,targets,path)) or 0)
    assert cli.main(["webcam","--capture","--model",str(model),"--no-display","--objects","사람","개"]) == 0
    assert calls[-1][1:] == (["person","dog"],None)
    assert calls[-1][0].capture and calls[-1][0].no_display
    assert cli.main(["video",str(video),"--events","--model",str(model),"--fast"]) == 0
    assert calls[-1][2] == video and calls[-1][0].events


def test_cli_rejects_event_tuning_without_bundle_mode(tmp_path,monkeypatch):
    import cli
    model=tmp_path/"model.pt"
    model.touch()
    with pytest.raises(SystemExit) as exc:
        cli.main(["webcam","--model",str(model),"--pre-sec","2"])
    assert exc.value.code == 2


def test_cli_help_and_doctor_work_when_runtime_dependencies_are_unavailable():
    root = Path(__file__).resolve().parents[1]
    # -S excludes site-packages: numpy/opencv/ultralytics cannot be imported.
    result = subprocess.run([sys.executable,"-S","cli.py","--help"],cwd=root,capture_output=True,text=True)
    assert result.returncode == 0 and "사용법:" in result.stdout
    result = subprocess.run([sys.executable,"-S","cli.py","doctor"],cwd=root,capture_output=True,text=True)
    assert result.returncode == 2 and "[필요] opencv-python" in result.stdout
    assert "Traceback" not in result.stderr
