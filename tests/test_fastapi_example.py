from dataclasses import replace
import json
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from examples.fastapi.app import MotionHost, RunRequest, create_app
from motion_person import MotionEngine
from motion_person.event_capture import EventCaptureEngine
from tests.test_engine import inputs as inputs
from tests.test_event_capture import feed


def test_host_owns_one_run_and_cached_preview_readers_do_not_consume_input(
    inputs, tmp_path, monkeypatch
):
    video, resources, observations, module = inputs
    ready, release = threading.Event(), threading.Event()
    original = module.VideoSource

    class HeldSource(original):
        def read(self):
            if self.i == 15:
                ready.set()
                release.wait(3)
            return super().read()

    monkeypatch.setattr(module, "VideoSource", HeldSource)
    creations = []

    def factory(request):
        creations.append(request)
        return MotionEngine.video(video, output_dir=tmp_path / "events", pace="fast")

    app = create_app(output_dir=tmp_path, engine_factory=factory)
    with TestClient(app) as client:
        try:
            assert client.get("/motion/status").json()["status"] == "idle"
            start = client.post("/motion/runs", json={})
            assert start.status_code == 202
            run = start.json()["run_id"]
            assert ready.wait(2)
            duplicate = client.post("/motion/runs", json={})
            assert (
                duplicate.json()["run_id"] == run
                and len(creations) == len(resources) == 1
            )
            assert client.post("/motion/runs", json={"camera": 1}).status_code == 409
            first = client.get(f"/motion/runs/{run}/preview.jpg")
            second = client.get(f"/motion/runs/{run}/preview.jpg")
            assert first.status_code == second.status_code == 200
            assert (
                first.content == second.content
                and first.headers["X-Frame-Index"] == "14"
            )
            assert first.headers["Cache-Control"] == "no-store"
            assert (
                len(observations) == 15 and app.state.motion.preview.encode_count == 1
            )
            records = client.get(f"/motion/runs/{run}/events").json()
            assert len(records) == 1 and records[0]["evidence_complete"]
            assert str(tmp_path) not in json.dumps(records)
            assert records[0]["analysis_status"] == "ready"
            for image in records[0]["images"]:
                response = client.get(image["url"])
                assert (
                    response.status_code == 200
                    and response.headers["content-type"] == "image/jpeg"
                )
            assert client.get("/motion/runs/unknown/preview.jpg").status_code == 404
            assert client.post(f"/motion/runs/{run}/stop").status_code == 202
        finally:
            release.set()
    assert all(source.closed for source in resources)
    assert app.state.motion.preview.closed and not app.state.motion.worker.is_alive()


def test_uploads_have_opaque_ids_and_empty_upload_is_cleaned_up(tmp_path):
    with TestClient(create_app(output_dir=tmp_path)) as client:
        response = client.post(
            "/motion/videos", files={"file": ("../../secret.mp4", b"example video")}
        )
        assert response.status_code == 201
        identifier = response.json()["video_id"]
        assert len(identifier) == 32
        assert (
            tmp_path / "inputs" / (identifier + ".video")
        ).read_bytes() == b"example video"
        empty = client.post("/motion/videos", files={"file": ("empty.mp4", b"")})
        assert (
            empty.status_code == 400 and len(list((tmp_path / "inputs").iterdir())) == 1
        )
        assert (
            client.post(
                "/motion/runs", json={"source": "video", "video_id": "../secret"}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/motion/runs", json={"source": "video", "video_id": "missing"}
            ).status_code
            == 404
        )
        assert client.post("/motion/runs", json={"objects": []}).status_code == 422
        assert client.post("/motion/runs", json={"objects": [" "]}).status_code == 422
    assert RunRequest(objects=[" Person ", "person"]).objects == ["person"]


def test_analysis_queue_full_is_bounded_without_blocking_capture_callback(tmp_path):
    (bundle,) = feed(
        EventCaptureEngine(tmp_path / "bundles", "run", "webcam:0"),
        [i / 2 for i in range(15)],
        lambda t: 3 <= t <= 3.5,
    )
    entered, release = threading.Event(), threading.Event()

    def analyze(bundle):
        entered.set()
        release.wait(3)
        return {"description": "example result"}

    host = MotionHost(tmp_path, tmp_path / "unused.pt", analyze=analyze)
    try:
        host._completed(bundle)
        assert entered.wait(1)
        for index in range(2, 42):
            host._completed(replace(bundle, part_index=index))
        assert len(host.records) == 32 and host.jobs.qsize() == 8
        latest = host.records[(bundle.event_id, 41)]
        assert (
            latest.analysis_status == "failed"
            and "queue is full" in latest.analysis_error
        )
        assert bundle.manifest_path.is_file()
    finally:
        release.set()
        host.close()
    assert not host.analysis_worker.is_alive()


def test_gemini_adapter_exception_does_not_invalidate_saved_evidence(tmp_path):
    (bundle,) = feed(
        EventCaptureEngine(tmp_path / "bundles", "run", "webcam:0"),
        [i / 2 for i in range(15)],
        lambda t: 3 <= t <= 3.5,
    )

    def fail(bundle):
        raise RuntimeError("external analysis unavailable")

    host = MotionHost(tmp_path, tmp_path / "unused.pt", analyze=fail)
    try:
        host._completed(bundle)
        host.jobs.join()
        record = host.records[(bundle.event_id, 1)]
        assert (
            record.analysis_status == "failed"
            and "unavailable" in record.analysis_error
        )
        assert record.bundle.manifest_path.is_file() and all(
            path.is_file() for path in bundle.image_paths
        )
    finally:
        host.close()
