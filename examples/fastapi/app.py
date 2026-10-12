"""Same-PC host example. Run with one Uvicorn worker; core has no FastAPI dependency."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
from typing import Callable, Literal
import uuid

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from motion_person import EventBundle, FramePreview, MotionEngine, PreviewConfig

ROOT = Path(__file__).resolve().parents[2]


class RunRequest(BaseModel):
    source: Literal["webcam", "video"] = "webcam"
    camera: int = Field(default=0, ge=0)
    video_id: str | None = None
    objects: list[str] = Field(default_factory=lambda: ["person"], min_length=1)
    mirror: bool = True
    overlay: bool = False
    fast: bool = False

    @field_validator("objects")
    @classmethod
    def normalize_objects(cls, values):
        result = list(dict.fromkeys(name.strip().lower() for name in values))
        if any(not name for name in result):
            raise ValueError("object labels cannot be blank")
        return result


@dataclass
class EventRecord:
    bundle: EventBundle
    analysis_status: str = "ready"
    analysis_result: object = None
    analysis_error: str | None = None

    def payload(self) -> dict:
        data = self.bundle.to_dict()
        data.pop("manifest_path")
        if not data["camera_id"].startswith("webcam:"):
            data["camera_id"] = "video"
        for index, image in enumerate(data["images"]):
            image.pop("path")
            image["url"] = (
                f"/motion/events/{self.bundle.event_id}/parts/{self.bundle.part_index}/images/{index}"
            )
        return {
            **data,
            "analysis_status": self.analysis_status,
            "analysis_result": self.analysis_result,
            "analysis_error": self.analysis_error,
        }


class MotionHost:
    """One camera run, bounded recent metadata and bounded optional analysis jobs."""

    def __init__(
        self,
        output_dir: Path,
        model_path: Path,
        *,
        engine_factory=None,
        analyze: Callable[[EventBundle], object] | None = None,
    ):
        self.output_dir, self.model_path = output_dir, model_path
        self.engine_factory, self.analyze = engine_factory, analyze
        self.lock = threading.RLock()
        self.engine = self.preview = self.worker = None
        self.run_id = self.request = self.summary = self.error = None
        self.stopping = False
        self.records: OrderedDict[tuple[str, int], EventRecord] = OrderedDict()
        self.jobs = queue.Queue(maxsize=8)
        self.shutdown = threading.Event()
        self.analysis_worker = None
        if analyze is not None:
            self.analysis_worker = threading.Thread(
                target=self._analyze_jobs, name="host-analysis", daemon=True
            )
            self.analysis_worker.start()

    def start(self, request: RunRequest, video_path: Path | None = None) -> str:
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                if self.request == request:
                    return self.run_id
                raise HTTPException(
                    409, "Stop the current run before changing its input or settings"
                )
            if self.engine_factory is not None:
                engine = self.engine_factory(request)
            else:
                options = dict(
                    objects=tuple(request.objects),
                    model_path=self.model_path,
                    output_dir=self.output_dir / "events",
                    capture=True,
                    display=False,
                )
                engine = (
                    MotionEngine.webcam(request.camera, **options)
                    if request.source == "webcam"
                    else MotionEngine.video(
                        video_path,
                        pace="fast" if request.fast else "realtime",
                        **options,
                    )
                )
            self.run_id = uuid.uuid4().hex
            self.request, self.engine = request, engine
            self.preview = FramePreview(
                PreviewConfig(mirror=request.mirror, overlay=request.overlay)
            )
            self.summary = self.error = None
            self.stopping = False
            self.records.clear()
            self.worker = threading.Thread(
                target=self._run, name="motion-run", daemon=True
            )
            self.worker.start()
            return self.run_id

    def _run(self) -> None:
        try:
            summary = self.engine.start(
                run_id=self.run_id, preview=self.preview, on_event=self._completed
            )
            with self.lock:
                self.summary = summary.to_dict()
                self.summary.pop("run_dir")
                self.summary["source"] = (
                    f"webcam:{self.request.camera}"
                    if self.request.source == "webcam"
                    else "video"
                )
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            self.preview.close()

    def _completed(self, bundle: EventBundle) -> None:
        record = EventRecord(bundle)
        with self.lock:
            self.records[(bundle.event_id, bundle.part_index)] = record
            while len(self.records) > 32:
                self.records.popitem(last=False)
            if self.analyze is not None:
                record.analysis_status = "queued"
                try:
                    self.jobs.put_nowait(record)
                except queue.Full:
                    record.analysis_status = "failed"
                    record.analysis_error = (
                        "Analysis queue is full; the saved manifest remains available"
                    )

    def _analyze_jobs(self) -> None:
        while not self.shutdown.is_set():
            try:
                record = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                with self.lock:
                    record.analysis_status = "analyzing"
                result = self.analyze(record.bundle)
                with self.lock:
                    record.analysis_result, record.analysis_status = result, "done"
            except Exception as exc:
                with self.lock:
                    record.analysis_error, record.analysis_status = str(exc), "failed"
            finally:
                self.jobs.task_done()

    def stop(self) -> None:
        with self.lock:
            if self.engine is not None:
                self.stopping = True
                self.engine.stop()

    def status(self) -> dict:
        with self.lock:
            state = (
                self.engine.progress()
                if self.engine is not None
                else {"status": "idle"}
            )
            if self.stopping and self.worker is not None and self.worker.is_alive():
                state["status"] = "stopping"
            return {
                **state,
                "run_id": self.run_id,
                "summary": self.summary,
                "error": self.error,
                "preview_error": state.get("preview_error")
                or (self.preview.last_error if self.preview is not None else None),
            }

    def close(self) -> None:
        self.stop()
        try:
            if self.worker is not None:
                self.worker.join(timeout=8)
                if self.worker.is_alive():
                    raise RuntimeError("camera worker did not finish shutdown")
        finally:
            self.shutdown.set()
            if self.analysis_worker is not None:
                self.analysis_worker.join(timeout=2)


def create_app(
    *,
    output_dir: Path = ROOT / "outputs/web",
    model_path: Path = ROOT / "models/yolo11n.pt",
    engine_factory=None,
    analyze=None,
) -> FastAPI:
    output_dir = Path(output_dir)

    @asynccontextmanager
    async def lifespan(app):
        app.state.motion = MotionHost(
            output_dir, Path(model_path), engine_factory=engine_factory, analyze=analyze
        )
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.motion.close)

    app = FastAPI(title="Motion capture host example", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://{host}:{port}"
            for host in ("localhost", "127.0.0.1")
            for port in (5173, 5174)
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Frame-Index", "X-Elapsed-Sec"],
    )

    def host(run_id: str | None = None) -> MotionHost:
        instance = app.state.motion
        if run_id is not None and instance.run_id != run_id:
            raise HTTPException(404, "Run is no longer active in this example")
        return instance

    @app.get("/motion/status")
    def current_status():
        return host().status()

    @app.post("/motion/videos", status_code=201)
    async def upload_video(file: UploadFile):
        identifier = uuid.uuid4().hex
        folder = output_dir / "inputs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (identifier + ".video")
        size = 0
        try:
            with path.open("xb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 256 * 1024 * 1024:
                        raise HTTPException(413, "Example upload limit is 256 MiB")
                    stream.write(chunk)
            if not size:
                raise HTTPException(400, "Video is empty")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        return {"video_id": identifier}

    @app.post("/motion/runs", status_code=202)
    def start_run(request: RunRequest):
        video = None
        if request.source == "video":
            if not request.video_id or not request.video_id.isalnum():
                raise HTTPException(400, "Upload a video first")
            video = output_dir / "inputs" / (request.video_id + ".video")
            if not video.is_file():
                raise HTTPException(404, "Uploaded video not found")
        return {"run_id": host().start(request, video)}

    @app.post("/motion/runs/{run_id}/stop", status_code=202)
    def stop_run(run_id: str):
        host(run_id).stop()
        return {"run_id": run_id, "status": "stopping"}

    @app.get("/motion/runs/{run_id}")
    def run_status(run_id: str):
        return host(run_id).status()

    @app.get("/motion/runs/{run_id}/events")
    def events(run_id: str):
        instance = host(run_id)
        with instance.lock:
            return [record.payload() for record in instance.records.values()]

    @app.get("/motion/runs/{run_id}/preview.jpg")
    async def snapshot(run_id: str):
        image = await asyncio.to_thread(host(run_id).preview.snapshot, timeout=1)
        if image is None:
            raise HTTPException(503, "Preview is preparing or has no frames")
        return Response(
            image.jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Frame-Index": str(image.frame_index),
                "X-Elapsed-Sec": str(image.elapsed_sec),
            },
        )

    @app.get("/motion/runs/{run_id}/preview.mjpg")
    async def stream(run_id: str):
        preview = host(run_id).preview

        async def images():
            with preview.subscribe():
                previous = -1
                while not preview.closed:
                    image = await asyncio.to_thread(
                        preview.snapshot, after=previous, timeout=0.5
                    )
                    if image is None:
                        continue
                    previous = image.frame_index
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(image.jpeg)).encode()
                        + b"\r\n\r\n"
                        + image.jpeg
                        + b"\r\n"
                    )

        return StreamingResponse(
            images(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/motion/events/{event_id}/parts/{part_index}/images/{image_id}")
    def event_image(event_id: str, part_index: int, image_id: int):
        instance = host()
        with instance.lock:
            record = instance.records.get((event_id, part_index))
            if record is None or not 0 <= image_id < len(record.bundle.images):
                raise HTTPException(404, "Photo not in the recent event list")
            path = record.bundle.images[image_id].path
        return FileResponse(path, media_type="image/jpeg")

    return app


app = create_app()
