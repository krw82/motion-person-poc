"""Blocking high-level API; host applications own cloud jobs and alerts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable
import uuid

import cv2

from motion_person.domain.settings import (
    Config,
    config_to_dict,
    validate_analysis_values,
)
from motion_person.domain.contracts import (
    Box,
    CaptureResult,
    ConfigError,
    VideoNotFoundError,
)
from motion_person.application.event_capture import (
    EventBundle,
    EventCaptureEngine,
    EventConfig,
)
from motion_person.application.processor import FrameProcessor
from motion_person.adapters.diagnostics import (
    RunLogger as RunLogger,
    OptionalRunDiagnostics,
)
from motion_person.application.ports import PreviewSink
from motion_person.adapters.video import END_OF_STREAM, VideoSource
from motion_person.adapters.webcam import WebcamSource


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    source: str
    end_reason: str
    frames_processed: int
    candidate_frames: int
    event_count: int
    image_count: int
    processing_fps: float
    run_dir: Path | None
    input_completion_verified: bool
    frames_skipped: int
    peak_event_buffer_bytes: int
    episode_count: int = 0
    diagnostic_errors: int = 0
    preview_errors: int = 0

    @property
    def bundle_count(self) -> int:
        return self.event_count

    def to_dict(self):
        data = asdict(self)
        data["bundle_count"] = self.bundle_count
        data["run_dir"] = str(self.run_dir) if self.run_dir is not None else None
        return data


class MotionEngine:
    """video(...).start() or webcam(..., capture=True).start(on_event=...).

    start() blocks until EOF/stop/duration/interrupt and releases all resources.
    Callbacks run on this processing thread *after* a complete bundle is saved;
    use a short callback to enqueue external analysis rather than waiting for it.
    An instance can be reused after a run, but cannot start two concurrent runs.
    """

    def __init__(
        self,
        source_kind,
        source_value,
        *,
        config,
        events,
        output_dir,
        capture,
        mirror=False,
        diagnostics=False,
        processor_factory=None,
    ):
        if not isinstance(capture, bool) or not isinstance(mirror, bool):
            raise ConfigError("capture and mirror must be boolean")
        if events is not None and not isinstance(events, EventConfig):
            raise ConfigError("events must be an EventConfig")
        validate_analysis_values(config)
        if config.show_mask and not config.display:
            raise ConfigError("show_mask requires display")
        if not isinstance(diagnostics, bool):
            raise ConfigError("diagnostics must be boolean")
        if processor_factory is not None and not callable(processor_factory):
            raise ConfigError("processor_factory must be callable")
        self.diagnostics = diagnostics
        self.processor_factory = processor_factory
        self.run_id = None
        self._progress = {"status": "idle", "frames_processed": 0}
        self.source_kind, self.source_value = source_kind, source_value
        self.config, self.event_config = config, events or EventConfig()
        self.output_dir = Path(output_dir).expanduser()
        self.capture, self.mirror = capture, mirror
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._running = False

    @staticmethod
    def _config(config, objects, model_path, display, overlay, pace):
        if config is not None and not isinstance(config, Config):
            raise ConfigError("config must be a Config")
        if not isinstance(display, bool):
            raise ConfigError("display must be boolean")
        if isinstance(objects, str):
            objects = tuple(name.strip() for name in objects.split(","))
        base = config or replace(
            Config(),
            tracking=True,
            person_confidence=0.4,
            min_person_motion_pixels_ref=1000,
            capture_cooldown_sec=0.5,
        )
        return replace(
            base,
            target_classes=tuple(objects)
            if objects is not None
            else base.target_classes,
            model_path=Path(model_path).expanduser()
            if model_path is not None
            else base.model_path,
            display=display,
            overlay_mode=overlay,
            pace=pace,
        )

    @classmethod
    def video(
        cls,
        path,
        *,
        objects=None,
        model_path=None,
        output_dir="outputs",
        capture=True,
        display=False,
        overlay="full",
        events=None,
        config=None,
        pace="realtime",
        diagnostics=False,
        processor_factory=None,
    ):
        settings = cls._config(config, objects, model_path, display, overlay, pace)
        path = Path(path).expanduser()
        return cls(
            "video",
            path,
            config=replace(settings, video_path=path),
            events=events,
            output_dir=output_dir,
            capture=capture,
            diagnostics=diagnostics,
            processor_factory=processor_factory,
        )

    @classmethod
    def webcam(
        cls,
        camera=0,
        *,
        objects=None,
        model_path=None,
        output_dir="outputs",
        capture=False,
        display=False,
        overlay="full",
        events=None,
        config=None,
        mirror=True,
        diagnostics=False,
        processor_factory=None,
    ):
        # Validate the camera index without opening a device.
        WebcamSource(camera)
        settings = cls._config(
            config, objects, model_path, display, overlay, "realtime"
        )
        return cls(
            "webcam",
            camera,
            config=settings,
            events=events,
            output_dir=output_dir,
            capture=capture,
            mirror=mirror,
            diagnostics=diagnostics,
            processor_factory=processor_factory,
        )

    webCam = webcam

    def stop(self):
        """Request exit at the next input/display boundary; safe from a callback/thread."""
        self._stop.set()

    def progress(self) -> dict:
        with self._lock:
            return dict(self._progress)

    def start(
        self,
        *,
        on_event: Callable[[EventBundle], None] | None = None,
        duration_sec: float | None = None,
        preview: PreviewSink | None = None,
        run_id: str | None = None,
    ) -> RunSummary:
        if on_event is not None and not callable(on_event):
            raise ConfigError("on_event must be callable")
        if on_event is not None and not self.capture:
            raise ConfigError("on_event requires capture=True")
        if duration_sec is not None and (
            not isinstance(duration_sec, (int, float))
            or isinstance(duration_sec, bool)
            or not math.isfinite(duration_sec)
            or duration_sec <= 0
        ):
            raise ConfigError("duration_sec must be a finite number > 0")
        if self.source_kind == "video" and not self.source_value.is_file():
            raise VideoNotFoundError(f"video file not found: {self.source_value}")
        if run_id is not None and (
            not isinstance(run_id, str)
            or not run_id
            or Path(run_id).name != run_id
            or run_id in (".", "..")
        ):
            raise ConfigError("run_id must be a filename component")
        with self._lock:
            if self._running:
                raise ConfigError("this MotionEngine is already running")
            self._running = True
            self._stop.clear()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = run_id or f"{stamp}_{uuid.uuid4().hex[:8]}"
        self.run_id = run_id
        self._progress = {
            "status": "preparing",
            "frames_processed": 0,
            "run_id": run_id,
        }
        run_dir = (self.output_dir / run_id).resolve() if self.capture else None
        source_name = (
            f"webcam:{self.source_value}"
            if self.source_kind == "webcam"
            else str(self.source_value.resolve())
        )
        source = window = logger = captures = None
        processed = moving = 0
        processing_seconds = 0.0
        reason = "ERROR"
        error = None
        completion_verified = False
        preview_errors = 0
        preview_error = None

        def preview_call(method, *args, closing=False):
            # Preview is optional: its adapter failures cannot discard evidence.
            nonlocal preview_errors, preview_error
            if preview is None or (preview_error is not None and not closing):
                return
            try:
                getattr(preview, method)(*args)
            except Exception as exc:
                preview_errors += 1
                preview_error = f"{type(exc).__name__}: {exc}"

        def record_cleanup(exc, label):
            nonlocal error, reason
            if error is None:
                error, reason = exc, "ERROR"
            else:
                error.add_note(f"{label} also failed: {exc!r}")

        def recover_completed():
            if captures is not None:
                try:
                    deliver(captures.drain_completed())
                except Exception as exc:
                    record_cleanup(exc, "Completed bundle delivery")

        def deliver(bundles, *, callbacks=True):
            for bundle in bundles:
                if logger is not None:
                    logger.write("EVENT_COMPLETED", bundle.to_dict())
                if on_event is not None and callbacks:
                    on_event(bundle)

        try:
            with tempfile.TemporaryDirectory(
                prefix="motion-person-tracker-"
            ) as temporary:
                factory = self.processor_factory or FrameProcessor
                processor = factory(self.config, run_id)
                tracker_path = Path(temporary) / "tracker.yaml"
                # Loading can fail without opening a webcam or creating output folders.
                processor.load(tracker_path)
                if self.source_kind == "video":
                    source = VideoSource(self.source_value, self.config.fallback_fps)
                    info = source.open()
                    width, height, fps = info.width, info.height, info.fps
                    input_info = info.to_dict()
                else:
                    source = WebcamSource(self.source_value)
                    width, height = source.open()
                    fps = None
                    input_info = {
                        "camera": self.source_value,
                        "width": width,
                        "height": height,
                        "timeline_source": "monotonic_receipt_time",
                    }
                preview_call("attach", source)
                if run_dir is not None and self.diagnostics:
                    logger = OptionalRunDiagnostics(run_dir.parent, run_id)
                    logger.start(
                        {
                            **config_to_dict(self.config),
                            "source_kind": self.source_kind,
                            "source": source_name,
                            "event_config": asdict(self.event_config),
                            "capture_mode": "event_bundles",
                        },
                        input_info,
                    )
                    logger.copy_tracker(tracker_path)
                if run_dir is not None:
                    captures = EventCaptureEngine(
                        run_dir / "events",
                        run_id,
                        source_name,
                        config=self.event_config,
                        jpeg_quality=self.config.jpeg_quality,
                        analysis_roi=self.config.analysis_roi,
                    )
                if self.config.display:
                    from motion_person.presentation.overlay import (
                        DisplayWindow,
                        render_mask_view,
                        render_overlay,
                    )

                    window = DisplayWindow(show_mask=self.config.show_mask)
                loop_started = time.perf_counter()
                reason = "USER_STOP"
                while not self._stop.is_set():
                    if (
                        self.source_kind == "webcam"
                        and duration_sec is not None
                        and time.perf_counter() - source.started_perf >= duration_sec
                    ):
                        reason = "DURATION_LIMIT"
                        break
                    item = source.read()
                    if item is None:
                        reason = "END_OF_VIDEO"
                        completion_verified = source.end_reason == END_OF_STREAM
                        break
                    index, elapsed, raw = item
                    if duration_sec is not None and elapsed >= duration_sec:
                        reason = "DURATION_LIMIT"
                        break
                    started = time.perf_counter()
                    analysis = processor.process(raw, index, elapsed)
                    preview_call("publish", analysis)
                    processed += 1
                    moving += analysis.decision.candidate
                    if captures is not None:
                        deliver(captures.process(analysis.packet, analysis.decision))
                    processing_seconds += time.perf_counter() - started
                    with self._lock:
                        self._progress = {
                            "run_id": run_id,
                            "status": "warming_up"
                            if analysis.decision.status == "WARMUP"
                            else "running",
                            "frames_processed": processed,
                            "candidate_frames": moving,
                            "bundle_count": captures.bundle_count if captures else 0,
                            "episode_count": captures.episode_count if captures else 0,
                            "event_active": captures.active if captures else False,
                            "processing_fps": processed / max(processing_seconds, 1e-9),
                            "preview_error": preview_error,
                        }
                    if window is not None:
                        metrics = {
                            "preview": not self.capture,
                            "event_capture": self.capture,
                            "event_active": captures.active if captures else False,
                            "event_count": captures.bundle_count if captures else 0,
                            "successful_capture_count": captures.image_count
                            if captures
                            else 0,
                            "processing_fps": processed / max(processing_seconds, 1e-9),
                            "tracking": self.config.tracking,
                            "frame_age_ms": max(
                                0,
                                (time.perf_counter() - source.started_perf - elapsed)
                                * 1000,
                            )
                            if self.source_kind == "webcam"
                            else 0,
                        }
                        view_packet, view_objects, view_motion, view_decision = (
                            analysis.packet,
                            analysis.objects,
                            analysis.motion,
                            analysis.decision,
                        )
                        if self.mirror:
                            view_packet, view_objects, view_motion, view_decision = (
                                _mirror_view(
                                    analysis.packet,
                                    analysis.objects,
                                    analysis.motion,
                                    analysis.decision,
                                )
                            )
                        display_frame = render_overlay(
                            view_packet,
                            view_objects,
                            view_motion,
                            view_decision,
                            CaptureResult.disabled(elapsed),
                            metrics,
                            overlay_mode=self.config.overlay_mode,
                        )
                        window.show(
                            display_frame,
                            render_mask_view(analysis.motion)
                            if self.config.show_mask
                            else None,
                        )
                        if window.poll_key() is not None or window.is_closed():
                            break
                    if fps is not None and self.config.pace == "realtime":
                        deadline = loop_started + (index + 1) / fps
                        while (
                            time.perf_counter() < deadline and not self._stop.is_set()
                        ):
                            if window is not None and (
                                window.poll_key() is not None or window.is_closed()
                            ):
                                self.stop()
                                break
                            self._stop.wait(
                                min(0.01, max(0, deadline - time.perf_counter()))
                            )
        except KeyboardInterrupt:
            reason = "USER_INTERRUPT"
            recover_completed()
        except Exception as exc:
            error, reason = exc, "ERROR"
            recover_completed()
        finally:
            # Persist observed tail on EOF/stop/error, without hiding the original error.
            if captures is not None:
                try:
                    deliver(captures.flush(reason), callbacks=error is None)
                except Exception as exc:
                    record_cleanup(exc, "Event tail publication")
            preview_call("close", closing=True)
            for resource in (source, window):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        record_cleanup(exc, "Resource cleanup")
            summary = RunSummary(
                run_id,
                source_name,
                reason,
                processed,
                moving,
                captures.bundle_count if captures else 0,
                captures.image_count if captures else 0,
                processed / max(processing_seconds, 1e-9),
                run_dir,
                completion_verified and reason == "END_OF_VIDEO",
                source.frames_skipped if self.source_kind == "webcam" and source else 0,
                captures.peak_buffer_bytes if captures else 0,
                captures.episode_count if captures else 0,
                preview_errors=preview_errors,
            )
            if logger is not None:
                try:
                    if error is not None:
                        logger.write(
                            "ERROR",
                            {"type": type(error).__name__, "message": str(error)},
                        )
                    logger.write("RUN_END", summary.to_dict())
                    logger.write_summary({"schema_version": 1, **summary.to_dict()})
                except Exception as exc:
                    if error is None:
                        error = exc
                finally:
                    try:
                        logger.close()
                    except Exception as exc:
                        if error is None:
                            error = exc
            if logger is not None:
                summary = replace(summary, diagnostic_errors=logger.failures)
            with self._lock:
                self._running = False
                self._progress = {
                    **self._progress,
                    "status": "error" if error else "stopped",
                    "end_reason": reason,
                    "bundle_count": summary.bundle_count,
                    "episode_count": summary.episode_count,
                    "frames_processed": summary.frames_processed,
                    "candidate_frames": summary.candidate_frames,
                    "event_active": False,
                    "processing_fps": summary.processing_fps,
                    "preview_error": preview_error,
                }
        if error is not None:
            raise error
        return summary


def _mirror_view(packet, objects, motion, decision):
    """Reflect pixels and boxes before drawing readable UI text; never mutate originals."""
    width = packet.analysis_size[0]

    def reflect(box):
        return Box(width - box.x2, box.y1, width - box.x1, box.y2)

    return (
        replace(packet, analysis_frame=cv2.flip(packet.analysis_frame, 1)),
        tuple(replace(obj, box=reflect(obj.box)) for obj in objects),
        replace(
            motion,
            valid_mask=cv2.flip(motion.valid_mask, 1),
            regions=tuple(
                replace(region, box=reflect(region.box)) for region in motion.regions
            ),
        ),
        replace(
            decision,
            persons=tuple(
                replace(obj, box=reflect(obj.box)) for obj in decision.objects
            ),
        ),
    )
