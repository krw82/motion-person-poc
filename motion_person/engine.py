"""Blocking high-level API; host applications own cloud jobs and alerts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Callable
import uuid

import cv2

from .config import Config, config_to_dict, validate_analysis_values
from .contracts import Box, ConfigError, VideoNotFoundError
from .event_capture import EventBundle, EventCaptureEngine, EventConfig
from .overlay_renderer import DisplayWindow, render_mask_view, render_overlay
from .processor import FrameProcessor
from .run_logger import RunLogger
from .video_source import END_OF_STREAM, VideoSource
from .webcam_source import WebcamSource


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

    def to_dict(self):
        data = asdict(self)
        data["run_dir"] = str(self.run_dir) if self.run_dir is not None else None
        return data


class MotionEngine:
    """video(...).start() or webcam(..., capture=True).start(on_event=...).

    start() blocks until EOF/stop/duration/interrupt and releases all resources.
    Callbacks run on this processing thread *after* a complete bundle is saved;
    use a short callback to enqueue external analysis rather than waiting for it.
    An instance can be reused after a run, but cannot start two concurrent runs.
    """

    def __init__(self, source_kind, source_value, *, config, events, output_dir,
                 capture, mirror=False):
        if not isinstance(capture, bool) or not isinstance(mirror, bool):
            raise ConfigError("capture and mirror must be boolean")
        if events is not None and not isinstance(events, EventConfig):
            raise ConfigError("events must be an EventConfig")
        validate_analysis_values(config)
        if config.show_mask and not config.display:
            raise ConfigError("show_mask requires display")
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
        base = config or replace(Config(), tracking=True, person_confidence=.4,
                                 min_person_motion_pixels_ref=1000, capture_cooldown_sec=.5)
        return replace(base, target_classes=tuple(objects) if objects is not None else base.target_classes,
                       model_path=Path(model_path).expanduser() if model_path is not None else base.model_path,
                       display=display, overlay_mode=overlay, pace=pace)

    @classmethod
    def video(cls, path, *, objects=None, model_path=None, output_dir="outputs",
              capture=True, display=True, overlay="full", events=None, config=None, pace="realtime"):
        settings = cls._config(config, objects, model_path, display, overlay, pace)
        path = Path(path).expanduser()
        return cls("video", path, config=replace(settings, video_path=path), events=events,
                   output_dir=output_dir, capture=capture)

    @classmethod
    def webcam(cls, camera=0, *, objects=None, model_path=None, output_dir="outputs",
               capture=False, display=True, overlay="full", events=None, config=None, mirror=True):
        # Validate the camera index without opening a device.
        WebcamSource(camera)
        settings = cls._config(config, objects, model_path, display, overlay, "realtime")
        return cls("webcam", camera, config=settings, events=events,
                   output_dir=output_dir, capture=capture, mirror=mirror)

    webCam = webcam

    def stop(self):
        """Request exit at the next input/display boundary; safe from a callback/thread."""
        self._stop.set()

    def start(self, *, on_event: Callable[[EventBundle], None] | None = None,
              duration_sec: float | None = None) -> RunSummary:
        if on_event is not None and not callable(on_event):
            raise ConfigError("on_event must be callable")
        if on_event is not None and not self.capture:
            raise ConfigError("on_event requires capture=True")
        if duration_sec is not None and (not isinstance(duration_sec, (int, float))
                or isinstance(duration_sec, bool) or not math.isfinite(duration_sec) or duration_sec <= 0):
            raise ConfigError("duration_sec must be a finite number > 0")
        if self.source_kind == "video" and not self.source_value.is_file():
            raise VideoNotFoundError(f"video file not found: {self.source_value}")
        with self._lock:
            if self._running:
                raise ConfigError("this MotionEngine is already running")
            self._running = True
            self._stop.clear()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
        run_dir = (self.output_dir / run_id).resolve() if self.capture else None
        source_name = f"webcam:{self.source_value}" if self.source_kind == "webcam" else str(self.source_value.resolve())
        source = window = logger = captures = None
        processed = moving = 0
        processing_seconds = 0.0
        reason = "ERROR"
        error = None
        completion_verified = False

        def deliver(bundles, *, callbacks=True):
            for bundle in bundles:
                if logger is not None:
                    logger.write("EVENT_COMPLETED", bundle.to_dict())
                if on_event is not None and callbacks:
                    on_event(bundle)

        try:
            with tempfile.TemporaryDirectory(prefix="motion-person-tracker-") as temporary:
                processor = FrameProcessor(self.config, run_id)
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
                    input_info = {"camera": self.source_value, "width": width, "height": height,
                                  "timeline_source": "monotonic_receipt_time"}
                if run_dir is not None:
                    logger = RunLogger(run_dir.parent, run_id)
                    logger.start({**config_to_dict(self.config), "source_kind": self.source_kind,
                                  "source": source_name, "event_config": asdict(self.event_config),
                                  "capture_mode": "event_bundles"}, input_info)
                    if tracker_path.is_file():
                        shutil.copyfile(tracker_path, run_dir / "tracker.yaml")
                    captures = EventCaptureEngine(run_dir / "events", run_id, source_name,
                                                  config=self.event_config,
                                                  jpeg_quality=self.config.jpeg_quality,
                                                  analysis_roi=self.config.analysis_roi)
                if self.config.display:
                    window = DisplayWindow(show_mask=self.config.show_mask)
                loop_started = time.perf_counter()
                reason = "USER_STOP"
                while not self._stop.is_set():
                    if (self.source_kind == "webcam" and duration_sec is not None
                            and time.perf_counter() - source.started_perf >= duration_sec):
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
                    packet, objects, motion, decision, disabled = processor.process(raw, index, elapsed)
                    processed += 1
                    moving += decision.candidate
                    if captures is not None:
                        deliver(captures.process(packet, decision))
                    processing_seconds += time.perf_counter() - started
                    if window is not None:
                        metrics = {"preview": not self.capture, "event_capture": self.capture,
                                   "event_active": captures.active if captures else False,
                                   "event_count": captures.bundle_count if captures else 0,
                                   "successful_capture_count": captures.image_count if captures else 0,
                                   "processing_fps": processed/max(processing_seconds, 1e-9),
                                   "tracking": self.config.tracking,
                                   "frame_age_ms": max(0, (time.perf_counter()-source.started_perf-elapsed)*1000)
                                   if self.source_kind == "webcam" else 0}
                        view_packet, view_objects, view_motion, view_decision = packet, objects, motion, decision
                        if self.mirror:
                            view_packet, view_objects, view_motion, view_decision = _mirror_view(packet, objects, motion, decision)
                        display_frame = render_overlay(view_packet, view_objects, view_motion, view_decision, disabled,
                                                       metrics, overlay_mode=self.config.overlay_mode)
                        window.show(display_frame, render_mask_view(motion) if self.config.show_mask else None)
                        if window.poll_key() is not None or window.is_closed():
                            break
                    if fps is not None and self.config.pace == "realtime":
                        deadline = loop_started + (index + 1) / fps
                        while time.perf_counter() < deadline and not self._stop.is_set():
                            if window is not None and (window.poll_key() is not None or window.is_closed()):
                                self.stop()
                                break
                            self._stop.wait(min(.01, max(0, deadline-time.perf_counter())))
        except KeyboardInterrupt:
            reason = "USER_INTERRUPT"
        except Exception as exc:
            error, reason = exc, "ERROR"
        finally:
            # Persist observed tail on EOF/stop/error, without hiding the original error.
            if captures is not None:
                try:
                    deliver(captures.flush(reason), callbacks=error is None)
                except Exception as exc:
                    if error is None:
                        error, reason = exc, "ERROR"
            for resource in (source, window):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        if error is None:
                            error, reason = exc, "ERROR"
            summary = RunSummary(run_id, source_name, reason, processed, moving,
                                 captures.bundle_count if captures else 0,
                                 captures.image_count if captures else 0,
                                 processed/max(processing_seconds, 1e-9), run_dir,
                                 completion_verified and reason == "END_OF_VIDEO",
                                 source.frames_skipped if self.source_kind == "webcam" and source else 0,
                                 captures.peak_buffer_bytes if captures else 0)
            if logger is not None:
                try:
                    if error is not None:
                        logger.write("ERROR", {"type": type(error).__name__, "message": str(error)})
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
            with self._lock:
                self._running = False
        if error is not None:
            raise error
        return summary


def _mirror_view(packet, objects, motion, decision):
    """Reflect pixels and boxes before drawing readable UI text; never mutate originals."""
    width = packet.analysis_size[0]
    def reflect(box):
        return Box(width-box.x2, box.y1, width-box.x1, box.y2)
    return (replace(packet, analysis_frame=cv2.flip(packet.analysis_frame, 1)),
            tuple(replace(obj, box=reflect(obj.box)) for obj in objects),
            replace(motion, valid_mask=cv2.flip(motion.valid_mask, 1),
                    regions=tuple(replace(region, box=reflect(region.box)) for region in motion.regions)),
            replace(decision, persons=tuple(replace(obj, box=reflect(obj.box)) for obj in decision.objects)))
