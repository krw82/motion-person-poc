"""Bounded, timestamp-based photo bundles shared by file and live inputs.

YOLO/MOG2 decide when to open or extend an episode. Once open, sampling also
keeps quiet/empty frames. No cloud requests or notification policy live here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path


from motion_person.domain.contracts import (
    CaptureSaveError,
    ConfigError,
    ContractError,
    FrameDecision,
    FramePacket,
)
from motion_person.domain.geometry import box_to_original
from motion_person.application.ports import BundleStore
from motion_person.composition import event_storage
from motion_person.domain.photo_selection import (
    ObjectKey,
    context_complete,
    eviction_candidate,
    select_photos,
)


from motion_person.domain.events import (
    EventBundle,
    EventConfig,
    EventImage,
    EventObject,
)


@dataclass(frozen=True)
class _Sample:
    frame_index: int
    timestamp: float
    jpeg: bytes
    size: tuple[int, int]
    objects: tuple[EventObject, ...]
    matched: bool
    motion_starts: frozenset[ObjectKey] = frozenset()

    @property
    def moving_objects(self):
        return frozenset((o.class_name, o.track_id) for o in self.objects if o.moving)

    @property
    def score(self):
        return max(
            ((o.motion_pixels, o.confidence) for o in self.objects if o.moving),
            default=(0, 0.0),
        )


@dataclass
class _Part:
    event_id: str
    index: int
    start: float
    trigger: float
    last_motion: float
    pre_complete: bool
    frames: list[_Sample] = field(default_factory=list)
    dropped: int = 0
    gaps: int = 0
    sample_count: int = 0
    observed_moving_objects: set = field(default_factory=set)


class EventCaptureEngine:
    """One camera/analysis region per instance. Track IDs never split an episode.

    process() returns only completed bundles, after directory publication.
    10 seconds limits the complete window, including pre/post context. A hard
    split keeps the event ID and increments part_index. flush() closes remaining
    observed data without fabricating future post-capture frames.
    """

    def __init__(
        self,
        output_dir: Path,
        run_id: str,
        camera_id: str,
        *,
        config: EventConfig | None = None,
        jpeg_quality: int = 95,
        analysis_roi=None,
        store: BundleStore | None = None,
        encoder=None,
    ):
        self.config = config or EventConfig()
        self.store, self.encoder = event_storage(store, encoder)
        if (
            not isinstance(run_id, str)
            or not run_id
            or Path(run_id).name != run_id
            or run_id in (".", "..")
        ):
            raise ContractError("run_id must be a nonempty filename component")
        if (
            not isinstance(jpeg_quality, int)
            or isinstance(jpeg_quality, bool)
            or not 1 <= jpeg_quality <= 100
        ):
            raise ConfigError("jpeg_quality must be an integer within [1, 100]")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.run_id, self.camera_id = run_id, str(camera_id)
        self.jpeg_quality, self.analysis_roi = jpeg_quality, analysis_roi
        self._ring: deque[_Sample] = deque()
        self._part: _Part | None = None
        self._continuation: tuple[str, int, float, float, float] | None = None
        self._last_index = -1
        self._last_time = -1.0
        self._last_sample_time = -math.inf
        self._event_sequence = 0
        self._moving_keys: frozenset[ObjectKey] = frozenset()
        self._pending: list[EventBundle] = []
        self.bundle_count = self.image_count = 0
        self.peak_buffer_bytes = 0

    @property
    def active(self):
        return self._part is not None or self._continuation is not None

    @property
    def episode_count(self):
        return self._event_sequence

    @property
    def buffer_bytes(self):
        samples = list(self._ring) + (self._part.frames if self._part else [])
        return sum(len(s.jpeg) for s in {s.frame_index: s for s in samples}.values())

    def process(
        self, packet: FramePacket, decision: FrameDecision
    ) -> tuple[EventBundle, ...]:
        try:
            bundles = self._process(packet, decision)
        except (Exception, KeyboardInterrupt):
            # Failed encoding/publication must not leave partly ingested future
            # frames to be published by an error flush with an older end time.
            self.discard()
            raise
        self._mark_delivered(bundles)
        return bundles

    def discard(self):
        """Release unfinished data after an input/capture failure; keep saved bundles."""
        self._part = self._continuation = None
        self._ring.clear()
        # Discard unfinished evidence while retaining the monotonic input timeline.
        self._moving_keys = frozenset()

    def drain_completed(self) -> tuple[EventBundle, ...]:
        """Bundles published to disk whose return was cut off by a later error."""
        pending, self._pending = tuple(self._pending), []
        return pending

    def _mark_delivered(self, bundles):
        if bundles:
            delivered = {id(bundle) for bundle in bundles}
            self._pending = [b for b in self._pending if id(b) not in delivered]

    def _process(
        self, packet: FramePacket, decision: FrameDecision
    ) -> tuple[EventBundle, ...]:
        now = packet.video_time_sec
        if (
            not isinstance(packet.frame_index, int)
            or isinstance(packet.frame_index, bool)
            or packet.frame_index <= self._last_index
            or not isinstance(now, (int, float))
            or isinstance(now, bool)
            or not math.isfinite(now)
            or now < 0
            or now <= self._last_time
            or decision.frame_index != packet.frame_index
            or decision.video_time_sec != now
        ):
            raise ContractError(
                "event frames and decisions must agree and increase in index/time"
            )
        now = float(now)
        moving_keys = frozenset(
            (e.class_name, e.track_id) for e in decision.objects if e.qualifies
        )
        starts = moving_keys - self._moving_keys
        completed = []
        sample = None
        new_part = False
        # Resolve elapsed deadlines before this frame can prolong the old part.
        if self._part is not None:
            part = self._part
            quiet_end = part.last_motion + self.config.quiet_sec
            hard_end = part.start + self.config.max_duration_sec
            boundary = min(quiet_end, hard_end)
            if now + 1e-9 >= boundary:
                reason = "QUIET" if quiet_end <= hard_end + 1e-9 else "MAX_DURATION"
                if abs(now - boundary) <= 1e-9 and not decision.candidate:
                    sample = self._encode(packet, decision, starts)
                    self._append(sample)
                completed.append(
                    self._finish(
                        boundary,
                        reason,
                        post_complete=reason == "QUIET"
                        and abs(now - boundary) <= self.config.sample_interval_sec,
                    )
                )
                if reason == "MAX_DURATION" and now < quiet_end - 1e-9:
                    self._continuation = (
                        part.event_id,
                        part.index + 1,
                        boundary,
                        part.trigger,
                        part.last_motion,
                    )

        if self._part is None and self._continuation is not None:
            event_id, index, start, trigger, last_motion = self._continuation
            self._continuation = None
            if now < last_motion + self.config.quiet_sec - 1e-9:
                # No repeated pre-roll in continuation parts, so every window <= 10s.
                self._part = _Part(
                    event_id,
                    index,
                    now if now - start >= self.config.max_duration_sec else start,
                    trigger,
                    last_motion,
                    False,
                )
                new_part = True
        if self._part is None and decision.candidate:
            self._event_sequence += 1
            event_id = f"{self.run_id}_e{self._event_sequence:06d}"
            wanted = max(0.0, now - self.config.pre_capture_sec)
            before = [s for s in self._ring if wanted - 1e-9 <= s.timestamp < now]
            start = before[0].timestamp if before else now
            pre_complete = self.config.pre_capture_sec == 0 or context_complete(
                [s.timestamp for s in before] + [now],
                start=now - self.config.pre_capture_sec,
                end=now,
                interval=self.config.sample_interval_sec,
            )
            self._part = _Part(
                event_id,
                1,
                start,
                now,
                now,
                pre_complete,
                frames=before.copy(),
                sample_count=len(before),
            )
            self._part.gaps = sum(
                b.timestamp - a.timestamp > self.config.sample_interval_sec * 2 + 1e-9
                for a, b in zip(before, before[1:])
            )
            new_part = True

        if self._part is not None and decision.candidate:
            self._part.last_motion = now
        first_motion = self._part is not None and not any(
            s.matched and s.timestamp >= self._part.trigger for s in self._part.frames
        )
        due = now - self._last_sample_time + 1e-9 >= self.config.sample_interval_sec
        if due or new_part or (decision.candidate and first_motion) or starts:
            sample = sample or self._encode(packet, decision, starts)
        if sample is not None:
            self._last_sample_time = now
            self._ring.append(sample)
            if self._part is not None:
                self._append(sample)
        while (
            self._ring
            and self._ring[0].timestamp < now - self.config.pre_capture_sec - 1e-9
        ):
            self._ring.popleft()
        self._bound_memory()
        if self._part is not None:
            self._part.observed_moving_objects.update(moving_keys)
        self._last_index, self._last_time, self._moving_keys = (
            packet.frame_index,
            now,
            moving_keys,
        )
        return tuple(completed)

    def flush(self, reason="END_OF_VIDEO") -> tuple[EventBundle, ...]:
        if reason not in (
            "END_OF_VIDEO",
            "USER_STOP",
            "USER_INTERRUPT",
            "ERROR",
            "DURATION_LIMIT",
        ):
            raise ContractError("unsupported event flush reason")
        completed = ()
        try:
            if self._part is not None and self._part.frames:
                completed = (
                    self._finish(
                        min(
                            self._last_time,
                            self._part.start + self.config.max_duration_sec,
                        ),
                        reason,
                        post_complete=False,
                    ),
                )
        finally:
            self.discard()
        self._mark_delivered(completed)
        return completed

    def _encode(self, packet, decision, starts=frozenset()):
        jpeg = self.encoder(packet.raw_frame, self.jpeg_quality)
        if len(jpeg) > self.config.max_buffer_bytes:
            raise CaptureSaveError(
                "one JPEG exceeds max_buffer_bytes; increase the event buffer budget"
            )
        objects = tuple(
            EventObject(
                e.class_name,
                int(e.track_id) if e.track_id is not None else None,
                float(e.confidence),
                tuple(box_to_original(e.box, packet).to_list()),
                bool(e.qualifies),
                int(e.motion_pixels),
                float(e.motion_ratio),
            )
            for e in decision.objects
        )
        return _Sample(
            packet.frame_index,
            float(packet.video_time_sec),
            jpeg,
            packet.original_size,
            objects,
            bool(decision.candidate),
            frozenset(starts),
        )

    def _append(self, sample):
        part = self._part
        if part.frames and part.frames[-1].frame_index == sample.frame_index:
            return
        if (
            part.frames
            and sample.timestamp - part.frames[-1].timestamp
            > self.config.sample_interval_sec * 2 + 1e-9
        ):
            part.gaps += 1
        part.frames.append(sample)
        part.sample_count += 1
        self._bound_memory()

    def _bound_memory(self):
        while self.buffer_bytes > self.config.max_buffer_bytes:
            if self._ring:
                self._ring.popleft()
                continue
            part = self._part
            if part is None or len(part.frames) <= 1:
                raise CaptureSaveError(
                    "event buffer budget cannot hold required evidence"
                )
            victim = eviction_candidate(part.frames)
            if victim is None:
                raise CaptureSaveError(
                    "event buffer budget cannot hold trigger and latest photo"
                )
            part.frames.remove(victim)
            part.dropped += 1
            part.pre_complete = False
        self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.buffer_bytes)

    def _finish(self, end, reason, *, post_complete):
        part = self._part
        if part is None or not part.frames:
            raise ContractError("cannot finish an empty event part")
        post_complete = bool(
            post_complete
            and context_complete(
                [s.timestamp for s in part.frames if s.timestamp >= part.last_motion],
                start=part.last_motion,
                end=end,
                interval=self.config.sample_interval_sec,
            )
        )
        selected = select_photos(
            part.frames,
            start=part.start,
            trigger=part.trigger,
            last_motion=part.last_motion,
            max_images=self.config.max_images,
        )
        observed = set(part.observed_moving_objects)
        evidence_start = max(part.start, part.trigger)
        covered = {
            key
            for sample, _ in selected
            if sample.timestamp >= evidence_start
            for key in sample.moving_objects
        }
        omitted = tuple(
            sorted(
                observed - covered,
                key=lambda key: (
                    key[0],
                    key[1] is None,
                    key[1] if key[1] is not None else -1,
                ),
            )
        )
        final_dir = self.output_dir / part.event_id / f"part_{part.index:04d}"
        images = []
        for i, (sample, roles) in enumerate(selected, 1):
            phase = (
                "before"
                if sample.timestamp < part.trigger
                else "after"
                if sample.timestamp > part.last_motion
                else "start"
                if "start" in roles
                else "during"
            )
            images.append(
                EventImage(
                    final_dir / f"{i:02d}_{phase}_f{sample.frame_index:08d}.jpg",
                    sample.frame_index,
                    sample.timestamp,
                    phase,
                    roles,
                    sample.size,
                    sample.objects,
                )
            )
        complete = bool(
            part.pre_complete and post_complete and part.dropped == 0 and part.gaps == 0
        )
        bundle = EventBundle(
            self.run_id,
            part.event_id,
            part.index,
            self.camera_id,
            part.start,
            end,
            part.trigger,
            part.last_motion,
            reason,
            reason != "MAX_DURATION",
            complete,
            bool(part.pre_complete),
            bool(post_complete),
            part.dropped,
            part.gaps,
            part.sample_count,
            tuple(images),
            self.analysis_roi,
            final_dir / "event.json",
            omitted_moving_objects=omitted,
        )
        self.store.publish(bundle, [sample.jpeg for sample, _ in selected])
        self.bundle_count += 1
        self.image_count += len(images)
        self._part = None
        self._pending.append(bundle)
        return bundle
