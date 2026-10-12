"""Photo selection and eviction policy, independent of encoding and disk I/O."""

from __future__ import annotations

from collections import Counter
from typing import Protocol, Sequence

ObjectKey = tuple[str, int | None]


class PhotoCandidate(Protocol):
    """Consumer-declared shape of a selectable frame sample."""

    frame_index: int
    timestamp: float
    matched: bool
    motion_starts: frozenset[ObjectKey]

    @property
    def moving_objects(self) -> frozenset[ObjectKey]: ...

    @property
    def score(self) -> tuple[int, float]: ...


def select_photos(
    frames: Sequence[PhotoCandidate],
    *,
    start: float,
    trigger: float,
    last_motion: float,
    max_images: int,
) -> list[tuple[PhotoCandidate, tuple[str, ...]]]:
    """Keep motion evidence even with two photos; larger budgets retain endpoints."""
    if not frames:
        raise ValueError("photo selection requires at least one frame")
    if (
        isinstance(max_images, bool)
        or not isinstance(max_images, int)
        or not 2 <= max_images <= 6
    ):
        raise ValueError("max_images must be within [2, 6]")
    roles: dict[int, set[str]] = {}

    def choose(sample: PhotoCandidate | None, role: str) -> None:
        if sample is not None:
            roles.setdefault(sample.frame_index, set()).add(role)

    evidence_start = max(start, trigger)
    active = [s for s in frames if s.matched and s.timestamp >= evidence_start]
    first = next(iter(active), frames[0]) if max_images == 2 else frames[0]
    choose(first, "before" if first.timestamp < trigger else "start")
    choose(frames[-1], "end")
    if max_images > 2:
        choose(next(iter(active), None), "start")

    uncovered = set().union(*(s.moving_objects for s in active))
    for sample in frames:
        if sample.frame_index in roles and sample.timestamp >= evidence_start:
            uncovered.difference_update(sample.moving_objects)
    while uncovered and len(roles) < max_images:
        remaining = [
            s
            for s in active
            if s.frame_index not in roles and s.moving_objects & uncovered
        ]
        sample = max(
            remaining,
            key=lambda s: (len(s.moving_objects & uncovered), s.score, -s.timestamp),
        )
        choose(sample, "object")
        uncovered.difference_update(sample.moving_objects)

    # Short resumes also deserve a photo even if the same object was seen earlier.
    for sample in active:
        if sample.motion_starts and (
            sample.frame_index in roles or len(roles) < max_images
        ):
            choose(sample, "motion_start")
    if len(roles) < max_images:
        choose(max(active, key=lambda s: s.score) if active else None, "peak")
    if len(roles) < max_images:
        choose(
            next(
                (s for s in frames if s.timestamp > last_motion and not s.matched), None
            ),
            "after",
        )
    while len(roles) < min(max_images, len(frames)):
        chosen_times = [s.timestamp for s in frames if s.frame_index in roles]
        remaining = [s for s in frames if s.frame_index not in roles]
        choose(
            max(
                remaining, key=lambda s: min(abs(s.timestamp - t) for t in chosen_times)
            ),
            "context",
        )
    return [
        (s, tuple(sorted(roles[s.frame_index])))
        for s in frames
        if s.frame_index in roles
    ]


def context_complete(
    timestamps: Sequence[float], *, start: float, end: float, interval: float
) -> bool:
    """Requested endpoints and sampled gaps must all be observed."""
    if start < 0 or not timestamps:
        return False
    return (
        timestamps[0] <= start + interval + 1e-9
        and timestamps[-1] >= end - interval - 1e-9
        and all(
            b - a <= 2 * interval + 1e-9 for a, b in zip(timestamps, timestamps[1:])
        )
    )


def eviction_candidate(frames: Sequence[PhotoCandidate]) -> PhotoCandidate | None:
    """Prefer replaceable context; protect the first motion and latest photo."""
    first_motion = next((s for s in frames if s.matched), None)
    candidates = [(i, s) for i, s in enumerate(frames[:-1]) if s is not first_motion]
    if not candidates:
        return None
    counts = Counter(key for sample in frames for key in sample.moving_objects)
    middle = (len(frames) - 1) / 2
    _, sample = min(
        candidates,
        key=lambda item: (
            sum(counts[key] == 1 for key in item[1].moving_objects),
            bool(item[1].motion_starts),
            abs(item[0] - middle),
            item[0],
        ),
    )
    return sample
