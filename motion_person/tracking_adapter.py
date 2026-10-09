"""Class-separated Ultralytics trackers, sharing one detector inference.

Each child advances on every processed frame, including empty detections.
Output detection indices are restored to the full inference result so the
Ultralytics callback can attach the correct boxes and native ReID features.
"""
from __future__ import annotations

import numpy as np


class ClassSeparatedTracker:
    def __init__(self, tracker_factory, args, class_ids):
        self.args = args
        self.trackers = {int(c): tracker_factory(args=args) for c in class_ids}
        self.frame_id = 0

    @property
    def tracked_stracks(self):
        return [track for child in self.trackers.values() for track in child.tracked_stracks]

    @property
    def removed_stracks_frame(self):
        return [track for child in self.trackers.values() for track in child.removed_stracks_frame]

    def update(self, results, img=None, feats=None, **kwargs):
        self.frame_id += 1
        rows = []
        for class_id, tracker in self.trackers.items():
            mask = np.asarray(results.cls) == class_id
            indices = np.flatnonzero(mask)
            if feats is None or not len(feats):
                selected_features = None
            elif hasattr(feats, "shape"):
                selected_features = feats[indices]
            else:
                # Child ByteTrack slices this container again with boolean masks.
                # Keep an indexable array/tensor, not a Python list of vectors.
                vectors = [f.cpu().numpy() if hasattr(f, "cpu") else f for f in feats]
                selected_features = np.asarray(vectors)[indices]
            output = tracker.update(results[mask], img, feats=selected_features, **kwargs)
            if len(output):
                output = output.copy()
                local_indices = output[:, -1].astype(int)
                if np.any(local_indices < 0) or np.any(local_indices >= len(indices)):
                    raise ValueError("tracker detection indices are outside their class subset")
                if not np.all(output[:, -2].astype(int) == class_id):
                    raise ValueError("class-separated tracker changed class")
                output[:, -1] = indices[local_indices]
                rows.append(output)
        if not rows:
            return np.empty((0, 8), dtype=np.float32)
        merged = np.concatenate(rows)
        return merged[np.argsort(merged[:, -1], kind="stable")]

    def reset(self):
        for tracker in self.trackers.values():
            tracker.reset()
        self.frame_id = 0
