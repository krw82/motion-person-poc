"""실행 내 추적 객체 상태. ID는 실제 개인 신원이나 재입장 식별을 보장하지 않는다."""
from __future__ import annotations

from contracts import CaptureResult, FrameDecision


class ObjectStateStore:
    """objects[track_id]로 종류, 현재 전경 판정, 위치와 저장 이력을 조회한다."""

    def __init__(self) -> None:
        self.objects: dict[int, dict] = {}

    def update(self, decision: FrameDecision, capture: CaptureResult) -> list[dict]:
        first_seen = []
        for state in self.objects.values():
            state["visible"] = False
            state["motion_qualified"] = False
        for evidence in decision.objects:
            track_id = evidence.track_id
            if track_id is None:
                continue
            if track_id not in self.objects:
                self.objects[track_id] = {
                    "track_id": track_id,
                    "tracker_id": evidence.tracker_id,
                    "class_id": evidence.class_id,
                    "class_name": evidence.class_name,
                    "label": evidence.label,
                    "first_seen_sec": decision.video_time_sec,
                    "observed_frames": 0,
                    "qualified_frames": 0,
                    "capture_count": 0,
                    "last_capture_sec": None,
                    "class_change_count": 0,
                }
                first_seen.append({"track_id": track_id, "class_name": evidence.class_name,
                                   "label": evidence.label})
            state = self.objects[track_id]
            if state["class_id"] != evidence.class_id:
                state["class_change_count"] += 1
            state.update({
                "class_id": evidence.class_id,
                "class_name": evidence.class_name,
                "label": evidence.label,
                "last_seen_sec": decision.video_time_sec,
                "last_seen_frame": decision.frame_index,
                "visible": True,
                "confidence": round(evidence.confidence, 4),
                "box_xyxy": evidence.box.to_list(),
                "motion_pixels": evidence.motion_pixels,
                "motion_ratio": round(evidence.motion_ratio, 6),
                "motion_qualified": evidence.qualifies,
                "observation_hits": evidence.observation_hits,
                "confirmed": evidence.confirmed,
                "warming_up": decision.status == "WARMUP",
                "rejection_reasons": list(evidence.rejection_reasons),
            })
            state["observed_frames"] += 1
            state["qualified_frames"] += int(evidence.qualifies)
            if capture.status == "SAVED" and track_id in capture.saved_track_ids:
                state["capture_count"] += 1
                state["last_capture_sec"] = decision.video_time_sec
        return first_seen

    def snapshot(self) -> dict[str, dict]:
        # JSON object keys are strings; callers of this module use integer track IDs.
        return {str(i): dict(state) for i, state in sorted(self.objects.items())}
