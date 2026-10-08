"""Audit logged identities/cooldowns/raw captures and render diagnostic videos."""
from collections import Counter, defaultdict
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from contracts import (Box, CaptureResult, FrameDecision, ObjectDetection, ObjectMotionEvidence)
from frame_processor import make_packet
from motion_detector import MotionDetector
from overlay_renderer import render_overlay

RESULTS = ROOT / "test_data/tracking_results"


def render_comparison(packet, objects, motion, decision, capture, metrics):
    """Compare display modes on one decision; crop sidebars, retain capture status below."""
    height, width = packet.analysis_frame.shape[:2]
    canvas = np.zeros((height + 68, width * 3, 3), dtype=np.uint8)
    for column, mode in enumerate(("full", "objects", "none")):
        rendered = render_overlay(packet, objects, motion, decision, capture, metrics,
                                  overlay_mode=mode)
        if mode == "none" and not np.array_equal(rendered, packet.analysis_frame):
            raise RuntimeError("Hidden overlay changed the analysis image")
        canvas[36:36 + height, column * width:(column + 1) * width] = rendered[:height, :width]
        cv2.putText(canvas, f"--overlay {mode}", (column * width + 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    status = (f"TIME {packet.video_time_sec:.2f}s | STATE {decision.status} | "
              f"CAPTURE {capture.status} | TOTAL {metrics['successful_capture_count']}")
    cv2.putText(canvas, status, (10, height + 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def evaluate(run):
    folder = Path(run["log_dir"])
    summary = json.loads((folder / "summary.json").read_text())
    settings = json.loads((folder / "run_config.json").read_text())
    values = {f.name: settings.get(f.name, f.default) for f in fields(Config)}
    for key in ["video_path", "model_path", "capture_dir", "log_dir"]: values[key] = Path(values[key])
    values["target_classes"] = tuple(values["target_classes"])
    if values["analysis_roi"] is not None: values["analysis_roi"] = tuple(values["analysis_roi"])
    config = Config(**values)
    records = [json.loads(s) for s in (folder / "events.jsonl").read_text().splitlines()]
    decisions = {r["frame_index"]: r for r in records if r["event_type"] == "FRAME_DECISION"}
    captures = {r["frame_index"]: r for r in records if r["event_type"] == "CAPTURE_SAVED"}
    seen_classes = defaultdict(set); trigger_counts = Counter(); times = defaultdict(list)
    class_frames = Counter(); observed_class_frames = Counter(); max_simultaneous = Counter()
    for r in decisions.values():
        classes = Counter(o["class_name"] for o in r["objects"])
        class_frames.update(classes.keys()); observed_class_frames.update(classes)
        for c, count in classes.items(): max_simultaneous[c] = max(max_simultaneous[c], count)
        for o in r["objects"]:
            if o["track_id"] is not None: seen_classes[o["track_id"]].add(o["class_name"])
    for r in captures.values():
        for i in r["saved_track_ids"]:
            trigger_counts[i] += 1; times[i].append(r["video_time_sec"])
    physical = list((ROOT / config.capture_dir / summary["run_id"]).glob("*.jpg"))
    minimum_gaps = {str(i): min(b-a for a,b in zip(t,t[1:]))
                    for i,t in times.items() if len(t)>1}
    global_times = [r["video_time_sec"] for r in captures.values()]
    checks = {
        "exit_zero": run["exit_code"] == 0,
        "all_input_processed": summary["input_completion_verified"] and
            summary["frames_read"] == summary["frames_processed"] == len(decisions),
        "capture_count_matches_files": len(physical) == len(captures) == summary["successful_capture_count"],
        "no_warmup_capture": all(r["video_time_sec"] >= config.warmup_sec for r in captures.values()),
        "one_jpg_per_frame": len({r["frame_index"] for r in captures.values()}) == len(physical),
        "selected_classes_only": set(class_frames).issubset(config.target_classes),
        "legacy_person_fields_contain_only_people": all(
            all(o["class_name"] == "person" for o in r.get("persons", r.get("matched_persons", [])))
            for r in [*decisions.values(), *captures.values()]),
        "object_class_stable": all(len(c)==1 for c in seen_classes.values()),
        "tracked_ids_valid": all(isinstance(o["track_id"],int) and o["track_id"]>0
                                 for r in decisions.values() for o in r["objects"]) if config.tracking else True,
        "cooldown_respected": all(g >= config.capture_cooldown_sec-1e-9 for g in minimum_gaps.values())
            if config.capture_scope == "object" else
            all(b-a >= config.capture_cooldown_sec-1e-9 for a,b in zip(global_times,global_times[1:])),
        "trigger_ids_are_qualified": all(set(r["saved_track_ids"]).issubset(
            {o["track_id"] for o in r["matched_objects"]}) for r in captures.values()),
        "summary_object_counts_match_triggers": all(
            state["capture_count"] == trigger_counts[int(i)]
            for i,state in summary.get("objects",{}).items()),
        "source_unchanged": hashlib.sha256(Path(run["source_video"]).read_bytes()).hexdigest() == run["source_sha256"],
        "pending_objects_do_not_trigger": all(
            all(o.get("confirmed", True) and o.get("observation_hits", 1) >= config.track_min_hits
                for o in r["matched_objects"] if o["track_id"] in r["saved_track_ids"])
            for r in captures.values()) if "track_min_hits" in settings else True,
        "native_ids_do_not_cross_classes": not summary.get("class_identity_splits", 0)
            if settings.get("tracking_profile") == "stable" else True,
    }
    raw_matches = True; decoded = 0; rendered = []; writer = None
    last_trigger_time = {}; running_capture_count = 0
    source = cv2.VideoCapture(run["source_video"])
    # Render the standard user runs. Alternative profiles remain in the numeric comparison.
    render = run.get("render", run["name"].startswith("tiktok_") and "_imgsz" not in run["name"])
    motion_detector = MotionDetector(config) if render else None
    overlay_path = RESULTS / f"{run['name']}_annotated.mp4"
    preview_index = (210 if "767523" in run["name"] else 126)
    try:
        while True:
            ok, raw = source.read()
            if not ok: break
            if decoded in captures:
                _, encoded = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
                raw_matches &= encoded.tobytes() == Path(captures[decoded]["capture_path"]).read_bytes()
            if render:
                record = decisions[decoded]
                packet = make_packet(summary["run_id"], decoded, record["video_time_sec"], raw,
                                     config.analysis_width, config.analysis_roi)
                motion = motion_detector.detect(packet.analysis_frame, packet.video_time_sec)
                objects = tuple(ObjectDetection(
                    o["detection_index"], Box(*o["box_xyxy"]), o["confidence"], o["class_id"],
                    o["class_name"], o["track_id"], o["tracker_id"], o.get("observation_hits",1), o.get("confirmed",True)
                ) for o in record["objects"])
                evidences = tuple(ObjectMotionEvidence(
                    o["detection_index"], Box(*o["box_xyxy"]), o["confidence"], o["motion_pixels"],
                    o["motion_ratio"], o["qualifies"], tuple(o["rejection_reasons"]),
                    o["class_id"], o["class_name"], o["track_id"], o["tracker_id"], o.get("observation_hits",1), o.get("confirmed",True)
                ) for o in record["objects"])
                decision = FrameDecision(decoded, packet.video_time_sec, record["status"], record["candidate"], evidences)
                saved = captures.get(decoded)
                eligible = {o.track_id for o in evidences if o.qualifies and o.track_id is not None}
                remaining = min((max(0, config.capture_cooldown_sec -
                    (packet.video_time_sec - last_trigger_time[i])) if i in last_trigger_time else 0
                    for i in eligible), default=0)
                if saved:
                    running_capture_count = saved["capture_sequence"]
                    for i in saved["saved_track_ids"]: last_trigger_time[i] = packet.video_time_sec
                result = CaptureResult(record["capture_status"], None, packet.video_time_sec, remaining,
                                       saved["capture_sequence"] if saved else None,
                                       tuple(record["saved_track_ids"]))
                metrics = {"tracking": True, "successful_capture_count": running_capture_count}
                if run.get("compare_overlays"):
                    display = render_comparison(packet, objects, motion, decision, result, metrics)
                else:
                    display = render_overlay(packet, objects, motion, decision, result, metrics,
                                             overlay_mode=run.get("overlay_mode", "full"))
                h, w = display.shape[:2]
                if h % 2 or w % 2:
                    display = cv2.copyMakeBorder(display, 0, h % 2, 0, w % 2,
                                                cv2.BORDER_CONSTANT, value=(0,0,0))
                if writer is None:
                    h,w = display.shape[:2]
                    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             source.get(cv2.CAP_PROP_FPS), (w,h))
                    if not writer.isOpened(): raise RuntimeError("Cannot write diagnostic video")
                writer.write(display)
                if decoded == preview_index:
                    path = RESULTS / f"{run['name']}_preview.png"
                    cv2.imwrite(str(path), display); rendered.append(str(path))
            decoded += 1
    finally:
        source.release()
        if writer is not None: writer.release()
    checks["captures_are_raw_frames"] = raw_matches
    checks["decoded_frames_match_log"] = decoded == len(decisions)
    if render:
        check_video = cv2.VideoCapture(str(overlay_path))
        checks["diagnostic_video_frames_match"] = int(check_video.get(cv2.CAP_PROP_FRAME_COUNT)) == decoded
        check_video.release()
    if settings.get("tracking_profile") == "stable" and config.tracking:
        checks["effective_tracker_yaml_recorded"] = (folder / "tracker.yaml").is_file()
    checks["source_code_matches_current"] = all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        for name, digest in run.get("source_code_sha256", {}).items()) if run.get(
            "require_current_code", RESULTS.name == "tracking_improvements") else True
    return {
        "name": run["name"], "log_dir": str(folder),
        "capture_dir": str((ROOT / config.capture_dir / summary["run_id"]).resolve()),
        "frames": summary["frames_processed"], "captures": len(captures),
        "classes": config.target_classes, "imgsz": config.yolo_imgsz,
        "fps": summary["processing_fps"], "checks": checks,
        "class_present_frames": dict(class_frames), "class_observations": dict(observed_class_frames),
        "maximum_simultaneous": dict(max_simultaneous),
        "track_ids_by_class": dict(Counter(o["class_name"] for o in summary.get("objects",{}).values())),
        "confirmed_ids_by_class": dict(Counter(o["class_name"] for o in summary.get("objects",{}).values()
                                              if o.get("confirmed",True))),
        "analysis_roi": config.analysis_roi,
        "minimum_trigger_interval_by_id": minimum_gaps,
        "capture_times_sec": [round(t,3) for t in global_times],
        "objects": summary.get("objects",{}),
        "tracking_profile": config.tracking_profile,
        "track_buffer": config.track_buffer,
        "track_min_hits": config.track_min_hits,
        "track_reid": config.track_reid,
        "diagnostic_video": str(overlay_path) if render else None,
        "previews": rendered,
        "caveat": "Tracking IDs are not counts of physical individuals; no independent person/dog box or identity ground truth. User requested evaluation within fixed camera scenes, ignoring edit cuts.",
    }


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    RESULTS = args.results_dir.resolve()
    runs = json.loads((RESULTS / "runs.json").read_text())
    results = []
    for run in runs:
        result = evaluate(run); results.append(result)
        print(json.dumps({"name":result["name"],"frames":result["frames"],"captures":result["captures"],
                          "checks_pass":all(result["checks"].values()),
                          "track_ids_by_class":result["track_ids_by_class"]}), flush=True)
    (RESULTS / "evaluation.json").write_text(json.dumps(results,indent=2)+"\n")
    if not all(all(r["checks"].values()) for r in results): raise SystemExit(1)


if __name__ == "__main__": main()
