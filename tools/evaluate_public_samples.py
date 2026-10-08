"""Evaluate scene movement and capture contracts independently of the detector.

CAVIAR: walking/movement = positive; active or conflicting top hypotheses = ambiguous;
inactive/no annotated object = negative. Unannotated static people may remain in view.
CDnet pixel scores use temporal ROI, spatial ROI, and labels 0, 50, 255 only.
"""
from pathlib import Path
import json
import math
import sys
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import Config
from motion_detector import MotionDetector


def read_events(record):
    path = Path(record["summary_path"]).parent / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def caviar_labels(sample):
    frames = ET.parse(ROOT / "test_data/public_sources" / sample["annotation"]).getroot().findall("frame")
    labels = {}
    boxes = {}
    for frame in frames:
        states = []
        gt_boxes = []
        for obj in frame.findall("objectlist/object"):
            hypotheses = obj.findall("hypothesislist/hypothesis")
            if hypotheses:
                best = max(float(h.get("evaluation", "0")) for h in hypotheses)
                moves = {h.findtext("movement", "") for h in hypotheses
                    if float(h.get("evaluation", "0")) == best}
                if moves <= {"walking", "movement"} and moves:
                    states.append("moving")
                elif moves == {"inactive"}:
                    states.append("stationary")
                else:
                    states.append("ambiguous")
            else:
                states.append("ambiguous")
            b = obj.find("box")
            if b is not None:
                x, y, w, h = (float(b.get(k)) for k in ["xc", "yc", "w", "h"])
                gt_boxes.append([max(0, x-w/2), max(0, y-h/2),
                    min(sample["size"][0], x+w/2), min(sample["size"][1], y+h/2)])
        label = "moving" if "moving" in states else (
            "ambiguous" if "ambiguous" in states else "negative")
        labels[int(frame.get("number"))] = label
        boxes[int(frame.get("number"))] = gt_boxes
    return labels, boxes


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    denom = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / denom if denom > 0 else 0.0


def segments(indices):
    result = []
    for index in sorted(indices):
        if not result or result[-1][-1] != index-1:
            result.append([index])
        else:
            result[-1].append(index)
    return result


def evaluate_scene(record, events):
    sample = record["sample"]
    decisions = {e["frame_index"]: e for e in events if e["event_type"] == "FRAME_DECISION"}
    saves = [e for e in events if e["event_type"] == "CAPTURE_SAVED"]
    checks = {
        "exit_zero": record["exit_code"] == 0,
        "all_frames_processed": record["summary"].get("frames_processed") == sample["frames"],
        "input_completion_verified": record["summary"].get("input_completion_verified") is True,
        "debug_decision_per_frame": len(decisions) == sample["frames"],
        "no_capture_in_warmup": all(e["video_time_sec"] >= 2.0 for e in saves),
        "cooldown_respected": all(b["video_time_sec"] - a["video_time_sec"] >= 3.0-1e-9
            for a, b in zip(saves, saves[1:])),
        "one_capture_per_frame": len({e["frame_index"] for e in saves}) == len(saves),
        "capture_count_matches_files": (
            len(saves) == record["summary"].get("successful_capture_count")
            == len(list((ROOT / "captures/public_tests" / sample["name"] /
                Path(record["summary_path"]).parent.name).glob("*.jpg")))
        ),
    }
    source = cv2.VideoCapture(str(ROOT / sample["path"]))
    jpeg_errors = []
    for e in saves:
        image_path = Path(e["capture_path"])
        decoded = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or [decoded.shape[1], decoded.shape[0]] != sample["size"]:
            jpeg_errors.append("dimension or decode mismatch")
            continue
        source.set(cv2.CAP_PROP_POS_FRAMES, e["frame_index"])
        ok, raw = source.read()
        encoded_ok, encoded = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 95]) if ok else (False, None)
        if not encoded_ok or image_path.read_bytes() != encoded.tobytes():
            jpeg_errors.append("saved image differs from re-encoding the source frame")
    source.release()
    checks["raw_capture_bytes_match"] = not jpeg_errors
    output = {"name": sample["name"], "contract_checks": checks,
        "captures": len(saves), "capture_times_sec": [e["video_time_sec"] for e in saves],
        "summary": record["summary"], "jpeg_errors": jpeg_errors}
    if sample["dataset"] == "CAVIAR" or "control_segments" in sample:
        if sample["dataset"] == "CAVIAR":
            labels, gt_boxes = caviar_labels(sample)
        else:
            labels = {i:s["label"] for s in sample["control_segments"]
                for i in range(s["start"],s["end"])}
            gt_boxes = {}
        valid = {i for i, e in decisions.items() if not e["warming_up"] and labels.get(i) != "ambiguous"}
        positive = valid & {i for i, label in labels.items() if label == "moving"}
        negative = valid - positive
        pos_segments = segments(positive)
        candidates = {i for i, e in decisions.items() if e["candidate"]}
        saved = {e["frame_index"] for e in saves}
        delays = [(min(set(segment)&candidates)-segment[0])/sample["fps"]
            for segment in pos_segments if set(segment)&candidates]
        matched_gt = gt_count = 0
        for i in valid:
            predicted = [p["box_xyxy"] for p in decisions[i]["persons"]]
            groundtruth = gt_boxes.get(i, [])
            gt_count += len(groundtruth)
            pairs = sorted((iou(a,b), ai, bi) for ai,a in enumerate(groundtruth) for bi,b in enumerate(predicted))
            used_a, used_b = set(), set()
            for score, ai, bi in reversed(pairs):
                if score >= .5 and ai not in used_a and bi not in used_b:
                    matched_gt += 1; used_a.add(ai); used_b.add(bi)
        output["movement_evaluation"] = {
            "positive_segments": len(pos_segments),
            "detected_segments": sum(bool(set(s)&candidates) for s in pos_segments),
            "saved_segments": sum(bool(set(s)&saved) for s in pos_segments),
            "positive_frames": len(positive), "negative_frames": len(negative),
            "excluded_frames": sample["frames"]-len(valid),
            "candidate_positive_frame_rate": len(candidates&positive)/len(positive) if positive else None,
            "candidate_negative_frame_rate": len(candidates&negative)/len(negative) if negative else None,
            "false_captures": len(saved&negative),
            "false_captures_per_min": len(saved&negative)/(len(negative)/sample["fps"]/60) if negative else None,
            "first_detection_delay_sec": delays,
            "annotated_person_boxes": gt_count, "matched_person_boxes_iou_0_5": matched_gt,
            "annotated_box_recall": matched_gt/gt_count if gt_count else None,
            "caveat": ("Partial CAVIAR annotations: box recall only; no full-scene precision claim."
                if sample["dataset"] == "CAVIAR" else sample["caveat"]),
        }
        output["cached_area_comparison"] = []
        for threshold_ref in [1000,2000,3000]:
            threshold = math.ceil(threshold_ref*sample["size"][0]*sample["size"][1]/(960*540))
            hypothetical = {i for i,e in decisions.items() if not e["warming_up"] and any(
                p["motion_pixels"] >= threshold and p["motion_ratio"] >= .03 for p in e["persons"])}
            output["cached_area_comparison"].append({"reference_pixels":threshold_ref,
                "effective_pixels":threshold, "candidate_frames":len(hypothetical),
                "detected_segments":sum(bool(set(s)&hypothetical) for s in pos_segments),
                "negative_candidates":len(hypothetical&negative),
                "caveat":"Counterfactual area filter on cached YOLO/MOG2 evidence; not an independent validation run."})
        if "control_segments" in sample:
            frozen = sample["control_segments"][2]
            frozen_indices = set(range(frozen["start"], frozen["end"]))
            output["frozen_person_evaluation"] = {
                "start_sec": frozen["start"]/sample["fps"],
                "end_sec": frozen["end"]/sample["fps"],
                "candidate_frames": len(candidates & frozen_indices),
                "capture_times_sec": sorted(i/sample["fps"] for i in saved&frozen_indices),
                "last_candidate_after_stop_sec": (
                    (max(candidates&frozen_indices)-frozen["start"])/sample["fps"]
                    if candidates&frozen_indices else None),
                "caveat": sample["caveat"],
            }
    return output


def evaluate_cdnet_pixels(sample):
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    detector = MotionDetector(Config())
    capture = cv2.VideoCapture(str(ROOT / sample["path"]))
    start, end = sample["temporal_roi"]
    with zipfile.ZipFile(ROOT / "test_data/public_sources" / sample["source_archive"]) as stream:
        name = sample["name"].removeprefix("cdnet_")
        roi = cv2.imdecode(np.frombuffer(stream.read(name+"/ROI.bmp"),np.uint8),cv2.IMREAD_GRAYSCALE) > 0
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok: break
            prediction = detector.detect(frame,index/sample["fps"]).valid_mask == 255
            official_index = index+1
            if start <= official_index <= end:
                gt = cv2.imdecode(np.frombuffer(stream.read(f"{name}/groundtruth/gt{official_index:06d}.png"),np.uint8),cv2.IMREAD_GRAYSCALE)
                valid = roi & np.isin(gt,[0,50,255])
                positive = gt == 255
                counts["tp"] += int(np.count_nonzero(valid & positive & prediction))
                counts["fp"] += int(np.count_nonzero(valid & ~positive & prediction))
                counts["fn"] += int(np.count_nonzero(valid & positive & ~prediction))
                counts["tn"] += int(np.count_nonzero(valid & ~positive & ~prediction))
            index += 1
    capture.release()
    tp,fp,fn = (counts[k] for k in ["tp","fp","fn"])
    return {**counts, "precision":tp/(tp+fp) if tp+fp else None,
        "recall":tp/(tp+fn) if tp+fn else None, "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        "iou":tp/(tp+fp+fn) if tp+fp+fn else None, "temporal_roi":sample["temporal_roi"],
        "caveat":"MOG2 postprocessed mask vs all foreground; does not measure person movement accuracy. MP4 re-encoding is lossy."}


def main():
    root = ROOT / "test_data/public_test_results"
    runs = json.loads((root / "runs.json").read_text())
    results = []
    for record in runs:
        result = evaluate_scene(record,read_events(record))
        if record["sample"]["dataset"] == "CDnet2014":
            result["mog2_pixel_evaluation"] = evaluate_cdnet_pixels(record["sample"])
        results.append(result)
        print(result["name"], "captures", result["captures"], "contracts", result["contract_checks"],
            "movement", result.get("movement_evaluation"), "pixels", result.get("mog2_pixel_evaluation"),flush=True)
    (root / "evaluation.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
