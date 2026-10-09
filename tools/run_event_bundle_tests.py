"""Run real YOLO/MOG2 bundles and audit each raw JPEG against its input frame.

Usage: .venv/bin/python tools/run_event_bundle_tests.py [--videos /path/to.mp4 ...]
Results are local/ignored; user videos are never copied into the package/repo.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "test_data/event_bundle_results"


def audit(video, run_dir):
    manifests = [json.loads(p.read_text()) for p in sorted(run_dir.glob("events/*/*/event.json"))]
    images = [im for bundle in manifests for im in bundle["images"]]
    by_index = {}
    for image in images:
        by_index.setdefault(image["frame_index"], []).append(image)
    cap = cv2.VideoCapture(str(video))
    index = 0
    verified = 0
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            for image in by_index.get(index, []):
                ok, expected = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY,95])
                assert ok and Path(image["path"]).read_bytes() == expected.tobytes(), image
                decoded = cv2.imdecode(np.frombuffer(expected,np.uint8),cv2.IMREAD_COLOR)
                assert (decoded.shape[1],decoded.shape[0]) == tuple(image["original_size"])
                verified += 1
            index += 1
    finally:
        cap.release()
    assert verified == len(images)
    for bundle in manifests:
        assert 0 <= bundle["duration_sec"] <= 10 + 1e-9
        assert 1 <= len(bundle["images"]) <= 6
        times = [im["timestamp_sec"] for im in bundle["images"]]
        assert times == sorted(set(times))
        assert all(bundle["start_sec"]-1e-9 <= t <= bundle["end_sec"]+1e-9 for t in times)
        assert len(list(Path(bundle["manifest_path"]).parent.glob("*.jpg"))) == len(bundle["images"])
    summary = json.loads((run_dir / "summary.json").read_text())
    log_entries = [json.loads(line) for line in (run_dir/"events.jsonl").read_text().splitlines()]
    logged_bundles = [entry for entry in log_entries if entry["event_type"] == "EVENT_COMPLETED"]
    assert len(logged_bundles) == len(manifests)
    assert [(b["event_id"],b["part_index"]) for b in logged_bundles] == [(b["event_id"],b["part_index"]) for b in manifests]
    for logged,bundle in zip(logged_bundles,manifests):
        assert all(logged[key] == value for key,value in bundle.items())
    assert summary["frames_processed"] == index
    assert summary["input_completion_verified"]
    assert summary["event_count"] == len(manifests) and summary["image_count"] == verified
    assert summary["peak_event_buffer_bytes"] <= 64*1024*1024
    return {"decoded_input_frames":index, "raw_jpegs_verified":verified, "jsonl_manifest_match":True,
            "max_bundle_duration_sec":max((b["duration_sec"] for b in manifests),default=0),
            "episode_count":len({b["event_id"] for b in manifests}),
            "end_reasons":{reason:sum(b["end_reason"] == reason for b in manifests)
                           for reason in sorted({b["end_reason"] for b in manifests})},
            "complete_context_bundles":sum(b["context_complete"] for b in manifests),
            "after_context_photos":sum(im["phase"] == "after" for im in images)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--videos", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    plans = [(name,ROOT/f"videos/public_tests/{name}.mp4") for name in
             ("empty_control","frozen_person_control","lighting_control","cdnet_single_pass","caviar_stop_resume","dog_walker_tokyo")]
    plans += [(p.stem.replace("-","_"),p.resolve()) for p in args.videos]
    RESULTS.mkdir(parents=True,exist_ok=True)
    environment = {**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    index_path = RESULTS/"runs.json"
    records = json.loads(index_path.read_text()) if index_path.exists() else []
    code_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT/"motion_person").glob("*.py"))}
    for name,video in plans:
        if args.only and name not in args.only:
            continue
        output = ROOT/"captures/event_verified"/name
        command = [sys.executable,"cli.py","video",str(video),"--events","--objects","person","dog",
                   "--model",str(ROOT/"models/yolo11n.pt"),"--output",str(output),"--no-display","--fast"]
        print(f"START {name}",flush=True)
        completed = subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True)
        (RESULTS/f"{name}_console.txt").write_text(completed.stdout+completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{name}: {completed.stdout}\n{completed.stderr}")
        run_dir = max(output.iterdir(),key=lambda p:p.stat().st_mtime)
        summary = json.loads((run_dir/"summary.json").read_text())
        checks = audit(video,run_dir)
        if name in ("empty_control","lighting_control"):
            assert summary["event_count"] == 0, (name,summary)
        if name == "frozen_person_control":
            # This fixture is move(5-10s), freeze(10-22s), move(22-27s), not all-static.
            bundles = [json.loads(p.read_text()) for p in sorted(run_dir.glob("events/*/*/event.json"))]
            starts = [b["trigger_sec"] for b in bundles if b["part_index"] == 1]
            assert len(starts) == 2 and 5 <= starts[0] < 10 and 22 <= starts[1] < 27, starts
            first_episode = [b for b in bundles if b["event_id"] == bundles[0]["event_id"]]
            checks["freeze_motion_tail_sec"] = max(b["last_motion_sec"] for b in first_episode)-10
            checks["freeze_bundle_close_delay_sec"] = max(b["end_sec"] for b in first_episode)-10
        records = [r for r in records if r["name"] != name]
        records.append({"name":name,"source_video":str(video),"source_sha256":hashlib.sha256(video.read_bytes()).hexdigest(),
                        "command":command,"source_code_sha256":code_hashes,
                        "run_dir":str(run_dir),"summary":summary,"audit":checks})
        (RESULTS/"runs.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n")
        print(f"END {name}: {summary['frames_processed']} frames, {summary['event_count']} bundles, "
              f"{summary['image_count']} photos, {summary['processing_fps']:.1f} FPS, checks passed",flush=True)


if __name__ == "__main__":
    main()
