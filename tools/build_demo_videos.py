"""Run bundled public inputs through main.py, audit raw captures, publish H.264 demos.

Requires requirements-dev.txt and models/yolo11n.pt. Local logs, JPGs and
intermediate MP4 files remain under ignored test_data/example_build/.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_tracking_tests as audit

WORK = ROOT / "test_data/example_build"
RESULTS = ROOT / "examples/results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_info(path: Path) -> dict:
    """Decode every frame; verify the count independently with FFmpeg."""
    capture = cv2.VideoCapture(str(path))
    frames = 0
    size = None
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_size = list(frame.shape[1::-1])
            if size is not None and current_size != size:
                raise RuntimeError(f"Resolution changed in {path}")
            size = current_size
            frames += 1
    finally:
        capture.release()
    result = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1",
    ], capture_output=True, text=True, check=True, timeout=120)
    counts = re.findall(r"^frame=(\d+)$", result.stdout, re.MULTILINE)
    if not frames or not counts or frames != int(counts[-1]) or result.stderr.strip():
        raise RuntimeError(f"Independent video verification failed: {path}")
    return {"path": str(path.relative_to(ROOT)), "frames": frames, "fps": fps,
            "size": size, "duration_sec": round(frames / fps, 3),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    audit.RESULTS = WORK
    sources = json.loads((ROOT / "examples/sources.json").read_text())
    report = {
        "runtime_code_sha256": {name: sha256(ROOT / name) for name in (
            "main.py", "config.py", "object_detector.py", "motion_detector.py",
            "event_detector.py", "capture_manager.py", "overlay_renderer.py",
        )},
        "model_sha256": sha256(ROOT / "models/yolo11n.pt"),
        "versions": {name: importlib.metadata.version(name) for name in (
            "opencv-python", "numpy", "ultralytics", "lap", "imageio-ffmpeg",
        )},
        "settings": {"confidence": 0.4, "min_object_motion_pixels_ref": 1000,
                     "warmup_sec": 2, "cooldown_sec": 0.5,
                     "capture_scope": "object", "tracker": "bytetrack",
                     "tracking_profile": "stable", "track_min_hits": 3,
                     "analysis_width": 960, "imgsz": 640, "device": "cpu"},
        "caveat": "Functional examples, not an accuracy benchmark. Track IDs are not physical individual counts. The dog clip has camera motion.",
        "runs": [],
    }
    for sample in sources:
        video = ROOT / sample["input"]["path"]
        actual = video_info(video)
        if actual != sample["input"]:
            raise RuntimeError(f"Bundled input changed: {video}")
        log_root = WORK / sample["name"] / "logs"
        before = set(log_root.glob("*/summary.json"))
        completed = subprocess.run([
            sys.executable, str(ROOT / "main.py"), "--video", str(video),
            "--model", str(ROOT / "models/yolo11n.pt"),
            "--classes", *sample["classes"], "--track", "--capture-scope", "object",
            "--cooldown-sec", "0.5", "--confidence", "0.4",
            "--min-object-motion-pixels", "1000", "--no-display", "--pace", "fast",
            "--debug-decisions", "--capture-dir", str(WORK / sample["name"] / "captures"),
            "--log-dir", str(log_root),
        ], cwd=ROOT, check=True, timeout=300)
        summaries = set(log_root.glob("*/summary.json")) - before
        if len(summaries) != 1:
            raise RuntimeError("Cannot identify the new run")
        log_dir = summaries.pop().parent
        result = audit.evaluate({
            "name": sample["name"], "log_dir": str(log_dir),
            "source_video": str(video), "source_sha256": actual["sha256"],
            "exit_code": completed.returncode, "render": True,
            "compare_overlays": sample["compare_overlays"],
            "overlay_mode": sample.get("overlay_mode", "full"),
            "source_code_sha256": report["runtime_code_sha256"],
            "require_current_code": True,
        })
        if not all(result["checks"].values()):
            raise RuntimeError(f"Demo audit failed: {result['checks']}")
        output = ROOT / sample["output"]
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
            "-i", result["diagnostic_video"], "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ], check=True, timeout=120)
        rendered = video_info(output)
        if rendered["frames"] != actual["frames"] or rendered["fps"] != actual["fps"]:
            raise RuntimeError("Output dropped frames or changed the timebase")
        result["checks"]["h264_all_frames_verified_independently"] = True
        if sample["compare_overlays"]:
            result["checks"]["none_overlay_matches_analysis_before_encoding"] = True
        row = {key: result[key] for key in (
            "name", "frames", "captures", "classes", "checks", "class_present_frames",
            "maximum_simultaneous", "track_ids_by_class", "confirmed_ids_by_class",
            "minimum_trigger_interval_by_id", "capture_times_sec",
        )}
        row["output"] = rendered
        row["display"] = "full / objects / none comparison" if sample["compare_overlays"] else sample["overlay_mode"]
        preview = ROOT / sample["preview"]
        player = cv2.VideoCapture(str(output))
        try:
            player.set(cv2.CAP_PROP_POS_FRAMES, sample["preview_frame"])
            ok, frame = player.read()
            if not ok or not cv2.imwrite(str(preview), frame):
                raise RuntimeError("Cannot create the demo preview")
        finally:
            player.release()
        row["preview"] = {"path": sample["preview"], "frame_index": sample["preview_frame"],
                          "bytes": preview.stat().st_size, "sha256": sha256(preview)}
        report["runs"].append(row)
        print(json.dumps({"name": row["name"], "frames": row["frames"],
                          "captures": row["captures"], "checks_pass": True,
                          "output": sample["output"]}), flush=True)
    (RESULTS / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
