"""Run the legacy/tracking smoke tests and optional user videos with recorded commands."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "test_data/tracking_results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="*", type=Path, default=[])
    parser.add_argument("--compare-imgsz", type=int, default=None,
                        help="Repeat the last user video with a different YOLO input size")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    common = ["--classes", "person", "dog", "--track", "--capture-scope", "object",
              "--cooldown-sec", "0.5"]
    profile = [*common, "--confidence", "0.4", "--min-object-motion-pixels", "1000"]
    plans = [
        ("single_pass", ROOT / "videos/public_tests/cdnet_single_pass.mp4", common),
        ("single_pass_legacy", ROOT / "videos/public_tests/cdnet_single_pass.mp4", ["--cooldown-sec", "0.5"]),
        ("person_dog", ROOT / "videos/public_tests/dog_walker_tokyo.mp4", profile),
    ]
    for video in args.videos:
        plans.append((video.stem.replace("-", "_"), video.resolve(), profile))
    if args.compare_imgsz is not None and args.videos:
        video = args.videos[-1].resolve()
        plans.append((video.stem.replace("-", "_") + f"_imgsz{args.compare_imgsz}",
                      video, [*profile, "--imgsz", str(args.compare_imgsz)]))
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    runs = []
    for name, video, options in plans:
        command = [sys.executable, "main.py", "--video", str(video), "--model", "models/yolo11n.pt",
                   *options, "--no-display", "--pace", "fast", "--debug-decisions",
                   "--capture-dir", f"captures/tracking_verified/{name}",
                   "--log-dir", f"logs/tracking_verified/{name}"]
        fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}
        completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        output = completed.stdout + completed.stderr
        (RESULTS / f"{name}_console.txt").write_text(output)
        match = re.search(r"^log dir: (.+)$", completed.stdout, re.MULTILINE)
        run = {"name": name, "command": command, "exit_code": completed.returncode,
               "source_video": str(video), "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
               "source_code_sha256": fingerprints,
               "environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
               "versions": {p: importlib.metadata.version(p) for p in ["ultralytics", "lap", "numpy", "opencv-python"]},
               "model_sha256": hashlib.sha256((ROOT / "models/yolo11n.pt").read_bytes()).hexdigest()}
        if match:
            run["log_dir"] = str((ROOT / match.group(1)).resolve())
        runs.append(run)
        (RESULTS / "runs.json").write_text(json.dumps(runs, indent=2)+"\n")
        print(f"{name}: exit={completed.returncode} " + completed.stdout.splitlines()[-1], flush=True)
        if completed.returncode: raise RuntimeError(output)


if __name__ == "__main__": main()
