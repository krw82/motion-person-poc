"""Run the actual CLI on the prepared public samples and record every run."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test_data/public_test_results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="Run only the named prepared samples")
    options = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "test_data/public_test_manifest.json").read_text())
    index_path = OUTPUT / "runs.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    for sample in manifest:
        name = sample["name"]
        if options.only and name not in options.only:
            continue
        log_root = ROOT / "logs/public_tests" / name
        args = [sys.executable, "main.py", "--video", sample["path"],
            "--model", "models/yolo11n.pt", "--no-display", "--pace", "fast",
            "--debug-decisions", "--capture-dir", str(ROOT / "captures/public_tests" / name),
            "--log-dir", str(log_root)]
        if "min_person_motion_pixels_ref" in sample:
            args += ["--min-person-motion-pixels", str(sample["min_person_motion_pixels_ref"])]
        print("START", name, flush=True)
        with (OUTPUT / f"{name}_console.txt").open("w") as console:
            completed = subprocess.run(args, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        summaries = sorted(log_root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
        summary_path = summaries[-1] if summaries else None
        summary = json.loads(summary_path.read_text()) if summary_path else {}
        record = dict(sample=sample, exit_code=completed.returncode, command=args,
            summary_path=str(summary_path) if summary_path else None, summary=summary)
        index = [r for r in index if r["sample"]["name"] != name] + [record]
        (OUTPUT / "runs.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        print("END", name, "exit", completed.returncode, "frames", summary.get("frames_processed"),
            "captures", summary.get("successful_capture_count"), "fps", summary.get("processing_fps"), flush=True)
        if completed.returncode:
            print((OUTPUT / f"{name}_console.txt").read_text()[-3000:], flush=True)


if __name__ == "__main__":
    main()
