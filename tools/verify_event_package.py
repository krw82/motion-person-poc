"""Verify a built wheel outside the checkout, then run its real video API."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from run_event_bundle_tests import audit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "test_data/event_bundle_results"

PROGRAM = '''
import sys, types, json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.modules["config"] = types.ModuleType("unrelated_host_config")
from motion_person import Config, MotionEngine
import motion_person
assert Path(motion_person.__file__).is_relative_to(Path(sys.argv[1]))
settings = json.loads(sys.argv[5])
config = Config(tracking=True, person_confidence=.4, min_person_motion_pixels_ref=1000,
                yolo_imgsz=settings["imgsz"],
                analysis_roi=tuple(settings["roi"]) if settings.get("roi") else None)
received=[]
def completed(bundle):
    assert bundle.manifest_path.is_file()
    assert all(path.is_file() for path in bundle.image_paths)
    received.append(bundle)
summary = MotionEngine.video(sys.argv[2], model_path=sys.argv[3], output_dir=sys.argv[4],
                             objects=("person", "dog"), config=config,
                             display=False, pace="fast").start(on_event=completed)
assert summary.event_count == len(received)
print(json.dumps(summary.to_dict()))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel",type=Path,default=ROOT/"dist/motion_person-0.2.0-py3-none-any.whl")
    parser.add_argument("--model",type=Path,default=ROOT/"models/yolo11n.pt")
    parser.add_argument("--roi-video",type=Path)
    parser.add_argument("--roi",type=float,nargs=4,default=(0,.34,1,.66),
                        help="ROI for the supplied 5396 sample; change for other inputs")
    args=parser.parse_args()
    wheel=args.wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        files=archive.namelist()
        assert all(name.startswith(("motion_person/","motion_person-")) for name in files)
        assert not any(name.endswith((".pt",".mp4")) or "/tests/" in name for name in files)
        for name in files:
            if name.startswith("motion_person/") and name.endswith(".py"):
                assert archive.read(name) == (ROOT/name).read_bytes(), name
    record={"wheel":str(wheel),"wheel_sha256":hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "isolated_installed_import":True,"unrelated_host_config":True,
            "implementation_matches_checkout":True,
            "weights_and_videos_excluded":True,"runs":[]}
    plans=[("installed_caviar",ROOT/"examples/videos/caviar_stop_resume.mp4",{"imgsz":640})]
    if args.roi_video:
        plans.append(("installed_5396_roi960",args.roi_video.resolve(),{"imgsz":960,"roi":args.roi}))
    with tempfile.TemporaryDirectory(prefix="motion-package-verification-") as folder:
        target=Path(folder)/"installed"
        installed=subprocess.run([sys.executable,"-m","pip","install","--no-deps","--target",str(target),str(wheel)],
                                 capture_output=True,text=True)
        assert installed.returncode == 0,installed.stdout+installed.stderr
        help_code='import sys; sys.path.insert(0,sys.argv[1]); from motion_person.cli import main; main(["--help"])'
        help_result=subprocess.run([sys.executable,"-I","-c",help_code,str(target)],cwd=folder,capture_output=True,text=True)
        assert help_result.returncode == 0 and "사용법:" in help_result.stdout
        record["installed_cli_help"]=True
        for name,video,settings in plans:
            output=ROOT/"captures/event_verified"/name
            print("START",name,flush=True)
            command=[sys.executable,"-I","-c",PROGRAM,str(target),str(video),str(args.model.resolve()),
                     str(output),json.dumps(settings)]
            result=subprocess.run(command,cwd=folder,capture_output=True,text=True,
                                  env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
            assert result.returncode == 0,result.stdout+result.stderr
            (RESULTS/f"{name}_console.txt").write_text(result.stdout+result.stderr)
            summary=json.loads(result.stdout.splitlines()[-1])
            checks=audit(video,Path(summary["run_dir"]))
            assert summary["event_count"] > 0
            record["runs"].append({"name":name,"source_video":str(video),"settings":settings,
                                   "summary":summary,"audit":checks})
            (RESULTS/"package_verification.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
            print(f"END {name}: {summary['event_count']} bundles, {summary['image_count']} photos, "
                  f"{summary['processing_fps']:.1f} FPS, all JPEGs matched input frames",flush=True)


if __name__ == "__main__":
    main()
