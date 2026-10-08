"""Compare baseline, class isolation/confirmation, buffers, native ReID and ROI.

All inputs and YOLO weights are unchanged. Runs are sequential, fingerprints
are recorded before execution, and older reports/captures remain intact.
"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', nargs=2, type=Path, required=True)
    parser.add_argument('--only', help='Re-run one named case and retain the other recorded cases')
    args = parser.parse_args()
    results = ROOT / 'test_data/tracking_improvements'
    results.mkdir(parents=True, exist_ok=True)
    common = ['--classes','person','dog','--track','--capture-scope','object','--cooldown-sec','.5']
    profile = [*common, '--confidence','.4','--min-object-motion-pixels','1000']
    baseline = ['--tracking-profile','baseline']
    buffer90 = ['--track-buffer','90']
    reid = [*buffer90, '--tracker','botsort','--reid']
    # Both fixed camera images in this supplied portrait file occupy this ROI.
    # Rounded outward to include the entire camera image; no automatic scene handling.
    roi = ['--roi','0','.285','1','.755']
    one,two = [p.resolve() for p in args.videos]
    stem1,stem2 = one.stem.replace('-','_'),two.stem.replace('-','_')
    plans = [
        ('single_pass_legacy', ROOT/'videos/public_tests/cdnet_single_pass.mp4', ['--cooldown-sec','.5'],False),
        ('single_pass_stable', ROOT/'videos/public_tests/cdnet_single_pass.mp4', common,False),
        (stem1+'_baseline',one,[*profile,*baseline],False),
        (stem1+'_stable',one,profile,True),
        (stem1+'_buffer90',one,[*profile,*buffer90],False),
        (stem1+'_reid',one,[*profile,*reid],True),
        (stem2+'_baseline',two,[*profile,*baseline],False),
        (stem2+'_stable',two,profile,True),
        (stem2+'_roi',two,[*profile,*roi,*buffer90],True),
        (stem2+'_roi_reid',two,[*profile,*roi,*reid],True),
        (stem2+'_imgsz960',two,[*profile,'--imgsz','960'],False),
    ]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    runs=[]
    if args.only:
        selected=[p for p in plans if p[0] == args.only]
        if not selected: parser.error('Unknown case: '+args.only)
        plans=selected
        path=results/'runs.json'
        if path.exists(): runs=[r for r in json.loads(path.read_text()) if r['name'] != args.only]
    for name,video,options,render in plans:
        command=[sys.executable,'main.py','--video',str(video),'--model','models/yolo11n.pt',
                 *options,'--no-display','--pace','fast','--debug-decisions',
                 '--capture-dir',f'captures/tracking_improvements/{name}',
                 '--log-dir',f'logs/tracking_improvements/{name}']
        fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
        completed=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        output=completed.stdout+completed.stderr
        (results/f'{name}_console.txt').write_text(output)
        match=re.search(r'^log dir: (.+)$',completed.stdout,re.MULTILINE)
        run={'name':name,'command':command,'render':render,'require_current_code':True,'exit_code':completed.returncode,
             'source_video':str(video),'source_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
             'source_code_sha256':fingerprints,'environment':{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
             'versions':{p:importlib.metadata.version(p) for p in ['ultralytics','lap','numpy','opencv-python']},
             'model_sha256':hashlib.sha256((ROOT/'models/yolo11n.pt').read_bytes()).hexdigest()}
        if match:run['log_dir']=str((ROOT/match.group(1)).resolve())
        runs.append(run)
        (results/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
        print(name,completed.returncode,completed.stdout.splitlines()[-1] if completed.stdout else output,flush=True)
        if completed.returncode:raise RuntimeError(output)


if __name__=='__main__':main()
