"""Fetch a short CC BY-SA dog-walking clip and convert all frames to silent CFR MP4."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import urllib.request

import cv2

ROOT = Path(__file__).resolve().parents[1]
URL = "https://upload.wikimedia.org/wikipedia/commons/0/0c/Pro_dog_walker_-_Tokyo_-_2024_Nov_1.webm"
PAGE = "https://commons.wikimedia.org/wiki/File:Pro_dog_walker_-_Tokyo_-_2024_Nov_1.webm"


def main():
    source = ROOT / "test_data/public_sources/dog_walker_tokyo.webm"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "MotionPersonPoC/1.0 (local video test)"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read(20_000_001)
        if len(content) > 20_000_000:
            raise RuntimeError("Unexpectedly large source")
        source.write_bytes(content)
    if hashlib.sha1(source.read_bytes()).hexdigest() != "269e3b92330e96dc90d06d8215ebd36be5ff175d":
        raise RuntimeError("Source checksum changed; inspect before use")
    output = ROOT / "videos/public_tests/dog_walker_tokyo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = capture.read()
    if not ok or fps <= 0:
        raise RuntimeError("Cannot decode source video")
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened(): raise RuntimeError("Cannot open MP4 writer")
    count = 0
    try:
        while ok:
            writer.write(frame); count += 1
            ok, frame = capture.read()
    finally:
        capture.release(); writer.release()
    verified_frames = reported
    if count != reported:
        # WebM metadata can round duration*FPS up. Verify actual decoded count independently.
        import imageio_ffmpeg
        result = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(source),
            "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1",
        ], capture_output=True, text=True, timeout=30, check=True)
        if result.stderr.strip(): raise RuntimeError(result.stderr)
        verified_frames = int(re.findall(r"^frame=(\d+)$", result.stdout, re.MULTILINE)[-1])
        if count != verified_frames:
            raise RuntimeError(f"Independent decoded frame count mismatch: {count}/{verified_frames}")
    metadata = {
        "source_url": URL, "source_page": PAGE, "author": "Nesnad",
        "license": "CC BY-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "source_path": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "video_path": str(output.relative_to(ROOT)), "frames": count, "fps": fps,
        "reported_source_frames": reported, "verified_source_frames": verified_frames,
        "size": [width, height], "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "changes": "All decoded frames retained at full dimensions; silent OpenCV mp4v CFR conversion at reported average FPS.",
        "caveat": "Camera motion and occlusion make this a multi-class tracking smoke test, not MOG2 accuracy ground truth.",
    }
    (ROOT / "test_data/tracking_sample_manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
