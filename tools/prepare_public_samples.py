"""Convert official frame archives to CFR MP4 without dropping frames.

Run from the project root after downloading files listed in public_sources/manifest.json.
CDnet's 25 FPS is an assigned test timebase; it is not a claim about original capture FPS.
"""
from pathlib import Path
import hashlib
import json
import tarfile
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "test_data/public_sources"
VIDEOS = ROOT / "videos/public_tests"
FPS = 25.0


def decode(data):
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode source frame")
    return frame


def write_video(name, frames, provenance):
    path = VIDEOS / f"{name}.mp4"
    iterator = iter(frames)
    first = next(iterator)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    count = 0
    try:
        writer.write(first)
        count += 1
        for frame in iterator:
            if frame.shape != first.shape:
                raise ValueError("Source frame dimensions changed")
            writer.write(frame)
            count += 1
    finally:
        writer.release()
    result = dict(name=name, path=str(path.relative_to(ROOT)), frames=count,
        fps=FPS, size=[width, height], encoding="OpenCV mp4v (lossy)",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **provenance)
    print(json.dumps(result), flush=True)
    return result


def main():
    VIDEOS.mkdir(parents=True, exist_ok=True)
    results = []
    for name, archive, xml in [
        ("caviar_walk", "Walk1_jpg.tar.gz", "wk1gt.xml"),
        ("caviar_multi", "Meet_WalkSplit_jpg.tar.gz", "mws1gt.xml"),
        ("caviar_stop_resume", "OneStopNoEnter1front.tar.gz", "fosne1gt.xml"),
    ]:
        with tarfile.open(SOURCES / archive) as stream:
            files = sorted((m for m in stream.getmembers() if m.name.endswith(".jpg")), key=lambda m: m.name)
            labels = ET.parse(SOURCES / xml).getroot().findall("frame")
            assert len(files) == len(labels), "Image/annotation frame count mismatch"
            # All three archives have contiguous zero-based filenames and matching XML indices.
            results.append(write_video(name, (decode(stream.extractfile(m).read()) for m in files),
                dict(dataset="CAVIAR", source_archive=archive, annotation=xml,
                    annotation_frame_offset=0, timebase="official 25 FPS")))
    for name in ["pedestrians", "fountain01"]:
        with zipfile.ZipFile(SOURCES / f"{name}.zip") as stream:
            files = sorted(n for n in stream.namelist() if "/input/" in n and n.endswith(".jpg"))
            roi = [int(v) for v in stream.read(f"{name}/temporalROI.txt").decode().split()]
            results.append(write_video("cdnet_" + name, (decode(stream.read(n)) for n in files),
                dict(dataset="CDnet2014", source_archive=f"{name}.zip", temporal_roi=roi,
                    annotation_frame_offset=1, timebase="assigned 25 FPS; original FPS unverified")))
    blank = np.full((288, 384, 3), 100, np.uint8)
    results.append(write_video("empty_control", (blank.copy() for _ in range(300)),
        dict(dataset="synthetic control", timebase="assigned 25 FPS")))
    results.append(write_video("lighting_control", (
        np.full_like(blank, 100 if i < 125 else (220 if i < 200 else 50)) for i in range(300)),
        dict(dataset="synthetic control", timebase="assigned 25 FPS")))
    # Exploratory freeze selected from the baseline pedestrian run; kept fixed for reproduction.
    capture = cv2.VideoCapture(str(VIDEOS / "cdnet_pedestrians.mp4"))
    source_frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        source_frames.append(frame)
    capture.release()
    chosen = 683
    def freeze_sequence():
        for _ in range(125):
            yield source_frames[0]
        yield from source_frames[chosen-125:chosen]
        for _ in range(300):
            yield source_frames[chosen]
        yield from source_frames[chosen+1:chosen+126]
    results.append(write_video("frozen_person_control", freeze_sequence(), dict(
        dataset="derived control from CDnet real frames", timebase="assigned 25 FPS",
        source_video="cdnet_pedestrians", source_frozen_frame=chosen,
        control_segments=[{"start":0,"end":125,"label":"negative"},
            {"start":125,"end":250,"label":"moving"},
            {"start":250,"end":550,"label":"negative"},
            {"start":550,"end":675,"label":"moving"}],
        caveat="Freeze is a controlled pixel-identical hold, not a real person stop under changing background.")))
    results.append(dict(next(s for s in results if s["name"] == "caviar_stop_resume"),
        name="caviar_stop_resume_area1000", min_person_motion_pixels_ref=1000))
    (ROOT / "test_data/public_test_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
