"""Make videos ready for the web app: upright, small, H.264 mp4 that any browser plays.

    python3 tools/web_videos.py [inputs ...] [--out runs/bdd100k/web] [--width 640] [--fps 30] [--seconds 0]

Inputs are video files or directories of them; the default is runs/bdd100k/videos (what
tools/fetch_bdd100k.py downloads). For each one it writes <out>/<name>.mp4:

    upright      phone video stored sideways with a rotation flag (most of BDD100K) is turned the
                 way a player shows it, so the frames themselves are upright and no flag is left
    small        --width pixels wide (640: the model sees 320 x 240 anyway), --fps frames a second
    H.264        yuv420p, the index at the front, no sound: plays and uploads in every browser
    --seconds    keep only the first N seconds (0: all)

The Video tab of tools/vla_app.py lists what is in runs/bdd100k/web as examples to click. Needs
ffmpeg with libx264 on the PATH (the container of docker/vla has it).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SUFFIXES = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm"}


def probe(path: Path) -> dict:
    """width, height, fps, seconds, rotation of the first video stream, as stored."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:stream_side_data=rotation:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    num, den = (int(v) for v in stream["r_frame_rate"].split("/"))
    rotation = next((int(s["rotation"]) for s in stream.get("side_data_list", []) if "rotation" in s), 0)
    return dict(width=stream["width"], height=stream["height"], fps=num / den,
                seconds=float(data["format"]["duration"]), rotation=rotation)


def convert(source: Path, target: Path, width: int, fps: int, seconds: float) -> None:
    # ffmpeg turns the frames by the rotation flag itself (autorotate) before the filters run.
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(source)]
    if seconds > 0:
        command += ["-t", str(seconds)]
    command += ["-vf", f"scale={width}:-2,fps={fps}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", "-metadata:s:v", "rotate=0", str(target)]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path, default=[Path("runs/bdd100k/videos")])
    parser.add_argument("--out", type=Path, default=Path("runs/bdd100k/web"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=0.0, help="keep only the first N seconds (0: all)")
    args = parser.parse_args()
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        print("needs ffmpeg and ffprobe on the PATH", file=sys.stderr)
        return 1

    sources = []
    for item in args.inputs:
        sources += sorted(p for p in item.rglob("*") if p.suffix.lower() in SUFFIXES) if item.is_dir() else [item]
    if not sources:
        print(f"no videos in {', '.join(map(str, args.inputs))}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = args.out / f"{source.stem}.mp4"
        before = probe(source)
        convert(source, target, args.width, args.fps, args.seconds)
        after = probe(target)
        turned = f", turned {before['rotation']}°" if before["rotation"] else ""
        print(f"{target}  {after['width']}x{after['height']} {after['fps']:.0f} fps {after['seconds']:.1f} s "
              f"{target.stat().st_size / 1e6:.1f} MB   (was {before['width']}x{before['height']} "
              f"{before['fps']:.0f} fps{turned}, {source.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
