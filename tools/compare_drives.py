"""The same routes driven by two models, side by side, from two `carla drive --video` runs.

    ./vla.sh tools/compare_drives.py \\
        runs/carla/eval/smolvla-carla-v3-6mps-replan500-town01-seed100.json \\
        runs/carla/videos/drive-smolvla-carla-v3-6mps-replan500-town01.mp4 \\
        runs/carla/eval/smolvla-carla-v3-dagger2-6mps-replan500-town01-seed100.json \\
        runs/carla/videos/drive-smolvla-carla-v3-dagger2-6mps-replan500-town01.mp4 \\
        runs/carla/videos/compare.mp4 --drives 5,7,8 --labels v3,DAgger2

Both runs must have driven the same routes (the same --seed and town). Each drive is found
in its video by the report's per-drive `frames`; the shorter of the two is held on its last
frame, bordered red if it failed, until the longer one ends.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FAILED = ("collision", "off_route", "stuck")


def spans(report: dict) -> list[tuple[int, int]]:
    out, start = [], 0
    for d in report["drives"]:
        if "frames" not in d:
            raise SystemExit("the report has no per-drive frames (made before they were recorded): "
                             "drive again, or cut with tools/failure_reel.py")
        out.append((start, start + d["frames"]))
        start += d["frames"]
    return out


def read(video: Path, wanted: dict[int, tuple[int, int]]) -> dict[int, list[np.ndarray]]:
    capture, clips, n = cv2.VideoCapture(str(video)), {i: [] for i in wanted}, 0
    last = max(b for _, b in wanted.values())
    while n < last:
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"{video}: ends at frame {n}, the report needs {last}")
        for i, (a, b) in wanted.items():
            if a <= n < b:
                clips[i].append(frame)
        n += 1
    capture.release()
    return clips


def caption(frame: np.ndarray, text: str, colour=(255, 255, 255)) -> np.ndarray:
    out = frame.copy()
    out[-44:] = (out[-44:] * 0.3).astype(np.uint8)
    cv2.putText(out, text, (12, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report_a", type=Path)
    parser.add_argument("video_a", type=Path)
    parser.add_argument("report_b", type=Path)
    parser.add_argument("video_b", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--drives", required=True, help="drive indices, comma-separated")
    parser.add_argument("--labels", default="A,B", help="what to call the two (default A,B)")
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    indices = [int(i) for i in args.drives.split(",")]
    labels = args.labels.split(",")
    reports = [json.loads(p.read_text()) for p in (args.report_a, args.report_b)]
    clips = [read(v, {i: spans(r)[i] for i in indices}) for v, r in zip((args.video_a, args.video_b), reports)]
    height, width = clips[0][indices[0]][0].shape[:2]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.hz, (2 * width, height))
    for i in indices:
        sides = []
        for k in range(2):
            d, frames = reports[k]["drives"][i], clips[k][i]
            marks = " ".join(f"{m[0].upper()}{'+' if ok else 'x'}" for m, ok in d["junctions"]) or "-"
            how = "to the end" if d["ended"] == "duration" else d["ended"].replace("_", " ")
            colour = (80, 80, 255) if d["ended"] in FAILED else (120, 255, 120)
            sides.append((frames, f"{labels[k]}: {how}, {d['driven_m']:.0f} m, junctions {marks}", colour,
                          d["ended"] in FAILED))
        for n in range(max(len(f) for f, *_ in sides)):
            pair = []
            for frames, text, colour, failed in sides:
                frame = caption(frames[min(n, len(frames) - 1)], text, colour)
                if n >= len(frames) - 1 and failed:
                    frame = cv2.copyMakeBorder(frame[8:-8, 8:-8], 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                pair.append(frame)
            writer.write(np.hstack(pair))
        print(f"  drive {i}: " + " | ".join(text for _, text, _, _ in sides))
    writer.release()
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
