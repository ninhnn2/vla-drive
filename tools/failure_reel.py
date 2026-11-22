"""Cut the end of every failed drive out of a `carla drive --video` recording, into one short reel.

    ./vla.sh tools/failure_reel.py runs/carla/eval/smolvla-carla-v1-town01-seed100.json \\
        runs/carla/videos/drive-v1-town01.mp4 runs/carla/videos/failures-v1-town01.mp4

The report and the video must come from the same `carla drive` run. The video holds every
drive's frames at `--hz` (10, `EvalConfig.hz`), drive after drive, so drive i starts at the
sum of the earlier drives' `frames`. A report older than that field has only `seconds`,
rounded to 0.1 s from a time on the 0.05 s tick, which can be a frame short per drive; then
each boundary is snapped to the scene cut (the largest change between frames) within
`--snap` frames of where the seconds put it. The total is checked against the video's frame
count before anything is cut.

Each clip is the last `--seconds` of a drive that ended in a collision, off the route or
stuck, with a banner naming the drive, how it ended and its junctions (L+ passed, Lx
failed), and its last frame held for a moment in a red frame. `--sheets DIR` also writes one
picture per clip: four moments before the end (6, 3 and 1 s, and the last frame).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FAILED = ("collision", "off_route", "stuck")
WHY = {"collision": "hit something", "off_route": "left the route", "stuck": "stuck"}


def drive_frames(drives: list[dict], hz: float) -> list[tuple[int, int]]:
    """[start, end) frame of each drive in the video, from `frames` or else the rounded `seconds`."""
    spans, start = [], 0
    for d in drives:
        end = start + (d["frames"] if "frames" in d else int(round(d["seconds"] * hz)))
        spans.append((start, end))
        start = end
    return spans


def changes(video: Path) -> np.ndarray:
    """change[n]: how much frame n differs from frame n-1 (a scene cut is a spike)."""
    capture, small, out = cv2.VideoCapture(str(video)), None, [0.0]
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        now = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60), interpolation=cv2.INTER_AREA)
        if small is not None:
            out.append(float(np.mean(cv2.absdiff(now, small))))
        small = now
    capture.release()
    return np.array(out)


def snap(spans: list[tuple[int, int]], change: np.ndarray, reach: int) -> list[tuple[int, int]]:
    """Move each boundary between drives onto the scene cut within `reach` frames of it."""
    total, out, start = len(change), [], 0
    lengths = [b - a for a, b in spans]
    for i, n in enumerate(lengths):
        if i == len(lengths) - 1:
            end = total
        else:
            guess = start + n
            near = range(max(start + 1, guess - reach), min(total - 1, guess + reach) + 1)
            end = max(near, key=lambda k: change[k])
        out.append((start, end))
        start = end
    return out


def banner(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    height = 16 + 26 * len(lines)
    out[:height] = (out[:height] * 0.35).astype(np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (12, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path, help="the JSON `carla drive` wrote")
    parser.add_argument("video", type=Path, help="the --video of the same run")
    parser.add_argument("out", type=Path, help="the reel to write (.mp4)")
    parser.add_argument("--seconds", type=float, default=8.0, help="how much of each drive's end (default 8)")
    parser.add_argument("--hold", type=float, default=1.5, help="seconds to hold each last frame (default 1.5)")
    parser.add_argument("--hz", type=float, default=10.0, help="frames per second of the video (default 10)")
    parser.add_argument("--sheets", type=Path, metavar="DIR", help="also write a picture of each failure here")
    parser.add_argument("--snap", type=int, default=3,
                        help="for a report without frames: frames to look either side for a cut (default 3)")
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    drives = report["drives"]
    spans = drive_frames(drives, args.hz)
    capture = cv2.VideoCapture(str(args.video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    exact = all("frames" in d for d in drives)
    if total != spans[-1][1] and (exact or abs(total - spans[-1][1]) > len(drives)):
        raise SystemExit(f"{args.video} has {total} frames, the report's drives make {spans[-1][1]}: not the same run?")
    if not exact:
        change = changes(args.video)
        if len(change) != total:
            raise SystemExit(f"{args.video}: read {len(change)} frames of {total}")
        spans = snap(spans, change, args.snap)
        print(f"  no frame counts in the report: drive boundaries snapped to the scene cuts "
              f"(change {np.median([change[a] for a, _ in spans[1:]]):.0f} there, "
              f"{np.median(change):.1f} elsewhere)")
    failed = [i for i, d in enumerate(drives) if d["ended"] in FAILED]
    if not failed:
        print("no drive failed: nothing to cut")
        return 0
    take = int(round(args.seconds * args.hz))
    wanted = {i: (max(spans[i][0], spans[i][1] - take), spans[i][1]) for i in failed}

    clips: dict[int, list[np.ndarray]] = {i: [] for i in failed}
    for n in range(total):
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"{args.video}: could not read frame {n} of {total}")
        for i, (a, b) in wanted.items():
            if a <= n < b:
                clips[i].append(frame)
    capture.release()

    height, width = clips[failed[0]][0].shape[:2]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.hz, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot write {args.out}")
    if args.sheets:
        args.sheets.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(failed, 1):
        d = drives[i]
        marks = " ".join(f"{m[0].upper()}{'+' if ok else 'x'}" for m, ok in d["junctions"]) or "-"
        lines = [f"{k}/{len(failed)}  {d['name']}  {WHY[d['ended']]} at {d['seconds']:.1f} s  ({d['driven_m']:.0f} m)",
                 f"junctions {marks}   {d['weather']}   lateral rms {d['lateral_rms_m']:.2f} m"]
        frames = clips[i]
        for frame in frames:
            writer.write(banner(frame, lines))
        last = cv2.copyMakeBorder(banner(frames[-1], lines)[8:-8, 8:-8], 8, 8, 8, 8, cv2.BORDER_CONSTANT,
                                  value=(0, 0, 255))
        for _ in range(int(round(args.hold * args.hz))):
            writer.write(last)
        if args.sheets:
            picks = [max(len(frames) - int(round(s * args.hz)) - 1, 0) for s in (6, 3, 1)] + [len(frames) - 1]
            small = [cv2.resize(frames[p], (width // 2, height // 2), interpolation=cv2.INTER_AREA) for p in picks]
            for img, s in zip(small, ("-6 s", "-3 s", "-1 s", "end")):
                cv2.putText(img, s, (width // 2 - 90, height // 2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2, cv2.LINE_AA)
            sheet = np.vstack([np.hstack(small[:2]), np.hstack(small[2:])])
            cv2.imwrite(str(args.sheets / f"{k:02d}-{d['name']}-{d['ended']}.jpg"), banner(sheet, lines))
        print(f"  {d['name']}: {WHY[d['ended']]} at {d['seconds']:.1f} s, junctions {marks}, {len(frames)} frames")
    writer.release()
    print(f"{len(failed)} failures, {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
