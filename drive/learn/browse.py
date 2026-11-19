"""Look at a dataset: every frame with its state, its action, and its label drawn on the road.

The label is a plan of future actions. Rolled forward with the car model it is a path,
and a good label's path follows the road in the picture.
"""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import cv2
import numpy as np

from ..config import CameraParams, VehicleParams
from ..sim import step_vehicle
from ..sim.render import Camera
from ..types import VehicleState
from ..viewer import Output
from .label import episodes, read_labels

SCALE = 3                                       # the 320x240 frame is shown at 960x720
CYAN, WHITE, GREY = (60, 220, 240), (240, 240, 240), (150, 150, 150)


def plan_path(plan: np.ndarray, dt: float, speed: float, steer: float, vehicle=VehicleParams()) -> np.ndarray:
    """(steps, 2) points (forward, left) in metres: where the plan takes the car from here."""
    state, points = VehicleState(0.0, 0.0, 0.0, speed, steer), []
    for action in plan:
        for _ in range(5):
            state = step_vehicle(state, action, vehicle, dt / 5)
        points.append((state.x, state.y))
    return np.array(points)


def draw(image, state, action, plan, plan_dt, title: str, camera: Camera) -> np.ndarray:
    """One annotated RGB frame."""
    frame = cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    if plan is not None:
        pixels = camera.project(*plan_path(plan, plan_dt, state[0], state[1]).T) * SCALE
        pixels = pixels[np.isfinite(pixels).all(axis=1)].round().astype(np.int32)
        if len(pixels) > 1:
            cv2.polylines(frame, [pixels], False, CYAN, 3, cv2.LINE_AA)
    lines = [title, f"speed {state[0]:.2f} m/s   steer {state[1]:+.2f}",
             f"action: steer {action[0]:+.2f}  throttle {action[1]:+.2f}",
             "cyan: the labelled plan" if plan is not None else "no usable label for this frame"]
    for row, line in enumerate(lines):
        cv2.putText(frame, line, (12, 28 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE if row < 3 else CYAN,
                    2, cv2.LINE_AA)
    cv2.putText(frame, "d/a: next/previous frame   n/p: episode   space: play   q: quit",
                (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    return frame


def browse(directory: Path, show: bool = True, video: Path | None = None, every: int = 1,
           match: str = "*") -> int:
    """Step through a dataset in a window, or write it to a video. Returns the frames shown.

    `match` picks episodes by name, as a shell pattern: 'sim2-*', 'ros-000'.
    """
    found = [e for e in episodes(directory) if fnmatch(e.name, match)]
    if not found:
        raise ValueError(f"no episodes under {directory}" + (f" named {match!r}" if match != "*" else ""))
    # Frames are numbered through the whole dataset; `starts[i]` is episode i's first.
    starts = np.concatenate([[0], np.cumsum([len(np.load(e / "episode.npz")["t"]) for e in found])])
    camera = Camera(CameraParams())
    output, loaded, shown = None, None, 0
    position, playing = 0, not show
    try:
        while 0 <= position < starts[-1]:
            index = int(np.searchsorted(starts, position, side="right")) - 1
            frame = position - starts[index]
            if loaded != index:
                with np.load(found[index] / "episode.npz") as archive:
                    data = {key: archive[key] for key in ("t", "images", "states", "actions")}
                labels, loaded = read_labels(found[index]), index
            usable = labels is not None and labels["valid"][frame]
            title = f"{found[index].name}  frame {frame + 1}/{len(data['t'])}  t {data['t'][frame]:.1f} s"
            picture = draw(data["images"][frame], data["states"][frame], data["actions"][frame],
                           labels["plans"][frame] if usable else None, labels["dt"] if labels else 0.0,
                           title, camera)
            output = output or Output(show, video, fps=10, size=picture.shape[1::-1])
            key = output.put(picture)
            shown += 1
            while show and not playing and not key:       # paused: wait for a key
                key = output.put(picture)
            if key == "q":
                break
            if key == " ":
                playing = not playing
            position = {"a": max(position - every, 0), "n": starts[index + 1],
                        "p": starts[max(index - 1, 0)]}.get(key, position + every)
        return shown
    finally:
        if output is not None:
            output.close()


def add_parser(sub):
    from .project import add_project_argument
    p = sub.add_parser("browse", help="look at the dataset's frames with their labels drawn on the road")
    add_project_argument(p)
    p.add_argument("--video", type=Path, help="write an .mp4 instead of opening a window")
    p.add_argument("--every", type=int, default=1, help="show every Nth frame")
    p.add_argument("--match", default="*", metavar="PATTERN",
                   help="only the episodes with this name, a shell pattern: 'sim2-*', ros-000")


def run(project, args) -> int:
    shown = browse(project.episodes, show=args.video is None, video=args.video, every=args.every,
                   match=args.match)
    if args.video:
        print(f"{shown} frames written to {args.video}")
    return 0
