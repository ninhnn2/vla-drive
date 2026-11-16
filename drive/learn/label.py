"""Labels: for every recorded frame, the plan a policy should output there.

A plan is the next `steps` actions, `dt` seconds apart. It comes from one of two places:

    teacher   the simulator's teacher states its plan while recording (collect.py)
    actions   the actions that really followed the frame: for drives recorded from a
              person or over ROS 2, where nobody states a plan

Labels are written to `labels.npz` beside `episode.npz`, so labelling again never
touches a recording.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PLAN_STEPS, PLAN_DT_S = 16, 0.1        # what a learner predicts: 1.5 s ahead


def episodes(directory: Path) -> list[Path]:
    """The episode directories under `directory`, in order."""
    return sorted(path.parent for path in Path(directory).glob("*/episode.npz"))


def plans_from_actions(t: np.ndarray, actions: np.ndarray, steps: int = PLAN_STEPS,
                       dt: float = PLAN_DT_S) -> tuple[np.ndarray, np.ndarray]:
    """(plans (N, steps, 2), valid (N,)): each frame's plan is the actions after it.

    A frame near the end of the drive has no full future; it is marked not valid and
    training skips it.
    """
    times = t[:, None] + np.arange(steps) * dt
    plans = np.stack([np.interp(times, t, actions[:, k]) for k in range(actions.shape[1])], axis=-1)
    return plans.astype(np.float32), times[:, -1] <= t[-1] + 1e-6


def write_labels(episode: Path, plans: np.ndarray, valid: np.ndarray, dt: float, source: str) -> None:
    np.savez_compressed(Path(episode) / "labels.npz", plans=np.asarray(plans, np.float32),
                        valid=np.asarray(valid, bool), dt=np.float64(dt), source=np.array(source))


def read_labels(episode: Path) -> dict | None:
    """{'plans', 'valid', 'dt', 'source'}, or None for an episode not labelled yet."""
    path = Path(episode) / "labels.npz"
    if not path.is_file():
        return None
    data = np.load(path)
    return {"plans": data["plans"], "valid": data["valid"], "dt": float(data["dt"]), "source": str(data["source"])}


def label(directory: Path, steps: int = PLAN_STEPS, dt: float = PLAN_DT_S, relabel: bool = False) -> dict:
    """Label every episode that has no labels yet (or all of them with `relabel`) from its actions."""
    found = episodes(directory)
    if not found:
        raise ValueError(f"no episodes under {directory}")
    labelled = kept = frames = 0
    for episode in found:
        if read_labels(episode) is not None and not relabel:
            kept += 1
            continue
        data = np.load(episode / "episode.npz")
        plans, valid = plans_from_actions(data["t"].astype(np.float64), data["actions"], steps, dt)
        write_labels(episode, plans, valid, dt, "actions")
        labelled += 1
        frames += int(valid.sum())
    return {"labelled": labelled, "already_labelled": kept, "frames": frames}


def add_parser(sub):
    from .project import add_project_argument
    p = sub.add_parser("label", help="label recorded drives from the actions that followed each frame")
    add_project_argument(p)
    p.add_argument("--relabel", action="store_true", help="also replace labels that exist (a teacher's too)")


def run(project, args) -> int:
    done = label(project.episodes, relabel=args.relabel)
    print(f"labelled {done['labelled']} episode(s) from their actions: {done['frames']} usable frames"
          + (f"; {done['already_labelled']} already had labels" if done["already_labelled"] else ""))
    return 0
