"""Check a dataset before training on it: is it complete, consistent and varied enough?

A FAIL means training on it would be wrong or impossible; a WARN is worth a look.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .label import episodes, read_labels


def check_episode(path: Path) -> tuple[list[tuple[str, str]], dict | None]:
    """([(level, message)], measurements) for one episode; no measurements if it cannot be read."""
    problems: list[tuple[str, str]] = []
    try:
        data = np.load(path / "episode.npz")
        t, images, states, actions = (data[k] for k in ("t", "images", "states", "actions"))
        fps = json.loads((path / "meta.json").read_text())["fps"]
    except (OSError, KeyError, ValueError) as error:
        return [("FAIL", f"cannot be read: {error}")], None

    n = len(t)
    if not (len(images) == len(states) == len(actions) == n) or n < 2:
        return [("FAIL", f"arrays differ in length or are too short: t {n}, images {len(images)}, "
                         f"states {len(states)}, actions {len(actions)}")], None
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        problems.append(("FAIL", f"images must be uint8 (N, H, W, 3), got {images.dtype} {images.shape}"))
    if not (np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(t).all()):
        problems.append(("FAIL", "NaN or infinity in the states, actions or timestamps"))
    if np.abs(actions).max() > 1.0 + 1e-6:
        problems.append(("FAIL", f"an action outside [-1, 1]: {np.abs(actions).max():.2f}"))
    if states[:, 0].min() < 0 or np.abs(states[:, 1]).max() > 1.0 + 1e-6:
        problems.append(("FAIL", "a state outside its range (speed < 0 or |steer| > 1)"))

    steps = np.diff(t.astype(np.float64))
    if (steps <= 0).any():
        problems.append(("FAIL", "timestamps do not increase"))
    else:
        typical = float(np.median(steps))
        if abs(typical * fps - 1.0) > 0.2:
            problems.append(("WARN", f"recorded at {1 / typical:.1f} Hz, meta.json says {fps}"))
        gaps = int((steps > 2.5 * typical).sum())
        if gaps:
            problems.append(("WARN", f"{gaps} gap(s) in time, the longest {steps.max():.2f} s"))
    frozen = int(((images[1:] == images[:-1]).all(axis=(1, 2, 3)) & (states[1:, 0] > 0.1)).sum())
    if frozen:
        problems.append(("WARN", f"{frozen} frame(s) identical to the one before while the car moved"))

    labels = read_labels(path)
    if labels is None:
        problems.append(("WARN", "not labelled yet (./drive.sh label)"))
    elif len(labels["plans"]) != n or np.abs(labels["plans"]).max() > 1.0 + 1e-6:
        problems.append(("FAIL", "labels do not match the recording, or hold an action outside [-1, 1]"))
        labels = None
    return problems, {"frames": n, "seconds": float(t[-1] - t[0]), "image": images.shape[1:],
                      "speed": states[:, 0], "steer": actions[:, 0], "labels": labels}


def validate(directory: Path, log=print) -> dict:
    """Print a report and return {'ok', 'failures', 'warnings', ...}."""
    found = episodes(directory)
    if not found:
        raise ValueError(f"no episodes under {directory}")
    problems, measured = [], []
    for path in found:
        issues, stats = check_episode(path)
        problems += [(level, f"{path.name}: {message}") for level, message in issues]
        if stats:
            measured.append(stats)

    def whole(level: str, message: str) -> None:
        problems.append((level, message))

    summary = {"episodes": len(found), "frames": sum(m["frames"] for m in measured)}
    if len(found) < 2:
        whole("FAIL", "one episode only: training holds whole episodes out for validation and needs two")
    if measured:
        speed = np.concatenate([m["speed"] for m in measured])
        steer = np.concatenate([m["steer"] for m in measured])
        labelled = [m["labels"] for m in measured if m["labels"]]
        summary.update(
            minutes=round(sum(m["seconds"] for m in measured) / 60, 1),
            image=sorted({m["image"] for m in measured}),
            speed_mps=[round(float(v), 2) for v in (speed.min(), speed.mean(), speed.max())],
            steer_pct={"left": round(100 * float((steer > 0.1).mean())),
                       "straight": round(100 * float((np.abs(steer) <= 0.1).mean())),
                       "right": round(100 * float((steer < -0.1).mean()))},
            labelled_frames=sum(int(l["valid"].sum()) for l in labelled),
            label_sources=sorted({l["source"] for l in labelled}))
        if len(summary["image"]) > 1:
            whole("FAIL", f"episodes differ in image size: {summary['image']}")
        if len({(l["plans"].shape[1], l["dt"]) for l in labelled}) > 1:
            whole("FAIL", "episodes differ in the plan's length or time step: label them again with one setting")
        if summary["frames"] < 2000:
            whole("WARN", f"{summary['frames']} frames is little; the example network was trained on 12 000")
        if (speed < 0.05).mean() > 0.2:
            whole("WARN", f"the car stands still in {100 * (speed < 0.05).mean():.0f} % of the frames")
        # A teacher's plan starts with what it wanted; the applied steering differs from it
        # only if the drive was disturbed, which is what teaches a learner to recover.
        taught = [(m["steer"], m["labels"]["plans"][:, 0, 0]) for m in measured
                  if m["labels"] and m["labels"]["source"] == "teacher"]
        if taught and np.mean([np.median(np.abs(applied - wanted)) for applied, wanted in taught]) < 0.02:
            whole("WARN", "perfect driving only: the car never leaves the centerline, so a network trained "
                          "on this cannot recover from its own errors (collect with --steer-noise)")

    failures = [m for level, m in problems if level == "FAIL"]
    warnings = [m for level, m in problems if level == "WARN"]
    log(f"{summary['episodes']} episodes, {summary['frames']} frames"
        + (f", {summary['minutes']} min, images {' / '.join('x'.join(map(str, s[:2])) for s in summary['image'])}"
           if measured else ""))
    if measured:
        low, mean, high = summary["speed_mps"]
        steer_pct = summary["steer_pct"]
        log(f"  speed      {low} to {high} m/s, mean {mean}")
        log(f"  steering   left {steer_pct['left']} %, straight {steer_pct['straight']} %, right {steer_pct['right']} %")
        log(f"  labels     {summary['labelled_frames']} usable frames"
            + (f", from {' and '.join(summary['label_sources'])}" if summary["label_sources"] else ""))
    for level, message in sorted(problems):
        log(f"  {level}  {message}")
    log("dataset: " + ("FAILED" if failures else "ok" + (f", {len(warnings)} warning(s)" if warnings else "")))
    return {**summary, "ok": not failures, "failures": failures, "warnings": warnings}


def add_parser(sub):
    from .project import add_project_argument
    add_project_argument(sub.add_parser("validate", help="check the dataset before training on it"))


def run(project, args) -> int:
    return 0 if validate(project.episodes)["ok"] else 1
