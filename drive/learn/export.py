"""export: the project's episodes as a LeRobotDataset, for fine-tuning a VLA (SmolVLA) with LeRobot.

    python -m drive export [--project DIR] [--out DIR] [--match PATTERN] [--action intended|applied]

Needs LeRobot, which the project's container does not have: run it with an environment
that has `lerobot` installed (see docs/study/13-vla-and-smolvla.md). Episodes must be
labelled first (./drive.sh label).

What goes where, per frame:

    episode.npz images[i]   ->  observation.images.front   video, 240x320 RGB
    episode.npz states[i]   ->  observation.state          [speed_mps, steer]
    labels.npz plans[i, 0]  ->  action                     [steer, throttle]   (--action intended)
    episode.npz actions[i]  ->  action                                         (--action applied)
    meta.json task          ->  task

Why `intended` is the default. LeRobot builds a training target for frame i from the
`action` column of frames i, i+1, ...: one value per frame. For episodes recorded by
`collect`, `actions[i]` is the steering that was *applied*, with the disturbance in it,
and training on it would teach the noise. `plans[i, 0]` is what the teacher *wanted*
from that state, which is DART's label. For episodes labelled from their actions
(ROS 2, a person), `plans[i, 0]` equals `actions[i]`, so `intended` is right for both.

LeRobot assumes frame i is at i / fps. Episodes from `collect` are exactly 0.1 s apart;
ROS 2 recordings jitter by up to about 40 ms. Frames are kept as recorded, and an
episode whose timing strays by half a period or more is refused.
"""
from __future__ import annotations

import json
import time
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

from .label import episodes, read_labels
from .project import Project, add_project_argument

CAMERA = "observation.images.front"
ACTIONS = ("intended", "applied")


def features(image_shape: tuple[int, int, int]) -> dict:
    """The LeRobot feature schema of one frame."""
    return {
        CAMERA: {"dtype": "video", "shape": tuple(image_shape), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["speed_mps", "steer"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["steer", "throttle"]},
    }


def frames(episode: Path, action: str = "intended") -> tuple[dict, list[dict]]:
    """(meta, one dict per frame in LeRobot's layout) for one episode directory. Pure numpy."""
    if action not in ACTIONS:
        raise ValueError(f"--action must be one of {', '.join(ACTIONS)}, got {action!r}")
    meta = json.loads((episode / "meta.json").read_text())
    labels = read_labels(episode)
    if labels is None:
        raise ValueError(f"{episode} is not labelled; run ./drive.sh label")
    with np.load(episode / "episode.npz") as data:
        t, images, states, applied = (data[k] for k in ("t", "images", "states", "actions"))
    fps = float(meta["fps"])
    drift = float(np.abs(t.astype(np.float64) - t[0] - np.arange(len(t)) / fps).max())
    if drift >= 0.5 / fps:
        raise ValueError(f"{episode.name}: frames stray {drift * 1000:.0f} ms from a steady {fps:g} Hz; "
                         "LeRobot would misplace them in time")
    actions = labels["plans"][:, 0] if action == "intended" else applied
    rows = [{CAMERA: images[i], "observation.state": states[i].astype(np.float32),
             "action": np.clip(actions[i], -1.0, 1.0).astype(np.float32), "task": meta["task"]}
            for i in range(len(t))]
    return meta, rows


def export(directory: Path, out: Path, repo_id: str = "local/vla-drive", match: str = "*",
           action: str = "intended", log=print) -> dict:
    """Write every episode under `directory` whose name matches `match` to a new LeRobotDataset at `out`."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise ValueError(f"export needs LeRobot ({error}); run it in an environment that has lerobot "
                         "installed (docs/study/13-vla-and-smolvla.md)") from error
    found = [e for e in episodes(directory) if fnmatch(e.name, match)]
    if not found:
        raise ValueError(f"no episodes under {directory}" + (f" named {match!r}" if match != "*" else ""))
    out = Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; a LeRobotDataset is written once. Choose another --out or remove it")
    first_meta, first = frames(found[0], action)
    fps, shape = first_meta["fps"], first[0][CAMERA].shape
    if float(fps) != int(fps):
        raise ValueError(f"LeRobot needs a whole number of frames per second, {found[0].name} has {fps}")

    started = time.perf_counter()
    dataset = LeRobotDataset.create(repo_id, fps=int(fps), features=features(shape), root=out,
                                    robot_type="vla-drive-sim", use_videos=True, image_writer_threads=4)
    total = 0
    try:
        for index, episode in enumerate(found):
            meta, rows = (first_meta, first) if index == 0 else frames(episode, action)
            if meta["fps"] != fps or rows[0][CAMERA].shape != shape:
                raise ValueError(f"{episode.name}: {meta['fps']} fps, images {rows[0][CAMERA].shape}; "
                                 f"the first episode has {fps} fps, images {shape}")
            for row in rows:
                dataset.add_frame(row)
            dataset.save_episode()
            total += len(rows)
            log(f"  {index + 1}/{len(found)} {episode.name}: {len(rows)} frames")
    finally:
        dataset.finalize()
    return {"episodes": len(found), "frames": total, "fps": int(fps), "image": list(shape), "action": action,
            "root": str(out), "repo_id": repo_id, "seconds": round(time.perf_counter() - started, 1)}


def add_parser(sub):
    p = sub.add_parser("export", help="write the episodes as a LeRobotDataset, for fine-tuning a VLA (needs lerobot)")
    add_project_argument(p)
    p.add_argument("--out", type=Path, help="where to write it (default <project>/lerobot); must not exist")
    p.add_argument("--repo-id", default="local/vla-drive",
                   help="the dataset's name inside LeRobot; nothing is uploaded (default local/vla-drive)")
    p.add_argument("--match", default="*", metavar="PATTERN",
                   help="only the episodes with this name, a shell pattern: 'sim*', ros-000")
    p.add_argument("--action", choices=ACTIONS, default="intended",
                   help="intended: the label's first action (the teacher's, not the disturbed one; default); "
                        "applied: what was on the motors")


def run(project: Project, args) -> int:
    out = args.out or project.home / "lerobot"
    print(f"exporting {project.episodes} to {out} (action: {args.action})")
    done = export(project.episodes, out, args.repo_id, args.match, args.action)
    print(f"{done['frames']} frames in {done['episodes']} episodes at {done['fps']} Hz, {done['seconds']} s\n"
          f"  dataset: {done['root']}\n"
          f"  train with: lerobot-train --policy.path=lerobot/smolvla_base "
          f"--dataset.repo_id={done['repo_id']} --dataset.root={Path(done['root']).resolve()} "
          f"--rename_map='{{\"{CAMERA}\": \"observation.images.camera1\"}}' "
          f"--policy.chunk_size=16 --policy.n_action_steps=16 --policy.push_to_hub=false ...")
    return 0
