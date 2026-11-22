"""export: the project's episodes as a LeRobotDataset, for fine-tuning a VLA (SmolVLA) with LeRobot.

    python -m drive export [--project DIR] [--out DIR] [--match PATTERN] [--action intended|applied]
                           [--state speed_mps,steer|speed_mps]

Needs LeRobot, which the project's container does not have: run it with an environment
that has `lerobot` installed (see docs/study/13-vla-and-smolvla.md). Episodes must be
labelled first (./drive.sh label).

What goes where, per frame:

    episode.npz images[i]   ->  observation.images.front   video, 240x320 RGB
    episode.npz states[i]   ->  observation.state          [speed_mps, steer], or [speed_mps] (--state)
    labels.npz plans[i, 0]  ->  action                     [steer, throttle]   (--action intended)
    episode.npz actions[i]  ->  action                                         (--action applied)
    episode.npz tasks[i]    ->  task       the frame's own sentence (CARLA: "turn left at ..."),
    meta.json task          ->  task       or, without one, the episode's

The dataset's robot_type is the episodes' (`carla:vehicle.tesla.model3`, or
`vla-drive-sim`); episodes from different platforms are not mixed into one dataset.

Why `intended` is the default. LeRobot builds a training target for frame i from the
`action` column of frames i, i+1, ...: one value per frame. For episodes recorded by
`collect`, `actions[i]` is the steering that was *applied*, with the disturbance in it,
and training on it would teach the noise. `plans[i, 0]` is what the teacher *wanted*
from that state, which is DART's label. For episodes labelled from their actions
(ROS 2, a person), `plans[i, 0]` equals `actions[i]`, so `intended` is right for both.

Why `--state speed_mps`. With the steer the car is holding in its state, a model learns to
keep holding it: steering changes slowly, so the next steer is mostly the present one
(correlation 0.83 in the CARLA episodes) and copying it pays in training. Driving, the
steer it is holding is its own last command, and it turns into a bend late and out of one
late (docs/carla.md). The teacher's label never reads that steer; without it in the state
the model has to read the road. The speed stays: a picture does not say how fast the car is.
A model reads the first n of [speed_mps, steer]: `LeRobotBackend` cuts the state to the
width its checkpoint was trained on, so --state is one of the two prefixes.

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
STATE = ("speed_mps", "steer")          # the columns of episode.npz states, in order


def check_state(state) -> tuple[str, ...]:
    """`state` as a tuple, if it is one of the prefixes of STATE a model can be given."""
    state = tuple(state)
    if not state or state != STATE[: len(state)]:
        raise ValueError(f"--state must be {' or '.join(','.join(STATE[:n]) for n in range(len(STATE), 0, -1))}, "
                         f"got {','.join(state) or 'nothing'}")
    return state


def features(image_shape: tuple[int, int, int], state=STATE) -> dict:
    """The LeRobot feature schema of one frame."""
    state = check_state(state)
    return {
        CAMERA: {"dtype": "video", "shape": tuple(image_shape), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(state),), "names": list(state)},
        "action": {"dtype": "float32", "shape": (2,), "names": ["steer", "throttle"]},
    }


def frames(episode: Path, action: str = "intended", state=STATE) -> tuple[dict, list[dict]]:
    """(meta, one dict per frame in LeRobot's layout) for one episode directory. Pure numpy."""
    width = len(check_state(state))
    if action not in ACTIONS:
        raise ValueError(f"--action must be one of {', '.join(ACTIONS)}, got {action!r}")
    meta = json.loads((episode / "meta.json").read_text())
    labels = read_labels(episode)
    if labels is None:
        raise ValueError(f"{episode} is not labelled; run ./drive.sh label")
    with np.load(episode / "episode.npz") as data:
        t, images, states, applied = (data[k] for k in ("t", "images", "states", "actions"))
        tasks = [str(task) for task in data["tasks"]] if "tasks" in data.files else [meta["task"]] * len(t)
    if len(tasks) != len(t):
        raise ValueError(f"{episode.name}: {len(tasks)} tasks for {len(t)} frames")
    fps = float(meta["fps"])
    drift = float(np.abs(t.astype(np.float64) - t[0] - np.arange(len(t)) / fps).max())
    if drift >= 0.5 / fps:
        raise ValueError(f"{episode.name}: frames stray {drift * 1000:.0f} ms from a steady {fps:g} Hz; "
                         "LeRobot would misplace them in time")
    actions = labels["plans"][:, 0] if action == "intended" else applied
    rows = [{CAMERA: images[i], "observation.state": states[i, :width].astype(np.float32),
             "action": np.clip(actions[i], -1.0, 1.0).astype(np.float32), "task": tasks[i]}
            for i in range(len(t))]
    return meta, rows


def matches(name: str, patterns: str) -> bool:
    """`name` matches one of the comma-separated shell patterns: 'carla3-*,dagger1-*'."""
    return any(fnmatch(name, pattern.strip()) for pattern in patterns.split(","))


def robot_type(meta: dict) -> str:
    return meta.get("robot_type", "vla-drive-sim")


def export(directory: Path, out: Path, repo_id: str = "local/vla-drive", match: str = "*",
           action: str = "intended", log=print, exclude: str | None = None, state=STATE) -> dict:
    """Write every episode under `directory` whose name matches `match`, and not `exclude`, to a new
    LeRobotDataset at `out`. `exclude` keeps episodes out for testing the model on afterwards;
    `state` is the columns of the state the model is given."""
    state = check_state(state)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise ValueError(f"export needs LeRobot ({error}); run it in an environment that has lerobot "
                         "installed (docs/study/13-vla-and-smolvla.md)") from error
    found = [e for e in episodes(directory) if matches(e.name, match) and not (exclude and matches(e.name, exclude))]
    if not found:
        raise ValueError(f"no episodes under {directory}" + (f" named {match!r}" if match != "*" else "")
                         + (f" once {exclude!r} is left out" if exclude else ""))
    out = Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; a LeRobotDataset is written once. Choose another --out or remove it")
    first_meta, first = frames(found[0], action, state)
    fps, shape = first_meta["fps"], first[0][CAMERA].shape
    if float(fps) != int(fps):
        raise ValueError(f"LeRobot needs a whole number of frames per second, {found[0].name} has {fps}")

    started = time.perf_counter()
    robot = robot_type(first_meta)
    dataset = LeRobotDataset.create(repo_id, fps=int(fps), features=features(shape, state), root=out,
                                    robot_type=robot, use_videos=True, image_writer_threads=4)
    total = 0
    try:
        for index, episode in enumerate(found):
            meta, rows = (first_meta, first) if index == 0 else frames(episode, action, state)
            if meta["fps"] != fps or rows[0][CAMERA].shape != shape or robot_type(meta) != robot:
                raise ValueError(f"{episode.name}: {robot_type(meta)}, {meta['fps']} fps, images {rows[0][CAMERA].shape}; "
                                 f"the first episode is {robot}, {fps} fps, images {shape}. Export one platform "
                                 "at a time (--match)")
            for row in rows:
                dataset.add_frame(row)
            dataset.save_episode()
            total += len(rows)
            log(f"  {index + 1}/{len(found)} {episode.name}: {len(rows)} frames")
    finally:
        dataset.finalize()
    return {"episodes": len(found), "frames": total, "fps": int(fps), "image": list(shape), "action": action,
            "state": list(state),
            "robot_type": robot, "root": str(out), "repo_id": repo_id,
            "seconds": round(time.perf_counter() - started, 1)}


def add_parser(sub):
    p = sub.add_parser("export", help="write the episodes as a LeRobotDataset, for fine-tuning a VLA (needs lerobot)")
    add_project_argument(p)
    p.add_argument("--out", type=Path, help="where to write it (default <project>/lerobot); must not exist")
    p.add_argument("--repo-id", default="local/vla-drive",
                   help="the dataset's name inside LeRobot; nothing is uploaded (default local/vla-drive)")
    p.add_argument("--match", default="*", metavar="PATTERN",
                   help="only the episodes with this name, shell patterns, comma-separated: 'sim*', 'carla3-*,dagger1-*'")
    p.add_argument("--exclude", metavar="PATTERN",
                   help="leave out the episodes with this name, to test the model on: 'carla-*-03[5-9]'")
    p.add_argument("--action", choices=ACTIONS, default="intended",
                   help="intended: the label's first action (the teacher's, not the disturbed one; default); "
                        "applied: what was on the motors")
    p.add_argument("--state", default=",".join(STATE), type=lambda text: check_state(text.split(",")),
                   help="the state the model is given: speed_mps,steer (default) or speed_mps, which leaves "
                        "out the steer the car is holding so the model cannot copy it")


def run(project: Project, args) -> int:
    out = args.out or project.home / "lerobot"
    print(f"exporting {project.episodes} to {out} (action: {args.action}, state: {','.join(args.state)})")
    done = export(project.episodes, out, args.repo_id, args.match, args.action, exclude=args.exclude,
                  state=args.state)
    print(f"{done['frames']} frames in {done['episodes']} episodes at {done['fps']} Hz, {done['seconds']} s\n"
          f"  dataset: {done['root']}\n"
          f"  train with: lerobot-train --policy.path=lerobot/smolvla_base "
          f"--dataset.repo_id={done['repo_id']} --dataset.root={Path(done['root']).resolve()} "
          f"--rename_map='{{\"{CAMERA}\": \"observation.images.camera1\"}}' "
          f"--policy.chunk_size=16 --policy.n_action_steps=16 --policy.push_to_hub=false ...")
    return 0
