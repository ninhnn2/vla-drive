"""Test a VLA fine-tuned on the CARLA episodes without driving it, on episodes it did not train on.

    ./vla.sh tools/probe_vla_carla.py --checkpoint smolvla-carla-v1 [--episodes 'carla-*-03[5-9]']

Needs no CARLA server; HF_HOME as for training. The episodes must be left out of the training
set (`drive export --exclude 'carla-*-03[5-9]'`), or this measures what it memorised.

    offline   per sentence kind (follow, left, right, straight): mean |steer error| against the
              teacher's clean plan at chunk steps 0, 5, 10 and 15, and |throttle error| at step 0
    listen    at frames near or in a junction: the same picture and the same noise with the
              sentence swapped for each maneuver. A model that reads the sentence steers further
              left for "turn left" than for "turn right"; one that does not gives all three the
              same plan. This is the test the dataset is for.
    speed     one frame at 0, 3, 6 and 9 m/s: does the throttle answer the speed?
    timing    one chunk on this machine
"""
from __future__ import annotations

import argparse
import json
import time
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

from drive.carla.route import MANEUVERS, PHRASES
from drive.learn.label import episodes, read_labels
from drive.learn.project import Project
from drive.policy.lerobot_backend import LeRobotBackend, synchronize
from drive.policy.vla import BackendObservation

KIND = {sentence: kind for kind, sentences in PHRASES.items() for sentence in sentences}


def chunk(backend, image, speed, steer, task, noise):
    obs = BackendObservation(index=0, images=[image], state=np.array([speed, steer], np.float32), task=task, noise=noise)
    return backend.infer(obs).chunk[:, :2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("runs/carla"))
    parser.add_argument("--checkpoint", default="smolvla-carla-v1",
                        help="a version in <project>/vla, or a checkpoint directory")
    parser.add_argument("--episodes", default="carla-*-03[5-9]", help="held-out episodes, a shell pattern")
    parser.add_argument("--every", type=int, default=10, help="use every Nth frame (default 10: one a second)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="cuda, mps or cpu (default: the fastest there is; or $VLA_DEVICE)")
    parser.add_argument("--json", type=Path, help="also write the numbers here")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    checkpoint = Project(args.project).checkpoint(args.checkpoint)
    started = time.perf_counter()
    backend = LeRobotBackend(checkpoint, args.device)
    steps = backend.config.chunk_size
    noise_shape = (1, steps, backend.config.max_action_dim)
    print(f"loaded {checkpoint} on {backend.device} in {time.perf_counter() - started:.1f} s; chunk {steps}")
    found = [e for e in episodes(args.project / "episodes") if fnmatch(e.name, args.episodes)]
    if not found:
        raise SystemExit(f"no episodes named {args.episodes!r} under {args.project / 'episodes'}")

    shown = [k for k in (0, 5, 10, 15) if k < steps]
    steer_err = {kind: {k: [] for k in shown} for kind in PHRASES}
    throttle_err = {kind: [] for kind in PHRASES}
    listen = {m: [] for m in MANEUVERS}          # mean steer over the chunk, by the sentence given
    listen_kind = []                             # the real maneuver at those frames
    firsts, sample = [], None
    for episode in found:
        data, labels = np.load(episode / "episode.npz"), read_labels(episode)
        meta = json.loads((episode / "meta.json").read_text())
        tasks = data["tasks"] if "tasks" in data.files else np.array([meta["task"]] * len(data["t"]))
        plans = labels["plans"]
        for i in range(0, len(data["t"]) - steps, args.every):
            image, (speed, steer), task = data["images"][i], data["states"][i], str(tasks[i])
            kind = KIND.get(task, "follow")
            noise = rng.standard_normal(noise_shape).astype(np.float32)
            a = chunk(backend, image, speed, steer, task, noise)
            for k in shown:
                if k < plans.shape[1]:
                    steer_err[kind][k].append(abs(a[k, 0] - plans[i, k, 0]))
            throttle_err[kind].append(abs(a[0, 1] - plans[i, 0, 1]))
            firsts.append((a[0, 0], plans[i, 0, 0]))
            if kind == "follow" and sample is None:
                sample = (image, steer, noise)
            if kind != "follow":
                for m in MANEUVERS:
                    listen[m].append(float(chunk(backend, image, speed, steer, PHRASES[m][0], noise)[:, 0].mean()))
                listen_kind.append(kind)
        print(f"  {episode.name}: done", flush=True)

    result = {"checkpoint": str(checkpoint), "episodes": len(found), "frames": len(firsts)}
    print(f"\n[offline] {len(firsts)} frames from {len(found)} held-out episodes: mean |steer error| "
          "against the teacher's clean plan")
    print("  sentence    frames  " + "  ".join(f"step {k:2d}" for k in shown) + "   |throttle| step 0")
    result["offline"] = {}
    for kind in PHRASES:
        n = len(throttle_err[kind])
        if not n:
            continue
        cells = [float(np.mean(steer_err[kind][k])) for k in shown]
        print(f"  {kind:10s} {n:6d}  " + "  ".join(f"{c:7.3f}" for c in cells) + f"   {np.mean(throttle_err[kind]):7.3f}")
        result["offline"][kind] = {"frames": n, "steer": dict(zip(map(str, shown), cells)),
                                   "throttle_0": float(np.mean(throttle_err[kind]))}
    model, teacher = np.array(firsts).T
    corr = float(np.corrcoef(model, teacher)[0, 1])
    print(f"  step 0: correlation with the teacher {corr:.3f}")
    result["offline_correlation_step0"] = corr

    if listen_kind:
        left, right, straight = (np.array(listen[m]) for m in ("left", "right", "straight"))
        order = float(np.mean(left > right))
        ordered = float(np.mean((left > straight) & (straight > right)))
        spread = float(np.mean(np.abs(left - right)))
        print(f"\n[listen] {len(listen_kind)} frames near or in a junction, the sentence swapped: mean steer over the chunk")
        print(f"  told left {left.mean():+.3f}   straight {straight.mean():+.3f}   right {right.mean():+.3f}")
        print(f"  'left' steers further left than 'right' at {100 * order:.0f} % of the frames; "
              f"left > straight > right at {100 * ordered:.0f} %; mean |left - right| {spread:.3f}")
        kinds = np.array(listen_kind)
        for real in MANEUVERS:
            mask = kinds == real
            if mask.any():
                print(f"  where the route really goes {real:8s} ({mask.sum():4d}): left {left[mask].mean():+.3f}  "
                      f"straight {straight[mask].mean():+.3f}  right {right[mask].mean():+.3f}")
        result["listen"] = {"frames": len(listen_kind), "left_beyond_right": order, "ordered": ordered,
                            "mean_abs_left_minus_right": spread,
                            "mean_steer": {m: float(np.mean(listen[m])) for m in MANEUVERS}}
        print("  (a model that ignores the sentence: 50 % and |left - right| near 0; the lane may still give "
              "the turn away in Town03 and Town05, docs/carla.md)")

    if sample is not None:
        image, steer, noise = sample
        print("\n[speed] one lane-following frame, the same noise: first throttle at speed")
        result["speed"] = {}
        for v in (0.0, 3.0, 6.0, 9.0):
            throttle = float(chunk(backend, image, v, steer, PHRASES["follow"][0], noise)[0, 1])
            print(f"  {v:3.0f} m/s -> throttle {throttle:+.3f}")
            result["speed"][str(v)] = throttle
        import torch
        chunk(backend, image, 5.0, steer, PHRASES["follow"][0], noise)
        synchronize(torch, backend.device)
        t0 = time.perf_counter()
        for _ in range(10):
            chunk(backend, image, 5.0, steer, PHRASES["follow"][0], noise)
        synchronize(torch, backend.device)
        ms = 1000 * (time.perf_counter() - t0) / 10
        print(f"\n[timing] one chunk {ms:.0f} ms on {backend.device}")
        result["timing_ms"] = ms
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"numbers written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
