"""Probe a fine-tuned LeRobot policy (smolvla-v1) without driving it: what does it output, and why?

    python tools/probe_vla.py [--checkpoint smolvla-v1] [--episodes 'sim2-03[1-9]']

Run it with the Python of an environment that has lerobot (docs/study/15-hands-on-labs.md,
Lab 15), from the repository root, with PYTHONPATH=. and HF_HOME set. Each probe prints
one block; docs/study/17-debugging-the-model.md explains how to read them.

    offline     per chunk step, the model's steering against the teacher's clean plan and
                against the target LeRobot trained it on, on episodes it did not train on
    noise       the same observation with different starting noise: how much does the chunk move?
    repeat      the same observation and the same noise twice: is it deterministic?
    mirror      the image flipped left-right and the steering negated: does the steering flip?
    blank       black and grey images: what does it do when it sees nothing?
    speed       the same image at other speeds: does throttle respond to the state?
    task        other task sentences: does the text change anything?
    timing      one chunk with 1 and with 10 denoising steps: where the time goes
    shapes      the tensors entering the network
"""
from __future__ import annotations

import argparse
import time
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

from drive.learn.label import episodes, read_labels
from drive.learn.project import Project
from drive.policy.lerobot_backend import LeRobotBackend, synchronize
from drive.policy.vla import BackendObservation


def observation(backend, image, speed, steer, task, noise):
    return BackendObservation(index=0, images=[image], state=np.array([speed, steer], np.float32),
                              task=task, noise=noise)


def chunk(backend, image, speed, steer, task="drive along the track", noise=None, rng=None):
    if noise is None:
        noise = rng.standard_normal((1, backend.config.chunk_size, backend.config.max_action_dim)).astype(np.float32)
    return backend.infer(observation(backend, image, speed, steer, task, noise)).chunk[:, :2]


def shown_steps(steps: int) -> list[int]:
    """The chunk steps a probe prints: 0, 1, 2, 3, 5, 10, 15 for smolvla-v1's 16, fewer or more for other chunks."""
    return [k for k in (0, 1, 2, 3, 5, 10, 15, 25, 49) if k < steps]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("runs/drive"))
    parser.add_argument("--checkpoint", default="smolvla-v1", help="a version in <project>/vla, or a checkpoint directory")
    parser.add_argument("--episodes", default="sim2-03[1-9]",
                        help="held-out episodes (LeRobot's eval_split=0.1 holds out the last 9: sim2-031 … sim2-039)")
    parser.add_argument("--every", type=int, default=10, help="use every Nth frame of each episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="cuda, mps or cpu (default: the fastest there is; or $VLA_DEVICE)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    checkpoint = Project(args.project).checkpoint(args.checkpoint)
    started = time.perf_counter()
    backend = LeRobotBackend(checkpoint, args.device)
    print(f"loaded {checkpoint} on {backend.device} in {time.perf_counter() - started:.1f} s; chunk {backend.config.chunk_size}, "
          f"{backend.config.num_steps} denoising steps, camera key {backend.camera!r}")

    found = [e for e in episodes(args.project / "episodes") if fnmatch(e.name, args.episodes)]
    if not found:
        raise SystemExit(f"no episodes named {args.episodes!r}")
    steps = backend.config.chunk_size
    vs_plan, vs_target, firsts = [[] for _ in range(steps)], [[] for _ in range(steps)], []
    sample = None
    for episode in found:
        data, labels = np.load(episode / "episode.npz"), read_labels(episode)
        plans, n = labels["plans"], len(data["t"])
        for i in range(0, n - steps, args.every):
            a = chunk(backend, data["images"][i], *data["states"][i], rng=rng)
            for k in range(steps):
                if k < plans.shape[1]:                                      # a plan has 16 steps, a chunk any number
                    vs_plan[k].append(abs(a[k, 0] - plans[i, k, 0]))       # the teacher's clean plan from frame i
                vs_target[k].append(abs(a[k, 0] - plans[i + k, 0, 0]))     # what LeRobot trained it to output
            firsts.append((a[0, 0], plans[i, 0, 0]))
            sample = sample or (data["images"][i], *data["states"][i], plans[i])
    if not firsts:
        raise SystemExit(f"no frame to probe: the episodes need more than {steps} frames (the chunk size)")
    shown = shown_steps(steps)
    mean = lambda errors: f"{np.mean(errors):5.3f}" if errors else "    -"
    print(f"\n[offline] {len(firsts)} frames from {len(found)} held-out episodes: mean |steer error| by chunk step")
    print("  step:          " + " ".join(f"{k:5d}" for k in shown))
    print("  vs clean plan: " + " ".join(mean(vs_plan[k]) for k in shown))
    print("  vs LeRobot's target: " + " ".join(mean(vs_target[k]) for k in shown))
    model, teacher = np.array(firsts).T
    print(f"  step 0: correlation with the teacher {np.corrcoef(model, teacher)[0, 1]:.3f}; "
          f"mean steer model {model.mean():+.3f}, teacher {teacher.mean():+.3f}")

    image, speed, steer, plan = sample
    runs = np.array([chunk(backend, image, speed, steer, rng=rng) for _ in range(10)])
    spread = runs[:, :, 0].std(axis=0)
    print(f"\n[noise] one frame, 10 different noises: std of steering by step "
          f"{' '.join(f'{spread[k]:.3f}' for k in shown if k != 2 and k != 3)}; "
          f"std of throttle at step 0 {runs[:, 0, 1].std():.3f}")

    fixed = rng.standard_normal((1, steps, backend.config.max_action_dim)).astype(np.float32)
    once, again = chunk(backend, image, speed, steer, noise=fixed), chunk(backend, image, speed, steer, noise=fixed)
    print(f"\n[repeat] same frame, same noise, twice: max difference {np.abs(once - again).max():.2e}")

    flipped = chunk(backend, image[:, ::-1].copy(), speed, -steer, noise=fixed)
    print(f"\n[mirror] steer step 0: original {once[0, 0]:+.3f}, mirrored {flipped[0, 0]:+.3f} "
          f"(a symmetric policy gives {-once[0, 0]:+.3f}); mean |orig + mirrored| over the chunk "
          f"{np.abs(once[:, 0] + flipped[:, 0]).mean():.3f}")

    for name, blank in (("black", np.zeros_like(image)), ("grey", np.full_like(image, 128))):
        b = chunk(backend, blank, speed, steer, noise=fixed)
        print(f"[blank] {name} image: steer {b[0, 0]:+.3f} … {b[-1, 0]:+.3f}, throttle {b[0, 1]:+.3f}")

    print(f"\n[speed] same frame, steer {steer:+.2f}: first throttle at speed")
    for v in (0.0, 0.5, 1.0, 1.4, 1.8, 2.5):
        print(f"  {v:3.1f} m/s -> throttle {chunk(backend, image, v, steer, noise=fixed)[0, 1]:+.3f}")

    print("\n[task] same frame, same noise, other sentences: mean |steer change| over the chunk")
    for task in ("drive slowly", "stop", "turn left", ""):
        other = chunk(backend, image, speed, steer, task=task, noise=fixed)
        print(f"  {task!r:24s} {np.abs(other[:, 0] - once[:, 0]).mean():.3f}   throttle {other[0, 1]:+.3f}")

    import torch
    timings = {}
    for n in (1, 10):
        backend.policy.config.num_steps = n
        chunk(backend, image, speed, steer, noise=fixed)                       # warm up
        synchronize(torch, backend.device)
        t0 = time.perf_counter()
        for _ in range(10):
            chunk(backend, image, speed, steer, noise=fixed)
        synchronize(torch, backend.device)
        timings[n] = (time.perf_counter() - t0) / 10
    backend.policy.config.num_steps = 10
    per_step = (timings[10] - timings[1]) / 9
    print(f"\n[timing] one chunk: {1000 * timings[1]:.0f} ms with 1 step, {1000 * timings[10]:.0f} ms with 10 "
          f"→ about {1000 * per_step:.1f} ms per denoising step, {1000 * (timings[1] - per_step):.0f} ms for "
          "everything else (pre-processing, image encoder, VLM prefix, post-processing)")

    model = backend.policy.model
    original = model.embed_prefix

    def traced(images, img_masks, lang_tokens, lang_masks, state=None):
        print("\n[shapes] into embed_prefix:")
        for i, img in enumerate(images):
            print(f"  images[{i}] {tuple(img.shape)} {img.dtype}  range [{img.min().item():.2f}, {img.max().item():.2f}]"
                  f"  present {img_masks[i].tolist()}")
        print(f"  lang_tokens {tuple(lang_tokens.shape)}, {int(lang_masks.sum())} real tokens")
        print(f"  state {tuple(state.shape)}: {np.round(state[0, :4].float().cpu().numpy(), 3).tolist()} …")
        out = original(images, img_masks, lang_tokens, lang_masks, state=state)
        print(f"  -> prefix embeddings {tuple(out[0].shape)}")
        return out

    model.embed_prefix = traced
    chunk(backend, image, speed, steer, noise=fixed)
    model.embed_prefix = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
