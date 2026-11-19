"""lrfind: the learning-rate range test (Smith 2015). Before a long training, find out which
learning rates make the loss fall, in under a minute.

    ./drive.sh lrfind [--project DIR] [--steps 150]

A fresh network is trained for `steps` batches while the learning rate grows exponentially
from 1e-6 to 1. Too small a rate: the loss stays put. Good rates: it falls fast. Too large:
it stops falling, then shoots up.

The loss alone can mislead: a rate that kills the last convolution's units still lets the
loss fall for a while, as the head learns the average plan with no picture at all (lesson 7
of docs/thuc-hanh). So the sweep also watches how many of the last convolution's channels
are silent on each batch, and the range it suggests ends where the picture starts to die.
Nothing is saved; `./drive.sh train --lr` takes the result. The sweep is short and noisy:
it narrows the search, short trainings decide.
"""
from __future__ import annotations

import math

import numpy as np

from . import dataset as datasets
from .project import Project, add_project_argument


def find_lr(data, start: float = 1e-6, end: float = 1.0, steps: int = 150, batch: int = 128, seed: int = 0,
            device: str | None = None, log=print) -> dict:
    """[(lr, smoothed loss)] along an exponential sweep, and the rates it suggests."""
    import torch
    from .model import ChunkNet
    from .train import augment, pick_device

    device = device or pick_device(log)
    torch.manual_seed(seed)
    training, _ = datasets.load(data).split()
    images, states, plans = (torch.from_numpy(a).to(device) for a in (training.images, training.states, training.plans))
    model = ChunkNet(plans.shape[1], images.shape[1:3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=start)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.05)
    generator = torch.Generator(device=device).manual_seed(seed)
    grow = (end / start) ** (1 / max(steps - 1, 1))
    swept, smooth, lowest = [], 0.0, math.inf
    silent = {}                                            # the last convolution's ReLU, on this batch
    eye = model.vision[-2].register_forward_hook(
        lambda m, i, out: silent.__setitem__("now", float((out.amax(dim=(0, 2, 3)) <= 0).float().mean())))
    log(f"{len(training)} training frames, on {device}: {steps} steps of {batch}, the learning rate x{grow:.3f} each")
    for k in range(steps):
        rate = start * grow ** k
        for group in optimizer.param_groups:
            group["lr"] = rate
        pick = torch.randint(0, len(images), (batch,), device=device, generator=generator)
        x, s, y = augment(images[pick], states[pick], plans[pick], generator)
        loss = loss_fn(model(x, s), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value = float(loss)
        if not math.isfinite(value):
            break
        smooth = 0.9 * smooth + 0.1 * value
        smoothed = smooth / (1 - 0.9 ** (k + 1))           # the average's start-up bias removed
        swept.append((rate, smoothed, silent["now"]))
        lowest = min(lowest, smoothed)
        if smoothed > 4 * lowest and k > 10:             # it has blown up: the rest says nothing new
            break
    eye.remove()
    rates, losses = np.array([r for r, _, _ in swept]), np.array([v for _, v, _ in swept])
    dead = np.array([d for _, _, d in swept])
    best = int(np.argmin(losses))
    # where the loss falls fastest per decade of learning rate, before its minimum
    slope = np.gradient(losses[:best + 1], np.log10(rates[:best + 1])) if best > 2 else np.zeros(1)
    steepest = int(np.argmin(slope))
    # the picture starts to die: the share of silent channels climbs 15 points above the lowest it
    # has been, and stays there for 5 steps. A full training spends thousands of steps at one rate,
    # the sweep three: what starts here finishes there (lesson 7, E15).
    dying = next((k for k in range(1, len(dead) - 5) if dead[k:k + 5].min() >= dead[:k].min() + 0.15), None)
    blind_at = float(rates[dying]) if dying is not None else None
    high = float(rates[best]) / 10
    if blind_at is not None:
        high = min(high, blind_at / 2)
    low = min(float(rates[steepest]), high)
    return {"swept": swept, "lowest_at": float(rates[best]), "steepest_at": float(rates[steepest]),
            "blind_at": blind_at, "range": (low, high)}


def add_parser(sub):
    p = sub.add_parser("lrfind", help="the learning-rate range test: which learning rates make the loss fall")
    add_project_argument(p)
    p.add_argument("--steps", type=int, default=150, help="batches in the sweep from 1e-6 to 1 (default 150)")
    p.add_argument("--device", help="cuda, mps or cpu (default: as train picks)")
    p.add_argument("--seed", type=int, default=0)


def run(project: Project, args) -> int:
    try:
        import torch.nn  # noqa: F401
    except ImportError as error:
        raise ValueError(f"lrfind needs PyTorch ({error}); make setup, or use the container") from error
    found = find_lr(project.episodes, steps=args.steps, device=args.device, seed=args.seed)
    swept = found["swept"]
    top = max(v for _, v, _ in swept)
    every = max(len(swept) // 30, 1)
    print(f"  {'learning rate':>13} {'loss':>8} {'vision.9 silent':>15}")
    for rate, loss, dead in swept[::every] + ([swept[-1]] if (len(swept) - 1) % every else []):
        print(f"  {rate:>13.1e} {loss:>8.4f} {100 * dead:>14.0f}%  " + "#" * max(1, round(40 * loss / top)))
    print(f"  the loss falls fastest at {found['steepest_at']:.1e} and is lowest at {found['lowest_at']:.1e}"
          + ("; it blew up after that" if len(swept) < args.steps else ""))
    if found["blind_at"] is not None:
        print(f"  from {found['blind_at']:.1e} more and more of the last convolution's channels go silent: the loss "
              "still falls there, but the network is going blind")
    low, high = found["range"]
    print(f"  worth trying: --lr between {low:.0e} and {high:.0e} (from the steepest fall, below where it goes blind "
          "and a decade below the lowest); a sweep of 150 steps is a hint, so confirm with short trainings")
    return 0
