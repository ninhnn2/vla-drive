"""train: behaviour cloning. Fit the network to the labelled plans; the model goes to
<project>/models/<version>.pt, with a report beside it as <version>.json.

    ./drive.sh train v1 [--epochs 15]

PyTorch is imported inside the functions that use it, so every other command runs
without it.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

from . import dataset as datasets
from .project import Project, add_project_argument


def augment(images, states, plans, generator):
    """Mirror half the batch (a left bend becomes a right bend) and vary the brightness."""
    import torch
    flip = torch.rand(len(images), device=images.device, generator=generator) < 0.5
    images = torch.where(flip[:, None, None, None], images.flip(2), images)
    sign = torch.where(flip, -1.0, 1.0)
    states = states * torch.stack([torch.ones_like(sign), sign], dim=1)             # steer changes sign
    plans = plans * torch.stack([sign, torch.ones_like(sign)], dim=1)[:, None, :]
    gain = 0.8 + 0.4 * torch.rand(len(images), 1, 1, 1, device=images.device, generator=generator)
    return (images.float() * gain).clamp(0, 255), states, plans


def pick_device(log=print) -> str:
    """cuda when it works, else mps (a Mac's GPU), else cpu, saying why when an NVIDIA GPU is
    there but will not start."""
    import torch
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if torch.cuda.is_available():
            return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if any("unknown error" in str(w.message) for w in caught):
        log("CUDA would not start. After a suspend the driver's UVM module needs reloading:\n"
            "  sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm\ntraining on the CPU instead")
    return "cpu"


def train(data: Path, out: Path, epochs: int = 15, batch: int = 128, lr: float = 1e-3,
          device: str | None = None, seed: int = 0, log=print) -> dict:
    """Train on the episodes under `data`, save the model to `out`, return the report."""
    import torch
    from .model import ChunkNet, save

    device = device or pick_device(log)
    torch.manual_seed(seed)
    started = time.perf_counter()
    training, validation = datasets.load(data).split()
    log(f"{len(training)} training frames, {len(validation)} validation frames, on {device}")

    def tensors(d):
        return (torch.from_numpy(d.images).to(device), torch.from_numpy(d.states).to(device),
                torch.from_numpy(d.plans).to(device))

    images, states, plans = tensors(training)
    held = tensors(validation)
    model = ChunkNet(plans.shape[1], images.shape[1:3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.05)
    generator = torch.Generator(device=device).manual_seed(seed)

    def check() -> tuple[float, float]:
        """(loss, mean |steer error|) on the held-out episodes."""
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(held[0][i:i + 512], held[1][i:i + 512]) for i in range(0, len(held[0]), 512)])
        return float(loss_fn(out, held[2])), float((out[..., 0] - held[2][..., 0]).abs().mean())

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(images), device=device, generator=generator)
        total = 0.0
        for i in range(0, len(order), batch):
            pick = order[i:i + batch]
            x, s, y = augment(images[pick], states[pick], plans[pick], generator)
            loss = loss_fn(model(x, s), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(pick)
        schedule.step()
        val_loss, steer_error = check()
        history.append({"epoch": epoch, "train_loss": round(total / len(order), 5),
                        "val_loss": round(val_loss, 5), "val_steer_error": round(steer_error, 4)})
        log(f"  epoch {epoch:2d}/{epochs}: train {history[-1]['train_loss']:.5f}   "
            f"validation {val_loss:.5f}   steer error {steer_error:.4f}")

    report = {"frames": len(training), "validation_frames": len(validation), "epochs": epochs,
              "parameters": sum(p.numel() for p in model.parameters()), "device": device,
              "seconds": round(time.perf_counter() - started, 1), "history": history}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, model, training.plan_dt, report)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def add_parser(sub):
    p = sub.add_parser("train", help="train a network on the labelled episodes -> models/<version>.pt")
    add_project_argument(p)
    p.add_argument("version", help="a name for the model, e.g. v1; evaluate and sim refer to it by this")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--device", help="cuda, mps or cpu (default: cuda when it works, else a Mac's GPU)")
    p.add_argument("--seed", type=int, default=0)


def run(project: Project, args) -> int:
    try:
        import torch.nn  # noqa: F401
    except ImportError as error:
        raise ValueError(f"training needs PyTorch ({error}); make setup (a .venv, on a Mac), or use the container: "
                         "make train") from error
    out = project.model(args.version)
    if out.exists():
        raise ValueError(f"{out} exists; a trained model is not replaced, choose another version")
    report = train(project.episodes, out, args.epochs, device=args.device, seed=args.seed)
    print(f"{report['parameters']:,} parameters trained in {report['seconds']} s; "
          f"validation steer error {report['history'][-1]['val_steer_error']}\n  model: {out}")
    return 0
