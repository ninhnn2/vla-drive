"""Is this the environment smolvla-v1 works in? Versions, the GPU, and the model on three known frames.

    python tools/check_env.py [--checkpoint smolvla-v1] [--device cuda|cpu] [--tolerance 0.01]

Run it with the Python of the LeRobot environment (the container of docker/vla, or a local one),
from the repository root with PYTHONPATH=. and HF_HOME set. Exit status 0 means it passed.

1. versions   each package against docker/vla/requirements.lock: a different torch, transformers
              or lerobot can change the actions without an error
2. GPU        what torch sees
3. the model  the three frames jetson/smolvla_trt/export.py saved with what LeRobot computed on the
              PC it was trained on (<checkpoint>-trt/samples): the same image, state, sentence and
              starting noise must give the same chunk. bf16 on another GPU or on the CPU rounds
              differently, so a few thousandths of steering is expected there; tenths are not.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "docker" / "vla" / "requirements.lock"
CHECKED = ("torch", "torchvision", "transformers", "lerobot", "numpy", "opencv-python-headless", "av", "safetensors",
           "tokenizers", "accelerate", "huggingface_hub", "onnx", "onnxruntime-gpu", "gradio")
TASK = "drive along the track"                 # what export.py ran the samples with (its ROOT_TASK)


def locked() -> dict[str, str]:
    """name -> version from the lock file; lerobot as its commit."""
    pins = {}
    if not LOCK.is_file():
        return pins
    for line in LOCK.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.lower().replace("_", "-")] = version.strip()
        elif line.startswith("lerobot"):
            pins["lerobot"] = "0.5.2"
    return pins


def versions() -> int:
    pins = locked()
    print(f"python {platform.python_version()} ({sys.executable})")
    differ = 0
    for name in CHECKED:
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        want = pins.get(name.lower().replace("_", "-"))
        mark = "  " if have == want or want is None else "!!"
        differ += mark == "!!"
        print(f"  {mark} {name:24s} {have or 'missing':14s}" + (f" lock {want}" if want and have != want else ""))
    if not pins:
        print(f"  (no {LOCK.relative_to(ROOT)} to compare with)")
    return differ


def gpu(torch) -> None:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        free, total = (v / 2**30 for v in torch.cuda.mem_get_info(0))
        print(f"GPU  {name}, {total:.1f} GB ({free:.1f} free), torch CUDA {torch.version.cuda}")
    else:
        print("GPU  none: torch.cuda.is_available() is False. In Docker that needs the NVIDIA container "
              "toolkit on the host (docker/vla/README.md)")


def samples(directory: Path) -> list[dict]:
    import cv2

    out = []
    for line in (directory / "samples.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, state, _ = line.split("\t")
        stem = directory / name
        out.append(dict(
            name=name,
            image=cv2.cvtColor(cv2.imread(str(stem) + ".png"), cv2.COLOR_BGR2RGB),
            state=np.array(state.split(), np.float32),
            noise=np.fromfile(f"{stem}.noise.f32", np.float32).reshape(1, 16, -1),
            bf16=np.fromfile(f"{stem}.bf16.f32", np.float32).reshape(16, 2),
        ))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("runs/drive"))
    parser.add_argument("--checkpoint", default="smolvla-v1")
    parser.add_argument("--samples", type=Path, help="default <project>/vla/<checkpoint>-trt/samples")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if there is one)")
    parser.add_argument("--tolerance", type=float, default=0.01, help="largest |difference| in any action that passes")
    args = parser.parse_args()

    print("1. versions")
    differ = versions()
    import torch

    print("2. GPU")
    gpu(torch)

    print("3. the model on known frames")
    from drive.learn.project import Project
    from drive.policy.lerobot_backend import LeRobotBackend
    from drive.policy.vla import BackendObservation

    project = Project(args.project)
    checkpoint = project.checkpoint(args.checkpoint)
    directory = args.samples or project.home / "vla" / f"{Path(args.checkpoint).name}-trt" / "samples"
    if not (directory / "samples.txt").is_file():
        print(f"  no reference samples in {directory}: run jetson/smolvla_trt/export.py once where the model works")
        return 1
    started = time.perf_counter()
    backend = LeRobotBackend(checkpoint, args.device)
    print(f"  loaded {checkpoint} on {backend.device} in {time.perf_counter() - started:.1f} s")
    worst = 0.0
    for s in samples(directory):
        started = time.perf_counter()
        chunk = backend.infer(BackendObservation(0, [s["image"]], s["state"], TASK, s["noise"])).chunk
        ms = 1000 * (time.perf_counter() - started)
        diff = float(np.abs(chunk - s["bf16"]).max())
        worst = max(worst, diff)
        print(f"  {s['name']:24s} steering {chunk[0, 0]:+.4f} (reference {s['bf16'][0, 0]:+.4f})  "
              f"throttle {chunk[0, 1]:+.4f} ({s['bf16'][0, 1]:+.4f})  max |diff| {diff:.2e}  {ms:.0f} ms")
    passed = worst <= args.tolerance
    print(f"\n{'PASS' if passed else 'FAIL'}: largest difference {worst:.2e} (tolerance {args.tolerance})"
          + (f"; {differ} package version(s) differ from the lock" if differ else ""))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
