"""A LeRobot policy checkpoint (a fine-tuned SmolVLA) as a backend for `BackendPolicy`.

This runs the PyTorch policy on the PC: the step before exporting to TensorRT for the
Jetson. It answers "can the fine-tuned model drive?" in the simulator, with the same
`BackendPolicy`, executor and supervisor the Jetson would use.

    ./drive.sh sim --policy vla --checkpoint smolvla-v1        (in an environment with lerobot)

The checkpoint is a directory written by `lerobot-train` (`.../checkpoints/last/pretrained_model`)
that holds the weights, the policy config and the pre/post-processors with the dataset's
normalisation statistics. LeRobot is imported only here.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def chunk_dt_s(checkpoint: Path, default: float = 0.1) -> float:
    """Seconds between the model's actions: 1 / fps of the dataset it was fine-tuned on."""
    for config in (Path(checkpoint) / "train_config.json", Path(checkpoint).parent / "train_config.json"):
        if config.is_file():
            root = json.loads(config.read_text()).get("dataset", {}).get("root")
            info = Path(root or "") / "meta" / "info.json"
            if root and info.is_file():
                return 1.0 / float(json.loads(info.read_text())["fps"])
    return default


def pick_device(torch) -> str:
    """$VLA_DEVICE if set, else cuda, else mps (a Mac's GPU), else cpu."""
    if os.environ.get("VLA_DEVICE"):
        return os.environ["VLA_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def synchronize(torch, device: str) -> None:
    """Wait for the GPU to finish, so that a timing covers the work and not just its launch."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class LeRobotBackend:
    """`infer(observation)` -> an object whose `.chunk` is `(chunk_size, action_dim)`, in robot units."""

    def __init__(self, checkpoint: Path, device: str | None = None):
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        except ImportError as error:
            raise ValueError(f"the vla policy needs LeRobot ({error}); run it in an environment that has "
                             "lerobot installed (docs/study/13-vla-and-smolvla.md)") from error
        checkpoint = Path(checkpoint)
        if not (checkpoint / "config.json").is_file():
            raise ValueError(f"{checkpoint} is not a LeRobot checkpoint (no config.json); "
                             "point at .../checkpoints/last/pretrained_model")
        self.torch = torch
        self.device = device or pick_device(torch)
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = self.device
        self.policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
        self.policy.to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config, str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}})
        self.config = self.policy.config
        self.camera = self._camera_key()

    def _camera_key(self) -> str:
        """The key to put the camera image under, before the preprocessor renames it."""
        expected = list(self.config.image_features)
        for step in getattr(self.preprocess, "steps", []):
            for source, target in (getattr(step, "rename_map", None) or {}).items():
                if target in expected:
                    return source
        return expected[0]

    def infer(self, obs):
        torch, started = self.torch, time.perf_counter()
        image = torch.from_numpy(np.ascontiguousarray(obs.images[0])).permute(2, 0, 1).float() / 255.0
        batch = self.preprocess({self.camera: image, "observation.state": torch.from_numpy(obs.state),
                                 "task": obs.task})
        noise = torch.from_numpy(obs.noise[:, : self.config.chunk_size, : self.config.max_action_dim])
        actions = self.policy.predict_action_chunk(batch, noise=noise.to(self.device))
        actions = self.postprocess(actions)
        return SimpleNamespace(chunk=actions[0].float().cpu().numpy(),
                               timings_ms={"total": 1000 * (time.perf_counter() - started)})


def make_vla_policy(checkpoint: Path, seed: int = 0):
    """`BackendPolicy` around a LeRobot checkpoint, with its chunk length and time step."""
    from .vla import BackendPolicy
    backend = LeRobotBackend(checkpoint)
    return BackendPolicy(backend, chunk_size=backend.config.chunk_size, noise_width=backend.config.max_action_dim,
                         chunk_dt_s=chunk_dt_s(checkpoint), seed=seed)
