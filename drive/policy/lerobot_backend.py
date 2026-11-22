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


def unknown_config_fields(raw: dict, config_class) -> list[str]:
    """The keys of a checkpoint's config.json that this LeRobot's config class does not have."""
    import dataclasses
    return sorted(set(raw) - {field.name for field in dataclasses.fields(config_class)} - {"type"})


def load_config(checkpoint: Path, log=print):
    """The checkpoint's policy config, also when a newer LeRobot wrote it.

    Lab 15 fine-tuned with a LeRobot checkout newer than 0.5.1, the release `make vla-setup`
    installs, and that checkout writes fields 0.5.1 refuses (`pretrained_revision`). Such
    fields are dropped, and named, instead of failing to load the checkpoint at all.
    """
    import tempfile
    from lerobot.configs.policies import PreTrainedConfig
    raw = json.loads((Path(checkpoint) / "config.json").read_text())
    unknown = unknown_config_fields(raw, PreTrainedConfig.get_choice_class(raw["type"]))
    if not unknown:
        return PreTrainedConfig.from_pretrained(checkpoint)
    log(f"note: {checkpoint} was written by a newer LeRobot; ignoring the config fields it alone has: "
        f"{', '.join(f'{name}={raw[name]!r}' for name in unknown)}")
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text(json.dumps({k: v for k, v in raw.items() if k not in unknown}))
        return PreTrainedConfig.from_pretrained(directory)


def state_width(checkpoint: Path) -> int | None:
    """How many numbers of state the checkpoint was trained on, from its normalizer's statistics.
    Not config.json: fine-tuned from smolvla_base, its input_features stay the base's (an arm, 6).
    None: no statistics to say (the state is passed whole)."""
    for path in sorted(Path(checkpoint).glob("policy_preprocessor_step_*_normalizer_processor.safetensors")):
        from safetensors.numpy import load_file
        stats = load_file(str(path))
        if "observation.state.mean" in stats:
            return int(stats["observation.state.mean"].shape[0])
    return None


def fit_state(state: np.ndarray, width: int | None) -> np.ndarray:
    """The first `width` of the state `[speed_mps, steer]`: what the checkpoint was trained on.
    `drive export --state speed_mps` trains a model on the speed alone (drive/learn/export.py)."""
    width = len(state) if width is None else width
    if width > len(state):
        raise ValueError(f"the checkpoint wants a state of {width}, the observation has {len(state)}")
    return np.ascontiguousarray(state[:width], dtype=np.float32)


class LeRobotBackend:
    """`infer(observation)` -> an object whose `.chunk` is `(chunk_size, action_dim)`, in robot units."""

    def __init__(self, checkpoint: Path, device: str | None = None):
        try:
            import torch
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
        config = load_config(checkpoint)
        config.device = self.device
        self.policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
        self.policy.to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config, str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}})
        self.config = self.policy.config
        self.camera = self._camera_key()
        self.state_width = state_width(checkpoint)

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
        state = torch.from_numpy(fit_state(obs.state, self.state_width))
        batch = self.preprocess({self.camera: image, "observation.state": state,
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
