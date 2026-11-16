"""A VLA on the Jetson as a driving policy.

`backend` is a loaded TensorRT backend from jetson-orin-nano-vla
(https://github.com/eetmie/jetson-orin-nano-vla), for example its
`TrtSplitSmolVLABackend`, built from a bundle whose checkpoint was fine-tuned to
output `[steer, throttle]` from the state `[speed, steer]`. That repository is not a
dependency: a backend is anything with `infer(observation)` returning an object whose
`.chunk` is the (chunk_size, action_width) array of actions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import ACTION_DIM, ActionChunk, Observation
from .base import Policy


@dataclass
class BackendObservation:
    """One timestep in the fields a jetson-orin-nano-vla backend reads (its bench.obs.Observation)."""

    index: int
    images: list[np.ndarray]   # uint8 HxWx3 each, in camera order
    state: np.ndarray          # float32, raw units
    task: str
    noise: np.ndarray          # float32 (1, chunk_size, action_width): where denoising starts


class BackendPolicy(Policy):
    name = "vla"

    def __init__(self, backend, *, chunk_size: int, noise_width: int, chunk_dt_s: float, seed: int = 0):
        self.backend = backend
        self.noise_shape = (1, chunk_size, noise_width)
        self.chunk_dt = chunk_dt_s
        self.rng = np.random.default_rng(seed)
        self.index = 0

    def infer(self, obs: Observation) -> ActionChunk:
        result = self.backend.infer(BackendObservation(
            index=self.index, images=obs.images, state=np.array([obs.speed, obs.steer], np.float32),
            task=obs.task, noise=self.rng.standard_normal(self.noise_shape).astype(np.float32)))
        self.index += 1
        actions = np.asarray(result.chunk, dtype=np.float64)[:, :ACTION_DIM]
        return ActionChunk(np.clip(actions, -1.0, 1.0), self.chunk_dt, obs.t)
