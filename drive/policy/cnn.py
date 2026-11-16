"""A policy learned from the teacher: `./drive.sh collect`, then `./drive.sh train`.

It sees what a real car would have: one camera frame, its speed and its steering.
"""
from __future__ import annotations

from pathlib import Path

from ..types import ActionChunk, Observation
from .base import Policy


class CnnPolicy(Policy):
    name = "cnn"

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        try:
            from ..learn import model
        except ImportError as error:
            raise ValueError(f"the cnn policy needs PyTorch ({error}); use the container: make shell") from error
        if not Path(checkpoint).is_file():
            raise ValueError(f"no checkpoint at {checkpoint}; train one with ./drive.sh train")
        self._predict = model.predict
        self.model, self.info = model.load(checkpoint, device)

    def infer(self, obs: Observation) -> ActionChunk:
        from ..learn.dataset import shrink
        image = shrink(obs.images[0], self.info["image_hw"])
        return ActionChunk(self._predict(self.model, image, obs.speed, obs.steer), self.info["plan_dt"], obs.t)
