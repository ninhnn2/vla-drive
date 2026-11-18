"""Labelled episodes as arrays ready for training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .label import episodes, read_labels

IMAGE_HW = (96, 128)        # what the network sees; the camera gives 240x320


def shrink(image: np.ndarray, hw=IMAGE_HW) -> np.ndarray:
    """The network's input size. Used for training and at inference: keep them identical."""
    return cv2.resize(image, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


@dataclass
class Dataset:
    images: np.ndarray       # uint8 (N, H, W, 3)
    states: np.ndarray       # float32 (N, 2): speed m/s, steer
    plans: np.ndarray        # float32 (N, steps, 2)
    episode: np.ndarray      # int (N,): which episode each frame is from
    plan_dt: float

    def __len__(self) -> int:
        return len(self.images)

    def split(self, validation_fraction: float = 0.1):
        """(train, validation) by whole episodes: frames 0.1 s apart are near-duplicates."""
        mask = np.isin(self.episode, held_out(np.unique(self.episode), validation_fraction))
        if mask.all():
            raise ValueError("need at least two episodes to hold one out for validation")
        return self._take(~mask), self._take(mask)

    def _take(self, mask) -> "Dataset":
        return Dataset(self.images[mask], self.states[mask], self.plans[mask], self.episode[mask], self.plan_dt)


def held_out(ids: np.ndarray, validation_fraction: float = 0.1) -> np.ndarray:
    """The episodes training holds out for validation: the last tenth, at least one."""
    return ids[-max(int(round(len(ids) * validation_fraction)), 1):]


def split_plans(directory: Path, validation_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """(training plans, validation plans), split as `Dataset.split` does, without loading a single image."""
    found = episodes(directory)
    plans = [labels["plans"][labels["valid"]] for labels in map(read_labels, found) if labels is not None]
    if len(plans) != len(found) or len(found) < 2:
        raise ValueError(f"need two or more labelled episodes under {directory}")
    held = set(held_out(np.arange(len(found)), validation_fraction).tolist())
    return (np.concatenate([p for i, p in enumerate(plans) if i not in held]),
            np.concatenate([p for i, p in enumerate(plans) if i in held]))


def load(directory: Path, hw=IMAGE_HW) -> Dataset:
    """Every labelled, valid frame under `directory`."""
    found = episodes(directory)
    if not found:
        raise ValueError(f"no episodes under {directory}; record some with ./drive.sh collect")
    images, states, plans, episode, steps = [], [], [], [], set()
    for index, path in enumerate(found):
        labels = read_labels(path)
        if labels is None:
            raise ValueError(f"{path} is not labelled; run ./drive.sh label")
        data, valid = np.load(path / "episode.npz"), labels["valid"]
        images.append(np.stack([shrink(image, hw) for image in data["images"][valid]]))
        states.append(data["states"][valid])
        plans.append(labels["plans"][valid])
        episode.append(np.full(int(valid.sum()), index))
        steps.add((labels["plans"].shape[1], labels["dt"]))
    if len(steps) != 1:
        raise ValueError(f"episodes disagree on the plan's (length, time step): {sorted(steps)}")
    return Dataset(np.concatenate(images), np.concatenate(states), np.concatenate(plans),
                   np.concatenate(episode), steps.pop()[1])
