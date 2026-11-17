"""Turn a stream of action chunks into one action per control tick."""
from __future__ import annotations

import math

import numpy as np

from ..types import ActionChunk


class ChunkExecutor:
    """Plays the newest chunk, fading it in over the previous one.

    A chunk describes the future as seen at its observation, and arrives one inference
    later. With `latency_compensation` it is read at `now - t_obs`, so the actions meant
    for the time spent computing are skipped; without, it is read from its first action
    on arrival and the car acts on a plan that is one latency out of date.
    """

    def __init__(self, blend_s: float = 0.10, latency_compensation: bool = True):
        self.blend_s = blend_s
        self.latency_compensation = latency_compensation
        self.chunk: ActionChunk | None = None
        self._origin = 0.0                     # the time chunk.actions[0] is played at
        self._previous: tuple[ActionChunk, float] | None = None
        self._swapped = 0.0

    def update(self, chunk: ActionChunk, now: float) -> None:
        if self.chunk is not None:
            self._previous = (self.chunk, self._origin)
        self.chunk, self._swapped = chunk, now
        self._origin = chunk.t_obs if self.latency_compensation else now

    def action(self, now: float) -> np.ndarray | None:
        """The action for `now`, or None when there is no plan or it has run out."""
        if self.chunk is None:
            return None
        action = self.chunk.sample(now - self._origin)
        fade = (now - self._swapped) / self.blend_s if self.blend_s > 0 else 1.0
        if action is not None and fade < 1.0 and self._previous is not None:
            old = self._previous[0].sample(now - self._previous[1])
            if old is not None:
                action = (1.0 - fade) * old + fade * action
        return action

    @property
    def origin(self) -> float:
        """The time `chunk.actions[0]` is played at: its observation's time, or its arrival without compensation."""
        return self._origin

    @property
    def arrived(self) -> float:
        """When the current chunk arrived."""
        return self._swapped

    def age_s(self, now: float) -> float:
        """Seconds since the observation the current plan was made from."""
        return math.inf if self.chunk is None else now - self.chunk.t_obs
