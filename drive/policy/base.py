from __future__ import annotations

from ..types import ActionChunk, Observation


class Policy:
    """One inference: an observation in, a plan for the next second or so out.

    `infer` may be slow (a VLA takes ~100 ms on the Orin Nano) and is called from the
    inference worker, never from the control loop.
    """

    name = "policy"

    def infer(self, obs: Observation) -> ActionChunk:
        raise NotImplementedError
