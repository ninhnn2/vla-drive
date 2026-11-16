"""Inference off the control loop: submit an observation, collect the chunk later.

Both workers run one inference at a time. The loop submits a fresh observation as soon
as the worker is free, so the policy runs back to back at its own rate.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..policy.base import Policy
from ..types import ActionChunk, Observation


class SimulatedWorker:
    """Deterministic latency on the simulator's clock, for experiments and tests."""

    def __init__(self, policy: Policy, latency_s: float = 0.1, jitter_s: float = 0.0, seed: int = 0):
        self.policy, self.latency_s, self.jitter_s = policy, latency_s, jitter_s
        self.rng = np.random.default_rng(seed)
        self._pending: tuple[float, ActionChunk] | None = None
        self.latencies_s: list[float] = []

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def submit(self, obs: Observation, now: float) -> None:
        latency = max(self.latency_s + self.jitter_s * float(self.rng.standard_normal()), 0.0)
        self.latencies_s.append(latency)
        self._pending = (now + latency, self.policy.infer(obs))

    def poll(self, now: float) -> ActionChunk | None:
        if self._pending is None or now < self._pending[0] - 1e-9:
            return None
        chunk, self._pending = self._pending[1], None
        return chunk

    def close(self) -> None: ...


class ThreadedWorker:
    """Real inference on a background thread: what runs on the car.

    `extra_latency_s` sleeps after each inference, to feel a slow model on a fast PC.
    A policy error is re-raised in the control loop by the next `poll`.
    """

    def __init__(self, policy: Policy, extra_latency_s: float = 0.0):
        self.policy, self.extra_latency_s = policy, extra_latency_s
        self.latencies_s: list[float] = []
        self._request: Observation | None = None
        self._result: ActionChunk | BaseException | None = None
        self._busy = False
        self._closed = False
        self._wake = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="policy-inference", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, obs: Observation, now: float) -> None:
        with self._wake:
            self._request, self._busy = obs, True
            self._wake.notify()

    def poll(self, now: float) -> ActionChunk | None:
        with self._wake:
            result, self._result = self._result, None
            if result is not None:
                self._busy = False
        if isinstance(result, BaseException):
            raise result
        return result

    def _run(self) -> None:
        while True:
            with self._wake:
                self._wake.wait_for(lambda: self._request is not None or self._closed)
                if self._closed:
                    return
                obs, self._request = self._request, None
            started = time.perf_counter()
            try:
                result = self.policy.infer(obs)
                time.sleep(self.extra_latency_s)
            except BaseException as error:        # surfaced by poll()
                result = error
            self.latencies_s.append(time.perf_counter() - started)
            with self._wake:
                self._result = result

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._wake.notify()
        self._thread.join(timeout=5)
