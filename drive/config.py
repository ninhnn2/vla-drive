"""Every tunable number, in one place."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleParams:
    """A 1:10 RC car."""

    wheelbase_m: float = 0.26
    width_m: float = 0.19
    length_m: float = 0.40
    max_steer_rad: float = 0.42
    max_speed_mps: float = 3.0
    max_accel_mps2: float = 2.5
    max_brake_mps2: float = 6.0
    steer_tau_s: float = 0.08      # first-order servo lag
    drag_per_s: float = 0.3


@dataclass(frozen=True)
class CameraParams:
    width: int = 320
    height: int = 240
    hfov_deg: float = 110.0
    height_m: float = 0.18
    pitch_down_deg: float = 15.0
    forward_m: float = 0.10        # ahead of the rear axle
    far_m: float = 12.0            # ground beyond this renders as sky


@dataclass(frozen=True)
class RangeSensorParams:
    """A forward time-of-flight sensor: the supervisor's own eyes."""

    half_fov_deg: float = 15.0
    max_range_m: float = 4.0


@dataclass(frozen=True)
class SafetyLimits:
    max_speed_mps: float = 2.0
    max_steer_rate_per_s: float = 8.0   # full lock to full lock in 0.25 s
    #: Brake when the plan in use was made from an observation older than this, or has
    #: run out. A plan's age swings between one and two inference latencies, so this
    #: must exceed twice the latency of the model you run.
    max_plan_age_s: float = 0.6
    #: Deceleration the supervisor plans its stops with; below what the brakes can do.
    planned_decel_mps2: float = 4.0
    stop_margin_m: float = 0.30


@dataclass(frozen=True)
class LoopConfig:
    control_hz: float = 50.0
    task: str = "drive along the track"
    #: Seconds over which a new chunk fades in over the previous one.
    blend_s: float = 0.10
    #: Index a chunk by the time of its observation, skipping the actions that went
    #: stale while the model was computing. Off: play every chunk from its start.
    latency_compensation: bool = True
    #: At least this long between two observations handed to the policy. 0: the next as soon
    #: as the last chunk is back, so only a chunk's first step or two is ever played.
    replan_s: float = 0.0
    #: Average every chunk that still covers the moment (ACT's temporal ensembling) instead of
    #: fading the newest in over `blend_s`; weights exp(-ensemble_k x rank), rank 0 the oldest.
    ensemble: bool = False
    ensemble_k: float = 0.0

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz
