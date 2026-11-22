"""CARLA's axes and pedals, in drive/'s terms. Every sign flip between the two lives here.

CARLA (Unreal) is left-handed: x forward/east, y to the RIGHT, z up, yaw in degrees
turning clockwise seen from above, and `VehicleControl.steer` +1 turns the wheels to the
right. drive/ is right-handed: y to the LEFT, yaw in radians counter-clockwise, and
steer +1 is full LEFT (`drive/types.py`). So y, yaw and steer change sign, and nothing
else does. A sign wrong here would not crash anything: the car would still drive, and
every label in the dataset would point the other way.

drive/'s single throttle in [-1, 1] (negative is the brake, there is no reverse) is
CARLA's two pedals.
"""
from __future__ import annotations

import math


def pose_from_carla(x: float, y: float, yaw_deg: float) -> tuple[float, float, float]:
    """(x, y, yaw) in drive/'s frame from a CARLA location and rotation yaw."""
    return float(x), -float(y), -math.radians(float(yaw_deg))


def xy_to_carla(x: float, y: float) -> tuple[float, float]:
    """A point in drive/'s frame as a CARLA (x, y)."""
    return float(x), -float(y)


def yaw_to_carla(yaw: float) -> float:
    """A drive/ yaw (rad, counter-clockwise) as a CARLA yaw (degrees, clockwise)."""
    return -math.degrees(float(yaw))


def steer_to_carla(steer: float) -> float:
    """drive/'s steer (+1 left) as CARLA's (+1 right)."""
    return -min(max(float(steer), -1.0), 1.0)


def steer_from_carla(steer: float) -> float:
    return -float(steer)


def pedals(throttle: float) -> tuple[float, float]:
    """(throttle, brake), each in [0, 1], from drive/'s throttle in [-1, 1]."""
    throttle = min(max(float(throttle), -1.0), 1.0)
    return (throttle, 0.0) if throttle >= 0.0 else (0.0, -throttle)


def throttle_from_pedals(throttle: float, brake: float) -> float:
    """drive/'s throttle from CARLA's pedals; the brake wins if both are pressed."""
    return -float(brake) if brake > 0.0 else float(throttle)
