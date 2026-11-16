"""Policies: anything that turns an observation into a chunk of future actions."""
from .base import Policy
from .lane_camera import LaneCameraPolicy
from .pure_pursuit import PurePursuitPolicy

POLICIES = ("pure-pursuit", "lane-camera", "cnn")


def make_policy(name: str, *, track, vehicle, camera, checkpoint=None, target_speed_mps=None) -> Policy:
    """Build a policy by name. `track` is the map the pure-pursuit teacher follows."""
    speed = {} if target_speed_mps is None else {"target_speed_mps": target_speed_mps}
    if name == "pure-pursuit":
        return PurePursuitPolicy(track, vehicle, **speed)
    if name == "lane-camera":
        return LaneCameraPolicy(camera, vehicle, **speed)
    if name == "cnn":
        from .cnn import CnnPolicy
        if not checkpoint:
            raise ValueError("the cnn policy needs a checkpoint (a file from ./drive.sh train)")
        if speed:
            raise ValueError("the cnn policy drives at the speeds it was shown; a target speed does not apply")
        return CnnPolicy(checkpoint)
    raise ValueError(f"unknown policy {name!r}; choose from {', '.join(POLICIES)}")


__all__ = ["POLICIES", "LaneCameraPolicy", "Policy", "PurePursuitPolicy", "make_policy"]
