"""The simulated car in its world: the stand-in for the real platform."""
from __future__ import annotations

import math

import numpy as np

from ..config import CameraParams, RangeSensorParams, VehicleParams
from ..types import Observation, VehicleState
from .render import Camera, GroundMap
from .track import Obstacle, Track
from .vehicle import step_vehicle


class World:
    def __init__(self, track: Track | None = None, obstacles: tuple[Obstacle, ...] = (),
                 vehicle: VehicleParams = VehicleParams(), camera: CameraParams = CameraParams(),
                 range_sensor: RangeSensorParams = RangeSensorParams(), start_s: float = 0.0, seed: int = 0):
        self.track = track or Track.default()
        self.obstacles = tuple(obstacles)
        self.vehicle = vehicle
        self.range_sensor = range_sensor
        self.ground = GroundMap(self.track, self.obstacles, seed=seed)
        self.camera = Camera(camera)
        self.t = 0.0
        self.place(start_s)

    def place(self, s: float, lateral_m: float = 0.0, yaw_offset_rad: float = 0.0, speed: float = 0.0) -> None:
        """Put the car `s` metres along the track, optionally off-centre and askew."""
        x, y, heading = (float(v) for v in self.track.at(s))
        self.state = VehicleState(x - math.sin(heading) * lateral_m, y + math.cos(heading) * lateral_m,
                                  heading + yaw_offset_rad, speed, 0.0)

    def apply(self, action, dt: float) -> None:
        """Hold `action` on the motors for `dt` seconds."""
        self.state = step_vehicle(self.state, action, self.vehicle, dt)
        self.t += dt

    def image(self) -> np.ndarray:
        return self.camera.render(self.ground, self.state)

    def observe(self, task: str) -> Observation:
        s = self.state
        return Observation(self.t, [self.image()], s.speed, s.steer, task, pose=(s.x, s.y, s.yaw))

    def _obstacles_in_car_frame(self):
        s = self.state
        cos, sin = math.cos(s.yaw), math.sin(s.yaw)
        for o in self.obstacles:
            dx, dy = o.x - s.x, o.y - s.y
            yield cos * dx + sin * dy, -sin * dx + cos * dy, o.radius_m

    def range_m(self) -> float:
        """Distance from the front bumper to the nearest obstacle inside the sensor's cone."""
        sensor = self.range_sensor
        front = self.vehicle.wheelbase_m + 0.07
        nearest = sensor.max_range_m
        for forward, left, radius in self._obstacles_in_car_frame():
            forward -= front
            distance = math.hypot(forward, left)
            if forward <= 0 or distance <= radius:
                nearest = 0.0 if distance <= radius else nearest
                continue
            if abs(math.atan2(left, forward)) <= math.radians(sensor.half_fov_deg) + math.asin(radius / distance):
                nearest = min(nearest, distance - radius)
        return nearest

    def collided(self) -> bool:
        """True while the car's body overlaps an obstacle."""
        v = self.vehicle
        rear, front, half = -(v.length_m - v.wheelbase_m) / 2, (v.length_m + v.wheelbase_m) / 2, v.width_m / 2
        return any(math.hypot(forward - min(max(forward, rear), front), left - min(max(left, -half), half)) < radius
                   for forward, left, radius in self._obstacles_in_car_frame())
