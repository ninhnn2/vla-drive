"""A car in CARLA as a `Platform` (drive/platform.py): what DriveLoop, the recorder and the
teacher need, and nothing they would have to know about CARLA.

Measured on CARLA 0.9.16 (docs/carla.md has the numbers):
- The actor's origin is on the ground, mid-car; the Model 3's rear axle is 1.386 m behind it.
  The state's (x, y) is the REAR AXLE, the point the car model and the teacher use.
- `get_wheel_steer_angle` is positive turning RIGHT, like CARLA's steer command.
- The camera renders every tick and a frame is taken by counting ticks: `sensor_tick`
  accumulates rounding error and drops or adds a tick about once every 100 frames.
- A camera image arrives on another thread after `world.tick()` returns. `observe` waits
  for the image of the CURRENT frame; taking the latest one would sometimes hand over the
  previous tick's picture with this tick's state.
"""
from __future__ import annotations

import math
import threading

import numpy as np

from ..types import Observation, VehicleState
from .camera import CarlaCameraParams
from .frames import pedals, pose_from_carla, steer_to_carla
from .model import CarModel
from .server import ROLE, Session


class CarlaPlatform:
    def __init__(self, session: Session, transform, model: CarModel, camera: CarlaCameraParams = CarlaCameraParams(),
                 range_max_m: float = 40.0):
        import carla
        self._carla = carla
        self.session, self.world, self.model, self.camera_params = session, session.world, model, camera
        self.range_max_m = range_max_m
        library = self.world.get_blueprint_library()

        blueprint = library.find(model.blueprint)
        blueprint.set_attribute("role_name", ROLE)
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is None:
            raise RuntimeError(f"cannot spawn {model.blueprint} at {transform.location}: the place is taken")
        self.vehicle = session.keep(vehicle)

        rgb = library.find("sensor.camera.rgb")
        for key, value in (("image_size_x", camera.width), ("image_size_y", camera.height),
                           ("fov", camera.hfov_deg), ("sensor_tick", 0.0), ("motion_blur_intensity", 0.0),
                           ("exposure_compensation", camera.exposure_compensation)):
            rgb.set_attribute(key, str(value))
        mount = carla.Transform(carla.Location(x=camera.forward_m - model.rear_axle_m, z=camera.height_m),
                                carla.Rotation(pitch=-camera.pitch_down_deg))
        self.camera = session.keep(self.world.spawn_actor(rgb, mount, attach_to=self.vehicle))
        self._images: dict[int, object] = {}
        self._arrived = threading.Condition()
        self.camera.listen(self._on_image)

        self.collisions: list[tuple[int, str]] = []
        bump = library.find("sensor.other.collision")
        self.collision_sensor = session.keep(self.world.spawn_actor(bump, carla.Transform(), attach_to=self.vehicle))
        self.collision_sensor.listen(lambda event: self.collisions.append((event.frame, event.other_actor.type_id)))

        self.frame = self.world.tick()
        # Time by counting ticks: exact multiples of tick_s, which the recorder's 10 Hz needs.
        self.ticks, self.t = 0, 0.0
        self._state: VehicleState | None = None

    # --- sensors ---------------------------------------------------------------------
    def _on_image(self, image) -> None:
        with self._arrived:
            self._images[image.frame] = image
            for old in [f for f in self._images if f < image.frame - 10]:
                del self._images[old]
            self._arrived.notify_all()

    def image(self, timeout_s: float = 10.0) -> np.ndarray:
        """The camera's picture of the current frame, uint8 HxWx3 RGB."""
        with self._arrived:
            if not self._arrived.wait_for(lambda: self.frame in self._images, timeout=timeout_s):
                raise RuntimeError(f"no camera image for frame {self.frame} after {timeout_s:.0f} s")
            raw = self._images[self.frame]
        bgra = np.frombuffer(raw.raw_data, dtype=np.uint8).reshape(raw.height, raw.width, 4)
        return np.ascontiguousarray(bgra[:, :, 2::-1])

    @property
    def state(self) -> VehicleState:
        if self._state is None:
            transform = self.vehicle.get_transform()
            x, y, yaw = pose_from_carla(transform.location.x, transform.location.y, transform.rotation.yaw)
            x -= self.model.rear_axle_m * math.cos(yaw)
            y -= self.model.rear_axle_m * math.sin(yaw)
            velocity, forward = self.vehicle.get_velocity(), transform.get_forward_vector()
            speed = max(velocity.x * forward.x + velocity.y * forward.y + velocity.z * forward.z, 0.0)
            wheels = self._carla.VehicleWheelLocation
            angle = (self.vehicle.get_wheel_steer_angle(wheels.FL_Wheel)
                     + self.vehicle.get_wheel_steer_angle(wheels.FR_Wheel)) / 2.0
            steer = min(max(-math.radians(angle) / self.model.max_steer_rad, -1.0), 1.0)
            self._state = VehicleState(x, y, yaw, speed, steer)
        return self._state

    def yaw_rate(self) -> float:
        """rad/s, counter-clockwise (drive/'s sign)."""
        return -math.radians(self.vehicle.get_angular_velocity().z)

    def observe(self, task: str) -> Observation:
        s = self.state
        return Observation(self.t, [self.image()], s.speed, s.steer, task, pose=(s.x, s.y, s.yaw))

    def range_m(self) -> float:
        """Distance from the front bumper to the nearest vehicle or person in the car's lane ahead."""
        s, model = self.state, self.model
        cos, sin = math.cos(s.yaw), math.sin(s.yaw)
        front = model.rear_axle_m + model.length_m / 2
        nearest = self.range_max_m
        for actor in self.world.get_actors():
            if actor.id == self.vehicle.id or not actor.type_id.startswith(("vehicle.", "walker.")):
                continue
            location, extent = actor.get_location(), actor.bounding_box.extent
            x, y, _ = pose_from_carla(location.x, location.y, 0.0)
            forward = cos * (x - s.x) + sin * (y - s.y) - front
            left = -sin * (x - s.x) + cos * (y - s.y)
            reach = max(extent.x, extent.y)
            if forward + reach > 0 and abs(left) < model.width_m / 2 + reach:
                nearest = min(nearest, max(forward - reach, 0.0))
        return nearest

    # --- actuators -------------------------------------------------------------------
    def apply(self, action, dt: float) -> None:
        """Hold `[steer, throttle]` for `dt`, a whole number of the server's ticks."""
        ticks = int(round(dt / self.session.tick_s))
        if ticks < 1 or abs(ticks * self.session.tick_s - dt) > 1e-6:
            raise ValueError(f"CARLA ticks every {self.session.tick_s} s; a step of {dt} s is not a whole number "
                             "of ticks (run the loop at LoopConfig(control_hz=1/tick_s))")
        gas, brake = pedals(action[1])
        control = self._carla.VehicleControl(throttle=gas, brake=brake, steer=steer_to_carla(action[0]))
        for _ in range(ticks):
            self.vehicle.apply_control(control)
            self.frame = self.world.tick()
            self.ticks += 1
            self.t = self.ticks * self.session.tick_s
            self._state = None

    def collided(self) -> bool:
        return bool(self.collisions)

    def close(self) -> None:
        """Destroy the car and its sensors, once the camera has no frame left in flight.

        Half the time a drive ends on a tick whose picture nobody asked for, and that picture
        is still being read back from the GPU. Destroying a camera with a frame in flight lets
        the render thread finish against freed memory: simulate_carlar's fisheye sensor
        segfaulted the server that way (patches/carla-src/0012). Waiting costs a frame."""
        with self._arrived:
            self._arrived.wait_for(lambda: self.frame in self._images, timeout=5.0)
        for actor in (self.collision_sensor, self.camera, self.vehicle):
            self.session.destroy(actor)
