"""What the car's camera sees: the ground plane, drawn once and resampled per frame."""
from __future__ import annotations

import cv2
import numpy as np

from ..config import CameraParams
from ..types import VehicleState
from .track import Obstacle, Track

GRASS, ASPHALT, LINE, CENTER, CONE, SKY = (
    (58, 132, 62), (72, 72, 76), (236, 236, 236), (228, 190, 40), (232, 96, 24), (150, 190, 230))


class GroundMap:
    """A top-down RGB picture of the world, `px_per_m` pixels to the metre."""

    def __init__(self, track: Track, obstacles: tuple[Obstacle, ...] = (), px_per_m: int = 100,
                 margin_m: float = 1.5, seed: int = 0):
        self.px_per_m = px_per_m
        self.x_min, self.y_min = track.xy.min(axis=0) - margin_m
        x_max, self.y_max = track.xy.max(axis=0) + margin_m
        w, h = int((x_max - self.x_min) * px_per_m), int((self.y_max - self.y_min) * px_per_m)
        rng = np.random.default_rng(seed)
        # Mottled grass, so the image is not a flat colour a policy could key on.
        mottle = cv2.GaussianBlur(rng.normal(0.0, 30.0, (h, w)).astype(np.float32), (0, 0), 3)
        image = np.clip(np.array(GRASS, np.float32) + mottle[..., None], 0, 255).astype(np.uint8)
        line = [self.to_px(track.xy).round().astype(np.int32)]
        road = int(track.width_m * px_per_m)
        cv2.polylines(image, line, True, LINE, road + int(0.10 * px_per_m), cv2.LINE_AA)
        cv2.polylines(image, line, True, ASPHALT, road, cv2.LINE_AA)
        dash = int(0.30 / track.ds)
        for start in range(0, len(track.xy) - dash, 2 * dash):
            cv2.polylines(image, [line[0][start:start + dash]], False, CENTER,
                          max(int(0.03 * px_per_m), 1), cv2.LINE_AA)
        for o in obstacles:
            u, v = self.to_px(np.array([o.x, o.y])).round().astype(int)
            cv2.circle(image, (int(u), int(v)), int(o.radius_m * px_per_m), CONE, -1, cv2.LINE_AA)
        self.image = image

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        """World (x, y) to map (column, row); y points up in the world, rows go down."""
        xy = np.asarray(xy, dtype=np.float64)
        return np.stack([(xy[..., 0] - self.x_min) * self.px_per_m,
                         (self.y_max - xy[..., 1]) * self.px_per_m], axis=-1)


class Camera:
    """A pinhole camera looking forward and slightly down.

    `forward_m` and `left_m` give, for every pixel that sees the ground, where on the
    ground it looks, in the car's frame. The same two arrays are an inverse-perspective
    map, which is what a camera-only lane follower needs from a calibrated real camera.
    """

    def __init__(self, params: CameraParams = CameraParams()):
        self.params = p = params
        self.focal = focal = (p.width / 2) / np.tan(np.radians(p.hfov_deg) / 2)
        u, v = np.meshgrid(np.arange(p.width) + 0.5, np.arange(p.height) + 0.5)
        left, up = -(u - p.width / 2) / focal, -(v - p.height / 2) / focal
        pitch = np.radians(p.pitch_down_deg)
        down = np.sin(pitch) - up * np.cos(pitch)            # how fast each ray descends
        with np.errstate(divide="ignore", invalid="ignore"):
            reach = np.where(down > 1e-6, p.height_m / down, np.inf)
        self.forward_m = (p.forward_m + reach * (np.cos(pitch) + up * np.sin(pitch))).astype(np.float32)
        self.left_m = (reach * left).astype(np.float32)
        self.ground = np.isfinite(reach) & (self.forward_m < p.far_m)
        self.forward_m[~self.ground] = 0.0
        self.left_m[~self.ground] = 0.0

    def project(self, forward_m, left_m) -> np.ndarray:
        """Pixels (column, row) of points on the ground, in the car's frame: for drawing a path.

        Points behind the camera come back as NaN.
        """
        p, pitch = self.params, np.radians(self.params.pitch_down_deg)
        ahead = np.asarray(forward_m, dtype=np.float64) - p.forward_m
        depth = ahead * np.cos(pitch) + p.height_m * np.sin(pitch)
        depth = np.where(depth > 0.05, depth, np.nan)
        up = (ahead * np.sin(pitch) - p.height_m * np.cos(pitch)) / depth
        return np.stack([p.width / 2 - self.focal * np.asarray(left_m) / depth,
                         p.height / 2 - self.focal * up], axis=-1)

    def render(self, ground: GroundMap, state: VehicleState) -> np.ndarray:
        cos, sin = np.cos(state.yaw), np.sin(state.yaw)
        x = state.x + cos * self.forward_m - sin * self.left_m
        y = state.y + sin * self.forward_m + cos * self.left_m
        map_x = ((x - ground.x_min) * ground.px_per_m).astype(np.float32)
        map_y = ((ground.y_max - y) * ground.px_per_m).astype(np.float32)
        image = cv2.remap(ground.image, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=GRASS)
        image[~self.ground] = SKY
        return image
