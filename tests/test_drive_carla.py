"""drive/carla without a CARLA server: the frames, routes, car model, teacher and camera, the
per-frame sentences through the recorder, validate, browse and export, and the route walk
on a junction made of fake waypoints."""
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drive import cli
from drive.carla.camera import CarlaCameraParams
from drive.carla.frames import pedals, pose_from_carla, steer_to_carla, throttle_from_pedals, xy_to_carla, yaw_to_carla
from drive.carla.model import CarModel
from drive.carla.route import PHRASES, Junction, Route, maneuver
from drive.carla.routing import random_route
from drive.carla.teacher import RouteTeacher
from drive.learn.browse import Geometry, draw, plan_path
from drive.learn.export import frames, robot_type
from drive.learn.validate import validate
from drive.recorder import Recorder
from drive.types import ActionChunk, VehicleState


def corner_route(radius=12.0, straight=60.0, after=60.0, step=0.5) -> Route:
    """East along a straight, a quarter circle to the RIGHT, then south: drive/'s frame."""
    points, headings = [], []
    for x in np.arange(0.0, straight, step):
        points.append((x, 0.0)), headings.append(0.0)
    for angle in np.arange(0.0, math.pi / 2, step / radius):     # centre at (straight, -radius)
        points.append((straight + radius * math.sin(angle), -radius + radius * math.cos(angle)))
        headings.append(-angle)
    for d in np.arange(0.0, after, step):
        points.append((straight + radius, -radius - d)), headings.append(-math.pi / 2)
    return Route(points, headings, width_m=4.0,
                 junctions=[Junction(straight - 2.0, straight + radius * math.pi / 2 + 2.0, "right")])


MODEL = CarModel(wheelbase_m=3.0, rear_axle_m=1.4, throttle_accel=(5.0, 0.2), resistance=(0.0, 0.3, 0.0),
                 brake_table=((0.0, 0.0), (0.1, 2.0), (0.3, 4.5), (1.0, 8.0)))


class FramesTests(unittest.TestCase):
    def test_y_yaw_and_steer_change_sign_nothing_else(self):
        x, y, yaw = pose_from_carla(1.0, 2.0, 90.0)        # CARLA: 2 m to the right, facing right
        self.assertEqual((x, y), (1.0, -2.0))
        self.assertAlmostEqual(yaw, -math.pi / 2)
        self.assertEqual(xy_to_carla(x, y), (1.0, 2.0))
        self.assertAlmostEqual(yaw_to_carla(yaw), 90.0)
        self.assertEqual(steer_to_carla(0.3), -0.3)         # drive/'s left is CARLA's negative
        self.assertEqual(steer_to_carla(2.0), -1.0)

    def test_one_throttle_is_two_pedals(self):
        self.assertEqual(pedals(0.4), (0.4, 0.0))
        self.assertEqual(pedals(-0.7), (0.0, 0.7))
        self.assertEqual(pedals(-3.0), (0.0, 1.0))
        self.assertEqual(throttle_from_pedals(0.5, 0.2), -0.2)


class RouteTests(unittest.TestCase):
    def test_open_ends_and_curvature(self):
        route = corner_route()
        self.assertAlmostEqual(route.length, 60 + 12 * math.pi / 2 + 60, delta=1.0)
        self.assertEqual(float(route.at(-5.0)[0]), 0.0)                  # held at the ends, not wrapped
        self.assertAlmostEqual(float(route.at(route.length + 5)[1]), -12.0 - 59.5, delta=0.6)
        middle = 60 + 12 * math.pi / 4
        self.assertAlmostEqual(abs(route.curvature[int(middle / route.ds)]), 1 / 12, delta=0.01)
        self.assertLess(route.sharpest(0.0, 30.0), 0.01)
        s, lateral, _ = route.project(20.0, 0.5)
        self.assertAlmostEqual(s, 20.0, delta=0.3)
        self.assertAlmostEqual(lateral, 0.5, delta=0.01)                 # positive to the LEFT

    def test_a_crossing_is_resolved_by_where_the_car_was(self):
        # A loop that crosses its own start: at (10, 0) the route is twice, at s=10 and near the end.
        t = np.linspace(0, 2 * math.pi, 400)
        points = np.stack([10 + 10 * np.sin(t) * np.cos(t) * 2, 10 * np.sin(t)], axis=1)
        route = Route(np.vstack([[(0.0, 0.0)], points]))
        early, _, _ = route.project(10.2, 0.1, near_s=12.0, window_m=8.0)
        late, _, _ = route.project(10.2, 0.1, near_s=route.length - 3.0, window_m=8.0)
        self.assertLess(early, 25.0)
        self.assertGreater(late, route.length - 25.0)

    def test_commands_cover_the_junction_and_only_it(self):
        route = corner_route()
        self.assertEqual(route.command(10.0, announce_m=30.0), "follow")
        self.assertEqual(route.command(40.0, announce_m=30.0), "right")       # announced 30 m before
        self.assertEqual(route.command(70.0, announce_m=30.0), "right")       # inside
        self.assertEqual(route.command(route.length - 5, announce_m=30.0), "follow")

    def test_maneuver_from_headings(self):
        self.assertEqual(maneuver(0.0, math.pi / 2), "left")
        self.assertEqual(maneuver(0.0, -math.pi / 2), "right")
        self.assertEqual(maneuver(3.0, -3.1), "straight")                    # across the +-pi seam
        self.assertTrue(all(len(words) >= 2 for words in PHRASES.values()))


class ModelTests(unittest.TestCase):
    def test_the_wheel_settles_at_the_speeds_share_of_the_lock(self):
        state = VehicleState(0, 0, 0, 10.0, 0.0)
        for _ in range(20):
            state = MODEL.step(state, (0.5, 0.0), 0.05)
        lock = np.interp(state.speed * 3.6, [0, 20, 60, 120], [1.0, 0.9, 0.8, 0.7])   # coasting: slower now
        self.assertAlmostEqual(state.steer, 0.5 * lock, delta=0.005)
        self.assertGreater(state.yaw, 0.0)                                   # +steer turns left

    def test_throttle_for_inverts_accel(self):
        for speed in (1.0, 5.0, 9.0):
            for accel in (-3.0, -1.0, 0.0, 1.5):
                throttle = MODEL.throttle_for(accel, speed)
                self.assertTrue(-1.0 <= throttle <= 1.0)
                self.assertAlmostEqual(MODEL.accel(throttle, speed), accel, delta=1e-6)
        # Beyond the brakes: full brake, and the model does not pretend to stop harder.
        self.assertEqual(MODEL.throttle_for(-20.0, 5.0), -1.0)
        self.assertGreater(MODEL.accel(-1.0, 5.0), -20.0)
        # The brake is a table: half way between two rows is half way between their decelerations.
        self.assertAlmostEqual(MODEL.brake_decel(0.2), 3.25)
        self.assertAlmostEqual(MODEL.throttle_for(MODEL.accel(-0.2, 5.0), 5.0), -0.2)

    def test_no_creeping_from_standstill(self):
        creeping = CarModel(resistance=(-0.2, 0.3, 0.0))       # an unphysical fit's model
        self.assertEqual(creeping.accel(0.0, 0.0), 0.0)
        self.assertEqual(MODEL.step(VehicleState(0, 0, 0, 0.0, 0.0), (0, 0), 1.0).speed, 0.0)

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = MODEL.save(Path(tmp) / "m.json")
            self.assertEqual(CarModel.load(str(path)), MODEL)
            with self.assertRaises(ValueError):
                CarModel.load("vehicle.not.fitted")


class TeacherTests(unittest.TestCase):
    def drive(self, seconds=40.0, speed=8.0):
        route = corner_route()
        teacher = RouteTeacher(route, MODEL, target_speed_mps=speed)
        state, laterals, speeds = VehicleState(0, 0, 0, 0, 0), [], []
        for _ in range(int(seconds / 0.05)):
            steer, throttle = teacher.act(state)
            state = MODEL.step(state, (steer, throttle), 0.05)
            laterals.append(teacher.locate(state)[1]), speeds.append((teacher.s, state.speed))
        return route, teacher, state, np.array(laterals), np.array(speeds)

    def test_follows_the_corner_and_slows_for_it(self):
        route, teacher, state, laterals, speeds = self.drive()
        self.assertLess(np.abs(laterals).max(), 0.5)
        on_straight = speeds[(speeds[:, 0] > 25) & (speeds[:, 0] < 35), 1]
        in_corner = speeds[(speeds[:, 0] > 65) & (speeds[:, 0] < 72), 1]
        self.assertGreater(on_straight.mean(), 7.0)
        self.assertLess(in_corner.max(), math.sqrt(2.0 * 12) + 0.6)        # lateral acceleration 2 m/s^2
        self.assertTrue(teacher.done(teacher.s, state.speed))               # stopped at the end

    def test_a_plan_starts_with_what_it_would_do_now(self):
        route = corner_route()
        teacher = RouteTeacher(route, MODEL, target_speed_mps=6.0)
        state = VehicleState(50.0, 0.3, 0.05, 6.0, 0.0)
        s, _ = teacher.locate(state, near_s=50.0)
        plan = teacher.plan(state, s, t_obs=3.0)
        self.assertEqual(plan.actions.shape, (16, 2))
        self.assertEqual((plan.dt, plan.t_obs), (0.1, 3.0))
        np.testing.assert_allclose(plan.actions[0], teacher.control(state, s))
        self.assertLess(plan.actions[-1, 0], -0.05)                          # 1.5 s ahead: into the right turn


class CameraTests(unittest.TestCase):
    def test_straight_ahead_is_the_middle_column_and_far_is_the_horizon(self):
        camera = CarlaCameraParams()
        (u, v), = camera.project([camera.forward_m + 10.0], [0.0])
        self.assertAlmostEqual(u, camera.width / 2)
        (_, v_far), = camera.project([camera.forward_m + 1e5], [0.0])
        horizon = camera.height / 2 - camera.focal_px * math.tan(math.radians(camera.pitch_down_deg))
        self.assertAlmostEqual(v_far, horizon, delta=0.5)
        self.assertGreater(v, v_far)                                         # nearer is lower in the picture
        (u_left, _), = camera.project([camera.forward_m + 10.0], [2.0])
        self.assertLess(u_left, camera.width / 2)                            # left of the car, left in the picture
        self.assertTrue(np.isnan(camera.project([camera.forward_m - 1.0], [0.0])).all())


def write_carla_episode(directory: Path, name="carla-000", frames_n=30, tasks=None) -> Path:
    camera = CarlaCameraParams()
    recorder = Recorder(directory / name, "follow the lane", 10.0,
                        meta={"robot_type": "carla:vehicle.tesla.model3",
                              "carla": {"vehicle": MODEL.to_json(), "camera": camera.to_json()}})
    for i in range(frames_n):
        task = tasks[i] if tasks else None
        plan = ActionChunk(np.tile([0.0, 0.2], (16, 1)), 0.1, i / 10)
        recorder.add(i / 10, np.full((camera.height, camera.width, 3), 90, np.uint8), 5.0, 0.0, (0.01, 0.2), plan,
                     task=task, extra={"poses": (i * 0.5, 0.0, 0.0), "lateral_m": 0.1})
    return recorder.save()


class EpisodeSentenceTests(unittest.TestCase):
    def test_recorder_keeps_a_sentence_per_frame_only_when_it_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            same = write_carla_episode(Path(tmp), "same")
            told = ["follow the lane"] * 10 + ["take the next left"] * 20
            changes = write_carla_episode(Path(tmp), "changes", tasks=told)
            self.assertNotIn("tasks", np.load(same / "episode.npz").files)
            data = np.load(changes / "episode.npz")
            self.assertEqual(list(data["tasks"]), told)
            self.assertEqual(data["poses"].shape, (30, 3))
            meta = json.loads((changes / "meta.json").read_text())
            self.assertEqual(meta["tasks"], ["follow the lane", "take the next left"])
            self.assertEqual(meta["robot_type"], "carla:vehicle.tesla.model3")

    def test_truncate_and_reserved_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(Path(tmp) / "e", "follow the lane")
            for i in range(10):
                recorder.add(i / 10, np.zeros((4, 4, 3), np.uint8), 1.0, 0.0, (0, 0), task=f"t{i}", extra={"s": i})
            recorder.truncate(3)
            self.assertEqual((len(recorder.t), len(recorder.tasks), len(recorder.extras["s"])), (7, 7, 7))
            recorder.extras["tasks"] = list(range(7))
            with self.assertRaises(ValueError):
                recorder.save()

    def test_export_validate_and_browse_read_the_sentences(self):
        with tempfile.TemporaryDirectory() as tmp:
            told = ["follow the lane"] * 10 + ["take the next left"] * 20
            episode = write_carla_episode(Path(tmp), tasks=told)
            meta, rows = frames(episode)
            self.assertEqual([row["task"] for row in rows], told)
            self.assertEqual(robot_type(meta), "carla:vehicle.tesla.model3")
            self.assertEqual(robot_type({}), "vla-drive-sim")
            write_carla_episode(Path(tmp), "carla-001")
            with redirect_stdout(io.StringIO()) as out:
                report = validate(Path(tmp))
            self.assertTrue(report["ok"])
            self.assertEqual(report["tasks"]["take the next left"], 20)
            self.assertIn("take the next left", out.getvalue())
            geometry = Geometry(meta)
            data = np.load(episode / "episode.npz")
            picture = draw(data["images"][0], data["states"][0], data["actions"][0],
                           np.tile([0.0, 0.2], (16, 1)), 0.1, "t", geometry, task=told[0])
            self.assertEqual(picture.shape, (720, 960, 3))
            path = plan_path(np.tile([0.0, 0.2], (16, 1)), 0.1, 5.0, 0.0, step=geometry.step)
            pixels = geometry.project(*path.T)
            finite = pixels[np.isfinite(pixels).all(axis=1)]
            np.testing.assert_allclose(finite[:, 0], CarlaCameraParams.width / 2, atol=1e-6)   # straight: up the middle


class FakeWaypoint:
    """Just what routing.py reads of a carla.Waypoint, in CARLA's left-handed axes."""

    def __init__(self, x, y, yaw_deg, junction=False):
        self.transform = SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.0),
                                         rotation=SimpleNamespace(yaw=yaw_deg))
        self.is_junction, self.lane_width, self.following = junction, 3.5, []

    def next(self, step):
        return list(self.following)


def t_junction_road():
    """East for 50 m, then a junction: straight on, a right turn (to CARLA +y), a left turn (to -y)."""
    approach = [FakeWaypoint(x, 0.0, 0.0) for x in range(51)]
    branches = {"straight": [FakeWaypoint(50 + d, 0.0, 0.0, junction=d <= 12) for d in range(1, 120)]}
    for name, sign in (("right", 1.0), ("left", -1.0)):
        arc = [FakeWaypoint(50 + 10 * math.sin(a), sign * (10 - 10 * math.cos(a)), sign * math.degrees(a), True)
               for a in np.linspace(0.1, math.pi / 2, 15)]
        out = [FakeWaypoint(60.0, sign * (10 + d), sign * 90.0) for d in range(1, 120)]
        branches[name] = arc + out
    for road in [approach, *branches.values()]:
        for a, b in zip(road, road[1:]):
            a.following = [b]
    approach[-1].following = [road[0] for road in branches.values()]
    return approach[0]


class RoutingTests(unittest.TestCase):
    def test_turns_are_named_from_carlas_axes_and_drawn_evenly(self):
        seen = {}
        for seed in range(60):
            route = random_route(t_junction_road(), 120.0, np.random.default_rng(seed))
            (junction,) = route.junctions
            end_x, end_y, _ = (float(v) for v in route.at(route.length))
            # CARLA +y is drive/'s right: the right turn ends at negative y.
            expected = "straight" if abs(end_y) < 1 else ("right" if end_y < 0 else "left")
            self.assertEqual(junction.maneuver, expected)
            self.assertAlmostEqual(junction.s_entry, 51.0, delta=1.5)
            seen[expected] = seen.get(expected, 0) + 1
        self.assertEqual(set(seen), {"left", "right", "straight"})
        self.assertTrue(all(10 <= n <= 30 for n in seen.values()), seen)     # about 20 each

    def test_a_route_never_ends_inside_a_junction(self):
        route = random_route(t_junction_road(), 55.0, np.random.default_rng(0))
        self.assertLess(route.length, 50.0)
        self.assertEqual(route.junctions, ())


    def test_speed_noise_drives_faster_into_the_corner_and_the_label_does_not(self):
        route = corner_route()
        teacher = RouteTeacher(route, MODEL, target_speed_mps=6.0)
        state = VehicleState(55.0, 0.0, 0.0, 4.0, 0.0)              # the corner (radius 12 m) is near
        s, _ = teacher.locate(state, near_s=55.0)
        bend = teacher.speed_target(s, state.speed)
        self.assertAlmostEqual(bend, math.sqrt(2.0 * 12), delta=0.3)
        self.assertAlmostEqual(teacher.speed_target(s, state.speed, 1.3), 1.3 * bend)
        self.assertGreater(teacher.control(state, s, speed_scale=1.3)[1], teacher.control(state, s)[1])
        self.assertEqual(teacher.control(state, s, speed_scale=1.3)[0], teacher.control(state, s)[0])
        np.testing.assert_allclose(teacher.plan(state, s, 0.0).actions[0], teacher.control(state, s))


class ChunkPlayTests(unittest.TestCase):
    """How chunks are played: the replan interval, and ACT's temporal ensembling."""

    def test_ensembling_averages_the_chunks_that_cover_the_moment(self):
        from drive.control.executor import ChunkExecutor
        from drive.types import ActionChunk
        executor = ChunkExecutor(ensemble=True)
        executor.update(ActionChunk(np.tile([0.0, 0.2], (16, 1)), 0.1, 0.0), 0.1)
        executor.update(ActionChunk(np.tile([0.4, 0.2], (16, 1)), 0.1, 0.5), 0.6)
        np.testing.assert_allclose(executor.action(0.7), [0.2, 0.2])
        np.testing.assert_allclose(executor.action(1.7), [0.4, 0.2])        # the first chunk has run out
        executor.update(ActionChunk(np.tile([0.8, 0.2], (16, 1)), 0.1, 1.8), 1.9)
        self.assertEqual(len(executor._live), 2)                            # the first is dropped
        weighted = ChunkExecutor(ensemble=True, ensemble_k=1.0)
        weighted.update(ActionChunk(np.tile([0.0, 0.0], (16, 1)), 0.1, 0.0), 0.1)
        weighted.update(ActionChunk(np.tile([1.0, 0.0], (16, 1)), 0.1, 0.1), 0.2)
        self.assertAlmostEqual(weighted.action(0.3)[0], math.exp(-1) / (1 + math.exp(-1)))  # the oldest weighs most

    def test_the_loop_waits_replan_s_between_observations(self):
        from drive.config import LoopConfig
        from drive.loop import DriveLoop

        class Worker:
            busy, asked = False, []

            def submit(self, obs, now):
                self.asked.append(now)

            def poll(self, now):
                return None

        class Car:
            t = 0.0

            def observe(self, task):
                return task

            def range_m(self):
                return math.inf

            def apply(self, action, dt):
                self.t += dt

        car, worker = Car(), Worker()
        loop = DriveLoop(car, worker, None, LoopConfig(control_hz=20.0, replan_s=0.5))
        for _ in range(25):
            loop.step()
        np.testing.assert_allclose(worker.asked, [0.0, 0.5, 1.0], atol=1e-9)


class HandoverTests(unittest.TestCase):
    def test_the_teacher_takes_over_before_the_lane_is_left_and_hands_back_when_settled(self):
        from drive.carla.dagger import DaggerConfig, Handover, drifting
        self.assertAlmostEqual(drifting(0.5, math.radians(10), 6.0, 0.5), 0.5 + 3.0 * math.sin(math.radians(10)))
        hand = Handover(DaggerConfig())
        self.assertFalse(hand.update(0.3, 0.0, 6.0, 0.05) or hand.teacher_driving)
        hand.update(0.6, math.radians(10), 6.0, 0.05)                       # 0.6 + 0.52: past 1 m
        self.assertEqual((hand.teacher_driving, hand.takeovers), (True, 1))
        backs = [hand.update(0.1, 0.02, 6.0, 0.05) for _ in range(19)]
        self.assertFalse(any(backs))                                        # 0.95 s settled: not yet
        self.assertTrue(hand.update(0.1, 0.02, 6.0, 0.05))
        self.assertFalse(hand.teacher_driving)


class CliTests(unittest.TestCase):
    def test_help_works_without_carla(self):
        with redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stop:
            cli.main(["carla", "collect", "--help"])
        self.assertEqual(stop.exception.code, 0)
        self.assertIn("--announce-m", out.getvalue())


class ResumeTests(unittest.TestCase):
    """`collect --resume` keeps what a crashed run finished and nothing it did not."""

    def write(self, out, index, seed, labels=True):
        directory = out / f"carla-town05-{index:03d}"
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text(json.dumps({"carla": {"seed": [seed, index]}}))
        if labels:
            (directory / "labels.npz").write_bytes(b"")

    def test_finished_episodes_are_kept_and_cut_off_ones_redone(self):
        from drive.carla.collect import CollectConfig, _written
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.write(out, 0, 2)
            self.write(out, 1, 2)
            self.write(out, 2, 2, labels=False)         # the server died while it was saved
            config = CollectConfig(name="carla-town05", seed=2, resume=True)
            self.assertEqual(_written(out, config), {0, 1})

    def test_another_seed_is_refused(self):
        from drive.carla.collect import CollectConfig, _written
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.write(out, 0, 7)
            with self.assertRaisesRegex(ValueError, "--seed"):
                _written(out, CollectConfig(name="carla-town05", seed=2, resume=True))

    def test_old_camera_metadata_still_reads(self):
        """browse draws plans on episodes recorded before a camera field existed (or after one went)."""
        old = {k: v for k, v in CarlaCameraParams(forward_m=2.4).to_json().items() if k != "exposure_compensation"}
        self.assertEqual(CarlaCameraParams.from_json({**old, "removed_later": 1}).forward_m, 2.4)



class DriveScoreTests(unittest.TestCase):
    """`carla drive` scores the junctions a drive reached, and only those."""

    J = [Junction(10.0, 30.0, "left"), Junction(80.0, 100.0, "right")]

    def score(self, s_end, failed):
        from drive.carla.evaluate import score_junctions
        return score_junctions(self.J, s_end, failed)

    def test_out_of_every_junction(self):
        self.assertEqual(self.score(120.0, False), [("left", True), ("right", True)])

    def test_left_the_route_inside_a_junction(self):
        self.assertEqual(self.score(90.0, True), [("left", True), ("right", False)])

    def test_failed_just_before_a_junction_fails_it(self):
        self.assertEqual(self.score(72.0, True), [("left", True), ("right", False)])

    def test_a_failure_on_a_straight_is_not_a_junction_failure(self):
        self.assertEqual(self.score(60.0, True), [("left", True)])

    def test_out_of_time_inside_a_junction_is_not_scored(self):
        self.assertEqual(self.score(90.0, False), [("left", True)])

    def test_summary_counts_by_maneuver(self):
        from drive.carla.evaluate import summarise
        drives = [{"ended": "duration", "driven_m": 500.0, "seconds": 60.0, "lateral_rms_m": 0.2,
                   "junctions": [("left", True), ("right", False)]},
                  {"ended": "off_route", "driven_m": 300.0, "seconds": 40.0, "lateral_rms_m": 0.4,
                   "junctions": [("left", False)]}]
        m = summarise(drives, [80.0, 120.0], 10.0)
        self.assertEqual(m["junctions"]["left"], {"passed": 1, "reached": 2})
        self.assertEqual(m["junctions"]["straight"], {"passed": 0, "reached": 0})
        self.assertEqual(m["km_per_failure"], 0.8)
        self.assertEqual(m["inference_ms_mean"], 100.0)


class PlanAgeTests(unittest.TestCase):
    def test_a_plan_may_age_by_the_time_between_asks(self):
        from drive.carla.evaluate import CAR_LIMITS, EvalConfig
        self.assertEqual(EvalConfig().limits().max_plan_age_s, CAR_LIMITS.max_plan_age_s)
        # asked every 500 ms, a chunk computed in 190 ms is 0.69 s old before the next: not stale
        self.assertGreater(EvalConfig(replan_s=0.5).limits().max_plan_age_s, 0.5 + 0.19 + 0.05)


class FailureReelTests(unittest.TestCase):
    """tools/failure_reel.py finds each drive in the video: by its frames, or by the scene cuts."""

    def reel(self):
        import importlib.util
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("no OpenCV")
        path = Path(__file__).resolve().parent.parent / "tools/failure_reel.py"
        spec = importlib.util.spec_from_file_location("failure_reel", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_frames_beat_the_rounded_seconds(self):
        reel = self.reel()
        # 23.15 s on the 0.05 s tick is 232 frames at 10 Hz; the report rounds it to 23.1 s
        drives = [{"seconds": 23.1, "frames": 232}, {"seconds": 60.0, "frames": 600}]
        self.assertEqual(reel.drive_frames(drives, 10.0), [(0, 232), (232, 832)])
        self.assertEqual(reel.drive_frames([{"seconds": 23.1}], 10.0), [(0, 231)])

    def test_a_boundary_a_frame_off_snaps_to_the_cut(self):
        reel = self.reel()
        change = np.full(832, 6.0)
        change[232] = 50.0                          # the second drive really starts here
        spans = reel.drive_frames([{"seconds": 23.1}, {"seconds": 60.0}], 10.0)   # says 231
        self.assertEqual(reel.snap(spans, change, 3), [(0, 232), (232, 832)])


class LoopTaskTests(unittest.TestCase):
    def test_the_sentence_of_the_tick_reaches_the_policy(self):
        from drive.config import LoopConfig
        from drive.control.inference import SimulatedWorker
        from drive.loop import DriveLoop
        seen = []

        class Platform:
            t, state = 0.0, VehicleState(0, 0, 0, 0, 0)

            def observe(self, task):
                seen.append(task)
                return SimpleNamespace(t=self.t, task=task)

            def range_m(self):
                return 40.0

            def apply(self, action, dt):
                self.t += dt

        class Policy:
            def infer(self, obs):
                return ActionChunk(np.zeros((16, 2)), 0.1, obs.t)

        loop = DriveLoop(Platform(), SimulatedWorker(Policy(), latency_s=0.1), None, LoopConfig(control_hz=20))
        loop.step("turn left at the next intersection")
        loop.step("ignored: an inference is under way")
        loop.step()
        self.assertEqual(seen, ["turn left at the next intersection", "drive along the track"])

    def test_browse_draws_the_reference_plan(self):
        plan = np.tile([0.0, 0.3], (16, 1))
        image = np.zeros((240, 320, 3), np.uint8)
        geometry = Geometry({"carla": {"camera": CarlaCameraParams().to_json(), "vehicle": MODEL.to_json()}})
        frame = draw(image, (5.0, 0.0), (0.0, 0.3), plan, 0.1, "t", geometry, reference=np.tile([0.3, 0.3], (16, 1)),
                     legend="cyan: the model's plan   green: the teacher's")
        green = (frame[:, :, 1] > 200) & (frame[:, :, 0] < 120) & (frame[:, :, 2] < 120)
        self.assertGreater(int(green.sum()), 50)


class ColabNotebookTests(unittest.TestCase):
    """tools/colab_train_carla.ipynb: its cells parse, and its helpers do what the training cell relies on."""

    def helpers(self):
        import ast
        notebook = json.loads((Path(__file__).resolve().parent.parent / "tools" / "colab_train_carla.ipynb").read_text())
        codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for source in codes:                               # IPython's ! lines are not Python
            ast.parse("\n".join(line for line in source.splitlines() if not line.lstrip().startswith("!")))
        namespace = {"WORK": "/content", "RUN": "smolvla-carla-v1", "DRIVE_DIR": "/drive", "STEPS": 8000, "BATCH": 16,
                     "SAVE_FREQ": 1000, "LOG_FREQ": 100, "VIDEO_BACKEND": "pyav", "data_root": Path("/content/data/x")}
        exec(next(c for c in codes if "def shown" in c), namespace)
        return namespace

    def test_the_step_line_is_kept_off_the_end_of_the_progress_bar(self):
        shown = self.helpers()["shown"]
        bar = "Training:  17%|##| 10/60 [00:05<00:19,  2.61step/s]INFO 2026 ot_train.py:451 step:10 loss:1.539\n"
        self.assertEqual(shown(bar), "INFO 2026 ot_train.py:451 step:10 loss:1.539\n")
        self.assertIsNone(shown("Training:  18%|##| 11/60 [00:05<00:18,  2.65step/s]\n"))
        self.assertIsNone(shown("  warnings.warn(\n"))
        self.assertEqual(shown("INFO ot_train.py:471 Checkpoint policy after step 1000\n"),
                         "INFO ot_train.py:471 Checkpoint policy after step 1000\n")

    def test_a_fresh_run_and_a_resumed_one(self):
        command = self.helpers()["train_command"]
        fresh = command(None)
        self.assertIn("--policy.path=lerobot/smolvla_base", fresh)
        self.assertIn('--rename_map={"observation.images.front": "observation.images.camera1"}', fresh)
        self.assertIn("--output_dir=/content/runs/smolvla-carla-v1", fresh)
        resumed = command(Path("/content/runs/smolvla-carla-v1/checkpoints/001000"))
        self.assertEqual(resumed[1], "--config_path=/content/runs/smolvla-carla-v1/checkpoints/last/pretrained_model/train_config.json")
        self.assertIn("--resume=true", resumed)


class ListeningTests(unittest.TestCase):
    """`connect` hands a port to carla.Client only once a server, not Docker's forwarder, holds it."""

    def serve(self, close_at_once):
        import socket
        import threading
        server = socket.create_server(("127.0.0.1", 0))
        held = []

        def accept():
            connection, _ = server.accept()
            if close_at_once:
                connection.close()                  # what docker-proxy does with nobody behind it
            else:
                held.append(connection)             # what an RPC server does: holds it, says nothing
        threading.Thread(target=accept, daemon=True).start()
        self.addCleanup(lambda: [c.close() for c in held] + [server.close()])
        return server.getsockname()[1]

    def test_a_server_that_holds_the_connection(self):
        from drive.carla.server import listening
        self.assertTrue(listening("127.0.0.1", self.serve(close_at_once=False), wait_s=0.3))

    def test_a_forwarder_with_nobody_behind_it(self):
        from drive.carla.server import listening
        self.assertFalse(listening("127.0.0.1", self.serve(close_at_once=True), wait_s=0.3))

    def test_nothing_there(self):
        import socket
        from drive.carla.server import listening
        with socket.create_server(("127.0.0.1", 0)) as probe:
            port = probe.getsockname()[1]
        self.assertFalse(listening("127.0.0.1", port, wait_s=0.3))


if __name__ == "__main__":
    unittest.main()
