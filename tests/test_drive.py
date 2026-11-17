"""drive/: the simulator, chunk execution, the safety supervisor and whole drives."""
import csv
import io
import json
import math
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drive import cli
from drive.config import SafetyLimits, VehicleParams
from drive.control import ChunkExecutor, SimulatedWorker, ThreadedWorker
from drive.episode import EpisodeConfig, build, run
from drive.policy.base import Policy
from drive.policy.lane_camera import road_mask, runs
from drive.policy.pure_pursuit import pursuit_steer
from drive.policy.vla import BackendPolicy
from drive.recorder import Recorder
from drive.safety import SafetySupervisor
from drive.sim import Obstacle, Track, World, step_vehicle
from drive.sim.render import ASPHALT
from drive.types import ActionChunk, Observation, VehicleState
from drive.viewer import HEIGHT, HUD_HEIGHT, WIDTH, Hud


def ramp(n=5, dt=0.1, t_obs=0.0) -> ActionChunk:
    """A chunk whose steer is its own index / 10, so a sample shows where it was read."""
    return ActionChunk(np.stack([np.arange(n) / 10, np.zeros(n)], axis=1), dt, t_obs)


class ConstantPolicy(Policy):
    def __init__(self, action=(0.0, 0.5)):
        self.action = action

    def infer(self, obs):
        return ActionChunk(np.tile(self.action, (10, 1)), 0.1, obs.t)


def observation(t=0.0) -> Observation:
    return Observation(t, [np.zeros((4, 4, 3), np.uint8)], 1.0, 0.2, "drive")


class ChunkTests(unittest.TestCase):
    def test_sample_interpolates_and_ends(self):
        chunk = ramp()
        self.assertAlmostEqual(chunk.sample(0.0)[0], 0.0)
        self.assertAlmostEqual(chunk.sample(0.25)[0], 0.25)
        self.assertAlmostEqual(chunk.sample(0.4)[0], 0.4)
        self.assertAlmostEqual(chunk.sample(-1.0)[0], 0.0)
        self.assertIsNone(chunk.sample(0.41))
        self.assertAlmostEqual(chunk.horizon_s, 0.4)

    def test_single_action_chunk(self):
        chunk = ActionChunk(np.array([[0.3, 0.1]]), 0.1, 0.0)
        np.testing.assert_allclose(chunk.sample(0.0), [0.3, 0.1])
        self.assertIsNone(chunk.sample(0.05))

    def test_rejects_wrong_shapes(self):
        for actions, dt in ((np.zeros((3, 3)), 0.1), (np.zeros((0, 2)), 0.1), (np.zeros((3, 2)), 0.0)):
            with self.assertRaises(ValueError):
                ActionChunk(actions, dt, 0.0)


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.track = Track.default()

    def test_closed_and_drivable(self):
        self.assertAlmostEqual(self.track.length, len(self.track.xy) * self.track.ds)
        tightest = 1 / np.abs(self.track.curvature).max()
        car = VehicleParams()
        self.assertGreater(tightest, 1.5 * car.wheelbase_m / math.tan(car.max_steer_rad))
        self.assertTrue((self.track.curvature > 0.1).any() and (self.track.curvature < -0.1).any())

    def test_project_inverts_at(self):
        for s in (0.0, 3.3, 11.0, self.track.length - 0.01):
            x, y, heading = self.track.at(s)
            found, lateral, found_heading = self.track.project(float(x), float(y))
            self.assertAlmostEqual(found, s, places=3)
            self.assertAlmostEqual(lateral, 0.0, places=3)
            self.assertAlmostEqual(found_heading, float(heading), places=6)

    def test_lateral_is_positive_to_the_left(self):
        left = self.track.obstacle(5.0, lateral_m=0.2)
        self.assertAlmostEqual(self.track.project(left.x, left.y)[1], 0.2, places=3)

    def test_at_wraps(self):
        np.testing.assert_allclose(self.track.at(1.0)[:2], self.track.at(1.0 + self.track.length)[:2], atol=1e-9)


class VehicleTests(unittest.TestCase):
    car = VehicleParams()

    def drive(self, action, seconds, state=VehicleState(0, 0, 0, 0, 0)):
        for _ in range(int(seconds / 0.02)):
            state = step_vehicle(state, action, self.car, 0.02)
        return state

    def test_straight(self):
        s = self.drive((0.0, 1.0), 2.0)
        self.assertGreater(s.x, 2.0)
        self.assertAlmostEqual(s.y, 0.0)
        self.assertLessEqual(s.speed, self.car.max_speed_mps)

    def test_brake_stops_without_reversing(self):
        s = self.drive((0.0, -1.0), 2.0, VehicleState(0, 0, 0, 2.0, 0))
        self.assertEqual(s.speed, 0.0)
        self.assertGreater(s.x, 0.0)

    def test_positive_steer_turns_left_after_a_lag(self):
        first = step_vehicle(VehicleState(0, 0, 0, 1.0, 0), (1.0, 0.0), self.car, 0.02)
        self.assertTrue(0.0 < first.steer < 0.5)
        self.assertGreater(self.drive((1.0, 0.3), 1.0).yaw, 0.5)

    def test_commands_are_clamped(self):
        self.assertEqual(self.drive((9.0, 9.0), 0.5), self.drive((1.0, 1.0), 0.5))


class WorldTests(unittest.TestCase):
    def test_camera_sees_road_below_and_sky_above(self):
        world = World()
        image = world.image()
        self.assertEqual((image.shape, image.dtype), ((240, 320, 3), np.uint8))
        self.assertLess(np.abs(image[230, 20].astype(int) - ASPHALT).max(), 12)
        self.assertFalse(world.camera.ground[0].any())
        self.assertTrue(world.camera.ground[-1].all())
        self.assertTrue(road_mask(image)[230, 20])
        self.assertFalse(road_mask(world.ground.image[:5, :5]).any())       # grass

    def test_range_sensor(self):
        world = World()
        s, front = world.state, world.vehicle.wheelbase_m + 0.07

        def cone(forward, left=0.0):
            return Obstacle(s.x + math.cos(s.yaw) * forward - math.sin(s.yaw) * left,
                            s.y + math.sin(s.yaw) * forward + math.cos(s.yaw) * left, 0.1)

        def seen(*obstacles):
            return World(world.track, obstacles).range_m()

        self.assertEqual(seen(), world.range_sensor.max_range_m)
        self.assertAlmostEqual(seen(cone(front + 1.5)), 1.4)
        self.assertAlmostEqual(seen(cone(front + 1.5), cone(front + 0.8)), 0.7)
        self.assertEqual(seen(cone(-1.0)), world.range_sensor.max_range_m)            # behind
        self.assertEqual(seen(cone(front + 1.0, left=1.0)), world.range_sensor.max_range_m)  # off to the side
        self.assertEqual(seen(cone(front + 9.0)), world.range_sensor.max_range_m)     # out of range

    def test_collision(self):
        world = World()
        s = world.state
        ahead = Obstacle(s.x + math.cos(s.yaw) * 0.3, s.y + math.sin(s.yaw) * 0.3, 0.1)
        self.assertTrue(World(world.track, (ahead,)).collided())
        self.assertFalse(World(world.track, (world.track.obstacle(5.0),)).collided())


class ExecutorTests(unittest.TestCase):
    def test_nothing_before_the_first_chunk(self):
        executor = ChunkExecutor()
        self.assertIsNone(executor.action(0.0))
        self.assertEqual(executor.age_s(1.0), math.inf)

    def test_latency_compensation_skips_stale_actions(self):
        executor = ChunkExecutor(blend_s=0.0)
        executor.update(ramp(t_obs=1.0), now=1.2)
        self.assertAlmostEqual(executor.action(1.2)[0], 0.2)
        self.assertAlmostEqual(executor.action(1.4)[0], 0.4)
        self.assertIsNone(executor.action(1.45))
        self.assertAlmostEqual(executor.age_s(1.3), 0.3)

    def test_without_compensation_plays_from_the_start(self):
        executor = ChunkExecutor(blend_s=0.0, latency_compensation=False)
        executor.update(ramp(t_obs=1.0), now=1.2)
        self.assertAlmostEqual(executor.action(1.2)[0], 0.0)
        self.assertAlmostEqual(executor.action(1.6)[0], 0.4)
        self.assertAlmostEqual(executor.age_s(1.3), 0.3)       # age is still measured from the observation

    def test_new_chunk_fades_in(self):
        executor = ChunkExecutor(blend_s=0.1)
        executor.update(ActionChunk(np.zeros((20, 2)), 0.1, 0.0), now=0.0)
        executor.update(ActionChunk(np.ones((20, 2)), 0.1, 1.0), now=1.0)
        self.assertAlmostEqual(executor.action(1.0)[0], 0.0)
        self.assertAlmostEqual(executor.action(1.05)[0], 0.5)
        self.assertAlmostEqual(executor.action(1.1)[0], 1.0)

    def test_fade_survives_an_exhausted_previous_chunk(self):
        executor = ChunkExecutor(blend_s=0.1)
        executor.update(ActionChunk(np.zeros((2, 2)), 0.1, 0.0), now=0.0)
        executor.update(ActionChunk(np.ones((20, 2)), 0.1, 1.0), now=1.0)
        self.assertAlmostEqual(executor.action(1.0)[0], 1.0)


class WorkerTests(unittest.TestCase):
    def test_simulated_latency(self):
        worker = SimulatedWorker(ConstantPolicy(), latency_s=0.1)
        self.assertFalse(worker.busy)
        worker.submit(observation(2.0), now=2.0)
        self.assertTrue(worker.busy)
        self.assertIsNone(worker.poll(2.08))
        chunk = worker.poll(2.1)
        self.assertEqual(chunk.t_obs, 2.0)
        self.assertFalse(worker.busy)
        self.assertIsNone(worker.poll(2.2))
        self.assertEqual(worker.latencies_s, [0.1])

    def test_jitter_is_seeded_and_never_negative(self):
        def draw(seed):
            worker = SimulatedWorker(ConstantPolicy(), latency_s=0.01, jitter_s=0.05, seed=seed)
            for i in range(50):
                worker.submit(observation(), now=0.0)
                worker.poll(math.inf)
            return worker.latencies_s

        self.assertEqual(draw(1), draw(1))
        self.assertNotEqual(draw(1), draw(2))
        self.assertGreaterEqual(min(draw(1)), 0.0)

    def wait(self, worker):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = worker.poll(0.0)
            if result is not None:
                return result
            time.sleep(0.002)
        self.fail("the worker never answered")

    def test_threaded_worker(self):
        worker = ThreadedWorker(ConstantPolicy(), extra_latency_s=0.02)
        try:
            for t in (1.0, 2.0):
                worker.submit(observation(t), now=t)
                self.assertTrue(worker.busy)
                self.assertEqual(self.wait(worker).t_obs, t)
                self.assertFalse(worker.busy)
            self.assertEqual(len(worker.latencies_s), 2)
            self.assertGreaterEqual(min(worker.latencies_s), 0.02)
        finally:
            worker.close()

    def test_threaded_worker_raises_policy_errors_in_the_loop(self):
        class Broken(Policy):
            def infer(self, obs):
                raise RuntimeError("engine fell over")

        worker = ThreadedWorker(Broken())
        try:
            worker.submit(observation(), now=0.0)
            with self.assertRaisesRegex(RuntimeError, "engine fell over"):
                self.wait(worker)
        finally:
            worker.close()


class SafetyTests(unittest.TestCase):
    def check(self, command, supervisor=None, **inputs):
        supervisor = supervisor or SafetySupervisor()
        inputs = {"speed": 1.0, "range_m": 4.0, "plan_age_s": 0.1, "dt": 0.02, **inputs}
        return supervisor.filter(None if command is None else np.array(command, float), **inputs)

    def test_passes_a_sane_command(self):
        decision = self.check([0.1, 0.5])
        np.testing.assert_allclose(decision.action, [0.1, 0.5])
        self.assertEqual(decision.reasons, ())

    def test_brakes_without_a_fresh_valid_plan(self):
        for command, inputs, reason in ((None, {}, "stale"),
                                        ([0.0, 0.5], {"plan_age_s": 0.7}, "stale"),
                                        ([float("nan"), 0.5], {}, "invalid"),
                                        ([0.0, float("inf")], {}, "invalid")):
            decision = self.check(command, **inputs)
            self.assertEqual((decision.action[1], decision.reasons), (-1.0, (reason,)))

    def test_estop_holds_until_released(self):
        supervisor = SafetySupervisor()
        supervisor.estop()
        self.assertEqual(self.check([0.0, 1.0], supervisor).reasons, ("estop",))
        self.assertEqual(self.check([0.0, 1.0], supervisor).action[1], -1.0)
        supervisor.estop(False)
        self.assertEqual(self.check([0.0, 1.0], supervisor).reasons, ())

    def test_clamps_and_limits_steering_rate(self):
        supervisor = SafetySupervisor(SafetyLimits(max_steer_rate_per_s=5.0))
        decision = self.check([3.0, 7.0], supervisor)
        np.testing.assert_allclose(decision.action, [0.1, 1.0])
        self.assertEqual(decision.reasons, ("steer_rate",))
        np.testing.assert_allclose(self.check([3.0, 0.0], supervisor).action[0], 0.2)
        # A brake keeps the wheels where they were, and the limit continues from there.
        np.testing.assert_allclose(self.check(None, supervisor).action[0], 0.2)

    def test_speed_limit(self):
        decision = self.check([0.0, 1.0], speed=2.3)
        self.assertLess(decision.action[1], 0.0)
        self.assertEqual(decision.reasons, ("speed",))
        self.assertEqual(self.check([0.0, 1.0], speed=9.0).action[1], -1.0)

    def test_obstacle_ladder(self):
        # At 1 m/s and 4 m/s^2 the car stops in 0.125 m, plus the 0.30 m margin.
        self.assertEqual(self.check([0.0, 0.5], range_m=1.0).reasons, ())
        slow = self.check([0.0, 0.5], range_m=0.47)
        self.assertEqual((slow.action[1], slow.reasons), (0.0, ("obstacle_slow",)))
        brake = self.check([0.0, 0.5], range_m=0.40)
        self.assertEqual((brake.action[1], brake.reasons), (-1.0, ("obstacle_brake",)))
        stop = self.check([0.0, 0.5], range_m=0.25, speed=0.0)
        self.assertEqual((stop.action[1], stop.reasons), (-1.0, ("obstacle_stop",)))


class DriveTests(unittest.TestCase):
    def test_teacher_laps_the_track(self):
        result = run(EpisodeConfig(duration_s=16.0))
        self.assertGreater(result["laps"], 1.0)
        self.assertLess(result["lateral_max_m"], 0.06)
        self.assertEqual(result["off_track_pct"], 0.0)
        self.assertEqual(result["inferences"], 160)
        # Only the wait for the first plan, and the first swing of the wheels.
        self.assertEqual(set(result["safety_s"]), {"stale", "steer_rate"})
        self.assertLess(result["safety_s"]["stale"], 0.15)

    def test_runs_are_reproducible(self):
        config = EpisodeConfig(duration_s=4.0, jitter_ms=30.0, seed=3)
        self.assertEqual(run(config), run(config))

    def test_camera_policy_stays_on_the_road(self):
        result = run(EpisodeConfig(policy="lane-camera", duration_s=27.0))
        self.assertGreater(result["laps"], 1.0)
        self.assertEqual(result["off_track_pct"], 0.0)
        self.assertLess(result["lateral_max_m"], 0.30)

    def test_supervisor_stops_for_an_obstacle_the_policy_ignores(self):
        guarded = run(EpisodeConfig(duration_s=10.0, obstacles_s=(8.0,)))
        self.assertFalse(guarded["collided"])
        self.assertAlmostEqual(guarded["min_range_m"], 0.30, delta=0.03)
        self.assertGreater(guarded["safety_s"]["obstacle_stop"], 3.0)
        unguarded = run(EpisodeConfig(duration_s=10.0, obstacles_s=(8.0,), safety=False))
        self.assertTrue(unguarded["collided"])

    def test_watchdog_stops_a_model_too_slow_for_it(self):
        result = run(EpisodeConfig(duration_s=5.0, latency_ms=650.0))
        self.assertEqual(result["safety_s"], {"stale": 5.0})
        self.assertEqual(result["laps"], 0.0)

    def test_realtime_thread_drives(self):
        result = run(EpisodeConfig(duration_s=1.5, latency_ms=20.0, realtime=True))
        self.assertGreater(result["mean_speed_mps"], 0.3)
        self.assertGreater(result["inferences"], 10)

    def test_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "unknown policy"):
            run(EpisodeConfig(policy="autopilot"))


class PolicyTests(unittest.TestCase):
    def test_pursuit_steer(self):
        car = VehicleParams()
        self.assertEqual(pursuit_steer(1.0, 0.0, car), 0.0)
        self.assertGreater(pursuit_steer(1.0, 0.3, car), 0.0)
        self.assertEqual(pursuit_steer(1.0, 0.3, car), -pursuit_steer(1.0, -0.3, car))
        self.assertEqual(pursuit_steer(0.1, 1.0, car), 1.0)
        self.assertEqual(pursuit_steer(0.0, 0.0, car), 0.0)

    def test_runs(self):
        row = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
        self.assertEqual(runs(row), [(0, 2), (3, 4), (6, 9)])
        self.assertEqual(runs(np.zeros(4, bool)), [])

    def test_backend_policy_adapts_a_vla_backend(self):
        seen = []

        class Backend:
            def infer(self, obs):
                seen.append(obs)
                chunk = np.zeros((50, 32), np.float32)
                chunk[:, 0], chunk[:, 1] = 0.25, 3.0
                return SimpleNamespace(chunk=chunk, timings_ms={"total": 100.0})

        policy = BackendPolicy(Backend(), chunk_size=50, noise_width=32, chunk_dt_s=1 / 30)
        chunk = policy.infer(observation(t=4.0))
        policy.infer(observation(t=4.1))
        self.assertEqual((chunk.actions.shape, chunk.dt, chunk.t_obs), ((50, 2), 1 / 30, 4.0))
        np.testing.assert_allclose(chunk.actions[0], [0.25, 1.0])            # clipped to the action range
        np.testing.assert_allclose(seen[0].state, [1.0, 0.2])
        self.assertEqual((seen[0].task, seen[0].noise.shape, seen[0].noise.dtype), ("drive", (1, 50, 32), np.float32))
        self.assertEqual([o.index for o in seen], [0, 1])
        self.assertFalse(np.array_equal(seen[0].noise, seen[1].noise))


class OutputTests(unittest.TestCase):
    def test_recorder(self):
        with tempfile.TemporaryDirectory() as d:
            recorder = Recorder(Path(d) / "episode", task="drive", hz=10.0)
            for i in range(100):                                  # 2 s at 50 Hz
                t = i * 0.02
                if recorder.due(t):
                    recorder.add(t, np.full((4, 6, 3), i, np.uint8), 1.0, 0.1, np.array([0.2, 0.3]))
            data = np.load(recorder.save() / "episode.npz")
            self.assertEqual(data["images"].shape, (20, 4, 6, 3))
            np.testing.assert_allclose(data["t"], np.arange(20) * 0.1, atol=1e-6)
            np.testing.assert_allclose(data["actions"][0], [0.2, 0.3])
            np.testing.assert_allclose(data["states"][0], [1.0, 0.1])
            meta = json.loads((Path(d) / "episode/meta.json").read_text())
            self.assertEqual((meta["frames"], meta["fps"], meta["action"]), (20, 10.0, ["steer", "throttle"]))

    def test_cli_writes_result_video_and_recording(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as out:
            d = Path(d)
            code = cli.main(["sim", "--duration-s", "2", "--obstacle", "8", "--out", str(d / "r.json"),
                             "--video", str(d / "v.mp4"), "--record", str(d / "rec")])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((d / "r.json").read_text())["config"]["obstacles_s"], [8.0])
            self.assertGreater((d / "v.mp4").stat().st_size, 10_000)
            self.assertEqual(len(np.load(d / "rec/episode.npz")["t"]), 20)
            self.assertIn("obstacles: not hit", out.getvalue())

    def test_action_log_and_debug_panel(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as out:
            log = Path(d) / "actions.csv"
            self.assertEqual(cli.main(["sim", "--duration-s", "1", "--log", str(log)]), 0)
            self.assertIn("action log", out.getvalue())
            rows = list(csv.DictReader(log.open()))
        self.assertEqual(len(rows), 50)                                    # every tick at 50 Hz
        self.assertEqual((rows[0]["overrides"], rows[0]["cmd_steer"], rows[0]["throttle"]), ("stale", "", "-1.0"))
        arrivals = [r for r in rows if r["new_chunk"] == "1"]
        self.assertEqual(len(arrivals), 9)                                 # one every 100 ms after the first
        self.assertEqual((arrivals[0]["t"], arrivals[0]["chunk_t_obs"], arrivals[0]["plan_age_ms"]), ("0.1", "0.0", "100.0"))
        self.assertNotEqual(arrivals[0]["cmd_steer"], "")
        # The HUD grows a debug panel underneath, for every policy.
        world, loop = build(EpisodeConfig())
        for _ in range(8):
            tick = loop.step()
        frame = Hud(world).frame(world, loop, tick)
        self.assertEqual(frame.shape, (HUD_HEIGHT, WIDTH, 3))
        self.assertGreater(frame[HEIGHT:].std(), 10)                       # something is drawn there
        loop.worker.close()

    def test_cli_sweep(self):
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["sweep", "--duration-s", "1", "--latencies", "0,100"]), 0)
        self.assertEqual(len(out.getvalue().strip().splitlines()), 6)


if __name__ == "__main__":
    unittest.main()
