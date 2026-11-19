"""drive/learn: collecting, labelling and checking data, training, and judging the result."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from drive import cli
from drive.episode import EpisodeConfig, run
from drive.learn import dataset as datasets
from drive.learn import export
from drive.learn.browse import browse, plan_path
from drive.learn.collect import collect
from drive.learn.evaluate import report_name
from drive.learn.label import PLAN_DT_S, PLAN_STEPS, label, plans_from_actions, read_labels
from drive.learn.project import Project
from drive.learn.status import next_step, summary
from drive.learn.validate import validate
from drive.recorder import Recorder
from drive.sim import SHAPES, Track, World
from drive.types import ActionChunk

try:
    import torch
except ImportError:                      # the simulator runs without PyTorch; training does not
    torch = None
if torch is not None and not hasattr(torch, "nn"):
    torch = None                         # an empty leftover directory named torch

try:
    import lerobot
except ImportError:                      # only an environment set up for VLA fine-tuning has it
    lerobot = None

needs_torch = unittest.skipIf(torch is None, "PyTorch is not installed (make setup, or use the container: make test)")


def quiet(*a, **k):
    pass


def small_dataset(directory: Path, episodes=3, duration_s=2.0) -> Path:
    collect(directory, episodes=episodes, duration_s=duration_s, log=quiet)
    return directory


class TrackShapeTests(unittest.TestCase):
    def test_every_shape_is_drivable(self):
        for name in SHAPES:
            track = Track.named(name)
            self.assertGreater(1 / np.abs(track.curvature).max(), 1.0, name)
            self.assertGreater(track.curvature.mean(), 0.0, name)             # counter-clockwise

    def test_reverse_swaps_the_bends(self):
        forward, backward = Track.named("peanut"), Track.named("peanut", reverse=True)
        self.assertAlmostEqual(forward.length, backward.length, places=6)
        self.assertAlmostEqual(forward.curvature.max(), -backward.curvature.min(), places=2)
        self.assertLess(backward.curvature.mean(), 0.0)

    def test_unknown_track(self):
        with self.assertRaisesRegex(ValueError, "unknown track"):
            Track.named("monaco")

    def test_place_off_centre(self):
        world = World()
        world.place(5.0, lateral_m=0.2, yaw_offset_rad=0.1, speed=1.0)
        s, lateral, heading = world.track.project(world.state.x, world.state.y)
        self.assertAlmostEqual(s, 5.0, places=2)
        self.assertAlmostEqual(lateral, 0.2, places=3)
        self.assertAlmostEqual(world.state.yaw - heading, 0.1, places=3)
        self.assertEqual(world.state.speed, 1.0)


class CollectTests(unittest.TestCase):
    def test_episodes_carry_the_teachers_plans(self):
        with tempfile.TemporaryDirectory() as d:
            summary = collect(Path(d) / "data", episodes=2, duration_s=2.0, log=quiet)
            self.assertEqual(summary["frames"], 40)
            self.assertLess(summary["max_lateral_m"], 0.45)
            self.assertEqual(summary["off_road_frames"], 0)                # 0.45 m is half the road's width
            episode = Path(d) / "data/sim-001"
            data, labels = np.load(episode / "episode.npz"), read_labels(episode)
            self.assertEqual(sorted(data), ["actions", "images", "states", "t"])
            self.assertEqual(data["images"].shape, (20, 240, 320, 3))
            self.assertEqual(labels["plans"].shape, (20, PLAN_STEPS, 2))
            self.assertEqual((labels["dt"], labels["source"], bool(labels["valid"].all())), (PLAN_DT_S, "teacher", True))
            self.assertLessEqual(np.abs(labels["plans"]).max(), 1.0)
            # The disturbance is in what was applied, never in the label.
            self.assertGreater(np.abs(data["actions"][1:, 0] - labels["plans"][1:, 0, 0]).mean(), 0.02)
            self.assertEqual(json.loads((episode / "meta.json").read_text())["frames"], 20)

    def test_reproducible_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a", "b"):
                collect(Path(d) / name, episodes=1, duration_s=1.0, seed=5, log=quiet)
            a, b = (np.load(Path(d) / name / "sim-000/episode.npz") for name in ("a", "b"))
            for key in ("images", "states", "actions"):
                np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(*(read_labels(Path(d) / name / "sim-000")["plans"] for name in "ab"))
            with self.assertRaisesRegex(ValueError, "another --name"):
                collect(Path(d) / "a", episodes=1, duration_s=1.0, log=quiet)
            collect(Path(d) / "a", episodes=1, duration_s=1.0, name="more", log=quiet)
            self.assertEqual(sorted(p.name for p in (Path(d) / "a").iterdir()), ["more-000", "sim-000"])

    def test_one_target_speed_for_every_drive(self):
        with tempfile.TemporaryDirectory() as d:
            collect(Path(d) / "fixed", episodes=2, duration_s=4.0, speeds=(1.2,), log=quiet)
            top = [np.load(e / "episode.npz")["states"][:, 0].max() for e in sorted((Path(d) / "fixed").iterdir())]
            np.testing.assert_allclose(top, [1.2, 1.2], atol=0.05)                 # both cruise at 1.2 m/s

    def test_watching_does_not_change_the_recording(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            collect(d / "plain", episodes=2, duration_s=1.0, seed=3, log=quiet)
            done = collect(d / "watched", episodes=2, duration_s=1.0, seed=3, log=quiet, video=d / "w.mp4")
            self.assertEqual((done["episodes"], done["frames"], done["stopped"]), (2, 20, False))
            for episode in ("sim-000", "sim-001"):
                a, b = (np.load(d / kind / episode / "episode.npz") for kind in ("plain", "watched"))
                for key in ("images", "states", "actions"):
                    np.testing.assert_array_equal(a[key], b[key])
            self.assertGreater((d / "w.mp4").stat().st_size, 20_000)       # 2 s at 25 frames a second

    def test_without_noise_the_applied_action_is_the_plan(self):
        with tempfile.TemporaryDirectory() as d:
            collect(Path(d) / "data", episodes=1, duration_s=2.0, steer_noise=0.0, log=quiet)
            data = np.load(Path(d) / "data/sim-000/episode.npz")
            plans = read_labels(Path(d) / "data/sim-000")["plans"]
            # actions[i] was applied just before frame i: one 20 ms tick behind plans[i][0].
            np.testing.assert_allclose(data["actions"][5:, 0], plans[5:, 0, 0], atol=0.05)


class DatasetTests(unittest.TestCase):
    def test_load_and_split_by_episode(self):
        with tempfile.TemporaryDirectory() as d:
            data = datasets.load(small_dataset(Path(d) / "data"))
            self.assertEqual(data.images.shape, (60, *datasets.IMAGE_HW, 3))
            self.assertEqual((data.states.shape, data.plans.shape), ((60, 2), (60, PLAN_STEPS, 2)))
            self.assertEqual(data.plan_dt, PLAN_DT_S)
            training, validation = data.split()
            self.assertEqual((len(training), len(validation)), (40, 20))
            self.assertEqual(set(validation.episode), {2})
            self.assertFalse(set(training.episode) & set(validation.episode))

    def test_errors(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "no episodes"):
                datasets.load(Path(d))
            recorder = Recorder(Path(d) / "plain/episode-000", "drive")
            recorder.add(0.0, np.zeros((4, 4, 3), np.uint8), 0.0, 0.0, np.zeros(2))
            recorder.save()
            with self.assertRaisesRegex(ValueError, "not labelled"):
                datasets.load(Path(d) / "plain")
            with self.assertRaisesRegex(ValueError, "two episodes"):
                datasets.load(small_dataset(Path(d) / "one", episodes=1, duration_s=1.0)).split()

    def test_recorder_wants_a_plan_for_every_frame(self):
        with tempfile.TemporaryDirectory() as d:
            recorder = Recorder(Path(d) / "e", "drive")
            image = np.zeros((4, 4, 3), np.uint8)
            recorder.add(0.0, image, 0.0, 0.0, np.zeros(2), ActionChunk(np.zeros((3, 2)), 0.1, 0.0))
            recorder.add(0.1, image, 0.0, 0.0, np.zeros(2))
            with self.assertRaisesRegex(ValueError, "every frame"):
                recorder.save()


def manual_drive(directory: Path, seconds=4.0, hz=10.0, steer=0.3) -> Path:
    """An episode as a person or a ROS 2 recording leaves it: no labels."""
    recorder = Recorder(directory, "drive", hz)
    for i in range(int(seconds * hz)):
        t = i / hz
        recorder.add(t, np.full((240, 320, 3), i, np.uint8), 1.0, steer, np.array([steer * np.sin(t), 0.2]))
    return recorder.save()


class LabelTests(unittest.TestCase):
    def test_a_plan_is_the_actions_that_followed(self):
        t = np.arange(40) / 10
        actions = np.stack([t / 10, np.full(40, 0.5)], axis=1)
        plans, valid = plans_from_actions(t, actions, steps=4, dt=0.2)
        self.assertEqual(plans.shape, (40, 4, 2))
        np.testing.assert_allclose(plans[5, :, 0], [0.05, 0.07, 0.09, 0.11], atol=1e-6)   # t = 0.5, 0.7, 0.9, 1.1
        np.testing.assert_allclose(plans[..., 1], 0.5)
        # The last plan that fits starts 0.6 s before the end.
        self.assertEqual(valid.tolist(), [True] * 34 + [False] * 6)

    def test_label_fills_in_only_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            collect(d, episodes=1, duration_s=2.0, log=quiet)
            (d / "sim-000").rename(d / "taught")
            manual_drive(d / "manual")
            self.assertEqual(label(d), {"labelled": 1, "already_labelled": 1, "frames": 25})
            self.assertEqual(read_labels(d / "taught")["source"], "teacher")
            labels = read_labels(d / "manual")
            self.assertEqual((labels["source"], labels["plans"].shape, int(labels["valid"].sum())),
                             ("actions", (40, PLAN_STEPS, 2), 25))
            actions = np.load(d / "manual/episode.npz")["actions"]
            np.testing.assert_allclose(labels["plans"][3], actions[3:3 + PLAN_STEPS], atol=1e-6)
            self.assertEqual(label(d)["labelled"], 0)
            self.assertEqual(label(d, relabel=True)["labelled"], 2)
            self.assertEqual(read_labels(d / "taught")["source"], "actions")
            self.assertEqual(len(datasets.load(d)), 25 + 5)               # only the frames with a full future

    def test_label_needs_episodes(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, "no episodes"):
            label(Path(d))


class ValidateTests(unittest.TestCase):
    def report(self, directory):
        lines = []
        return validate(directory, log=lines.append), "\n".join(lines)

    def test_a_good_dataset_passes(self):
        with tempfile.TemporaryDirectory() as d:
            result, text = self.report(small_dataset(Path(d) / "data"))
            self.assertTrue(result["ok"])
            self.assertEqual((result["episodes"], result["frames"], result["labelled_frames"]), (3, 60, 60))
            self.assertEqual(result["label_sources"], ["teacher"])
            self.assertEqual(result["warnings"], ["60 frames is little; the example network was trained on 12 000"])
            self.assertIn("dataset: ok, 1 warning(s)", text)
            self.assertIn("images 240x320", text)

    def test_what_it_catches(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            manual_drive(d / "unlabelled")
            manual_drive(d / "frozen")
            manual_drive(d / "broken")
            manual_drive(d / "small")
            for name, change in (("frozen", lambda a: a["images"].__setitem__(slice(5, 9), a["images"][4])),
                                 ("broken", lambda a: (a["actions"].__setitem__((3, 0), 4.0),
                                                       a["t"].__setitem__(10, a["t"][9]))),
                                 ("small", lambda a: a.__setitem__("images", a["images"][:, ::2, ::2]))):
                arrays = dict(np.load(d / name / "episode.npz"))
                change(arrays)
                np.savez_compressed(d / name / "episode.npz", **arrays)
            label(d)
            (d / "unlabelled/labels.npz").unlink()
            result, text = self.report(d)
            self.assertFalse(result["ok"])
            self.assertIn("dataset: FAILED", text)
            found = " | ".join(result["failures"] + result["warnings"])
            for expected in ("unlabelled: not labelled yet", "frozen: 4 frame(s) identical",
                             "broken: an action outside [-1, 1]: 4.00", "broken: timestamps do not increase",
                             "episodes differ in image size"):
                self.assertIn(expected, found)

    def test_perfect_driving_and_single_episode_are_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            collect(Path(d) / "data", episodes=1, duration_s=3.0, steer_noise=0.0, log=quiet)
            result, _ = self.report(Path(d) / "data")
            self.assertIn("one episode only", result["failures"][0])
            self.assertTrue(any("perfect driving only" in w for w in result["warnings"]))

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            small_dataset(Path(d) / "good/episodes")
            self.assertEqual(cli.main(["validate", "--project", str(Path(d) / "good")]), 0)
            manual_drive(Path(d) / "bad/episodes/only")
            self.assertEqual(cli.main(["validate", "--project", str(Path(d) / "bad")]), 1)


class BrowseTests(unittest.TestCase):
    def test_a_straight_plan_draws_up_the_middle(self):
        world = World()
        path = plan_path(np.tile([0.0, 0.3], (PLAN_STEPS, 1)), PLAN_DT_S, speed=1.0, steer=0.0)
        self.assertTrue((np.diff(path[:, 0]) > 0).all() and np.allclose(path[:, 1], 0.0))
        pixels = world.camera.project(*path.T)[1:]          # the first point is under the camera, out of view
        np.testing.assert_allclose(pixels[:, 0], 160.0)
        self.assertTrue((np.diff(pixels[:, 1]) < 0).all())            # farther is higher in the image
        self.assertTrue(((pixels[:, 1] > 90) & (pixels[:, 1] < 240)).all())   # below the horizon (row 90)
        left = world.camera.project(*plan_path(np.tile([0.8, 0.3], (PLAN_STEPS, 1)), PLAN_DT_S, 1.0, 0.0).T)
        self.assertLess(left[-1, 0], 100.0)                           # a left turn bends left on screen

    def test_projection_inverts_the_cameras_ground_map(self):
        camera = World().camera
        for row, column in ((200, 40), (150, 160), (230, 300)):
            pixel = camera.project(camera.forward_m[row, column], camera.left_m[row, column])
            np.testing.assert_allclose(pixel, [column + 0.5, row + 0.5], atol=1e-2)
        self.assertTrue(np.isnan(camera.project(-1.0, 0.0)).all())

    def test_video_of_a_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            small_dataset(d / "data", episodes=2, duration_s=1.0)
            self.assertEqual(browse(d / "data", show=False, video=d / "b.mp4"), 20)
            self.assertEqual(browse(d / "data", show=False, video=d / "c.mp4", every=4), 5)
            self.assertGreater((d / "b.mp4").stat().st_size, 20_000)
            self.assertEqual(browse(d / "data", show=False, video=d / "d.mp4", match="sim-001"), 10)
            with self.assertRaisesRegex(ValueError, "named 'ros-\\*'"):
                browse(d / "data", show=False, video=d / "e.mp4", match="ros-*")


class EvaluateTests(unittest.TestCase):
    def test_teacher_passes_every_track(self):
        from drive.learn.evaluate import evaluate
        lines = []
        report = evaluate(EpisodeConfig(duration_s=6.0), tracks=("peanut", "oval"), log=lines.append)
        self.assertTrue(report["passed"])
        self.assertEqual([(d["track"], d["reverse"]) for d in report["drives"]],
                         [("peanut", False), ("peanut", True), ("oval", False), ("oval", True)])
        self.assertEqual(lines[-1], "4/4 drives passed")

    def test_a_drive_that_hits_something_fails(self):
        from drive.learn.evaluate import evaluate
        lines = []
        report = evaluate(EpisodeConfig(duration_s=10.0, obstacles_s=(8.0,), safety=False), tracks=("peanut",),
                          log=lines.append)
        self.assertFalse(report["passed"])
        self.assertEqual(report["summary"]["drives_passed"], 0)
        self.assertIn("FAIL", lines[1])

    def test_cli_compares_models_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as out:
            code = cli.main(["evaluate", "pure-pursuit", "lane-camera", "--project", d, "--tracks", "oval",
                             "--duration-s", "5"])
            self.assertEqual(code, 0)
            report = json.loads((Path(d) / "reports/lane-camera.json").read_text())
            self.assertEqual((report["policy"], report["summary"]["drives_passed"]), ("lane-camera", 2))
            self.assertEqual(report["max_plan_age_ms"], 600.0)
            cli.main(["evaluate", "pure-pursuit", "--project", d, "--tracks", "oval", "--duration-s", "2",
                      "--latency-ms", "400", "--max-plan-age-ms", "1000"])
            slow = json.loads((Path(d) / "reports/pure-pursuit.json").read_text())
            self.assertEqual((slow["latency_ms"], slow["max_plan_age_ms"]), (400.0, 1000.0))
            self.assertTrue((Path(d) / "reports/pure-pursuit.json").is_file())
        self.assertRegex(out.getvalue(), r"pure-pursuit\s+2/2")

    def test_a_report_is_named_after_the_whole_version(self):
        with tempfile.TemporaryDirectory() as d:
            project = Project(Path(d))
            for version in ("v1", "v1.2"):
                project.model(version).parent.mkdir(parents=True, exist_ok=True)
                project.model(version).write_bytes(b"")
            # v1.2 must not overwrite v1's report: status looks it up as reports/v1.2.json
            self.assertEqual(report_name("v1.2", project.checkpoint("v1.2")), "v1.2")
            self.assertEqual(report_name("v1", project.checkpoint("v1")), "v1")
            self.assertEqual(report_name(str(project.model("v1.2")), project.model("v1.2")), "v1.2")


class VlaDeviceTests(unittest.TestCase):
    """drive/policy/lerobot_backend.py picks where the VLA runs; it needs neither LeRobot nor a GPU to test."""

    @staticmethod
    def fake_torch(cuda: bool, mps: bool):
        from types import SimpleNamespace
        calls = []
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda, synchronize=lambda: calls.append("cuda")),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
            mps=SimpleNamespace(synchronize=lambda: calls.append("mps")), calls=calls)

    def test_cuda_then_the_macs_gpu_then_the_cpu_unless_told(self):
        from unittest import mock
        from drive.policy.lerobot_backend import pick_device
        with mock.patch.dict("os.environ", {}, clear=False) as environ:
            environ.pop("VLA_DEVICE", None)
            self.assertEqual(pick_device(self.fake_torch(cuda=True, mps=True)), "cuda")
            self.assertEqual(pick_device(self.fake_torch(cuda=False, mps=True)), "mps")
            self.assertEqual(pick_device(self.fake_torch(cuda=False, mps=False)), "cpu")
            environ["VLA_DEVICE"] = "cpu"
            self.assertEqual(pick_device(self.fake_torch(cuda=True, mps=True)), "cpu")

    def test_timings_wait_for_the_gpu_in_use(self):
        from drive.policy.lerobot_backend import synchronize
        torch = self.fake_torch(cuda=True, mps=True)
        for device in ("cuda:0", "mps", "cpu"):
            synchronize(torch, device)
        self.assertEqual(torch.calls, ["cuda", "mps"])

    def test_config_fields_only_a_newer_lerobot_knows_are_found(self):
        from dataclasses import dataclass
        from drive.policy.lerobot_backend import unknown_config_fields

        @dataclass
        class Base:
            device: str = "cpu"

        @dataclass
        class Config(Base):
            chunk_size: int = 16

        raw = {"type": "smolvla", "device": "cuda", "chunk_size": 16, "pretrained_revision": None, "z_new": 1}
        self.assertEqual(unknown_config_fields(raw, Config), ["pretrained_revision", "z_new"])
        self.assertEqual(unknown_config_fields({"type": "smolvla", "chunk_size": 8}, Config), [])


class ExportTests(unittest.TestCase):
    """Episodes as a LeRobotDataset. The per-frame mapping needs only numpy; writing needs lerobot."""

    def test_frames_carry_the_teachers_intended_action_not_the_disturbed_one(self):
        with tempfile.TemporaryDirectory() as d:
            episode = small_dataset(Path(d), episodes=1, duration_s=2.0) / "sim-000"
            data, labels = np.load(episode / "episode.npz"), read_labels(episode)
            meta, rows = export.frames(episode)
            self.assertEqual((meta["fps"], len(rows)), (10.0, 20))
            self.assertEqual(sorted(rows[0]), sorted([export.CAMERA, "observation.state", "action", "task"]))
            np.testing.assert_array_equal(rows[5][export.CAMERA], data["images"][5])
            np.testing.assert_allclose(rows[5]["observation.state"], data["states"][5])
            np.testing.assert_allclose([r["action"] for r in rows], labels["plans"][:, 0], atol=1e-6)
            _, applied = export.frames(episode, "applied")
            np.testing.assert_allclose([r["action"] for r in applied], data["actions"], atol=1e-6)
            self.assertGreater(np.abs(labels["plans"][1:, 0, 0] - data["actions"][1:, 0]).mean(), 0.02)
            self.assertEqual(export.features((240, 320, 3))[export.CAMERA]["shape"], (240, 320, 3))

    def test_frames_refuse_unlabelled_and_unsteady_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            episode = manual_drive(Path(d) / "manual")
            with self.assertRaisesRegex(ValueError, "not labelled"):
                export.frames(episode)
            label(Path(d))
            _, rows = export.frames(episode)                          # labelled from its actions
            np.testing.assert_allclose([r["action"] for r in rows], np.load(episode / "episode.npz")["actions"],
                                       atol=1e-6)
            recorder = Recorder(Path(d) / "jittery", "drive", 10.0)
            for i in range(20):                                       # every other frame 60 ms late
                recorder.add(i / 10 + 0.06 * (i % 2), np.zeros((240, 320, 3), np.uint8), 1.0, 0.0, [0.0, 0.2])
            jittery = recorder.save()
            label(Path(d))
            with self.assertRaisesRegex(ValueError, "stray 60 ms"):
                export.frames(jittery)
            with self.assertRaisesRegex(ValueError, "--action"):
                export.frames(episode, "wanted")

    def test_export_writes_a_dataset_lerobot_can_read(self):
        with tempfile.TemporaryDirectory() as d:
            data = small_dataset(Path(d) / "episodes", episodes=2, duration_s=1.0)
            if lerobot is None:
                with self.assertRaisesRegex(ValueError, "needs LeRobot"):
                    export.export(data, Path(d) / "lerobot", log=quiet)
                return
            done = export.export(data, Path(d) / "lerobot", match="sim-*", log=quiet)
            self.assertEqual((done["episodes"], done["frames"], done["fps"]), (2, 20, 10))
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset("local/test", root=Path(d) / "lerobot")
            self.assertEqual((len(dataset), dataset.meta.total_episodes), (20, 2))
            plans = read_labels(data / "sim-001")["plans"]
            np.testing.assert_allclose(dataset[13]["action"].numpy(), plans[3, 0], atol=1e-6)
            with self.assertRaisesRegex(ValueError, "exists"):
                export.export(data, Path(d) / "lerobot", log=quiet)

    def test_a_vla_version_is_found_and_driven_by_the_vla_policy(self):
        with tempfile.TemporaryDirectory() as d:
            project = Project(Path(d))
            checkpoint = project.vla("smolvla-v1")
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
            self.assertEqual(project.checkpoint("smolvla-v1"), checkpoint)
            self.assertEqual(project.checkpoint(str(checkpoint)), checkpoint)
            self.assertEqual(project.policy_for(checkpoint), "vla")
            self.assertEqual(report_name("smolvla-v1", checkpoint), "smolvla-v1")
            self.assertEqual(report_name(str(checkpoint), checkpoint), "smolvla-v1")
            with self.assertRaisesRegex(ValueError, "needs a checkpoint"):
                run(EpisodeConfig(policy="vla", duration_s=0.1))


class SeeVlaTests(unittest.TestCase):
    """tools/see_vla.py: the camera pixels each SmolVLA image token covers. Needs no model."""

    def test_a_4_3_camera_fills_the_lower_six_rows_of_tokens(self):
        from tools.see_vla import cells
        boxes = cells(240, 320, (512, 512), 8)
        self.assertEqual(len(boxes), 48)                       # 16 of the 64 tokens are padding
        self.assertEqual({row for row, *_ in boxes}, set(range(2, 8)))
        self.assertEqual(boxes[0], (2, 0, 0, 40, 0, 40))
        self.assertEqual(boxes[-1], (7, 7, 200, 240, 280, 320))
        covered = np.zeros((240, 320), int)
        for _, _, y0, y1, x0, x1 in boxes:
            covered[y0:y1, x0:x1] += 1
        self.assertTrue((covered == 1).all())                   # every pixel in exactly one cell

    def test_padding_goes_left_for_a_tall_image_and_vanishes_for_a_square_one(self):
        from tools.see_vla import cells
        self.assertEqual({col for _, col, *_ in cells(320, 240, (512, 512), 8)}, set(range(2, 8)))
        self.assertEqual(len(cells(256, 256, (512, 512), 8)), 64)


class VlaAppTests(unittest.TestCase):
    """tools/vla_app.py: the edits it makes to an image before the model sees it. Needs no model or gradio."""

    def setUp(self):
        self.image = np.zeros((240, 320, 3), np.uint8)
        self.image[:, :160] = (200, 30, 30)                    # left half red, right half black
        self.image[200:210, 100:110] = 255                      # a white mark on the near road

    def test_hiding_fills_exactly_the_band_and_reports_it(self):
        from tools.vla_app import prepare
        out, changed = prepare(self.image, True, False, 1.0, "the near road (rows 160-240)", "grey", (512, 512))
        self.assertTrue((out[160:] == 128).all())
        self.assertTrue((out[:160] == self.image[:160]).all())
        self.assertEqual(changed, 80 * 320)
        out, changed = prepare(self.image, True, False, 1.0, "the sky (rows 0-80)", "mean colour", (512, 512))
        self.assertEqual(changed, 0)                            # uniform cells do not change under their mean

    def test_mirror_brightness_and_fit(self):
        from tools.vla_app import prepare
        out, _ = prepare(self.image, True, True, 0.5, "nothing", "grey", (512, 512))
        self.assertEqual(tuple(out[0, 0]), (0, 0, 0))           # the red half moved right and was halved
        self.assertEqual(tuple(out[0, -1]), (100, 15, 15))
        big = np.zeros((600, 800, 3), np.uint8)
        self.assertEqual(prepare(big, True, False, 1.0, "nothing", "grey", (512, 512))[0].shape, (240, 320, 3))
        self.assertEqual(prepare(big, False, False, 1.0, "nothing", "grey", (512, 512))[0].shape, (600, 800, 3))

    def test_a_painted_picture_is_told_apart_from_browser_rounding(self):
        from tools.vla_app import changed_pixels, same_picture
        rounded = np.clip(self.image.astype(int) + 3, 0, 255).astype(np.uint8)
        self.assertEqual(changed_pixels(rounded, self.image), 0)
        painted = self.image.copy()
        painted[100:105, 100:140] = (230, 194, 30)
        self.assertEqual(changed_pixels(painted, self.image), 200)
        self.assertTrue(same_picture(painted, self.image))      # still the same frame: its teacher plan applies
        self.assertFalse(same_picture(np.zeros_like(self.image), self.image))


@needs_torch
class ProbeTests(unittest.TestCase):
    def test_every_probe_runs_and_draws(self):
        from drive.learn.probe import WHAT
        from drive.learn.train import train
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as out:
            project = Project(Path(d))
            small_dataset(project.episodes)
            train(project.episodes, project.model("v1.0"), epochs=1, device="cpu", log=quiet)
            for what in WHAT:
                self.assertEqual(cli.main(["probe", "v1.0", what, "--project", d, "--worst", "2",
                                           "--duration-s", "1"]), 0, what)
            pictures = sorted(p.name.split("-")[0] for p in (project.home / "probe/v1.0").glob("*.png"))
            self.assertEqual(pictures, ["activations", "saliency", "worst", "worst"])
            page = (project.home / "probe/v1.0/report.html").read_text()
            self.assertEqual(page.count("<section>"), 9)
            self.assertIn("<svg", page)
            with self.assertRaises(SystemExit):
                cli.main(["probe", "v1.0", "saliency", "--project", d, "--frame", "sim-000"])
        text = out.getvalue()
        for expected in ("323,124 parameters", "63x63 px of the 96x128 input", "held out: 1 episodes (sim-002)",
                         "never active on 20 frames", "mirror:", "+0.12 throttle per m/s", "update / weight",
                         "the picture", "6 drives of 1 s", "by distance from the centerline", "report: "):
            self.assertIn(expected, text)

    def test_history_names_what_the_curves_show(self):
        from drive.learn.probe import history

        def report(train_losses, val_losses):
            return {"frames": 100, "validation_frames": 10, "epochs": len(train_losses), "seconds": 1.0,
                    "device": "cpu", "history": [{"epoch": i + 1, "train_loss": t, "val_loss": v, "val_steer_error": v}
                                                 for i, (t, v) in enumerate(zip(train_losses, val_losses))]}
        lines = []
        overfit = history(report([0.1, 0.05, 0.03, 0.02, 0.01, 0.005], [0.1, 0.06, 0.05, 0.06, 0.07, 0.08]), lines.append)
        self.assertTrue(overfit["verdicts"][0].startswith("OVERFITTING"))
        self.assertEqual(overfit["best_epoch"], 3)
        healthy = history(report([0.1, 0.05, 0.03, 0.02, 0.019, 0.018], [0.09, 0.05, 0.035, 0.03, 0.029, 0.029]),
                          lines.append)
        self.assertTrue(healthy["verdicts"][0].startswith("HEALTHY"))
        stuck = history(report([0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]), lines.append)
        self.assertTrue(any(v.startswith("NOT LEARNING") for v in stuck["verdicts"]))
        # Against the average plan's loss: falling curves can still mean it learned nothing.
        falling = report([0.1, 0.08, 0.077, 0.076], [0.095, 0.088, 0.086, 0.085])
        self.assertTrue(history(falling, lines.append, baseline=0.12)["verdicts"][0].startswith("NOT LEARNING"))
        self.assertTrue(history(falling, lines.append, baseline=0.25)["verdicts"][0].startswith("UNDERFITTING"))
        own = history({**falling, "average_plan_val_loss": 1.0}, lines.append, baseline=0.12)
        self.assertEqual(own["average_plan_val_loss"], 1.0)                    # the report's own figure wins
        self.assertFalse(any(v.startswith(("NOT LEARNING", "UNDERFITTING")) for v in own["verdicts"]))


@needs_torch
class LrFindTests(unittest.TestCase):
    def test_the_sweep_rises_through_the_rates_and_names_a_range(self):
        from drive.learn.lrfind import find_lr
        with tempfile.TemporaryDirectory() as d:
            found = find_lr(small_dataset(Path(d) / "data"), steps=30, device="cpu", log=quiet)
        rates = [r for r, _, _ in found["swept"]]
        self.assertTrue(all(0 <= d <= 1 for _, _, d in found["swept"]))
        self.assertEqual(rates, sorted(rates))
        self.assertAlmostEqual(rates[0], 1e-6)
        low, high = found["range"]
        self.assertLessEqual(low, high)
        self.assertTrue(rates[0] <= found["steepest_at"] <= found["lowest_at"] <= rates[-1])


@needs_torch
class DaggerTests(unittest.TestCase):
    def test_the_network_drives_and_the_teacher_labels(self):
        from drive.learn.train import train
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            train(small_dataset(d / "data"), d / "cnn.pt", epochs=1, device="cpu", log=quiet)
            done = collect(d / "dagger", episodes=1, duration_s=1.0, steer_noise=0.0, driver=d / "cnn.pt", log=quiet)
            self.assertEqual((done["episodes"], done["frames"]), (1, 10))
            self.assertEqual(read_labels(d / "dagger/sim-000")["source"], "teacher")
            # With beta 1 the teacher has every tick: the drive is the teacher's, undisturbed.
            collect(d / "taught", episodes=1, duration_s=1.0, steer_noise=0.0, driver=d / "cnn.pt", beta=1.0,
                    log=quiet)
            collect(d / "teacher", episodes=1, duration_s=1.0, steer_noise=0.0, log=quiet)
            for key in ("states", "actions"):
                np.testing.assert_array_equal(np.load(d / "taught/sim-000/episode.npz")[key],
                                              np.load(d / "teacher/sim-000/episode.npz")[key])
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(["collect", "--project", str(d), "--checkpoint", "v1"])        # no --driver cnn
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(["collect", "--project", str(d), "--driver", "cnn", "--checkpoint", str(d / "cnn.pt"),
                          "--beta", "2"])


class StatusTests(unittest.TestCase):
    def test_follows_the_project_through_its_stages(self):
        with tempfile.TemporaryDirectory() as d:
            project = Project(Path(d))
            self.assertIn("record drives", next_step(summary(project)))
            small_dataset(project.episodes, episodes=2, duration_s=1.0)
            manual_drive(project.episodes / "manual")
            s = summary(project)
            self.assertEqual((s["episodes"], s["frames"], s["labelled_frames"]), (3, 60, 20))
            self.assertEqual(s["label_sources"], {"teacher": 2, "unlabelled": 1})
            self.assertEqual(next_step(s), "./drive.sh label")
            label(project.episodes)
            self.assertIn("train v1", next_step(summary(project)))
            # A trained model and its evaluation, as train and evaluate leave them.
            project.model("v1").parent.mkdir()
            project.model("v1").write_bytes(b"weights")
            project.model("v1").with_suffix(".json").write_text(json.dumps(
                {"frames": 40, "epochs": 2, "history": [{"val_steer_error": 0.03}]}))
            self.assertEqual(next_step(summary(project)), "./drive.sh evaluate v1")
            project.report("v1").parent.mkdir()
            project.report("v1").write_text(json.dumps(
                {"summary": {"drives": 6, "drives_passed": 6, "lateral_rms_cm": 1.2}}))
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli.main(["status", "--project", d]), 0)
            self.assertIn("model      v1: 40 frames, 2 epochs, validation steer error 0.03; drives 6/6 passed",
                          out.getvalue())
            self.assertIn("more or more varied episodes", out.getvalue())
            self.assertEqual(project.checkpoint("v1"), project.model("v1"))
            with self.assertRaisesRegex(ValueError, "no model 'v9'"):
                project.checkpoint("v9")


@needs_torch
class LearnTests(unittest.TestCase):
    def test_network_shape_and_range(self):
        from drive.learn.model import ChunkNet
        model = ChunkNet(PLAN_STEPS)
        out = model(torch.randint(0, 255, (3, 96, 128, 3), dtype=torch.uint8), torch.randn(3, 2) * 50)
        self.assertEqual(tuple(out.shape), (3, PLAN_STEPS, 2))
        self.assertLessEqual(float(out.abs().max()), 1.0)
        self.assertLess(sum(p.numel() for p in model.parameters()), 1_000_000)

    def test_mirroring_flips_image_and_steering_only(self):
        from drive.learn.train import augment
        images = torch.arange(4 * 2 * 3 * 3, dtype=torch.float32).reshape(4, 2, 3, 3).repeat(8, 1, 1, 1) % 200
        states = torch.tensor([[1.5, 0.4]]).repeat(32, 1)
        plans = torch.tensor([[[0.3, 0.6]]]).repeat(32, 5, 1)
        x, s, p = augment(images, states, plans, torch.Generator().manual_seed(0))
        flipped = s[:, 1] < 0
        self.assertTrue(flipped.any() and (~flipped).any())
        self.assertTrue((s[:, 0] == 1.5).all() and (p[..., 1] == 0.6).all())      # speed, throttle untouched
        self.assertTrue((p[flipped][..., 0] == -0.3).all() and (p[~flipped][..., 0] == 0.3).all())
        gain = x[:, 0, 1, 0] / images[:, 0, 1, 0]                                  # the middle column stays put
        np.testing.assert_allclose(x[~flipped][:, 0, 0, 0], (images[~flipped][:, 0, 0, 0] * gain[~flipped]), rtol=1e-5)
        np.testing.assert_allclose(x[flipped][:, 0, 0, 0], (images[flipped][:, 0, 2, 0] * gain[flipped]), rtol=1e-5)
        self.assertTrue(((gain >= 0.8) & (gain <= 1.2)).all())

    @unittest.skipUnless(torch is not None and torch.backends.mps.is_available(), "no Mac GPU (MPS)")
    def test_mps_reads_a_slice_of_the_batch(self):
        # Validation runs the held-out frames 512 at a time. On MPS, permuting such a uint8
        # slice and then casting it read the batch from frame 0: a wrong validation score.
        from drive.learn.model import ChunkNet
        model = ChunkNet(PLAN_STEPS).eval()
        images = torch.randint(0, 255, (6, 96, 128, 3), dtype=torch.uint8)
        states = torch.randn(6, 2)
        with torch.no_grad():
            expected = model(images[3:], states[3:])
            out = model.to("mps")(images.to("mps")[3:], states.to("mps")[3:]).cpu()
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)

    def test_train_then_drive(self):
        from drive.learn.train import train
        from drive.policy.cnn import CnnPolicy
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            report = train(small_dataset(d / "data"), d / "cnn.pt", epochs=2, device="cpu", log=quiet)
            self.assertEqual((report["frames"], report["validation_frames"], len(report["history"])), (40, 20, 2))
            self.assertEqual(json.loads((d / "cnn.json").read_text()), report)
            again = train(d / "data", d / "again.pt", epochs=2, device="cpu", log=quiet)
            self.assertEqual(report["history"], again["history"])
            self.assertEqual((report["lr"], report["batch"], report["augmentation"], report["seed"]),
                             (1e-3, 128, True, 0))
            self.assertGreater(report["average_plan_val_loss"], report["history"][-1]["val_loss"])
            inside = report["history"][0]
            self.assertEqual(sorted(inside["grad_norm"]), sorted(inside["update_ratio"]))
            self.assertEqual(len(inside["grad_norm"]), 8)                       # 5 convolutions, 3 linear layers
            self.assertEqual(len(inside["dead"]), 7)                            # the ReLUs
            self.assertGreater(min(inside["update_ratio"].values()), 0)
            self.assertEqual(inside["lr"], 1e-3)
            plain = train(d / "data", d / "plain.pt", epochs=2, device="cpu", augmentation=False, log=quiet)
            self.assertFalse(plain["augmentation"])
            self.assertNotEqual(plain["history"], report["history"])

            policy = CnnPolicy(d / "cnn.pt")
            world = World()
            chunk = policy.infer(world.observe("drive"))
            self.assertEqual((chunk.actions.shape, chunk.dt, chunk.t_obs), ((PLAN_STEPS, 2), PLAN_DT_S, 0.0))
            np.testing.assert_array_equal(chunk.actions, policy.infer(world.observe("drive")).actions)

            result = run(EpisodeConfig(policy="cnn", checkpoint=str(d / "cnn.pt"), duration_s=1.0))
            self.assertEqual(result["inferences"], 10)

    def test_cnn_policy_errors(self):
        for config, message in ((EpisodeConfig(policy="cnn"), "needs a checkpoint"),
                                (EpisodeConfig(policy="cnn", checkpoint="/nonexistent.pt"), "no checkpoint"),
                                (EpisodeConfig(policy="cnn", checkpoint="x.pt", target_speed_mps=1.0), "target speed")):
            with self.assertRaisesRegex(ValueError, message):
                run(config)

    def test_cli_from_collecting_to_status(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()) as out:
            project = ["--project", d]
            self.assertEqual(cli.main(["collect", *project, "--episodes", "2", "--duration-s", "2",
                                       "--tracks", "oval,trefoil"]), 0)
            self.assertEqual(cli.main(["validate", *project]), 0)
            self.assertEqual(cli.main(["train", "v1", *project, "--epochs", "1", "--device", "cpu"]), 0)
            with self.assertRaises(SystemExit):                    # a trained version is never replaced
                cli.main(["train", "v1", *project, "--epochs", "1", "--device", "cpu"])
            self.assertEqual(cli.main(["sim", "--policy", "cnn", "--checkpoint", "v1", *project,
                                       "--track", "oval", "--reverse", "--duration-s", "1"]), 0)
            self.assertIn(cli.main(["evaluate", "v1", *project, "--tracks", "oval", "--duration-s", "2"]), (0, 1))
            self.assertTrue((Path(d) / "reports/v1.json").is_file())
            self.assertEqual(cli.main(["status", *project]), 0)
        self.assertIn("cnn on oval reversed", out.getvalue())
        self.assertIn("trefoil", out.getvalue())
        self.assertRegex(out.getvalue(), r"model      v1: 20 frames, 1 epochs, .*; drives \d/2 passed")


if __name__ == "__main__":
    unittest.main()
