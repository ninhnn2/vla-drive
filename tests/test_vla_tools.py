"""The VLA tools end to end, on a tiny SmolVLA that nobody trained.

drive/policy/lerobot_backend.py, tools/probe_vla.py, tools/see_vla.py and tools/vla_app.py
run here on a SmolVLA with 2 VLM layers and random weights, built in a few seconds, that
sees 256 x 256 instead of 512 x 512: 4 x 4 image tokens instead of 8 x 8, which also checks
that no tool assumes the grid (tests/test_drive_learn.py checks smolvla-v1's 8 x 8). They
must load it, run it (on CUDA, the Apple GPU or the CPU: VLA_DEVICE chooses) and write
their outputs. That is what has to hold on Linux and on a Mac alike; the numbers a
trained model gives are in docs/study/17-debugging-the-model.md.

Needs LeRobot: `make vla-setup`, then `./vla.sh -m unittest tests.test_vla_tools`. The
first run downloads SmolVLM2's configuration and tokenizer (a few MB) from the Hugging
Face Hub. Without LeRobot (in the container, say) these tests are skipped.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import lerobot  # noqa: F401
except ImportError:
    lerobot = None

ROOT = Path(__file__).resolve().parent.parent
FRAMES = 12
TASK = "drive along the track"


def tiny_checkpoint(directory: Path, state_width: int = 2) -> None:
    """A SmolVLA checkpoint as lerobot-train writes one, for a 320x240 camera and [speed, steer] -> [steer, throttle]
    ([speed] alone with `state_width` 1, as `export --state speed_mps` gives)."""
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = SmolVLAConfig(
        input_features={"observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 240, 320)),
                        "observation.state": PolicyFeature(FeatureType.STATE, (state_width,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        chunk_size=4, n_action_steps=4, num_steps=2, num_vlm_layers=2, resize_imgs_with_padding=(256, 256),
        load_vlm_weights=False, push_to_hub=False, device="cpu")
    torch.manual_seed(0)
    policy = SmolVLAPolicy(config)
    stats = {"observation.state": {"mean": torch.tensor([1.3, 0.0][:state_width]),
                                   "std": torch.tensor([0.26, 0.32][:state_width])},
             "action": {"mean": torch.tensor([0.0, 0.18]), "std": torch.tensor([0.36, 0.12])},
             "observation.images.front": {"mean": torch.full((3, 1, 1), 0.45), "std": torch.full((3, 1, 1), 0.22)}}
    preprocess, postprocess = make_pre_post_processors(config, dataset_stats=stats)
    for part in (policy, preprocess, postprocess):
        part.save_pretrained(directory)


def road_image(rng, shift: int = 0) -> np.ndarray:
    """Sky, grass and a grey road with a yellow dash: enough like the simulator's camera to draw on."""
    image = np.empty((240, 320, 3), np.uint8)
    image[:90], image[90:110], image[110:] = (150, 190, 230), (60, 140, 60), (72, 72, 76)
    image[160:200, 150 + shift:165 + shift] = (230, 194, 30)
    return np.clip(image + rng.integers(-6, 7, image.shape), 0, 255).astype(np.uint8)


def tiny_episode(directory: Path) -> None:
    """A labelled episode in the layout collect and label write."""
    rng = np.random.default_rng(0)
    directory.mkdir(parents=True)
    np.savez(directory / "episode.npz", t=(np.arange(FRAMES) * 0.1).astype(np.float32),
             images=np.stack([road_image(rng, 3 * i) for i in range(FRAMES)]),
             states=np.column_stack([np.full(FRAMES, 1.3), np.linspace(-0.2, 0.2, FRAMES)]).astype(np.float32),
             actions=rng.uniform(-0.3, 0.3, (FRAMES, 2)).astype(np.float32))
    #debug
    print(f"tiny_episode: {directory / 'episode.npz'} contains {list(np.load(directory / 'episode.npz').keys())}")
    
    np.savez(directory / "labels.npz", plans=rng.uniform(-0.3, 0.3, (FRAMES, 16, 2)).astype(np.float32),
             valid=np.ones(FRAMES, bool), dt=np.float32(0.1), source="test")


def run_tool(*args) -> str:
    """Run a tool as `./vla.sh tools/...` would, and return what it printed."""
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise AssertionError(f"{args[0]} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return result.stdout


@unittest.skipIf(lerobot is None, "needs LeRobot: make vla-setup, then ./vla.sh -m unittest tests.test_vla_tools")
class TinyVlaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.project = Path(cls.tmp.name)
        cls.checkpoint = cls.project / "vla" / "tiny" / "checkpoints" / "last" / "pretrained_model"
        cls.checkpoint.mkdir(parents=True)
        try:
            tiny_checkpoint(cls.checkpoint)
        except OSError as error:                     # offline, and SmolVLM2's config not cached
            cls.tmp.cleanup()
            raise unittest.SkipTest(f"SmolVLM2's configuration is not available: {error}")
        tiny_episode(cls.project / "episodes" / "tiny-000")
        from tools.see_vla import Seer
        cls.seer = Seer(cls.checkpoint)
        cls.rng = np.random.default_rng(1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def frame(self, index: int = 5):
        data = np.load(self.project / "episodes" / "tiny-000" / "episode.npz")
        return data["images"][index], data["states"][index]

    def test_the_backend_answers_a_chunk_in_robot_units(self):
        from drive.policy.vla import BackendObservation
        image, state = self.frame()
        noise = self.seer.noise(self.rng, 1)
        answer = self.seer.backend.infer(BackendObservation(index=0, images=[image], state=state, task=TASK,
                                                            noise=noise))
        self.assertEqual(answer.chunk.shape, (4, 2))
        self.assertTrue(np.isfinite(answer.chunk).all())
        again = self.seer.backend.infer(BackendObservation(index=1, images=[image], state=state, task=TASK,
                                                           noise=noise))
        np.testing.assert_allclose(again.chunk, answer.chunk, atol=1e-5)    # the same noise, the same chunk

    def test_look_sees_the_padded_image_and_records_the_attention(self):
        image, state = self.frame()
        seen = self.seer.look(image, state, TASK, self.seer.noise(self.rng, 2))
        self.assertEqual(seen["chunks"].shape, (2, 4, 2))
        self.assertEqual(seen["input"].shape, (256, 256, 3))
        self.assertEqual(seen["input"][:64].max(), 0)              # a 4:3 camera is padded at the top
        self.assertGreater(seen["input"][64:].max(), 0)
        self.assertEqual(seen["grid"], 4)                          # found from the prefix, not assumed
        self.assertEqual(seen["layer_maps"].shape, (2, 4, 4))      # 2 layers, 4 x 4 image tokens
        np.testing.assert_allclose(seen["layer_mass"].sum(1), 1.0, atol=1e-3)
        self.assertEqual(seen["layer_mass"][1, 3], 0.0)            # layer 1 is cross-attention: no chunk keys

    def test_see_vla_writes_its_panels_and_summary(self):
        out = self.project / "see"
        printed = run_tool("tools/see_vla.py", "--project", self.project, "--checkpoint", "tiny",
                           "--episode", "tiny-000", "--frames", "0.6s", "--noises", "1", "--steps", "2",
                           "--summary", "20", "--out", out)
        for name in ("tiny-000-006.png", "tiny-000-006-layers.png", "tiny-000-summary-mean.png"):
            self.assertTrue((out / name).is_file(), f"{name} missing; printed:\n{printed}")
        numbers = json.loads((out / "tiny-000-006.json").read_text())
        self.assertEqual(len(numbers["occlusion"]["cells"]), 12)  # 4 x 4 tokens, the top row padding
        self.assertAlmostEqual(sum(numbers["attention_share"].values()), 1.0, places=2)

    def test_probe_vla_runs_every_probe(self):
        printed = run_tool("tools/probe_vla.py", "--project", self.project, "--checkpoint", "tiny",
                           "--episodes", "tiny-*", "--every", "4")
        for probe in ("[offline]", "[noise]", "[repeat]", "[mirror]", "[blank]", "[speed]", "[task]", "[timing]",
                      "[shapes]"):
            self.assertIn(probe, printed)

    def test_the_app_answers_a_frame_and_an_upload(self):
        from tools.vla_app import AppError, Lab, answer, load_frame
        lab = Lab(self.project, "tiny")
        editor, speed, steer, loaded, note, frame, last, task = load_frame(lab, "tiny-000", 50)
        self.assertEqual(task, TASK)                               # no meta.json: the simulator's sentence
        self.assertEqual((frame, last), (FRAMES - 1, FRAMES - 1))
        self.assertIn("teacher", note)
        views = answer(lab, editor, loaded, speed, steer, TASK, 0, 2, 2, True, True, 0.8,
                       "the near road (rows 160-240)", "grey", True)
        path, summary, chunk, model_input, attention, occlusion, table = views
        self.assertEqual(path.shape[1], 640)
        self.assertIn("teacher", summary)
        self.assertIn("mirrored", summary)
        self.assertIn("**Occlusion**", summary)
        self.assertIsNotNone(occlusion)
        self.assertEqual(len(table), 4)
        upload = self.rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)    # any picture, any size
        views = answer(lab, {"background": upload, "layers": [], "composite": upload}, loaded, 1.0, 0.0, TASK,
                       0, 1, 2, True, False, 1.0, "nothing", "mean colour", False)
        self.assertNotIn("teacher", views[1])
        self.assertIsNone(views[5])
        with self.assertRaises(AppError):
            answer(lab, None, None, 1.0, 0.0, TASK, 0, 1, 2, True, False, 1.0, "nothing", "mean colour", False)
        try:
            import gradio  # noqa: F401
        except ImportError:
            return
        from tools.vla_app import build
        self.assertIsNotNone(build(lab))                           # the page is built; it is not served

    def test_the_app_debugs_a_carla_episode_with_a_speed_only_model(self):
        from drive.carla.camera import CarlaCameraParams
        from drive.carla.model import CarModel
        from tools.vla_app import Lab, answer, load_frame
        project = self.project / "carla"
        checkpoint = project / "vla" / "tiny1" / "checkpoints" / "last" / "pretrained_model"
        checkpoint.mkdir(parents=True)
        tiny_checkpoint(checkpoint, state_width=1)
        episode = project / "episodes" / "dagger1-town01-000"
        tiny_episode(episode)
        told = ["follow the lane"] * 6 + ["turn left at the next intersection"] * (FRAMES - 6)
        data = dict(np.load(episode / "episode.npz"))
        np.savez(episode / "episode.npz", **data, tasks=np.array(told), lateral_m=np.full(FRAMES, 0.8),
                 teacher_driving=(np.arange(FRAMES) >= 8).astype(np.float32))
        (episode / "meta.json").write_text(json.dumps({
            "task": "follow the lane", "tasks": sorted(set(told)),
            "carla": {"camera": CarlaCameraParams().to_json(), "vehicle": CarModel.load("vehicle.tesla.model3").to_json()}}))
        lab = Lab(project, "tiny1")
        self.assertEqual((lab.carla, lab.width, lab.played), (True, 1, 5))
        from tools.vla_app import ROAD
        self.assertEqual(ROAD[lab.carla], "#999493")                            # the brush paints CARLA's road
        editor, speed, steer, loaded, note, frame, last, task = load_frame(lab, "dagger1-town01-000", 9)
        self.assertEqual(task, "turn left at the next intersection")            # this frame's, not the episode's
        self.assertIn("the teacher was driving", note)
        self.assertIn("+0.80 m", note)
        views = answer(lab, editor, loaded, speed, steer, "turn right at the next intersection", 0, 2, 2, True,
                       False, 1.0, "nothing", "mean colour", False)
        self.assertEqual(views[0].shape[1], 640)
        self.assertIn("not an input of this model", views[1])                    # given the speed alone
        self.assertIn("first 5 actions", views[1])
        try:
            import gradio  # noqa: F401
        except ImportError:
            return
        from tools.vla_app import build
        self.assertIsNotNone(build(lab))

    def test_a_checkpoint_from_a_newer_lerobot_still_loads(self):
        from drive.policy.lerobot_backend import load_config
        raw = json.loads((self.checkpoint / "config.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(dict(raw, pretrained_revision=None)))
            said = []
            config = load_config(Path(directory), log=said.append)
            self.assertEqual((config.type, config.chunk_size), ("smolvla", 4))
            self.assertEqual(len(said), 1)
            self.assertIn("pretrained_revision=None", said[0])
        said = []
        load_config(self.checkpoint, log=said.append)
        self.assertEqual(said, [])


if __name__ == "__main__":
    unittest.main()
