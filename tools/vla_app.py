"""A small web app to study and test a fine-tuned VLA (smolvla-v1): give it an image, see what it does.

    python tools/vla_app.py [--checkpoint smolvla-v1] [--port 7860]

Run it with the LeRobot environment's Python, from the repository root, with PYTHONPATH=.
and HF_HOME set, like tools/probe_vla.py. It needs gradio as well (`uv pip install gradio`
into that environment). Then open http://127.0.0.1:7860. It listens on this computer only.

Load a frame from an episode, or upload, paste or photograph any image. Paint on it,
mirror it, darken it, hide a band of it, and change the state and the sentence. Then see
what the model would do:

    the path       where the chunk would take the car, drawn on the road (orange: the mean of
                   the samples, thin: each sample, cyan: the teacher's plan for a loaded frame)
    the numbers    steering and throttle, their spread across starting noises, the time,
                   and the state as the normaliser sees it (is it inside the training data?)
    the chunk      steering and throttle over the chunk, and where the attention goes
    model input    the 512 x 512 image SigLIP gets, and the attention on each image token
    occlusion      optional: which cells of the image move the steering (about 5 s)

docs/study/17-debugging-the-model.md explains the views; tools/see_vla.py draws them.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from drive.config import CameraParams
from drive.learn.browse import plan_path
from drive.learn.label import episodes, read_labels
from drive.learn.project import Project
from drive.policy.lerobot_backend import chunk_dt_s
from drive.sim.render import Camera
from drive.viewer import CYAN, DARK, ORANGE, WHITE, text
from tools.see_vla import (FLOOR, TILE_H, TILE_W, Seer, attention_view, cells, chunk_view, diverging,
                           model_input_view, overlay_cells, signed, steer_bar)

CAMERA_W, CAMERA_H = 320, 240                  # the simulator's camera, which the model was trained on
BANDS = {                                      # camera rows of a 320 x 240 frame (token rows 2-3, 4, 5, 6-7)
    "nothing": None,
    "the sky (rows 0-80)": (0, 80),
    "the horizon band (rows 80-120)": (80, 120),
    "the middle road (rows 120-160)": (120, 160),
    "the near road (rows 160-240)": (160, 240),
}
STEPS_PLAYED = 4                               # chunk steps the executor plays before the next chunk


class Lab:
    """The model, the statistics it was normalised with, and a lock: Gradio serves from threads."""

    def __init__(self, project: Path, checkpoint: str, device: str | None = None):
        self.project = Project(project)
        path = self.project.checkpoint(checkpoint)
        self.name = Path(checkpoint).name
        self.seer = Seer(path, device)
        self.dt = chunk_dt_s(path)
        self.default_steps = self.seer.config.num_steps
        self.camera = Camera(CameraParams())
        self.lock = threading.Lock()
        self.state_stats = self._state_stats(path)

    @staticmethod
    def _state_stats(path: Path) -> dict | None:
        """mean, std, min, max of observation.state, from the checkpoint's normaliser."""
        try:
            from safetensors.numpy import load_file
            found = sorted(path.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
            stats = load_file(found[0])
            return {k: stats[f"observation.state.{k}"] for k in ("mean", "std", "min", "max")}
        except (ImportError, IndexError, KeyError):
            return None

    def episode_names(self) -> list[str]:
        return [e.name for e in episodes(self.project.episodes)]

    def run(self, image, speed, steer, task, seed, samples, steps, occlusion):
        state = np.array([speed, steer], np.float32)
        noise = self.seer.noise(np.random.default_rng(int(seed)), int(samples))
        with self.lock:
            self.seer.config.num_steps = int(steps)          # what sample_actions reads
            try:
                started = time.perf_counter()
                seen = self.seer.look(image, state, task, noise)
                elapsed = time.perf_counter() - started
                boxes = cells(*image.shape[:2], self.seer.side, seen["grid"])
                moved = (self.seer.occlusion(image, state, task, noise, seen["chunks"], boxes, "mean", STEPS_PLAYED)
                         if occlusion else None)
            finally:
                self.seer.config.num_steps = self.default_steps
        return seen, boxes, moved, elapsed


# ---- preparing the input ----

def editor_image(value) -> np.ndarray | None:
    """The picture in the editor, painting included, as uint8 RGB."""
    if value is None:
        return None
    image = value.get("composite") if isinstance(value, dict) else value
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return np.ascontiguousarray(image[:, :, :3]).astype(np.uint8)


def changed_pixels(a: np.ndarray, b: np.ndarray) -> int:
    """Pixels that differ visibly (a browser may round a few colour values)."""
    return int((np.abs(a.astype(int) - b) > 8).any(axis=2).sum())


def same_picture(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    return (a is not None and b is not None and a.shape == b.shape
            and changed_pixels(a, b) < 0.01 * a.shape[0] * a.shape[1])


def hide_band(image: np.ndarray, band, side, fill: str = "mean colour") -> np.ndarray:
    """Fill every token cell inside a band of camera rows: with its own mean colour (as see_vla's occlusion),
    grey or black."""
    if band is None:
        return image
    top, bottom = (v * image.shape[0] / CAMERA_H for v in band)
    out = image.copy()
    for _, _, y0, y1, x0, x1 in cells(*image.shape[:2], side, 8):
        if y0 >= top - 1 and y1 <= bottom + 1:
            out[y0:y1, x0:x1] = {"mean colour": image[y0:y1, x0:x1].reshape(-1, 3).mean(0),
                                 "grey": 128, "black": 0}[fill]
    return out


def prepare(image, fit, mirror, brightness, hide, fill, side) -> tuple[np.ndarray, int]:
    """The image the model gets after the edits under "Change the input", and how many pixels the hiding changed."""
    if fit and image.shape[:2] != (CAMERA_H, CAMERA_W):
        image = cv2.resize(image, (CAMERA_W, CAMERA_H), interpolation=cv2.INTER_AREA)
    if mirror:
        image = image[:, ::-1]
    if brightness != 1.0:
        image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    hidden = hide_band(image, BANDS[hide], side, fill)
    return hidden, int((hidden != image).any(axis=2).sum())


# ---- drawing ----

LABEL_BAND = 30                                # Gradio draws an image's label over its top-left corner


def banded(image: np.ndarray, legend: str = "") -> np.ndarray:
    """The image under a dark band for Gradio's label, with an optional legend right of the label."""
    band = np.full((LABEL_BAND, image.shape[1], 3), DARK, np.uint8)
    if legend:
        text(band, legend, image.shape[1] - 9 * len(legend) - 10, 20, WHITE, 0.5)
    return np.vstack([band, image])

def path_view(lab: Lab, image, chunks, plan, speed, steer) -> np.ndarray:
    """The camera image at 640 wide with the paths the chunks would drive, projected onto the road."""
    view = cv2.resize(image, (640, int(round(image.shape[0] * 640 / image.shape[1]))), interpolation=cv2.INTER_LINEAR)
    sx, sy = view.shape[1] / lab.camera.params.width, view.shape[0] / lab.camera.params.height

    def line(actions, colour, thickness):
        pixels = lab.camera.project(*plan_path(np.asarray(actions, float), lab.dt, speed, steer).T)
        pixels = pixels[np.isfinite(pixels).all(axis=1)] * (sx, sy)
        if len(pixels) > 1:
            cv2.polylines(view, [pixels.round().astype(np.int32)], False, colour, thickness, cv2.LINE_AA)

    for chunk in chunks:
        line(chunk, (190, 120, 60), 1)
    if plan is not None:
        line(plan, CYAN, 2)
    line(chunks.mean(0), ORANGE, 3)
    model = chunks[:, :STEPS_PLAYED, 0].mean()
    teacher = None if plan is None else plan[:STEPS_PLAYED, 0].mean()
    steer_bar(view, 60, view.shape[0] - 30, view.shape[1] - 120, model, teacher)
    if image.shape[:2] != (CAMERA_H, CAMERA_W):
        text(view, "paths assume the simulator's camera", 10, 22, WHITE, 0.5)
    return banded(view, "orange: model, thin: each sample" + ("  cyan: teacher" if plan is not None else ""))


def chunk_plot(seen, plan, dt) -> np.ndarray:
    canvas = np.full((TILE_H, TILE_W, 3), DARK, np.uint8)
    chunk_view(canvas, 0, 0, seen, plan, dt)
    return banded(canvas)


def occlusion_view(image, boxes, moved) -> np.ndarray:
    scale = TILE_W / image.shape[1]
    camera = cv2.resize(image, (TILE_W, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    top = max(abs(moved[:, 0]).max(), FLOOR)
    return banded(overlay_cells(camera, moved[:, 0] / top, boxes, scale, diverging,
                                label=lambda v: signed(v * top) if abs(v * top) >= 0.01 else ""))


def state_line(name, value, unit, stats, k) -> str:
    if stats is None:
        return f"{name} {value:+.2f}{unit}"
    mean, std, low, high = (float(stats[s][k]) for s in ("mean", "std", "min", "max"))
    z = (value - mean) / std
    inside = low <= value <= high
    warning = "" if inside and abs(z) < 3 else "  ⚠ **outside what it was trained on**"
    return (f"{name} {value:+.2f}{unit} → normalised **{z:+.2f}** (training data: mean {mean:+.2f}, "
            f"std {std:.2f}, range {low:+.2f} … {high:+.2f}){warning}")


def summary(lab, seen, plan, speed, steer, steps, elapsed, edits, boxes=None, moved=None) -> str:
    chunks = seen["chunks"]
    played = chunks[:, :STEPS_PLAYED]
    mass = seen["layer_mass"].mean(0)
    mass = mass / mass.sum()
    lines = [
        f"**Steering** (+ left, − right): first action **{chunks[:, 0, 0].mean():+.3f}**; "
        f"first {STEPS_PLAYED} actions ({STEPS_PLAYED * lab.dt:.1f} s, what the car plays) "
        f"**{played[..., 0].mean():+.3f}**, spread across samples ±{played[..., 0].mean(1).std():.3f}"
        + ("" if plan is None else f"; teacher **{plan[:STEPS_PLAYED, 0].mean():+.3f}**"),
        f"**Throttle**: first action **{chunks[:, 0, 1].mean():+.3f}**; first {STEPS_PLAYED} "
        f"**{played[..., 1].mean():+.3f}**" + ("" if plan is None else f"; teacher **{plan[:STEPS_PLAYED, 1].mean():+.3f}**"),
        f"**Time**: {len(chunks)} samples in one batch, {steps} denoising steps: {1000 * elapsed:.0f} ms",
        state_line("**Speed**", speed, " m/s", lab.state_stats, 0),
        state_line("**Wheels**", steer, "", lab.state_stats, 1),
        f"**Attention of the action expert**: image {100 * mass[0]:.0f} %, sentence {100 * mass[1]:.0f} %, "
        f"state {100 * mass[2]:.0f} %, its own chunk {100 * mass[3]:.0f} %",
    ]
    if moved is not None:
        k = int(abs(moved[:, 0]).argmax())
        lines.append(f"**Occlusion**: hiding one cell moves the first {STEPS_PLAYED} steering actions by at most "
                     f"**{signed(moved[k, 0])}** (token row {boxes[k][0]}, column {boxes[k][1]}); the throttle by at "
                     f"most {abs(moved[:, 1]).max():.3f}. Colours span ±{max(abs(moved[:, 0]).max(), FLOOR):.2f}")
    if edits:
        lines.append("**Input edits**: " + "; ".join(edits)
                     + (". The teacher's plan is for the unedited frame" if plan is not None else ""))
    return "\n\n".join(lines)


def table(seen, plan, dt) -> list[list]:
    chunks = seen["chunks"]
    rows = []
    for k in range(chunks.shape[1]):
        row = [round(k * dt, 2), round(float(chunks[:, k, 0].mean()), 3), round(float(chunks[:, k, 0].std()), 3),
               round(float(chunks[:, k, 1].mean()), 3)]
        row += ([round(float(plan[k, 0]), 3), round(float(plan[k, 1]), 3)] if plan is not None and k < len(plan)
                else [None, None])
        rows.append(row)
    return rows


# ---- the app ----

def build(lab: Lab):
    import gradio as gr

    names = lab.episode_names()
    default_episode = "sim2-031" if "sim2-031" in names else (names[0] if names else None)

    def load(episode, frame):
        if episode is None:
            raise gr.Error("no episodes in this project")
        directory = lab.project.episodes / episode
        data, labels = np.load(directory / "episode.npz"), read_labels(directory)
        frame = int(min(max(frame, 0), len(data["t"]) - 1))
        image = data["images"][frame]
        speed, steer = (float(v) for v in data["states"][frame])
        plan = None if labels is None else labels["plans"][frame]
        loaded = {"image": image, "plan": plan, "name": f"{episode} frame {frame}"}
        note = (f"Loaded **{episode} frame {frame}** (t {data['t'][frame]:.1f} s). "
                + ("The teacher's plan is shown in cyan." if plan is not None else "This episode has no labels."))
        editor = {"background": image, "layers": [], "composite": image}
        return editor, speed, steer, loaded, note, gr.Slider(maximum=len(data["t"]) - 1, value=frame)

    def infer(value, loaded, speed, steer, task, seed, samples, steps, fit, mirror, brightness, hide, fill, occlusion):
        image = editor_image(value)
        if image is None:
            raise gr.Error("load a frame or upload an image first")
        background = editor_image({"composite": value.get("background")}) if isinstance(value, dict) else None
        plan = loaded["plan"] if loaded and same_picture(background, loaded["image"]) else None
        edits = []
        if plan is not None and image.shape == loaded["image"].shape and changed_pixels(image, loaded["image"]):
            edits.append(f"painted ({changed_pixels(image, loaded['image'])} pixels)")
        if fit and image.shape[:2] != (CAMERA_H, CAMERA_W):
            edits.append(f"resized from {image.shape[1]}x{image.shape[0]} to {CAMERA_W}x{CAMERA_H}")
        if mirror:
            edits.append("mirrored, wheels and the teacher's steering negated")
            steer = -steer
            plan = None if plan is None else plan * np.array([-1.0, 1.0])
        if brightness != 1.0:
            edits.append(f"brightness x{brightness:.2f}")
        image, hidden = prepare(image, fit, mirror, brightness, hide, fill, lab.seer.side)
        if BANDS[hide] is not None:
            edits.append(f"{hide} filled with {fill}: {hidden} pixels changed"
                         + (" (it is uniform here: try grey)" if hidden == 0 else ""))
        seen, boxes, moved, elapsed = lab.run(image, speed, steer, task, seed, samples, steps, occlusion)
        return (path_view(lab, image, seen["chunks"], plan, speed, steer),
                summary(lab, seen, plan, speed, steer, int(steps), elapsed, edits, boxes, moved),
                chunk_plot(seen, plan, lab.dt),
                banded(model_input_view(seen, TILE_H, seen["grid"])),
                banded(attention_view(seen, TILE_H, seen["grid"])[0]),
                occlusion_view(image, boxes, moved) if moved is not None else None,
                table(seen, plan, lab.dt))

    with gr.Blocks(title=f"{lab.name}: input an image, see the output") as app:
        gr.Markdown(f"## {lab.name}: give the VLA an image, see what it would do\n"
                    "Pick a frame (the model runs on it at once) or upload/paste any image, paint on it, change "
                    "the state or the input, then **Run the model**. "
                    "Steering +1 is full left. Views are explained in `docs/study/17-debugging-the-model.md`.")
        loaded = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row():
                    episode = gr.Dropdown(names, value=default_episode, label="episode", scale=2)
                    frame = gr.Slider(0, 299, value=235, step=1, label="frame", scale=3)
                    load_button = gr.Button("Load frame", scale=1)
                note = gr.Markdown()
                editor = gr.ImageEditor(label="input image: paint on it with the brush", type="numpy",
                                        image_mode="RGB", format="png", sources=("upload", "clipboard", "webcam"),
                                        brush=gr.Brush(colors=["#e6c21e", "#f0f0f0", "#46464b", "#3c8c3c", "#000000"],
                                                       default_color="#e6c21e", default_size=6),
                                        height=420)
                with gr.Accordion("State, sentence, sampling", open=True):
                    with gr.Row():
                        speed = gr.Slider(0.0, 3.0, value=1.35, step=0.05, label="speed (m/s)")
                        steer = gr.Slider(-1.0, 1.0, value=0.0, step=0.01, label="wheels now (+ left)")
                    task = gr.Textbox("drive along the track", label="task sentence")
                    with gr.Row():
                        seed = gr.Number(0, precision=0, label="noise seed")
                        samples = gr.Slider(1, 16, value=4, step=1, label="samples (starting noises)")
                        steps = gr.Slider(1, 20, value=lab.default_steps, step=1, label="denoising steps")
                with gr.Accordion("Change the input (tests)", open=True):
                    with gr.Row():
                        fit = gr.Checkbox(True, label=f"resize to the camera's {CAMERA_W}x{CAMERA_H}")
                        mirror = gr.Checkbox(False, label="mirror left-right")
                        occlusion = gr.Checkbox(False, label="occlusion map (about 5 s)")
                    with gr.Row():
                        brightness = gr.Slider(0.2, 2.0, value=1.0, step=0.05, label="brightness")
                        hide = gr.Dropdown(list(BANDS), value="nothing", label="hide a band of the image")
                        fill = gr.Dropdown(["mean colour", "grey", "black"], value="mean colour",
                                           label="fill each hidden cell with")
                run = gr.Button("Run the model", variant="primary")
            with gr.Column(scale=6):
                path = gr.Image(label="where the chunk would take the car", type="numpy", format="png")
                numbers = gr.Markdown()
                chunk = gr.Image(label="the chunk, and where the attention goes", type="numpy", format="png")
        with gr.Row():
            model_input = gr.Image(label="model input: 512x512, 8x8 image tokens", type="numpy", format="png")
            attention = gr.Image(label="attention on each image token (%)", type="numpy", format="png")
            occluded = gr.Image(label="occlusion -> steering (red: more left, blue: more right)", type="numpy",
                                format="png")
        rows = gr.Dataframe(headers=["t (s)", "steer", "steer sd", "throttle", "teacher steer", "teacher throttle"],
                            label="the chunk, step by step (mean over samples)")

        loading = dict(fn=load, inputs=[episode, frame], outputs=[editor, speed, steer, loaded, note, frame])
        running = dict(fn=infer, inputs=[editor, loaded, speed, steer, task, seed, samples, steps, fit, mirror,
                                         brightness, hide, fill, occlusion],
                       outputs=[path, numbers, chunk, model_input, attention, occluded, rows], concurrency_limit=1)
        run.click(**running)
        # Loading a frame runs the model on it too, so the frame slider scrubs through an episode.
        for event in (load_button.click, frame.release, episode.input, app.load):
            event(**loading).then(**running)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("runs/drive"))
    parser.add_argument("--checkpoint", default="smolvla-v1", help="a version in <project>/vla, or a checkpoint directory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="open it in the browser")
    parser.add_argument("--device", help="cuda, mps or cpu (default: the fastest there is; or $VLA_DEVICE)")
    args = parser.parse_args()
    started = time.perf_counter()
    lab = Lab(args.project, args.checkpoint, args.device)
    print(f"loaded {args.checkpoint} on {lab.seer.backend.device} in {time.perf_counter() - started:.1f} s")
    build(lab).launch(server_name="127.0.0.1", server_port=args.port, inbrowser=args.open, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
