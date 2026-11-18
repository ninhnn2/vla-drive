"""Where does a fine-tuned LeRobot policy (smolvla-v1) look? Debug images of what the model sees.

    python tools/see_vla.py [--checkpoint smolvla-v1] [--episode sim2-031] [--frames 40,120]
                            [--out runs/drive/reports/see-smolvla-v1] [--video]

Run it like tools/probe_vla.py: with the Python of an environment that has lerobot, from the
repository root, with PYTHONPATH=. and HF_HOME set. docs/study/17-debugging-the-model.md
explains how to read the images. For each frame it writes <out>/<episode>-<frame>.png:

    camera        the frame, with the steering the model and the teacher (if labelled) chose
    model input   the image SigLIP gets after LeRobot's resize_with_pad, cut into the cells that
                  become the image tokens (8 x 8 = 64 for SmolVLM2; with a 4:3 camera the top
                  rows are padding)
    attention     the share of the action expert's attention each image token gets, averaged
                  over its layers, heads, chunk steps, denoising steps and noises
    occlusion     each cell of the camera image replaced by its mean colour in turn: how far the
                  first actions of the chunk move, for steering and for throttle. Attention is
                  where the model reads; occlusion is what changes its answer
    chunk         the chunk for each noise against the teacher's plan, and where the attention
                  goes: image, sentence, state, or the action tokens themselves

and <out>/<episode>-<frame>-layers.png, the attention on the image layer by layer, and
<out>/<episode>-<frame>.json with the numbers. With --summary N it averages attention and
occlusion over every Nth frame of the episode into <out>/<episode>-summary-<fill>.png (and
.json). With --video it writes <out>/<episode>-attention.mp4: the attention on every frame.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from drive.learn.label import read_labels
from drive.learn.project import Project
from drive.policy.lerobot_backend import LeRobotBackend
from drive.viewer import CYAN, DARK, GREEN, GREY, MAGENTA, ORANGE, WHITE, YELLOW, Plot, text

TILE_W, TILE_H, TITLE = 480, 360, 34       # one view, and the caption above it
BATCH = 16                                 # images per forward pass
BLUE, RED = (70, 130, 240), (235, 70, 60)
FLOOR = 0.05                               # occlusion colours span at least +-0.05, so noise stays pale
FILLS = {"mean": "its mean colour", "grey": "grey", "black": "black"}


def cells(height: int, width: int, side: tuple[int, int], grid: int) -> list[tuple[int, int, int, int, int, int]]:
    """(row, col, y0, y1, x0, x1): the camera pixels each image token covers, for a camera image of
    height x width that lerobot's resize_with_pad fits into side = (width, height) and the vision
    encoder cuts into grid x grid tokens. Padding tokens cover no camera pixel and are left out."""
    side_w, side_h = side
    ratio = max(width / side_w, height / side_h)              # as resize_with_pad
    top, left = side_h - int(height / ratio), side_w - int(width / ratio)     # it pads the top and the left
    boxes = []
    for row in range(grid):
        for col in range(grid):
            y0, y1 = [int(round(np.clip((v * side_h / grid - top) * ratio, 0, height))) for v in (row, row + 1)]
            x0, x1 = [int(round(np.clip((v * side_w / grid - left) * ratio, 0, width))) for v in (col, col + 1)]
            if y1 > y0 and x1 > x0:
                boxes.append((row, col, y0, y1, x0, x1))
    return boxes


class Seer:
    """A checkpoint run in batches, recording on request what the action expert attends to."""

    def __init__(self, checkpoint: Path):
        self.backend = LeRobotBackend(checkpoint)
        self.torch, self.config = self.backend.torch, self.backend.config
        self.vlm = self.backend.policy.model.vlm_with_expert
        self.side = tuple(self.config.resize_imgs_with_padding)      # (width, height) SigLIP gets

    def noise(self, rng, n: int) -> np.ndarray:
        return rng.standard_normal((n, self.config.chunk_size, self.config.max_action_dim)).astype(np.float32)

    def run(self, images: np.ndarray, state: np.ndarray, task: str, noise: np.ndarray) -> np.ndarray:
        """`images` (B, H, W, 3) uint8 and `noise` (B, chunk, width), one state and task -> (B, chunk, 2)."""
        torch, out = self.torch, []
        for start in range(0, len(images), BATCH):
            part = torch.from_numpy(np.ascontiguousarray(images[start:start + BATCH])).permute(0, 3, 1, 2).float()
            n = len(part)
            batch = self.backend.preprocess({
                self.backend.camera: part / 255.0,
                "observation.state": torch.from_numpy(np.tile(state, (n, 1)).astype(np.float32)),
                "task": [task] * n})
            actions = self.backend.policy.predict_action_chunk(
                batch, noise=torch.from_numpy(noise[start:start + BATCH]).to(self.backend.device))
            out.append(self.backend.postprocess(actions).float().cpu().numpy())
        return np.concatenate(out)[:, :, :2]

    def look(self, image: np.ndarray, state: np.ndarray, task: str, noise: np.ndarray) -> dict:
        """One frame under each noise, recording the image the model got and the expert's attention."""
        if len(noise) > BATCH:
            raise ValueError(f"at most {BATCH} noises: the attention is recorded for one forward pass")
        torch, vlm, policy = self.torch, self.vlm, self.backend.policy
        groups = vlm.num_attention_heads // vlm.num_key_value_heads
        records, seen = [], {}
        attend, prepare = vlm.eager_attention_forward, policy.prepare_images

        def recording(mask, batch_size, head_dim, query, key, value):
            # The prefix pass has as many queries as keys. In a denoising step the queries are the
            # action expert's chunk tokens and the keys the cached prefix (and, in self-attention
            # layers, the chunk tokens too). Same arithmetic as eager_attention_forward.
            if query.shape[1] < key.shape[1]:
                keys = key.repeat_interleave(groups, dim=2)
                weights = torch.einsum("bqhd,bkhd->bhqk", query.float(), keys.float()) * head_dim ** -0.5
                weights = weights.masked_fill(~mask[:, None], torch.finfo(weights.dtype).min)
                records.append(weights.softmax(-1).mean(1).cpu().numpy())     # (B, queries, keys)
            return attend(mask, batch_size, head_dim, query, key, value)

        def preparing(batch):
            images, masks = prepare(batch)
            seen["image"] = images[0][0].float().cpu().numpy()                 # (3, H, W) in [-1, 1]
            seen["sentence"] = batch["observation.language.tokens"].shape[1]
            return images, masks

        vlm.eager_attention_forward, policy.prepare_images = recording, preparing
        try:
            chunks = self.run(np.repeat(image[None], len(noise), 0), state, task, noise)
        finally:
            del vlm.eager_attention_forward, policy.prepare_images

        steps = self.config.num_steps
        layers = len(records) // steps
        prefix = min(r.shape[-1] for r in records)
        on_prefix = np.stack([r[..., :prefix] for r in records]).reshape(steps, layers, *records[0].shape[:2], prefix)
        on_chunk = np.stack([r[..., prefix:].sum(-1) for r in records]).reshape(steps, layers, *records[0].shape[:2])
        sentence, state_tokens = seen["sentence"], 1
        tokens = prefix - sentence - state_tokens
        grid = int(round(tokens ** 0.5))
        if grid * grid != tokens:
            raise ValueError(f"{tokens} image tokens is not a square grid: is the prefix not image, sentence, state?")
        by_layer = on_prefix.mean(axis=(0, 2, 3))                              # (layers, prefix)
        chunk_by_layer = on_chunk.mean(axis=(0, 2, 3))
        mass = lambda a, b: by_layer[:, a:b].sum(-1)
        return {
            "chunks": chunks,
            "input": ((seen["image"].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8),
            "grid": grid,
            "layer_maps": by_layer[:, :tokens].reshape(layers, grid, grid),    # share of all attention
            "layer_mass": np.stack([mass(0, tokens), mass(tokens, tokens + sentence),
                                    mass(tokens + sentence, prefix), chunk_by_layer], axis=1),
        }

    def cells(self, height: int, width: int, grid: int) -> list[tuple[int, int, int, int, int, int]]:
        return cells(height, width, self.side, grid)

    def occlusion(self, image, state, task, noise, base, boxes, fill: str, steps: int) -> np.ndarray:
        """(len(boxes), 2): how far the mean of the first `steps` actions moves with each cell hidden."""
        hidden = np.repeat(image[None], len(boxes) * len(noise), 0)
        for b, (_, _, y0, y1, x0, x1) in enumerate(boxes):
            patch = hidden[b * len(noise):(b + 1) * len(noise), y0:y1, x0:x1]
            patch[:] = {"mean": image[y0:y1, x0:x1].reshape(-1, 3).mean(0), "grey": 128, "black": 0}[fill]
        moved = self.run(hidden, state, task, np.tile(noise, (len(boxes), 1, 1)))
        moved = moved.reshape(len(boxes), len(noise), *moved.shape[1:])[:, :, :steps].mean(2)
        return (moved - base[None, :, :steps].mean(2)).mean(1)          # (boxes, noises, 2) - (1, noises, 2)


# ---- drawing (RGB canvases) ----

def caption(canvas, x, y, lines):
    for k, line in enumerate(lines):
        text(canvas, line, x + 6, y + 14 + 15 * k, WHITE if k == 0 else GREY, 0.45 if k == 0 else 0.38)


def heat(values: np.ndarray) -> np.ndarray:
    """Values in [0, 1] -> RGB colours (inferno)."""
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0, ::-1]
    return lut[(np.clip(values, 0, 1) * 255).astype(np.uint8)]


def signed(value: float) -> str:
    """+0.25 -> '+.25': short enough for a cell."""
    return f"{value:+.2f}".replace("0.", ".", 1)


def diverging(value: float) -> np.ndarray:
    """[-1, 1] -> blue, white, red."""
    end = np.array(RED if value > 0 else BLUE, float)
    return (np.array((245, 245, 245), float) * (1 - abs(value)) + end * abs(value)).astype(np.uint8)


def steer_bar(canvas, x, y, width, model, teacher):
    """A [-1, 1] steering scale, +1 (left) on the left, with the model's and the teacher's steering."""
    cv2.line(canvas, (x, y), (x + width, y), GREY, 1)
    for value, label in ((1, "L"), (0, ""), (-1, "R")):
        px = int(x + (1 - value) / 2 * width)
        cv2.line(canvas, (px, y - 4), (px, y + 4), GREY, 1)
        text(canvas, label, px - 4, y + 16, GREY, 0.38)
    for value, colour in ((teacher, CYAN), (model, ORANGE)):
        if value is None:
            continue
        cv2.circle(canvas, (int(x + (1 - value) / 2 * width), y), 6, colour, -1, cv2.LINE_AA)


def paste(canvas, image, x, y, size):
    canvas[y:y + size[1], x:x + size[0]] = cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def overlay_cells(base, values, boxes, scale, colour_of, alpha=0.55, label=None, ink=lambda v: DARK):
    """Blend one colour per cell over `base` (already scaled by `scale`), with an optional label per cell."""
    out = base.copy()
    for (_, _, y0, y1, x0, x1), value in zip(boxes, values):
        y0, y1, x0, x1 = (int(round(v * scale)) for v in (y0, y1, x0, x1))
        out[y0:y1, x0:x1] = (out[y0:y1, x0:x1] * (1 - alpha) + colour_of(value) * alpha).astype(np.uint8)
    for (_, _, y0, y1, x0, x1), value in zip(boxes, values):
        if label:
            y0, x0, x1 = (int(round(v * scale)) for v in (y0, x0, x1))
            text(out, label(value), x0 + 3, y0 + 14, ink(value), 0.36)
    for (_, _, y0, y1, x0, x1) in boxes:
        cv2.rectangle(out, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale) - 1, int(y1 * scale) - 1), (90, 90, 96), 1)
    return out


def model_input_view(seen, size, grid):
    view = cv2.resize(seen["input"], (size, size), interpolation=cv2.INTER_AREA)
    step = size / grid
    for k in range(1, grid):
        cv2.line(view, (int(k * step), 0), (int(k * step), size), (110, 110, 118), 1)
        cv2.line(view, (0, int(k * step)), (size, int(k * step)), (110, 110, 118), 1)
    return view


def attention_view(seen, size, grid):
    """The model input in grey with the share of image attention per token."""
    grey = cv2.cvtColor(cv2.cvtColor(seen["input"], cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    view = cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)
    share = seen["layer_maps"].mean(0)
    share = share / share.sum()
    colours = heat(share / share.max())
    step = size / grid
    for r in range(grid):
        for c in range(grid):
            y0, y1, x0, x1 = int(r * step), int((r + 1) * step), int(c * step), int((c + 1) * step)
            view[y0:y1, x0:x1] = (view[y0:y1, x0:x1] * 0.35 + colours[r, c] * 0.65).astype(np.uint8)
            text(view, f"{100 * share[r, c]:.0f}", x0 + 3, y0 + 14, WHITE if share[r, c] < 0.6 * share.max() else DARK, 0.36)
    return view, share


def chunk_view(canvas, x, y, seen, plan, dt):
    chunks = seen["chunks"]
    steps = np.arange(chunks.shape[1]) * dt
    for k, (name, top) in enumerate((("steer", 18), ("throttle", 136))):
        plot = Plot(canvas, x + 34, y + top, TILE_W - 50, 100, 0.0, steps[-1])
        text(canvas, name, x + 40, y + top + 14, GREY, 0.4)
        for run in chunks:
            plot.line(steps, run[:, k], (150, 100, 50), 1)
        plot.line(steps, chunks[:, :, k].mean(0), ORANGE, 2)
        if plan is not None:
            plot.line(steps[: len(plan)], plan[:, k], CYAN, 2)
    teacher = "teacher plan (cyan)" if plan is not None else "no labels: no teacher plan"
    text(canvas, f"model (orange; thin: each noise)   {teacher}   0 .. {steps[-1]:.1f} s", x + 34, y + 256, GREY, 0.38)
    mass = seen["layer_mass"].mean(0)
    left, width = x + 34, TILE_W - 50
    text(canvas, "where the action expert's attention goes (mean of all layers)", left, y + 284, WHITE, 0.4)
    for (name, colour), share in zip((("image", YELLOW), ("sentence", MAGENTA), ("state", GREEN),
                                      ("chunk", GREY)), mass / mass.sum()):
        w = int(round(share * width))
        cv2.rectangle(canvas, (left, y + 294), (left + w, y + 316), colour, -1)
        if w > 50:
            text(canvas, f"{name} {100 * share:.0f}%", left + 4, y + 310, DARK, 0.38)
        left += w
    labels = "  ".join(f"{n} {100 * s:.1f}%" for n, s in zip(("image", "sentence", "state", "chunk"), mass / mass.sum()))
    text(canvas, labels, x + 34, y + 336, GREY, 0.38)


def panel(name, frame, t, image, state, task, seen, plan, dt, boxes, moved, fill, steps):
    canvas = np.full((40 + 2 * (TITLE + TILE_H) + 10, 3 * TILE_W, 3), DARK, np.uint8)
    model_steer = seen["chunks"][:, :steps, 0].mean()
    teacher_steer = None if plan is None else plan[:steps, 0].mean()
    text(canvas, f"{name}  frame {frame}  t {t:.1f} s   speed {state[0]:.2f} m/s, wheels {state[1]:+.2f}   "
         f"task '{task}'   {len(seen['chunks'])} noises", 10, 26, WHITE, 0.55)
    scale = TILE_W / image.shape[1]
    camera = cv2.resize(image, (TILE_W, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    grid, size = seen["grid"], TILE_H
    row1, row2 = 40, 40 + TITLE + TILE_H

    teacher = "no labels" if teacher_steer is None else f"teacher {teacher_steer:+.2f} (cyan)"
    caption(canvas, 0, row1, ["camera", f"first {steps} actions steer: model {model_steer:+.2f} (orange), {teacher}"])
    view = camera.copy()
    steer_bar(view, 40, view.shape[0] - 24, TILE_W - 80, model_steer, teacher_steer)
    canvas[row1 + TITLE:row1 + TITLE + view.shape[0], 0:TILE_W] = view

    side = seen["input"].shape[0]
    padding = sum(1 for r in range(grid) for c in range(grid)) - len(boxes)
    caption(canvas, TILE_W, row1, ["model input", f"{side}x{side} after resize_with_pad: {grid}x{grid} = "
                                                  f"{grid * grid} tokens, {padding} padding"])
    paste(canvas, model_input_view(seen, size, grid), TILE_W + (TILE_W - size) // 2, row1 + TITLE, (size, size))

    view, share = attention_view(seen, size, grid)
    padded = 1 - sum(share[r, c] for r, c, *_ in boxes)
    image_mass = seen["layer_mass"].mean(0)
    caption(canvas, 2 * TILE_W, row1, ["attention on the image (% per token)",
                                       f"image {100 * image_mass[0] / image_mass.sum():.0f}% of all attention, padding "
                                       f"{100 * padded:.0f}% of that; even: {100 / grid ** 2:.1f}%/token"])
    paste(canvas, view, 2 * TILE_W + (TILE_W - size) // 2, row1 + TITLE, (size, size))

    top = max(abs(moved).max(), FLOOR)          # one colour scale for both, so they compare
    for k, (what, plus, minus) in enumerate((("steer", "left", "right"), ("throttle", "more", "less"))):
        caption(canvas, k * TILE_W, row2, [f"occlusion -> {what}: each cell filled with {FILLS[fill]}",
                                           f"red {plus}, blue {minus}; first {steps} actions; max "
                                           f"{abs(moved[:, k]).max():.3f}, colours +-{top:.2f}"])
        view = overlay_cells(camera, moved[:, k] / top, boxes, scale, diverging,
                             label=lambda v, top=top: signed(v * top) if abs(v * top) >= 0.01 else "")
        canvas[row2 + TITLE:row2 + TITLE + view.shape[0], k * TILE_W:(k + 1) * TILE_W] = view

    caption(canvas, 2 * TILE_W, row2, ["the chunk", "steering and throttle for each starting noise"])
    chunk_view(canvas, 2 * TILE_W, row2 + TITLE, seen, plan, dt)
    return canvas


def layers_panel(name, frame, seen):
    maps, mass = seen["layer_maps"], seen["layer_mass"]
    layers, grid, size, gap = len(maps), seen["grid"], 200, 44
    cols = 4
    rows = (layers + cols - 1) // cols
    canvas = np.full((52 + rows * (size + gap), cols * (size + 20) + 10, 3), DARK, np.uint8)
    text(canvas, f"{name} frame {frame}: the action expert's attention on the image, per layer", 10, 20, WHITE, 0.5)
    text(canvas, "each map scaled to its own max; under it, where that layer's attention goes", 10, 38, GREY, 0.4)
    grey = cv2.cvtColor(cv2.cvtColor(seen["input"], cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    grey = cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)
    step = size / grid
    for layer in range(layers):
        x, y = 10 + (layer % cols) * (size + 20), 52 + (layer // cols) * (size + gap)
        share = maps[layer] / max(maps[layer].max(), 1e-9)
        view = grey.copy()
        colours = heat(share)
        for r in range(grid):
            for c in range(grid):
                y0, y1, x0, x1 = int(r * step), int((r + 1) * step), int(c * step), int((c + 1) * step)
                view[y0:y1, x0:x1] = (view[y0:y1, x0:x1] * 0.3 + colours[r, c] * 0.7).astype(np.uint8)
        canvas[y:y + size, x:x + size] = view
        m = mass[layer] / mass[layer].sum()
        kind = "self" if layer % 2 == 0 else "cross"
        text(canvas, f"L{layer} {kind}: image {100 * m[0]:.0f}%  sent {100 * m[1]:.0f}%", x, y + size + 15, WHITE, 0.38)
        text(canvas, f"state {100 * m[2]:.0f}%  chunk {100 * m[3]:.0f}%", x, y + size + 30, GREY, 0.36)
    return canvas


def video(seer, name, data, labels, task, path, every, rng, boxes):
    dt = float(np.median(np.diff(data["t"])))
    scale = TILE_W / data["images"].shape[2]
    height = int(data["images"].shape[1] * scale)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1 / (dt * every), (2 * TILE_W, height + 30))
    noise = seer.noise(rng, 1)                       # one noise for the whole episode: only the frame changes
    for i in range(0, len(data["t"]), every):
        image, state = data["images"][i], data["states"][i]
        seen = seer.look(image, state, task, noise)
        share = seen["layer_maps"].mean(0)
        share = share / share.sum()
        values = np.array([share[r, c] for r, c, *_ in boxes])
        camera = cv2.resize(image, (TILE_W, height), interpolation=cv2.INTER_AREA)
        view = overlay_cells(camera, values / values.max(), boxes, scale, lambda v: heat(np.array([v]))[0], 0.6)
        frame = np.full((height + 30, 2 * TILE_W, 3), DARK, np.uint8)
        frame[30:, :TILE_W], frame[30:, TILE_W:] = camera, view
        model = seen["chunks"][0, :4, 0].mean()
        teacher = None if labels is None else labels["plans"][i, :4, 0].mean()
        steer_bar(frame, 40, 30 + height - 24, TILE_W - 80, model, teacher)
        mass = seen["layer_mass"].mean(0)
        reference = "" if teacher is None else f" teacher {teacher:+.2f}"
        text(frame, f"{name} t {data['t'][i]:5.1f} s  steer model {model:+.2f}{reference}   "
                    f"image {100 * mass[0] / mass.sum():.0f}% of attention, padding {100 * (1 - values.sum()):.0f}% "
                    "of that", 10, 21, WHITE, 0.5)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def summary(seer, name, episode, data, task, noise, fill, steps, every):
    """Attention and occlusion averaged over every Nth frame: where the model looks on the whole drive."""
    frames = list(range(0, len(data["t"]) - steps, every))
    maps, moved, mass = [], [], []
    for i in frames:
        image, state = data["images"][i], data["states"][i]
        seen = seer.look(image, state, task, noise)
        boxes = seer.cells(*image.shape[:2], seen["grid"])
        moved.append(np.abs(seer.occlusion(image, state, task, noise, seen["chunks"], boxes, fill, steps)))
        maps.append(seen["layer_maps"].mean(0) / seen["layer_maps"].mean(0).sum())
        mass.append(seen["layer_mass"].mean(0) / seen["layer_mass"].mean(0).sum())
    grid, share, moved, mass = seen["grid"], np.mean(maps, 0), np.mean(moved, 0), np.mean(mass, 0)

    canvas = np.full((40 + TITLE + TILE_H + 10, 3 * TILE_W, 3), DARK, np.uint8)
    text(canvas, f"{name}  {episode}: {len(frames)} frames (every {every}th), {len(noise)} noises, task '{task}'",
         10, 26, WHITE, 0.55)
    first = dict(seen, input=seer.look(data["images"][frames[0]], data["states"][frames[0]], task, noise[:1])["input"],
                 layer_maps=share[None])
    view, _ = attention_view(first, TILE_H, grid)
    padded = 1 - sum(share[r, c] for r, c, *_ in boxes)
    caption(canvas, 0, 40, ["mean attention on the image (% per token)",
                            f"image {100 * mass[0]:.0f}% of all attention, padding {100 * padded:.0f}% of that"])
    paste(canvas, view, (TILE_W - TILE_H) // 2, 40 + TITLE, (TILE_H, TILE_H))

    image = data["images"][frames[len(frames) // 2]]
    scale = TILE_W / image.shape[1]
    camera = cv2.resize(image, (TILE_W, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    top = moved[:, 0].max()
    caption(canvas, TILE_W, 40, [f"mean |occlusion -> steer|, cells filled with {FILLS[fill]}",
                                 f"first {steps} actions; brightest {top:.3f}"])
    view = overlay_cells(camera, moved[:, 0] / top, boxes, scale, lambda v: heat(np.array([v]))[0], 0.7,
                         label=lambda v: f"{v * top:.3f}"[1:], ink=lambda v: WHITE if v < 0.5 else DARK)
    canvas[40 + TITLE:40 + TITLE + view.shape[0], TILE_W:2 * TILE_W] = view

    x0 = 2 * TILE_W
    caption(canvas, x0, 40, ["by token row (rows as in the model input)",
                             "attention (yellow) and steering sensitivity (orange), % of total"])
    rows_attention = share.sum(1)
    rows_moved = np.zeros(grid)
    for (r, *_), m in zip(boxes, moved[:, 0]):
        rows_moved[r] += m
    rows_moved /= rows_moved.sum()
    band, width = TILE_H / grid, TILE_W - 150
    for r in range(grid):
        y = int(40 + TITLE + r * band)
        text(canvas, f"row {r}", x0 + 8, y + int(band / 2) + 4, GREY, 0.4)
        for k, (value, colour) in enumerate(((rows_attention[r], YELLOW), (rows_moved[r], ORANGE))):
            yy = y + 8 + k * 16
            cv2.rectangle(canvas, (x0 + 60, yy), (x0 + 60 + int(value * width), yy + 12), colour, -1)
            text(canvas, f"{100 * value:.0f}%", x0 + 66 + int(value * width), yy + 11, colour, 0.38)
        if not any(b[0] == r for b in boxes):
            text(canvas, "padding", x0 + 140, y + int(band / 2) + 4, GREY, 0.38)
    numbers = {"checkpoint": name, "episode": episode, "frames": frames, "noises": len(noise), "fill": fill,
               "steps": steps, "attention_share": dict(zip(("image", "sentence", "state", "chunk"), mass.round(4).tolist())),
               "attention_on_image_tokens": share.round(4).tolist(),
               "attention_by_row": rows_attention.round(4).tolist(), "steer_sensitivity_by_row": rows_moved.round(4).tolist(),
               "mean_abs_occlusion": [[r, c, round(float(s), 4), round(float(t), 4)] for (r, c, *_), (s, t) in zip(boxes, moved)]}
    return canvas, numbers


def default_frames(steer: np.ndarray, speeds: np.ndarray) -> list[int]:
    """The hardest left, hardest right, and straightest moment at speed, by the teacher's (or the recorded) steering."""
    moving = np.flatnonzero(speeds > 0.5 * speeds.max())
    return sorted({int(steer.argmax()), int(steer.argmin()), int(moving[np.abs(steer[moving]).argmin()])})


def frame_at(item: str, t: np.ndarray) -> int:
    """'235' is frame 235; '41.7s' is the frame nearest 41.7 s."""
    return int(np.abs(t - float(item[:-1])).argmin()) if item.endswith("s") else int(item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("runs/drive"))
    parser.add_argument("--checkpoint", default="smolvla-v1", help="a version in <project>/vla, or a checkpoint directory")
    parser.add_argument("--episode", default="sim2-031",
                        help="an episode name in <project>/episodes (default: one LeRobot held out of training), or "
                             "a directory with an episode.npz, such as one written by `sim --record`")
    parser.add_argument("--frames", default="", help="comma-separated frame indices, or times such as 41.7s "
                                                     "(default: hardest left, hardest right, straightest at speed)")
    parser.add_argument("--task", default="drive along the track")
    parser.add_argument("--noises", type=int, default=4, help="starting noises to average over (at most 16)")
    parser.add_argument("--steps", type=int, default=4, help="chunk steps the occlusion measures (the ones played)")
    parser.add_argument("--fill", choices=("mean", "grey", "black"), default="mean", help="what hides a cell")
    parser.add_argument("--out", type=Path, help="default <project>/reports/see-<checkpoint>")
    parser.add_argument("--video", action="store_true", help="also write the attention on every frame as an mp4")
    parser.add_argument("--every", type=int, default=1, help="with --video, use every Nth frame")
    parser.add_argument("--summary", type=int, default=0, metavar="N",
                        help="average attention and occlusion over every Nth frame of the episode "
                             "(then no per-frame panels unless --frames is given)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    checkpoint = Project(args.project).checkpoint(args.checkpoint)
    name = Path(args.checkpoint).name
    out = args.out or args.project / "reports" / f"see-{name}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seer = Seer(checkpoint)
    print(f"loaded {checkpoint} in {time.perf_counter() - started:.1f} s")

    episode = Path(args.episode) if Path(args.episode, "episode.npz").is_file() else args.project / "episodes" / args.episode
    data, labels = np.load(episode / "episode.npz"), read_labels(episode)
    args.episode = episode.name
    dt = float(np.median(np.diff(data["t"])))
    frames = ([frame_at(f, data["t"]) for f in args.frames.split(",")] if args.frames else
              [] if args.summary else
              default_frames(data["actions"][:, 0] if labels is None else labels["plans"][:, 0, 0], data["states"][:, 0]))
    noise = seer.noise(rng, args.noises)
    boxes = None
    for frame in frames:
        started = time.perf_counter()
        image, state = data["images"][frame], data["states"][frame]
        plan = None if labels is None else labels["plans"][frame]
        seen = seer.look(image, state, args.task, noise)
        boxes = seer.cells(*image.shape[:2], seen["grid"])
        moved = seer.occlusion(image, state, args.task, noise, seen["chunks"], boxes, args.fill, args.steps)
        stem = out / f"{args.episode}-{frame:03d}"
        canvas = panel(name, frame, float(data["t"][frame]), image, state, args.task, seen, plan, dt, boxes, moved,
                       args.fill, args.steps)
        cv2.imwrite(f"{stem}.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{stem}-layers.png", cv2.cvtColor(layers_panel(name, frame, seen), cv2.COLOR_RGB2BGR))
        share = seen["layer_maps"].mean(0)
        mass = seen["layer_mass"].mean(0)
        stem.with_suffix(".json").write_text(json.dumps({
            "checkpoint": str(checkpoint), "episode": args.episode, "frame": frame, "t": float(data["t"][frame]),
            "state": state.tolist(), "task": args.task, "noises": args.noises, "seed": args.seed,
            "steer_model": float(seen["chunks"][:, :args.steps, 0].mean()),
            "steer_teacher": None if plan is None else float(plan[:args.steps, 0].mean()),
            "attention_share": dict(zip(("image", "sentence", "state", "chunk"), (mass / mass.sum()).round(4).tolist())),
            "attention_on_image_tokens": (share / share.sum()).round(4).tolist(),
            "attention_image_share_by_layer": (seen["layer_mass"][:, 0] / seen["layer_mass"].sum(1)).round(4).tolist(),
            "occlusion": {"fill": args.fill, "steps": args.steps,
                          "cells": [[r, c, round(float(s), 4), round(float(t), 4)]
                                    for (r, c, *_), (s, t) in zip(boxes, moved)]},
        }, indent=1))
        print(f"{stem}.png  ({time.perf_counter() - started:.1f} s)  steer model "
              f"{seen['chunks'][:, :args.steps, 0].mean():+.3f}"
              f"{'' if plan is None else f', teacher {plan[:args.steps, 0].mean():+.3f}'}; "
              f"attention on the image {100 * mass[0] / mass.sum():.0f}%; occlusion moves steering at most "
              f"{abs(moved[:, 0]).max():.3f}")

    if args.summary:
        started = time.perf_counter()
        canvas, numbers = summary(seer, name, args.episode, data, args.task, noise, args.fill, args.steps,
                                  args.summary)
        stem = out / f"{args.episode}-summary-{args.fill}"
        cv2.imwrite(f"{stem}.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        stem.with_suffix(".json").write_text(json.dumps(numbers, indent=1))
        print(f"{stem}.png  ({time.perf_counter() - started:.0f} s)  steering sensitivity by token row "
              f"{np.round(100 * np.array(numbers['steer_sensitivity_by_row'])).astype(int).tolist()} %, attention by row "
              f"{np.round(100 * np.array(numbers['attention_by_row'])).astype(int).tolist()} %")

    if args.video:
        path = out / f"{args.episode}-attention.mp4"
        started = time.perf_counter()
        if boxes is None:
            boxes = seer.cells(*data["images"].shape[1:3], seer.look(data["images"][0], data["states"][0],
                                                                      args.task, noise[:1])["grid"])
        video(seer, name, data, labels, args.task, path, args.every, rng, boxes)
        print(f"{path}  ({time.perf_counter() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
