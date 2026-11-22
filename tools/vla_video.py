"""Run a fine-tuned VLA (smolvla-v1) on a video, frame by frame, and watch what it would do.

    python tools/vla_video.py --episode sim2-031 [--rate 5] [--out runs/debug/vla-sim2-031.mp4]
    python tools/vla_video.py --video any.mp4 [--speed 1.35] [--start 0 --seconds 20]

Run it like tools/vla_app.py: with the LeRobot environment's Python, from the repository root,
with PYTHONPATH=. and HF_HOME set. The app has the same thing in its "Video" tab.

The source is a recorded episode (with its state and, if labelled, the teacher's plan) or any
video file. The model runs on `--rate` frames per second of it. An uploaded video has no state,
so the speed is fixed (--speed) and the wheels follow the model: each frame gets the steering
the model chose on the frame before, as a car's wheels would (--no-follow keeps --steer).
Frames are cropped to 4:3 and resized to the simulator's 320 x 240 camera first.

It writes an H.264 mp4 (plays in a browser), one frame per model run, at the source's time:

    top left      the frame with the path the chunk would drive (orange: mean of the samples,
                  thin: each sample, cyan: the teacher), and the steering it plays now
    top right     the chunk over its 1.6 s, and where the action expert's attention goes
    bottom        steering and throttle over the whole video, model against teacher, and a
                  cursor at the frame shown

and a .csv next to it with one row per model run.

The path is drawn with the simulator's camera geometry, so on a real video it is only a guide.
The steering is what this model would command; it was trained on one simulated track, so on
anything else it is a probe of the model, not a prediction of how a car would drive.
"""
from __future__ import annotations

import argparse
import csv
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from drive.learn.label import read_labels
from drive.viewer import CYAN, DARK, GREY, ORANGE, WHITE, Plot, text
from tools.vla_app import CAMERA_H, CAMERA_W, STEPS_PLAYED, Lab, chunk_plot, path_view

TIMELINE_H = 200                               # the strip chart at the bottom of each frame


# ---- the frames ----

def to_camera(image: np.ndarray) -> np.ndarray:
    """Centre-crop to the camera's 4:3 and resize to its 320 x 240, as uint8 RGB."""
    h, w = image.shape[:2]
    aspect = CAMERA_W / CAMERA_H
    if w > h * aspect:                                         # too wide: keep the middle
        cw = int(round(h * aspect))
        image = image[:, (w - cw) // 2:(w - cw) // 2 + cw]
    elif w < h * aspect:                                       # too tall: keep the middle
        ch = int(round(w / aspect))
        image = image[(h - ch) // 2:(h - ch) // 2 + ch]
    if image.shape[:2] != (CAMERA_H, CAMERA_W):
        image = cv2.resize(image, (CAMERA_W, CAMERA_H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image[:, :, :3]).astype(np.uint8)


def read_video(path: Path, rate: float, start: float = 0.0, seconds: float = 20.0):
    """Frames of any video file `rate` times per second from `start` for `seconds`: (images, times)."""
    import av

    images, times, next_t = [], [], start
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            t = frame.time
            if t is None or t < next_t - 1e-6:
                continue
            if t > start + seconds:
                break
            image = frame.to_ndarray(format="rgb24")
            # Phones store portrait-mounted video sideways with a display matrix (rotation=90 in ffprobe):
            # turn the frame as a player would, counter-clockwise by `rotation` degrees.
            turn = int(round(float(getattr(frame, "rotation", 0) or 0) / 90)) % 4
            images.append(to_camera(np.rot90(image, turn) if turn else image))
            times.append(t)
            next_t += 1.0 / rate
            while next_t <= t:                                 # the video skipped ahead
                next_t += 1.0 / rate
    if not images:
        raise ValueError(f"no frames in {path} between {start:.1f} s and {start + seconds:.1f} s")
    return np.stack(images), np.array(times)


def read_episode(directory: Path, rate: float, start: float = 0.0, seconds: float = 30.0):
    """Every Nth frame of a recorded episode: (images, times, states, plans or None)."""
    data, labels = np.load(directory / "episode.npz"), read_labels(directory)
    t = data["t"] - data["t"][0]
    every = max(1, int(round(1.0 / (rate * float(np.median(np.diff(t)))))))
    keep = np.flatnonzero((t >= start - 1e-6) & (t <= start + seconds + 1e-6))[::every]
    if not len(keep):
        raise ValueError(f"{directory.name} has no frames between {start:.1f} s and {start + seconds:.1f} s")
    plans = None if labels is None else labels["plans"][keep]
    return data["images"][keep], t[keep], data["states"][keep].astype(np.float32), plans


# ---- the model ----

def run_frames(lab: Lab, images, states, task, seed, samples, steps, speed, steer, follow, tick=None) -> list[dict]:
    """The model on every frame. `states` (N, 2) from an episode, or None: then `speed` is fixed and the
    wheels start at `steer` and, with `follow`, take the steering the model played on the frame before."""
    results = []
    for i, image in enumerate(images):
        if states is not None:
            speed_i, steer_i = (float(v) for v in states[i])
        else:
            speed_i = speed
            steer_i = (float(np.clip(results[-1]["played"][0], -1, 1)) if follow and results else steer)
        seen, _, _, elapsed = lab.run(image, speed_i, steer_i, task, seed, samples, steps, False)
        chunks = seen["chunks"]
        mass = seen["layer_mass"].mean(0)
        results.append({
            "chunks": chunks, "layer_mass": seen["layer_mass"],
            "speed": speed_i, "steer": steer_i,
            "played": chunks[:, :STEPS_PLAYED].mean(axis=(0, 1)),              # (steer, throttle) the car plays
            "first": chunks[:, 0].mean(0),
            "spread": float(chunks[:, :STEPS_PLAYED, 0].mean(1).std()),
            "image_share": float(mass[0] / mass.sum()),
            "ms": 1000 * elapsed,
        })
        if tick is not None:
            tick(i + 1, len(images))
    return results


# ---- drawing ----

def timeline(width, times, results, plans, now) -> np.ndarray:
    """Steering and throttle over the whole run (orange: model, cyan: teacher), a cursor at `now`."""
    canvas = np.full((TIMELINE_H, width, 3), DARK, np.uint8)
    begin, end = float(times[0]), max(float(times[-1]), float(times[0]) + 1e-3)
    played = np.array([r["played"] for r in results])
    teacher = None if plans is None else plans[:, :STEPS_PLAYED].mean(1)
    for k, (name, top) in enumerate((("steering (+ left)", 12), ("throttle", 108))):
        plot = Plot(canvas, 40, top, width - 60, 76, begin, end)
        text(canvas, name, 46, top + 14, GREY, 0.4)
        if teacher is not None:
            plot.line(times, teacher[:, k], CYAN, 2)
        plot.line(times, played[:, k], ORANGE, 2)
        x = plot.x(now)
        cv2.line(canvas, (x, top), (x, top + 76), WHITE, 1)
    legend = "model: the first 0.4 s of each chunk (what the car plays)" + ("   teacher (cyan)" if teacher is not None else "")
    text(canvas, f"{legend}   {begin:.1f} .. {end:.1f} s", 40, TIMELINE_H - 6, GREY, 0.38)
    return canvas


def frame_view(lab: Lab, image, t, result, plan, strip, stats, source) -> np.ndarray:
    """One frame of the output video: path | chunk and numbers, over the timeline."""
    seen = {"chunks": result["chunks"], "layer_mass": result["layer_mass"]}
    left = path_view(lab, image, result["chunks"], plan, result["speed"], result["steer"])
    right = chunk_plot(seen, plan, lab.dt)
    height = max(left.shape[0], right.shape[0] + 120)
    width = left.shape[1] + right.shape[1]
    canvas = np.full((height + strip.shape[0], width, 3), DARK, np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:] = right
    text(canvas, f"{source}  t {t:5.1f} s", 10, 20, WHITE, 0.5)
    x, y = left.shape[1] + 34, right.shape[0] + 10
    steer, throttle = result["played"]
    lines = [f"steering {steer:+.2f}  throttle {throttle:+.2f}  (spread {result['spread']:.3f})",
             f"state: speed {result['speed']:.2f} m/s, wheels {result['steer']:+.2f}"]
    if stats is not None:
        values = (result["speed"], result["steer"])[:len(stats["mean"])]    # [speed] alone: export --state speed_mps
        z = [(v - float(stats["mean"][k])) / float(stats["std"][k]) for k, v in enumerate(values)]
        lines.append("normalised: " + ", ".join(f"{name} {v:+.1f} sd" for name, v in zip(("speed", "wheels"), z))
                     + ("  OUTSIDE TRAINING" if max(map(abs, z)) > 3 else ""))
    if plan is not None:
        lines.append(f"teacher steering {plan[:STEPS_PLAYED, 0].mean():+.2f}  throttle {plan[:STEPS_PLAYED, 1].mean():+.2f}")
    lines.append(f"{result['ms']:.0f} ms for this frame")
    for k, line in enumerate(lines):
        text(canvas, line, x, y + 20 * k, WHITE if k == 0 else GREY, 0.42)
    canvas[height:] = strip
    return canvas


def write_mp4(path: Path, frames, fps: float) -> None:
    """H.264 in yuv420p with the index at the front: plays in a browser."""
    import av

    h, w = frames[0].shape[:2]
    with av.open(str(path), mode="w", options={"movflags": "faststart"}) as container:
        stream = container.add_stream("libx264", rate=Fraction(float(fps)).limit_denominator(1000))
        stream.width, stream.height, stream.pix_fmt = w - w % 2, h - h % 2, "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for frame in frames:
            picture = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame[:h - h % 2, :w - w % 2]), format="rgb24")
            for packet in stream.encode(picture):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_csv(path: Path, times, results, plans) -> None:
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["t", "speed", "wheels", "steer_played", "throttle_played", "steer_first", "throttle_first",
                      "steer_spread", "teacher_steer", "teacher_throttle", "image_attention", "ms"])
        for i, (t, r) in enumerate(zip(times, results)):
            teacher = ["", ""] if plans is None else [f"{v:.4f}" for v in plans[i, :STEPS_PLAYED].mean(0)]
            out.writerow([f"{t:.3f}", f"{r['speed']:.3f}", f"{r['steer']:.4f}", f"{r['played'][0]:.4f}",
                          f"{r['played'][1]:.4f}", f"{r['first'][0]:.4f}", f"{r['first'][1]:.4f}",
                          f"{r['spread']:.4f}", *teacher, f"{r['image_share']:.3f}", f"{r['ms']:.0f}"])


def report(times, results, plans, source, follow) -> str:
    """What happened over the run, in Markdown."""
    played = np.array([r["played"] for r in results])
    steer = played[:, 0]
    ms = np.array([r["ms"] for r in results])
    lines = [f"**{source}**: the model ran on **{len(results)} frames** from {times[0]:.1f} to {times[-1]:.1f} s, "
             f"{ms.mean():.0f} ms per frame (total {ms.sum() / 1000:.0f} s)",
             f"**Steering** it would play: mean {steer.mean():+.2f}, from {steer.min():+.2f} to {steer.max():+.2f}; "
             f"**throttle** mean {played[:, 1].mean():+.2f}",
             f"**Change of steering between frames** (weaving): mean |Δ| **{np.abs(np.diff(steer)).mean():.3f}**"
             if len(steer) > 1 else ""]
    if plans is not None:
        teacher = plans[:, :STEPS_PLAYED].mean(1)
        error = np.abs(steer - teacher[:, 0])
        corr = np.corrcoef(steer, teacher[:, 0])[0, 1] if len(steer) > 2 else float("nan")
        lines.append(f"**Against the teacher**: steering error mean {error.mean():.3f}, worst {error.max():.3f} "
                     f"(t {times[int(error.argmax())]:.1f} s), correlation {corr:.3f}; the teacher's mean |Δ| between "
                     f"frames is {np.abs(np.diff(teacher[:, 0])).mean():.3f}" if len(steer) > 1 else "")
    elif follow:
        lines.append("No state in a video: the speed is fixed and the wheels follow the model's previous steering.")
    lines.append(f"**Attention on the image**: mean {100 * np.mean([r['image_share'] for r in results]):.0f} %")
    return "\n\n".join(line for line in lines if line)


def render(lab: Lab, images, times, results, plans, source, out: Path, rate: float, tick=None) -> Path:
    """The annotated mp4 and its csv. Returns the mp4's path."""
    frames = []
    for i, (image, t, result) in enumerate(zip(images, times, results)):
        strip = timeline(640 + 480, times, results, plans, t)
        frames.append(frame_view(lab, image, t, result, None if plans is None else plans[i], strip,
                                 lab.state_stats, source))
        if tick is not None:
            tick(i + 1, len(images))
    out.parent.mkdir(parents=True, exist_ok=True)
    span = float(times[-1] - times[0])
    fps = (len(frames) - 1) / span if len(frames) > 1 and span > 0 else rate
    write_mp4(out, frames, fps)
    write_csv(out.with_suffix(".csv"), times, results, plans)
    return out


# ---- the app's tab ----

def video_tab(lab: Lab, names: list[str], default_episode: str | None, out_dir: Path,
              examples_dir: Path = Path("runs/bdd100k/web")) -> None:
    """The "Video" tab of tools/vla_app.py. Call inside a gr.Blocks. The mp4s in `examples_dir`
    (tools/web_videos.py writes them) are offered as examples to click."""
    import gradio as gr

    def go(source, episode, upload, rate, start, seconds, task, seed, samples, steps, speed, steer, follow,
           progress=gr.Progress()):
        started = time.perf_counter()
        if source == "episode":
            if episode is None:
                raise gr.Error("pick an episode")
            images, times, states, plans = read_episode(lab.project.episodes / episode, rate, start, seconds)
            name = episode
        else:
            if not upload:
                raise gr.Error("upload or record a video first")
            images, times = read_video(Path(upload), rate, start, seconds)
            states, plans, name = None, None, Path(upload).stem
        progress(0, desc=f"the model on {len(images)} frames")
        results = run_frames(lab, images, states, task, seed, samples, steps, speed, steer, follow,
                             tick=lambda i, n: progress(i / n * 0.9, desc=f"the model: frame {i} of {n}"))
        out = out_dir / f"{name}-{time.strftime('%H%M%S')}.mp4"
        render(lab, images, times, results, plans, name, out, rate,
               tick=lambda i, n: progress(0.9 + 0.1 * i / n, desc=f"drawing frame {i} of {n}"))
        note = report(times, results, plans, name, follow and states is None)
        note += f"\n\nWritten in {time.perf_counter() - started:.0f} s to `{out}` and `{out.with_suffix('.csv')}`"
        return str(out), note, str(out.with_suffix(".csv"))

    gr.Markdown("Run the model on every frame of a video and watch the steering it would choose. "
                "A recorded episode brings its state and the teacher; any other video gets a fixed speed and wheels "
                "that follow the model. Frames are cropped to 4:3 and resized to 320 x 240. "
                "On an RTX 2060 it takes about 0.7 s per frame with 2 samples: 30 s at 5 per second is ~2 minutes.")
    with gr.Row():
        with gr.Column(scale=4):
            source = gr.Radio(["episode", "uploaded video"], value="episode", label="source")
            episode = gr.Dropdown(names, value=default_episode, label="episode")
            upload = gr.Video(label="any video (upload or webcam)", sources=["upload", "webcam"], height=240)
            examples = sorted(examples_dir.glob("*.mp4"))
            if examples:
                gr.Examples([["uploaded video", str(e)] for e in examples], inputs=[source, upload],
                            example_labels=[e.stem for e in examples],
                            label=f"ready to try: {examples_dir} (tools/fetch_bdd100k.py, then tools/web_videos.py)")
            with gr.Row():
                rate = gr.Slider(1, 10, value=5, step=1, label="model runs per second of video")
                start = gr.Number(0, label="start at (s)")
                seconds = gr.Slider(1, 60, value=30, step=1, label="seconds")
            task = gr.Textbox("drive along the track", label="task sentence")
            with gr.Row():
                seed = gr.Number(0, precision=0, label="noise seed (the same for every frame)")
                samples = gr.Slider(1, 8, value=2, step=1, label="samples")
                steps = gr.Slider(1, 20, value=lab.default_steps, step=1, label="denoising steps")
            with gr.Accordion("State for an uploaded video", open=False):
                with gr.Row():
                    speed = gr.Slider(0.0, 3.0, value=1.35, step=0.05, label="speed (m/s)")
                    steer = gr.Slider(-1.0, 1.0, value=0.0, step=0.01, label="wheels at the start (+ left)")
                follow = gr.Checkbox(True, label="the wheels follow the model's previous steering")
            button = gr.Button("Run the model on the video", variant="primary")
        with gr.Column(scale=6):
            output = gr.Video(label="what the model would do, frame by frame", autoplay=True)
            numbers = gr.Markdown()
            table = gr.File(label="one row per frame (csv)")
    button.click(go, inputs=[source, episode, upload, rate, start, seconds, task, seed, samples, steps, speed,
                             steer, follow], outputs=[output, numbers, table], concurrency_limit=1)
    upload.change(lambda v: gr.Radio(value="uploaded video") if v else gr.Radio(), inputs=upload, outputs=source)


# ---- the command line ----

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("runs/drive"))
    parser.add_argument("--checkpoint", default="smolvla-v1", help="a version in <project>/vla, or a checkpoint directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", help="a recorded episode, e.g. sim2-031 (with its state and teacher)")
    source.add_argument("--video", type=Path, help="any video file")
    parser.add_argument("--rate", type=float, default=5.0, help="model runs per second of video")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the video")
    parser.add_argument("--seconds", type=float, default=30.0, help="how much of it")
    parser.add_argument("--task", default="drive along the track")
    parser.add_argument("--samples", type=int, default=2, help="starting noises per frame")
    parser.add_argument("--steps", type=int, default=0, help="denoising steps (default: the checkpoint's)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.35, help="the speed given with --video (m/s)")
    parser.add_argument("--steer", type=float, default=0.0, help="the wheels at the start with --video")
    parser.add_argument("--no-follow", action="store_true", help="with --video, keep the wheels at --steer")
    parser.add_argument("--out", type=Path, help="default runs/debug/vla-<source>.mp4")
    args = parser.parse_args()

    started = time.perf_counter()
    lab = Lab(args.project, args.checkpoint)
    print(f"loaded {args.checkpoint} in {time.perf_counter() - started:.1f} s")
    if args.episode:
        images, times, states, plans = read_episode(lab.project.episodes / args.episode, args.rate, args.start, args.seconds)
        name = args.episode
    else:
        images, times = read_video(args.video, args.rate, args.start, args.seconds)
        states, plans, name = None, None, args.video.stem
    out = args.out or Path("runs/debug") / f"vla-{name}.mp4"
    steps = args.steps or lab.default_steps

    def tick(i, n):
        print(f"\r  {i}/{n}", end="" if i < n else "\n", flush=True)

    print(f"the model on {len(images)} frames of {name}")
    results = run_frames(lab, images, states, args.task, args.seed, args.samples, steps, args.speed, args.steer,
                         not args.no_follow, tick)
    print("drawing")
    render(lab, images, times, results, plans, name, out, args.rate, tick)
    print(report(times, results, plans, name, not args.no_follow and states is None).replace("**", ""))
    print(f"wrote {out} and {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
