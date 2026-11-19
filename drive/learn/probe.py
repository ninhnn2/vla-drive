"""probe: look inside a trained CNN, and judge its output before driving it.

    ./drive.sh probe v1 layers       every layer: output shape, parameters, how much of the image one unit sees
    ./drive.sh probe v1 history      the training curves, and what they say: still learning? overfitting?
    ./drive.sh probe v1 training     inside training: gradient size, update/weight ratio, dead units, per epoch
    ./drive.sh probe v1 errors       accuracy on the held-out episodes: per step of the plan, per kind of
                                     road, against two baselines; the worst frames, drawn
    ./drive.sh probe v1 activations  what each layer outputs; units that never fire; the feature maps, drawn
    ./drive.sh probe v1 saliency     which part of the image the steering depends on, drawn as a heat map
    ./drive.sh probe v1 ablation     what each input is worth: the error with the picture, the speed or the
                                     steering state replaced by its average
    ./drive.sh probe v1 behaviour    mirror the image, dim it, blank it, change the speed: does it react
                                     as a driver should?
    ./drive.sh probe v1 onpolicy     let it drive, and ask the teacher at every tick what it would steer:
                                     the error where the model takes the car, against the distance off-centre
    ./drive.sh probe v1 report       all of the above in one HTML page, with charts

Run it with the project the model was trained on: `errors`, `activations` and `behaviour`
use the same held-out episodes as training did. Pictures go to <project>/probe/<model>/.
Everything runs on the CPU and is deterministic. The verdicts are rules of thumb, not
proof; docs/thuc-hanh explains how to read every block. For a VLA, see tools/probe_vla.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import dataset as datasets
from .label import episodes, read_labels
from .project import Project, add_project_argument

WHAT = ("layers", "history", "training", "errors", "ablation", "activations", "saliency", "behaviour", "onpolicy",
        "report")
MAGENTA, CYAN, WHITE = (240, 80, 210), (60, 220, 240), (240, 240, 240)
#: Bins of the label's first steering command: the kind of road the frame is on.
ROADS = (("straight", 0.0, 0.1), ("gentle bend", 0.1, 0.3), ("sharp bend", 0.3, 1.01))


# --- loading ---------------------------------------------------------------------------

def load_model(project: Project, ref: str):
    """(checkpoint path, model in eval mode on the CPU, its metadata)."""
    from .model import load
    path = project.checkpoint(ref)
    if path.is_dir():
        raise ValueError(f"{ref} is a LeRobot (VLA) checkpoint; probe it with tools/probe_vla.py")
    model, info = load(path)
    return path, model, info


def held_out(project: Project, hw):
    """(validation Dataset, episode names): the frames training held out, as train.py split them."""
    data = datasets.load(project.episodes, hw)
    _, validation = data.split()
    return validation, [path.name for path in episodes(project.episodes)]


def predict(model, images: np.ndarray, states: np.ndarray, batch: int = 512) -> np.ndarray:
    """(N, steps, 2) plans for shrunk uint8 images (N, H, W, 3) and states (N, 2)."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            out.append(model(torch.from_numpy(np.ascontiguousarray(images[i:i + batch])),
                             torch.from_numpy(np.asarray(states[i:i + batch], np.float32))).numpy())
    return np.concatenate(out).astype(np.float64)


def frame_index(project: Project, name: str, position: int) -> int:
    """A held-out frame's index in its recorded episode (training drops the invalid ones)."""
    labels = read_labels(project.episodes / name)
    return int(np.flatnonzero(labels["valid"])[position])


def pick_frame(project: Project, ref: str | None, hw) -> dict:
    """One recorded frame: `ref` is 'sim-038:120', or None for the sharpest bend of the first held-out episode."""
    if ref:
        name, _, index = ref.partition(":")
        if not (project.episodes / name / "episode.npz").is_file() or not index.isdigit():
            raise ValueError(f"--frame {ref!r}: want EPISODE:INDEX, such as sim-038:120, with an episode under "
                             f"{project.episodes}")
        index = int(index)
    else:
        validation, names = held_out(project, hw)
        name = names[int(validation.episode[0])]
        index = None
    labels = read_labels(project.episodes / name)
    with np.load(project.episodes / name / "episode.npz") as data:
        if index is None:                         # the sharpest bend, among frames with a usable label
            valid = np.flatnonzero(labels["valid"])
            index = int(valid[np.argmax(np.abs(labels["plans"][valid, 0, 0]))])
        if index >= len(data["t"]):
            raise ValueError(f"{name} has {len(data['t'])} frames; there is no frame {index}")
        frame = {"name": name, "index": index, "image": data["images"][index], "state": data["states"][index],
                 "action": data["actions"][index], "t": float(data["t"][index])}
    frame["plan"] = labels["plans"][index] if labels is not None and labels["valid"][index] else None
    frame["plan_dt"] = labels["dt"] if labels is not None else 0.0
    frame["small"] = datasets.shrink(frame["image"], hw)
    return frame


def describe(module) -> str:
    from torch import nn
    if isinstance(module, nn.Conv2d):
        return (f"Conv2d {module.in_channels}->{module.out_channels}, {module.kernel_size[0]}x{module.kernel_size[1]}, "
                f"stride {module.stride[0]}")
    if isinstance(module, nn.Linear):
        return f"Linear {module.in_features}->{module.out_features}"
    return type(module).__name__


def named_layers(model) -> list[tuple[str, object]]:
    from .model import named_layers as names
    return names(model)


def outputs(model, images: np.ndarray, states: np.ndarray) -> dict:
    """{layer name: its output as a numpy array} for one batch."""
    import torch
    captured, hooks = {}, []
    for name, module in named_layers(model):
        hooks.append(module.register_forward_hook(
            lambda m, inputs, out, name=name: captured.__setitem__(name, out.detach().numpy())))
    try:
        with torch.no_grad():
            model(torch.from_numpy(np.ascontiguousarray(images)), torch.from_numpy(np.asarray(states, np.float32)))
    finally:
        for hook in hooks:
            hook.remove()
    return captured


def shape(array_shape) -> str:
    return "(" + ", ".join(str(n) for n in array_shape) + ")"


# --- layers ----------------------------------------------------------------------------

def layers(model, info: dict, log=print) -> dict:
    """Every layer with its output shape for one frame, its parameter count and its receptive field."""
    from torch import nn
    h, w = model.image_hw
    out = outputs(model, np.zeros((1, h, w, 3), np.uint8), np.zeros((1, 2), np.float32))
    total = sum(p.numel() for p in model.parameters())
    log(f"ChunkNet: {total:,} parameters. In: one {h}x{w} RGB frame (the camera's 240x320, shrunk) and the "
        f"state [speed, steer]. Out: {model.steps} steps x [steer, throttle], {info['plan_dt']:g} s apart")
    log("  before the first layer: pixels/255 - 0.5 (so -0.5..0.5), channels first; speed / 3")
    log(f"  {'layer':<10} {'what it is':<32} {'output shape':<18} {'parameters':>10}   one unit sees")
    log(f"  {'input':<10} {'image':<32} {shape((1, 3, h, w)):<18} {0:>10}   1x1 px")
    rf, jump, rows = 1, 1, []
    for name, module in named_layers(model):
        params = sum(p.numel() for p in module.parameters())
        sees = ""
        if isinstance(module, nn.Conv2d):
            rf, jump = rf + (module.kernel_size[0] - 1) * jump, jump * module.stride[0]
            sees = f"{rf}x{rf} px of the {h}x{w} input"
        if name == "head.0":
            log(f"  {'(concat)':<10} {'vision features + state':<32} {shape((1, module.in_features)):<18} {0:>10}")
        log(f"  {name:<10} {describe(module):<32} {shape(out[name].shape):<18} {params:>10,}   {sees}".rstrip())
        rows.append({"layer": name, "kind": describe(module), "shape": list(out[name].shape), "parameters": params})
    log(f"  {'(view)':<10} {'reshape to the plan':<32} {shape((1, model.steps, 2)):<18} {0:>10}")
    vision = sum(p.numel() for p in model.vision.parameters())
    biggest = max(rows, key=lambda r: r["parameters"])
    log(f"  vision (the 5 convolutions): {vision:,} parameters ({100 * vision / total:.0f} %); "
        f"head (the 3 linear layers): {total - vision:,} ({100 * (total - vision) / total:.0f} %)")
    log(f"  the biggest layer is {biggest['layer']} ({biggest['kind']}): {biggest['parameters']:,} parameters, "
        f"{100 * biggest['parameters'] / total:.0f} % of the network")
    return {"parameters": total, "vision_parameters": vision, "layers": rows}


# --- history ---------------------------------------------------------------------------

def history(report: dict, log=print, baseline: float | None = None) -> dict:
    """The training log as a table, and what it says.

    `baseline` is the validation loss of the average plan (train.average_plan_loss); reports
    written since it was added carry it, for older ones the caller works it out.
    """
    h = report["history"]
    baseline = report.get("average_plan_val_loss", baseline)
    log(f"{report['frames']} training frames, {report['validation_frames']} validation frames; {report['epochs']} epochs, "
        f"batch {report.get('batch', 128)}, lr {report.get('lr', 0.001):g}, augmentation "
        f"{'on' if report.get('augmentation', True) else 'off'}, seed {report.get('seed', 0)}; "
        f"{report['seconds']} s on {report['device']}")
    worst = max(max(e["train_loss"], e["val_loss"]) for e in h)
    log(f"  {'epoch':>5} {'train loss':>10} {'val loss':>9} {'val/train':>9} {'steer err':>9}   validation loss")
    for e in h:
        bar = "#" * max(1, round(30 * e["val_loss"] / worst))
        log(f"  {e['epoch']:>5} {e['train_loss']:>10.5f} {e['val_loss']:>9.5f} {e['val_loss'] / e['train_loss']:>9.2f} "
            f"{e['val_steer_error']:>9.4f}   {bar}")
    best = min(h, key=lambda e: e["val_loss"])
    last = h[-1]
    reach = last["val_loss"] / baseline if baseline else None
    if reach is not None:
        log(f"  the average plan, the same for every frame, scores validation {baseline:.5f}; this model ends at "
            f"{last['val_loss']:.5f}, {100 * reach:.0f} % of that")
    verdicts = []
    if reach is not None and reach > 0.5:
        verdicts.append(f"NOT LEARNING: still {100 * reach:.0f} % of the average plan's loss; it learned little more "
                        "than the average, not to read the picture. The learning rate (too high or too low), or the "
                        "labels")
    elif reach is not None and reach > 0.25:
        verdicts.append(f"UNDERFITTING: still {100 * reach:.0f} % of the average plan's loss; it learned a little. "
                        "A learning rate nearer 0.001, more epochs, or more data")
    elif reach is None and h[0]["train_loss"] < 1.2 * last["train_loss"]:
        verdicts.append("NOT LEARNING: training loss hardly fell. The learning rate (too high or too low), or the "
                        "labels")
    if last["val_loss"] > 1.1 * best["val_loss"] and best["epoch"] < last["epoch"]:
        verdicts.append(f"OVERFITTING: validation loss was lowest at epoch {best['epoch']} ({best['val_loss']:.5f}) "
                        f"and ended {last['val_loss'] / best['val_loss']:.1f}x higher while training loss fell. "
                        "Fewer epochs, more or more varied data")
    if last["val_loss"] > 3 * last["train_loss"]:
        verdicts.append(f"GAP: validation loss is {last['val_loss'] / last['train_loss']:.1f}x the training loss: the "
                        "model knows the training episodes much better than new ones. More varied data")
    if len(h) >= 4 and h[-4]["val_loss"] > 1.1 * last["val_loss"] and best is last:
        verdicts.append("STILL LEARNING: validation loss fell more than 10 % over the last 3 epochs. More epochs may "
                        "help (the learning rate decays to 0 at the last epoch, so curves flatten there anyway)")
    if not verdicts:
        verdicts.append("HEALTHY: both losses fell and flattened together; validation ended near its best"
                        + (" and far below the average plan's" if reach is not None else ""))
    for v in verdicts:
        log("  " + v)
    log("  Low validation error does not prove the model drives: ./drive.sh evaluate decides")
    return {"best_epoch": best["epoch"], "verdicts": verdicts, "average_plan_val_loss": baseline}


# --- training --------------------------------------------------------------------------

#: An update moves a weight matrix by about this fraction of its size when learning goes well
#: (a rule of thumb from Karpathy, "A Recipe for Training Neural Networks", 2019).
HEALTHY_UPDATE = 1e-3


def training(report: dict, log=print) -> dict:
    """Per epoch: the learning rate, gradient norms, update/weight ratios and dead units the training recorded."""
    h = report["history"]
    if "grad_norm" not in h[0]:
        raise ValueError("this model was trained before train.py recorded what happens inside training; "
                         "train it again (with the same seed it learns the same weights)")
    layers = list(h[0]["grad_norm"])
    shown = [layers[0], layers[len(layers) // 2 - 1], layers[-3], layers[-1]] if len(layers) > 4 else layers
    relus = list(h[0]["dead"])
    log("gradient norm, the mean per step: how hard the loss pulls on each layer's weights")
    log(f"  {'epoch':>5} {'lr':>9}   " + "  ".join(f"{name:>9}" for name in shown))
    for e in h:
        log(f"  {e['epoch']:>5} {e['lr']:>9.2e}   " + "  ".join(f"{e['grad_norm'][n]:>9.3g}" for n in shown))
    log("update / weight: how far one step moves each weight matrix, against its size (healthy: about 1e-3)")
    log(f"  {'epoch':>5}   " + "  ".join(f"{name:>9}" for name in shown))
    for e in h:
        log(f"  {e['epoch']:>5}   " + "  ".join(f"{e['update_ratio'][n]:>9.1e}" for n in shown))
    log("units that never fire on the validation frames (dead ReLU), per layer")
    log(f"  {'epoch':>5}   " + "  ".join(f"{name:>9}" for name in relus))
    for e in h:
        log(f"  {e['epoch']:>5}   " + "  ".join(f"{100 * e['dead'][n]:>8.0f}%" for n in relus))

    first, last = h[0], h[-1]
    verdicts = []
    early = max(first["update_ratio"].values())
    if early > 10 * HEALTHY_UPDATE:
        verdicts.append(f"STEPS TOO BIG: in the first epoch a step moved a layer by {early:.0e} of its size, "
                        f"{early / HEALTHY_UPDATE:.0f}x the healthy 1e-3. Lower the learning rate")
    elif early < HEALTHY_UPDATE / 10:
        verdicts.append(f"STEPS TOO SMALL: in the first epoch the biggest step moved a layer by {early:.0e} of its "
                        "size, far below 1e-3. Raise the learning rate")
    jump = max(last["dead"][n] - first["dead"][n] for n in relus)
    worst = max(relus, key=lambda n: last["dead"][n])
    if last["dead"][worst] >= 0.999:
        verdicts.append(f"A DEAD LAYER: every unit of {worst} is silent, so nothing reaches the layers after it")
    elif last["dead"][worst] > 0.5:
        verdicts.append(f"MANY DEAD UNITS: {100 * last['dead'][worst]:.0f} % of {worst} never fire; capacity is wasted "
                        "(often a learning rate that was high early on)")
    if first["grad_norm"][layers[0]] > 0 and last["grad_norm"][layers[0]] / first["grad_norm"][layers[0]] < 1e-3:
        verdicts.append(f"VANISHING: the gradient reaching {layers[0]} fell a thousandfold; the first layers stopped "
                        "learning")
    if not verdicts:
        verdicts.append(f"HEALTHY: steps of {early:.0e} at the start, shrinking as the learning rate decays; "
                        f"{100 * last['dead'][worst]:.0f} % of {worst} dead at most")
    for v in verdicts:
        log("  " + v)
    return {"verdicts": verdicts, "dead_jump": jump}


# --- errors ----------------------------------------------------------------------------

def draw_frame(frame: dict, plan_model: np.ndarray, plan_dt: float, title: str, extra: list[str]) -> np.ndarray:
    """The camera frame at 3x with the label's path in cyan and the model's in magenta (RGB)."""
    from ..config import CameraParams
    from ..sim.render import Camera
    from .browse import SCALE, draw, plan_path
    camera = Camera(CameraParams())
    picture = draw(frame["image"], frame["state"], frame["action"], frame["plan"], plan_dt, title, camera)
    clean = cv2.resize(frame["image"], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    picture[-34:] = clean[-34:]                    # browse's key help means nothing in a still picture
    path = plan_path(plan_model, plan_dt, frame["state"][0], frame["state"][1])
    pixels = camera.project(*path.T) * SCALE
    pixels = pixels[np.isfinite(pixels).all(axis=1)].round().astype(np.int32)
    if len(pixels) > 1:
        cv2.polylines(picture, [pixels], False, MAGENTA, 3, cv2.LINE_AA)
    for row, line in enumerate(["magenta: the model's plan"] + extra):
        cv2.putText(picture, line, (12, 132 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    MAGENTA if row == 0 else WHITE, 2, cv2.LINE_AA)
    return picture


def errors(project: Project, model, info: dict, out: Path, worst: int = 5, log=print) -> dict:
    """Mean |error| on the held-out frames, by plan step, kind of road and speed, against two baselines."""
    validation, names = held_out(project, info["image_hw"])
    held = sorted({names[int(i)] for i in np.unique(validation.episode)})
    pred = predict(model, validation.images, validation.states)
    label = validation.plans.astype(np.float64)
    err = np.abs(pred - label)                                     # (N, steps, 2)
    dt = info["plan_dt"]
    # Two predictors that learned nothing about the image, to measure the model against:
    #   average  the mean plan of the held-out frames, whatever the picture
    #   keep     hold the steering the car has now, for the whole plan
    average = np.abs(label - label.mean(axis=0, keepdims=True))
    keep = np.abs(label[..., 0] - validation.states[:, None, 1])
    log(f"held out: {len(held)} episodes ({', '.join(held)}), {len(label)} frames that training never used")
    log("mean |error| per step of the plan; steer and throttle are in [-1, 1]")
    log(f"  {'step':>4} {'ahead':>6}   {'steer: model':>12} {'average':>8} {'keep':>6}   {'throttle: model':>15} {'average':>8}")
    for k in list(range(0, model.steps, 3)) + ([model.steps - 1] if (model.steps - 1) % 3 else []):
        log(f"  {k:>4} {k * dt:>5.1f}s   {err[:, k, 0].mean():>12.4f} {average[:, k, 0].mean():>8.4f} "
            f"{keep[:, k].mean():>6.4f}   {err[:, k, 1].mean():>15.4f} {average[:, k, 1].mean():>8.4f}")
    overall = {"steer": float(err[..., 0].mean()), "throttle": float(err[..., 1].mean()),
               "steer_average": float(average[..., 0].mean()), "steer_keep": float(keep.mean())}
    log(f"  {'all':>4} {'':>6}   {overall['steer']:>12.4f} {overall['steer_average']:>8.4f} {overall['steer_keep']:>6.4f}"
        f"   {overall['throttle']:>15.4f} {average[..., 1].mean():>8.4f}")
    skill = 1 - overall["steer"] / overall["steer_keep"]
    log(f"  the model's steering error is {100 * skill:.0f} % below 'keep' (the better baseline)"
        if skill > 0 else "  the model is no better than holding the steering it has: it has not learned to see")

    first = label[:, 0, 0]
    log("by the road the frame is on (the label's first steering command)")
    log(f"  {'road':<12} {'frames':>6} {'steer error':>11} {'bias':>7}")
    roads = {}
    for road, low, high in ROADS:
        pick = (np.abs(first) >= low) & (np.abs(first) < high)
        if pick.any():
            # bias > 0: steers more than the label on the bend's own side (oversteer); < 0: understeers
            bias = float(np.mean((pred[pick, :, 0] - label[pick, :, 0]) * np.sign(first[pick])[:, None]))
            roads[road] = {"frames": int(pick.sum()), "steer_error": float(err[pick, :, 0].mean()), "bias": bias}
            log(f"  {road:<12} {pick.sum():>6} {err[pick, :, 0].mean():>11.4f} {bias:>+7.3f}")
    log("  bias: + steers harder into the bend than the teacher, - less (cuts it short, drifts wide)")
    speed = validation.states[:, 0]
    log("by speed")
    for low, high in ((0.0, 1.2), (1.2, 1.5), (1.5, 9.0)):
        pick = (speed >= low) & (speed < high)
        if pick.any():
            log(f"  {low:.1f}-{high if high < 9 else 3.0:.1f} m/s {pick.sum():>6} frames   steer error "
                f"{err[pick, :, 0].mean():.4f}   throttle error {err[pick, :, 1].mean():.4f}")

    out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-err[..., 0].mean(axis=1))[:worst]
    log(f"the {len(order)} worst frames (mean steer error over the plan), drawn to {out}/")
    worst_frames = []
    for rank, i in enumerate(order, 1):
        name = names[int(validation.episode[i])]
        position = int(i - np.flatnonzero(validation.episode == validation.episode[i])[0])
        index = frame_index(project, name, position)
        frame = pick_frame(project, f"{name}:{index}", info["image_hw"])
        path = out / f"worst-{rank}-{name}-{index:04d}.png"
        picture = draw_frame(frame, pred[i], dt, f"{name} frame {index}  t {frame['t']:.1f} s",
                             [f"steer error {err[i, :, 0].mean():.3f}: label {label[i, 0, 0]:+.2f}, "
                              f"model {pred[i, 0, 0]:+.2f} now"])
        cv2.imwrite(str(path), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
        log(f"  {rank}. {name} frame {index:>3}: steer label {label[i, 0, 0]:+.2f} model {pred[i, 0, 0]:+.2f}, "
            f"mean error {err[i, :, 0].mean():.3f}   {path.name}")
        worst_frames.append({"episode": name, "frame": index, "steer_error": float(err[i, :, 0].mean())})
    log(f"  see them in context: ./drive.sh browse --project {project.home} --match {names[int(validation.episode[order[0]])]}")
    return {"held_out": held, "frames": len(label), "overall": overall, "plan_dt": dt,
            "per_step_steer": err[..., 0].mean(axis=0).tolist(),
            "per_step_average": average[..., 0].mean(axis=0).tolist(), "per_step_keep": keep.mean(axis=0).tolist(),
            "roads": roads, "worst": worst_frames}


# --- ablation --------------------------------------------------------------------------

def ablation(project: Project, model, info: dict, log=print) -> dict:
    """How much worse the plan gets when one input is replaced by its average: what each input is worth."""
    validation, _ = held_out(project, info["image_hw"])
    images, states = validation.images, validation.states.astype(np.float32)
    label = validation.plans.astype(np.float64)

    def score(pictures, measured) -> tuple[float, float]:
        pred = predict(model, pictures, measured)
        return float(np.abs(pred[..., 0] - label[..., 0]).mean()), float(np.abs(pred[..., 1] - label[..., 1]).mean())

    average_picture = np.broadcast_to(images.mean(axis=0).round().astype(np.uint8), images.shape)
    cases = [("nothing (the model as it is)", images, states),
             ("the picture", average_picture, states),
             ("the speed", images, np.column_stack([np.full(len(states), states[:, 0].mean()), states[:, 1]])),
             ("the steering state", images, np.column_stack([states[:, 0], np.full(len(states), states[:, 1].mean())]))]
    log(f"each input in turn replaced by its average over the {len(label)} held-out frames; "
        "the worse the plan gets, the more the model relies on that input")
    log(f"  {'replaced':<30} {'steer error':>11} {'':>6} {'throttle error':>14}")
    result, base = {}, None
    for name, pictures, measured in cases:
        steer, throttle = score(pictures, measured.astype(np.float32))
        base = base or (steer, throttle)
        result[name] = {"steer": steer, "throttle": throttle}
        log(f"  {name:<30} {steer:>11.4f} {steer / base[0]:>5.1f}x {throttle:>14.4f} {throttle / base[1]:>5.1f}x")
    picture = result["the picture"]["steer"] / base[0]
    log(f"  the steering comes from the picture: without it the error is {picture:.0f}x"
        if picture > 3 else "  WEAK PICTURE: the steering barely needs the image; it leans on the state")
    return result


# --- on-policy -------------------------------------------------------------------------

#: Bins of the distance from the centerline, in metres; the road's edge is at 0.45.
OFFSETS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.45), (0.45, 1e9))


def onpolicy(project: Project, path: Path, model, info: dict, duration_s: float = 40.0, log=print) -> dict:
    """Let the model drive every track both ways and ask the teacher, at every tick, what it would steer.

    Offline errors are measured where the teacher drove; this measures them where the model drives,
    which is where its own mistakes take it (covariate shift, DAgger's reason to exist).
    """
    from ..episode import EpisodeConfig, build
    from ..policy.pure_pursuit import PurePursuitPolicy
    from ..sim import SHAPES
    validation, _ = held_out(project, info["image_hw"])
    offline = float(np.abs(predict(model, validation.images, validation.states)[:, 0, 0]
                           - validation.plans[:, 0, 0]).mean())
    errors_, offsets, drives = [], [], []
    for track in SHAPES:
        for reverse in (False, True):
            world, loop = build(EpisodeConfig(policy="cnn", checkpoint=str(path), track=track, reverse=reverse,
                                              duration_s=duration_s))
            teacher = PurePursuitPolicy(world.track, world.vehicle)
            mine = []
            try:
                while world.t < duration_s - 1e-9:
                    wanted = teacher.act(world.state)[0]
                    lateral = abs(world.track.project(world.state.x, world.state.y)[1])
                    tick = loop.step()
                    if tick.command is not None:
                        mine.append(abs(tick.command[0] - wanted))
                        offsets.append(lateral)
            finally:
                loop.worker.close()
            errors_ += mine
            drives.append((f"{track}{' reversed' if reverse else ''}", float(np.mean(mine))))
    errors_, offsets = np.array(errors_), np.array(offsets)
    mean = float(errors_.mean())
    log(f"{len(drives)} drives of {duration_s:.0f} s with the model at the wheel; at every tick (50 a second) the "
        "teacher, who knows the map, says what it would steer from where the car is")
    log("  " + "   ".join(f"{name} {e:.3f}" for name, e in drives))
    log(f"  mean |model steer - teacher steer| on its own drives: {mean:.4f}; on the recorded held-out frames "
        f"(step 0): {offline:.4f}, so {mean / offline:.1f}x")
    log("by distance from the centerline (the road's edge is at 45 cm)")
    log(f"  {'off-centre':<12} {'time':>6} {'steer error':>11}")
    bins = []
    for low, high in OFFSETS:
        pick = (offsets >= low) & (offsets < high)
        label = f"{100 * low:.0f}-{100 * high:.0f} cm" if high < 1 else "off the road"
        share = float(pick.mean())
        error = float(errors_[pick].mean()) if pick.any() else None
        bins.append({"bin": label, "share": share, "steer_error": error})
        log(f"  {label:<12} {100 * share:>5.1f}% " + (f"{error:>11.4f}" if error is not None else f"{'-':>11}"))
    verdicts = []
    if mean > 2 * offline:
        verdicts.append(f"COMPOUNDING: on its own path the model is {mean / offline:.1f}x worse than on recorded frames. "
                        "Its mistakes take it where the data never went (covariate shift). Record where it goes and "
                        "let the teacher label it: ./drive.sh collect --driver cnn --checkpoint …  (DAgger)")
    far = [b for b in bins[2:] if b["steer_error"] is not None and b["share"] > 0.01]
    if far and bins[0]["steer_error"] and max(b["steer_error"] for b in far) > 3 * bins[0]["steer_error"]:
        verdicts.append("LOST WHEN OFF-CENTRE: the error grows with the distance from the centerline, so it does not "
                        "know how to come back")
    if not verdicts:
        verdicts.append("CONSISTENT: on its own path the model is about as accurate as on the recorded frames")
    for v in verdicts:
        log("  " + v)
    return {"mean": mean, "offline": offline, "drives": drives, "bins": bins, "verdicts": verdicts}


# --- activations -----------------------------------------------------------------------

def activations(project: Project, model, info: dict, frame: dict, out: Path, log=print) -> dict:
    """What each layer outputs: on one frame, and which units stay silent on 512 held-out frames."""
    validation, _ = held_out(project, info["image_hw"])
    take = np.linspace(0, len(validation) - 1, min(512, len(validation))).round().astype(int)
    many = outputs(model, validation.images[take], validation.states[take])
    one = outputs(model, frame["small"][None], frame["state"][None])
    log(f"frame {frame['name']}:{frame['index']} (for the picture), and {len(take)} held-out frames (for 'never active')")
    log(f"  {'layer':<9} {'what it is':<26} {'shape (one frame)':<18} {'mean':>6} {'zero %':>7}   never active on {len(take)} frames")
    rows = []
    for name, module in named_layers(model):
        kind = describe(module)
        a1, a = one[name][0], many[name]
        if kind == "ReLU":
            # a channel (conv) or unit (linear) that is 0 for every frame and every position is dead
            axes = (0, 2, 3) if a.ndim == 4 else (0,)
            dead = int((a.max(axis=axes) <= 0).sum())
            units = a.shape[1]
            what = f"{dead} of {units} {'channels' if a.ndim == 4 else 'units'}"
            rows.append({"layer": name, "mean": float(a1.mean()), "zero_pct": float(100 * (a1 <= 0).mean()),
                         "dead": dead, "units": units})
            log(f"  {name:<9} {'ReLU after ' + previous:<26} {shape(a1.shape):<18} {a1.mean():>6.3f} "
                f"{100 * (a1 <= 0).mean():>6.0f}%   {what}")
        elif kind == "Tanh":
            saturated = float(100 * (np.abs(a) > 0.99).mean())
            rows.append({"layer": name, "saturated_pct": saturated})
            log(f"  {name:<9} {'Tanh: the plan':<26} {shape(a1.shape):<18} {a1.mean():>6.3f} {'':>7}   "
                f"range {a.min():+.2f}..{a.max():+.2f}, saturated (|x|>0.99) {saturated:.1f} %")
        previous = kind.split(",")[0]

    # The picture: the frame the network saw, then the 8 most active channels of each convolution.
    h, w = model.image_hw
    tiles = [cv2.resize(cv2.cvtColor(frame["small"], cv2.COLOR_RGB2BGR), (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)]
    for name in ("vision.1", "vision.3", "vision.5", "vision.7", "vision.9"):
        maps = one[name][0]
        top = np.argsort(-maps.mean(axis=(1, 2)))[:8]
        peak = max(float(maps[top].max()), 1e-6)
        row = [cv2.applyColorMap(cv2.resize((255 * maps[c] / peak).astype(np.uint8), (w // 2, h // 2),
                                            interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_VIRIDIS) for c in top]
        strip = np.hstack(row + [np.zeros((h // 2, w // 2, 3), np.uint8)] * (8 - len(row)))
        label = np.zeros((h // 2, 150, 3), np.uint8)
        cv2.putText(label, name, (8, h // 4 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(label, f"{maps.shape[1]}x{maps.shape[2]}", (8, h // 4 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    WHITE, 1, cv2.LINE_AA)
        tiles.append(np.hstack([label, strip]))
    width = max(t.shape[1] for t in tiles)
    picture = np.vstack([np.pad(t, ((0, 4), (0, width - t.shape[1]), (0, 0))) for t in tiles])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"activations-{frame['name']}-{frame['index']:04d}.png"
    cv2.imwrite(str(path), picture)
    log(f"  feature maps (top: the input; then per layer the 8 most active channels, bright = strong): {path}")
    return {"rows": rows, "picture": str(path)}


# --- saliency --------------------------------------------------------------------------

def saliency(model, info: dict, frame: dict, out: Path, patch: int = 16, stride: int = 8, log=print) -> dict:
    """Cover the image patch by patch with grey; where covering moves the steering most is what it looks at."""
    small, state = frame["small"], frame["state"]
    h, w = small.shape[:2]
    base = predict(model, small[None], state[None])[0]
    corners = [(y, x) for y in range(0, h - patch + 1, stride) for x in range(0, w - patch + 1, stride)]
    covered = np.repeat(small[None], len(corners), axis=0)
    for n, (y, x) in enumerate(corners):
        covered[n, y:y + patch, x:x + patch] = 128
    moved = np.abs(predict(model, covered, np.repeat(state[None], len(corners), axis=0))[..., 0] - base[:, 0]).mean(axis=1)
    heat, count = np.zeros((h, w)), np.zeros((h, w))
    for (y, x), m in zip(corners, moved):
        heat[y:y + patch, x:x + patch] += m
        count[y:y + patch, x:x + patch] += 1
    heat /= np.maximum(count, 1)
    share = heat / max(heat.sum(), 1e-12)
    thirds = [float(100 * share[i * h // 3:(i + 1) * h // 3].sum()) for i in range(3)]
    halves = [float(100 * share[:, :w // 2].sum()), float(100 * share[:, w // 2:].sum())]
    log(f"frame {frame['name']}:{frame['index']}: model steers {base[0, 0]:+.3f} now"
        + (f", the label {frame['plan'][0, 0]:+.3f}" if frame["plan"] is not None else "")
        + f"; covering one {patch}x{patch} px patch moves the steering by up to {moved.max():.3f} (mean over the plan)")
    log(f"  where the effect is: top third {thirds[0]:.0f} %, middle {thirds[1]:.0f} %, bottom {thirds[2]:.0f} %;"
        f" left half {halves[0]:.0f} %, right half {halves[1]:.0f} %")
    image = frame["image"]
    colour = cv2.applyColorMap((255 * heat / max(heat.max(), 1e-12)).astype(np.uint8), cv2.COLORMAP_JET)
    colour = cv2.resize(colour, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    blend = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.55, colour, 0.45, 0)
    picture = np.hstack([cv2.cvtColor(image, cv2.COLOR_RGB2BGR), blend])
    picture = cv2.resize(picture, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cv2.putText(picture, "red: covering this changes the steering most", (12, picture.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"saliency-{frame['name']}-{frame['index']:04d}.png"
    cv2.imwrite(str(path), picture)
    log(f"  heat map: {path}")
    return {"steer": float(base[0, 0]), "max_effect": float(moved.max()), "thirds_pct": thirds, "halves_pct": halves,
            "picture": str(path)}


# --- behaviour -------------------------------------------------------------------------

def behaviour(project: Project, model, info: dict, frame: dict, log=print) -> dict:
    """Tests a driver should pass, whatever its accuracy."""
    validation, _ = held_out(project, info["image_hw"])
    take = np.linspace(0, len(validation) - 1, min(512, len(validation))).round().astype(int)
    images, states = validation.images[take], validation.states[take].astype(np.float32)
    plan = predict(model, images, states)
    size = float(np.abs(plan[..., 0]).mean())
    result = {}

    mirrored = predict(model, images[:, :, ::-1], states * np.array([1, -1], np.float32))
    asymmetry = float(np.abs(plan[..., 0] + mirrored[..., 0]).mean())
    result["mirror"] = asymmetry / size
    log(f"mirror: the image flipped left-right and the steering state negated should negate the steering")
    log(f"  mean |steer + mirrored steer| = {asymmetry:.4f}, {100 * asymmetry / size:.0f} % of the mean |steer| "
        f"({size:.3f})   {'ok' if asymmetry < 0.25 * size else 'ASYMMETRIC: it turns one way better than the other'}")

    log("brightness: the same frames darker and lighter (training saw 0.8x to 1.2x)")
    result["brightness"] = {}
    for gain in (0.5, 0.8, 1.2, 1.5):
        changed = predict(model, np.clip(images.astype(np.float32) * gain, 0, 255).astype(np.uint8), states)
        moved = float(np.abs(changed[..., 0] - plan[..., 0]).mean())
        result["brightness"][gain] = moved / size
        log(f"  x{gain:<4} steering moves by {moved:.4f} ({100 * moved / size:3.0f} % of its size)")

    log("blank: nothing to see, at 1.5 m/s going straight. A cautious driver would slow down; this one:")
    h, w = model.image_hw
    for name, value in (("black", 0), ("grey", 128), ("white", 255)):
        p = predict(model, np.full((1, h, w, 3), value, np.uint8), np.array([[1.5, 0.0]], np.float32))[0]
        result[f"blank_{name}"] = {"steer": float(p[:, 0].mean()), "throttle": float(p[:, 1].mean())}
        log(f"  {name:<5} steer {p[:, 0].mean():+.2f}, throttle {p[:, 1].mean():+.2f} (mean over the plan)")

    from ..config import VehicleParams
    car = VehicleParams()
    hold = car.drag_per_s / car.max_accel_mps2          # throttle per m/s that cancels the drag
    small, state = frame["small"], frame["state"]
    log(f"speed: frame {frame['name']}:{frame['index']} (the car went {state[0]:.2f} m/s), told other speeds. "
        f"Holding the speed it is told takes {hold:+.2f} throttle per m/s, to cancel the drag; driving towards "
        "one target speed takes less throttle the faster it goes")
    speeds = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    p = predict(model, np.repeat(small[None], len(speeds), axis=0),
                np.array([[s, state[1]] for s in speeds], np.float32))
    slope = float(np.polyfit(speeds, p[:, 0, 1], 1)[0])
    result["speed"] = {s: float(p[i, 0, 1]) for i, s in enumerate(speeds)}
    result["throttle_per_mps"] = slope
    log("  throttle now: " + "   ".join(f"{s:.1f} m/s {p[i, 0, 1]:+.2f}" for i, s in enumerate(speeds)))
    if abs(slope) < 0.25 * hold:
        verdict = "IGNORES THE SPEED: the same throttle whatever it is told"
    elif 0.5 * hold <= slope <= 1.5 * hold:
        verdict = ("HOLDS the speed it is told, whatever it is: what a teacher with another target speed every "
                   "drive teaches. From a standstill it will barely move")
    elif slope < -1.0 * hold:
        verdict = "DRIVES TOWARDS one speed: more throttle when slow, less when fast"
    else:
        verdict = "in between: neither holds the speed nor drives towards one"
    log(f"  {slope:+.3f} per m/s: {verdict}")

    log(f"steering state: the same frame, told the wheels are at other angles. A model that copies the "
        "state instead of reading the road follows it")
    steers = (-0.5, 0.0, 0.5)
    p = predict(model, np.repeat(small[None], 3, axis=0), np.array([[state[0], s] for s in steers], np.float32))
    follow = float((p[2, 0, 0] - p[0, 0, 0]) / (steers[2] - steers[0]))
    result["copies_state"] = follow
    log("  " + "   ".join(f"state {s:+.1f}: steers {p[i, 0, 0]:+.2f}" for i, s in enumerate(steers))
        + f"   (moves {follow:.2f} per 1.0 of state)")
    log(f"  {'mostly the road' if abs(follow) < 0.5 else 'COPIES THE STATE: it leans on the steering it has, not the picture'}")
    return result


# --- the HTML report -------------------------------------------------------------------

PALETTE = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")


def svg_lines(series: dict, title: str, xlabel: str, log_y: bool = False, width: int = 640, height: int = 260) -> str:
    """A line chart as inline SVG: {name: (xs, ys)}, no plotting library needed."""
    import math
    left, right, top, bottom = 64, 150, 30, 40
    xs = [x for xs_, _ in series.values() for x in xs_]
    ys = [y for _, ys_ in series.values() for y in ys_ if y is not None and (y > 0 or not log_y)]
    if not xs or not ys:
        return ""
    f = (lambda v: math.log10(v)) if log_y else (lambda v: v)
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    y0, y1 = f(min(ys)), f(max(ys))
    if y1 - y0 < 1e-12:
        y0, y1 = y0 - 1, y1 + 1
    if not log_y:
        y0 = min(0.0, y0)

    def px(x, y):
        return (left + (x - x0) / (x1 - x0) * (width - left - right),
                top + (1 - (f(y) - y0) / (y1 - y0)) * (height - top - bottom))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" '
             f'font-size="11"><text x="{left}" y="18" font-size="13" font-weight="bold">{title}</text>',
             f'<rect x="{left}" y="{top}" width="{width - left - right}" height="{height - top - bottom}" '
             'fill="none" stroke="#bbb"/>']
    ticks = [y0 + (y1 - y0) * k / 4 for k in range(5)]
    for t in ticks:
        value = 10 ** t if log_y else t
        y = top + (1 - (t - y0) / (y1 - y0)) * (height - top - bottom)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eee"/>'
                     f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{value:.3g}</text>')
    for k in range(5):
        x = x0 + (x1 - x0) * k / 4
        parts.append(f'<text x="{px(x, 10 ** y0 if log_y else y0)[0]:.1f}" y="{height - bottom + 16}" '
                     f'text-anchor="middle">{x:.3g}</text>')
    parts.append(f'<text x="{(left + width - right) / 2}" y="{height - 6}" text-anchor="middle">{xlabel}</text>')
    for k, (name, (xs_, ys_)) in enumerate(series.items()):
        colour = PALETTE[k % len(PALETTE)]
        points = " ".join("%.1f,%.1f" % px(x, y) for x, y in zip(xs_, ys_) if y is not None and (y > 0 or not log_y))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2"/>')
        parts.append(f'<rect x="{width - right + 12}" y="{top + 16 * k}" width="12" height="3" fill="{colour}"/>'
                     f'<text x="{width - right + 28}" y="{top + 16 * k + 5}">{name}</text>')
    return "".join(parts) + "</svg>"


def html_report(project: Project, path: Path, model, info: dict, out: Path, frame_ref: str | None = None,
                worst: int = 5, duration_s: float = 40.0) -> Path:
    """Every probe on one page: the text each prints, with charts and the pictures they draw."""
    import html
    import io
    import re
    from contextlib import redirect_stdout
    out.mkdir(parents=True, exist_ok=True)
    sections, verdicts = [], []

    def run_probe(title: str, call, chart: str = "", pictures=()):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            try:
                result = call(print)
            except ValueError as error:
                print(f"(skipped: {error})")
                result = None
        text = buffer.getvalue()
        # a verdict is a line with a word in capitals: HEALTHY, STEPS TOO BIG, DRIVES TOWARDS, ...
        verdicts.extend(f"{title}: {line.strip()}" for line in text.splitlines()
                        if re.search(r"\b[A-Z]{4,}(?: [A-Z]{2,})*\b", line))
        images = "".join(f'<img src="{Path(p).name}" alt="{html.escape(Path(p).name)}">' for p in pictures if p)
        sections.append(f"<section><h2>{html.escape(title)}</h2>{chart}<pre>{html.escape(text)}</pre>{images}</section>")
        return result

    report = training_report(project, path, info)
    h = report["history"]
    epochs = [e["epoch"] for e in h]
    curves = {"train loss": (epochs, [e["train_loss"] for e in h]), "validation loss": (epochs, [e["val_loss"] for e in h])}
    if report.get("average_plan_val_loss"):
        curves["average plan"] = (epochs, [report["average_plan_val_loss"]] * len(epochs))
    run_probe("Layers", lambda log: layers(model, info, log))
    run_probe("Training curves", lambda log: history(report, log), svg_lines(curves, "loss per epoch", "epoch", log_y=True))
    if "grad_norm" in h[0]:
        ratios = {name: (epochs, [e["update_ratio"][name] for e in h]) for name in list(h[0]["update_ratio"])[::3]}
        ratios["healthy 1e-3"] = (epochs, [HEALTHY_UPDATE] * len(epochs))
        run_probe("Inside training", lambda log: training(report, log),
                  svg_lines(ratios, "update / weight per step", "epoch", log_y=True))
    found = run_probe("Errors on held-out episodes", lambda log: errors(project, model, info, out, worst, log))
    if found:
        steps = [k * found["plan_dt"] for k in range(len(found["per_step_steer"]))]
        sections[-1] = sections[-1].replace("<pre>", svg_lines(
            {"model": (steps, found["per_step_steer"]), "keep": (steps, found["per_step_keep"]),
             "average": (steps, found["per_step_average"])}, "steer error per step of the plan", "seconds ahead") + "<pre>", 1)
        sections[-1] = sections[-1].replace("</section>", "".join(
            f'<img src="worst-{k}-{w["episode"]}-{w["frame"]:04d}.png" alt="worst {k}">'
            for k, w in enumerate(found["worst"][:3], 1)) + "</section>")
    run_probe("What each input is worth", lambda log: ablation(project, model, info, log))
    frame = pick_frame(project, frame_ref, info["image_hw"])
    run_probe("Behaviour", lambda log: behaviour(project, model, info, frame, log))
    seen = run_probe("Activations", lambda log: activations(project, model, info, frame, out, log))
    run_probe("Saliency", lambda log: saliency(model, info, frame, out, log=log),
              pictures=[out / f"saliency-{frame['name']}-{frame['index']:04d}.png"])
    if seen:
        sections[-2] = sections[-2].replace("</section>", f'<img src="{Path(seen["picture"]).name}" alt="activations"></section>')
    driven = run_probe("On its own drives", lambda log: onpolicy(project, path, model, info, duration_s, log=log))
    if driven:
        middle = [i for i, b in enumerate(driven["bins"]) if b["steer_error"] is not None]
        sections[-1] = sections[-1].replace("<pre>", svg_lines(
            {"steer error": (middle, [driven["bins"][i]["steer_error"] for i in middle])},
            "steer error against distance off-centre (bins 0: <2 cm ... 5: off the road)", "bin") + "<pre>", 1)

    summary = "".join(f"<li>{html.escape(v)}</li>" for v in verdicts)
    page = out / "report.html"
    page.write_text(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(path.stem)}: probe report</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:2em auto;color:#222}}pre{{background:#f6f6f6;
padding:10px;overflow-x:auto;font-size:12px}}img{{max-width:100%;margin:6px 0;display:block}}section{{margin-bottom:2em}}
h2{{border-bottom:1px solid #ddd}}li{{margin:4px 0}}</style></head>
<body><h1>{html.escape(path.stem)}</h1><p>{html.escape(str(path))}, project {html.escape(str(project.home))}.
Every section is the output of <code>./drive.sh probe {html.escape(path.stem)} &lt;what&gt;</code>.</p>
<h2>Verdicts</h2><ul>{summary}</ul>{''.join(sections)}</body></html>
""")
    return page


# --- the stage -------------------------------------------------------------------------

def add_parser(sub):
    p = sub.add_parser("probe", help="look inside a trained CNN: layers, training curves, errors, activations, "
                                     "saliency, behaviour")
    add_project_argument(p)
    p.add_argument("model", help="a trained version (v1) or a checkpoint file")
    p.add_argument("what", choices=WHAT, help="what to look at")
    p.add_argument("--frame", metavar="EPISODE:INDEX",
                   help="the frame for activations, saliency and behaviour, such as sim-038:120 "
                        "(default: the sharpest bend in the first held-out episode)")
    p.add_argument("--worst", type=int, default=5, help="errors: how many of the worst frames to list and draw")
    p.add_argument("--out", type=Path, help="where pictures go (default <project>/probe/<model>)")
    p.add_argument("--duration-s", type=float, default=40.0, help="onpolicy and report: seconds per drive (default 40)")


def run(project: Project, args) -> int:
    try:
        import torch.nn  # noqa: F401
    except ImportError as error:
        raise ValueError(f"probe needs PyTorch ({error}); make setup, or use the container: make shell") from error
    path, model, info = load_model(project, args.model)
    out = args.out or project.home / "probe" / path.stem
    if args.what == "layers":
        layers(model, info)
    elif args.what in ("history", "training"):
        report = training_report(project, path, info)
        history(report) if args.what == "history" else training(report)
    elif args.what == "errors":
        errors(project, model, info, out, args.worst)
    elif args.what == "ablation":
        ablation(project, model, info)
    elif args.what == "onpolicy":
        onpolicy(project, path, model, info, args.duration_s)
    elif args.what == "report":
        page = html_report(project, path, model, info, out, args.frame, args.worst, args.duration_s)
        print(f"report: {page}")
    else:
        frame = pick_frame(project, args.frame, info["image_hw"])
        if args.what == "activations":
            activations(project, model, info, frame, out)
        elif args.what == "saliency":
            saliency(model, info, frame, out)
        else:
            behaviour(project, model, info, frame)
    return 0


def training_report(project: Project, path: Path, info: dict) -> dict:
    """The training report beside the checkpoint, with the average plan's loss worked out if it predates it."""
    report = path.with_suffix(".json")
    report = json.loads(report.read_text()) if report.is_file() else info["report"]
    if "average_plan_val_loss" not in report:            # trained before train.py logged it
        from .train import average_plan_loss
        try:
            report = {**report, "average_plan_val_loss": round(average_plan_loss(*datasets.split_plans(project.episodes)), 5)}
        except ValueError:
            pass                                          # no episodes here: judge the curves alone
    return report
