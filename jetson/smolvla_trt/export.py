"""Export the fine-tuned SmolVLA (smolvla-v1) for vla_probe, the C++ TensorRT runtime beside this file.

Run it on the PC with the Python of the LeRobot environment (docs/study/15-hands-on-labs.md,
Lab 15), from the repository root, with PYTHONPATH=. and HF_HOME set:

    $V jetson/smolvla_trt/export.py [--checkpoint smolvla-v1] [--out runs/drive/vla/smolvla-v1-trt]

The network is cut where the work changes, so each part is one static-shaped graph:

    vision.onnx   pixel_values [1,3,512,512] in [-1,1]            -> image_tokens [1,64,960]
                  SigLIP (12 layers) and the connector that turns 1024 patches into 64 tokens
    prefix.onnx   image_tokens, lang_ids [1,48] int32, lang_mask [1,48] int32,
                  state [1,32] (normalised, zero-padded)           -> kv_cache [16,2,1,113,5,64]
                  the 16 language-model layers over 64 image + 48 text + 1 state tokens, run
                  once per observation; only their keys and values are kept
    expert.onnx   x_t [1,16,32], time_emb [1,720], kv_cache,
                  prefix_mask [1,113] int32                       -> v_t [1,16,32]
                  one denoising step of the action expert; vla_probe runs it 10 times

Everything around the graphs is left to C++, in the open: tokenising the task, resizing and
padding the image, normalising the state, the time embedding, the Euler loop, unnormalising.

It also writes what vla_probe needs, and what it checks itself against:

    meta.txt      sizes and normalisation statistics
    tokenizer/    vocab.txt, merges.txt, added.txt, and tests.txt: strings with the ids Python gives them
    samples/      frames of held-out episodes, each with the noise used and what Python computed:
                  the model input, the image tokens, and the chunk in fp32 and in the bf16 LeRobot runs

Three edits make the graphs exportable and safe in FP16; none changes the result in fp32:
the attention mask's -3.4e38 becomes -1e4 (in FP16 it would be -inf, and a fully masked row,
a padding token's, would turn into NaN and spread), RoPE is written without in-place slices,
and the vision patches' position ids, always 0..1023 for a 512x512 image, become a constant.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT_TASK = "drive along the track"
TOKENIZER_TESTS = [
    ROOT_TASK, "drive along the track\n", "turn left at the next corner\n", "stop\n", "STOP NOW!!\n",
    "drive at 1.5 m/s, then 0.75\n", "  two  spaces   and\ttabs\n", "it's the car's lane, we'll see\n",
    "lái xe theo làn đường\n", "đi chậm lại 20%\n", "émoji 🚗 ok\n", "<|im_start|>user\n", "a\n\nb\n",
    "x" * 80 + "\n", "",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", type=Path, default=Path("runs/drive"))
    p.add_argument("--checkpoint", default="smolvla-v1", help="a version in <project>/vla or a pretrained_model dir")
    p.add_argument("--out", type=Path, help="default <project>/vla/<checkpoint>-trt")
    p.add_argument("--episode", default="sim2-031", help="a held-out episode to take sample frames from")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------
# Edits that make the graphs exportable. Each is checked against the original below.

def rope_without_slices(x, positions, max_wavelength=10_000):
    """smolvlm_with_expert.apply_rope, with torch.cat instead of writes into slices of an empty tensor."""
    import torch
    d_half = x.shape[-1] // 2
    dtype = x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=x.device)
    timescale = max_wavelength ** freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


MASKED = -1.0e4   # instead of finfo(float32).min: finite in FP16, and exp(-1e4) is still exactly 0


def attention_with_finite_mask(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
    """SmolVLMWithExpertModel.eager_attention_forward with MASKED for masked scores."""
    import torch
    groups = self.num_attention_heads // self.num_key_value_heads
    length = key_states.shape[1]
    key_states = key_states[:, :, :, None, :].expand(batch_size, length, self.num_key_value_heads, groups, head_dim)
    key_states = key_states.reshape(batch_size, length, self.num_key_value_heads * groups, head_dim)
    value_states = value_states[:, :, :, None, :].expand(batch_size, length, self.num_key_value_heads, groups, head_dim)
    value_states = value_states.reshape(batch_size, length, self.num_key_value_heads * groups, head_dim)
    query_states = query_states.to(torch.float32).transpose(1, 2)
    key_states = key_states.to(torch.float32).transpose(1, 2)
    weights = torch.matmul(query_states, key_states.transpose(2, 3)) * head_dim ** -0.5
    weights = torch.where(attention_mask[:, None, :, :], weights, torch.tensor(MASKED, dtype=weights.dtype))
    probs = torch.softmax(weights, dim=-1).to(value_states.dtype)
    out = torch.matmul(probs, value_states.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    return out.reshape(batch_size, -1, self.num_key_value_heads * groups * head_dim)


def rmsnorm_fp16_safe(self, hidden_states):
    """LlamaRMSNorm, scaled by max|x| before squaring.

    The residual stream of smolvla-v1 reaches |x| = 2600 in the last layers, and 2600^2 is
    far past FP16's 65504: the plain RMSNorm turns into inf in a TensorRT FP16 engine. With
    s = max|x|,  x * rsqrt(mean(x^2) + eps)  =  (x/s) * rsqrt(mean((x/s)^2) + eps/s^2),
    and every term on the right stays within FP16.
    """
    import torch
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    y = x / scale
    variance = y.pow(2).mean(-1, keepdim=True)
    y = y * torch.rsqrt(variance + self.variance_epsilon / scale / scale)
    return self.weight * y.to(input_dtype)


def patch_lerobot(model):
    from lerobot.policies.smolvla import smolvlm_with_expert
    smolvlm_with_expert.apply_rope = rope_without_slices
    smolvlm_with_expert.SmolVLMWithExpertModel.eager_attention_forward = attention_with_finite_mask
    norms = {type(m) for m in model.modules() if type(m).__name__.endswith("RMSNorm")}
    for cls in norms:
        cls.forward = rmsnorm_fp16_safe
    return sorted(cls.__name__ for cls in norms)


# ---------------------------------------------------------------------------------------------
# The three graphs, written with the policy's own modules.

def graphs(model):
    """(Vision, Prefix, Expert) modules around a VLAFlowMatching `model`."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    vwe = model.vlm_with_expert
    vlm = vwe.get_vlm_model()
    config = model.config
    layers = vwe.num_vlm_layers
    chunk = config.chunk_size

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            vision = vlm.vision_model
            self.patch, self.position = vision.embeddings.patch_embedding, vision.embeddings.position_embedding
            self.encoder, self.post_layernorm, self.connector = vision.encoder, vision.post_layernorm, vlm.connector
            self.register_buffer("position_ids", torch.arange(vision.embeddings.num_positions)[None], persistent=False)

        def forward(self, pixel_values):
            x = self.patch(pixel_values).flatten(2).transpose(1, 2) + self.position(self.position_ids)
            x = self.encoder(inputs_embeds=x, attention_mask=None).last_hidden_state
            return self.connector(self.post_layernorm(x))

    class Prefix(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model           # registered, so the ONNX node names carry the module path

        def forward(self, image_tokens, lang_ids, lang_mask, state):
            hidden = image_tokens.shape[-1]
            images = image_tokens * math.sqrt(hidden)
            text = vwe.embed_language_tokens(lang_ids.long()) * math.sqrt(hidden)
            embs = torch.cat([images, text, model.state_proj(state)[:, None, :]], dim=1)
            n_img = image_tokens.shape[1]
            ones = torch.ones(1, n_img, dtype=torch.bool, device=embs.device)
            pad = torch.cat([ones, lang_mask.bool(), ones[:, :1]], dim=1)                 # [1, 113]
            # make_att_2d_masks with att_masks = 0 for image and text, 1 for the state: image and
            # text tokens see each other, the state token sees them all, nothing sees the state.
            n = pad.shape[1]
            block = torch.zeros(n, dtype=torch.int64, device=embs.device)
            block[-1] = 1
            att_2d = (block[None, :] <= block[:, None])[None] & pad[:, None, :] & pad[:, :, None]
            positions = torch.cumsum(pad.float(), dim=1) - 1
            _, cache = vwe.forward(attention_mask=att_2d, position_ids=positions, past_key_values=None,
                                   inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True)
            return torch.stack([torch.stack([cache[i]["key_states"], cache[i]["value_states"]]) for i in range(layers)])

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x_t, time_emb, kv_cache, prefix_mask):
            actions = model.action_in_proj(x_t)
            fused = torch.cat([actions, time_emb[:, None, :].expand_as(actions)], dim=2)
            suffix = model.action_time_mlp_out(F.silu(model.action_time_mlp_in(fused)))
            pad = prefix_mask.bool()
            n_prefix = pad.shape[1]
            causal = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=x_t.device))[None]
            mask = torch.cat([pad[:, None, :].expand(1, chunk, n_prefix), causal], dim=2)
            positions = pad.float().sum(dim=1, keepdim=True) + torch.arange(chunk, device=x_t.device)[None].float()
            past = {i: {"key_states": kv_cache[i, 0], "value_states": kv_cache[i, 1]} for i in range(layers)}
            out, _ = vwe.forward(attention_mask=mask, position_ids=positions, past_key_values=past,
                                 inputs_embeds=[None, suffix], use_cache=True, fill_kv_cache=False)
            return model.action_out_proj(out[1][:, -chunk:].to(torch.float32))

    return Vision().eval(), Prefix().eval(), Expert().eval()


def time_embedding(t: float, dim: int, min_period: float, max_period: float) -> np.ndarray:
    """create_sinusoidal_pos_embedding for one time, in float64 as LeRobot computes it on the GPU."""
    fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = min_period * (max_period / min_period) ** fraction
    angle = (2 * math.pi / period) * t
    return np.concatenate([np.sin(angle), np.cos(angle)]).astype(np.float32)[None]


def run_split(parts, pixel_values, lang_ids, lang_mask, state, noise, config, hidden):
    """The chunk (normalised, padded) from the three graphs and an Euler loop: what vla_probe does."""
    import torch
    vision, prefix, expert = parts
    with torch.no_grad():
        image_tokens = vision(pixel_values)
        kv = prefix(image_tokens, lang_ids, lang_mask, state)
        n_img = image_tokens.shape[1]
        prefix_mask = torch.cat([torch.ones(1, n_img, dtype=torch.int32, device=kv.device), lang_mask,
                                 torch.ones(1, 1, dtype=torch.int32, device=kv.device)], dim=1)
        x, dt = noise, -1.0 / config.num_steps
        for step in range(config.num_steps):
            t = 1.0 + step * dt
            emb = torch.from_numpy(time_embedding(t, hidden, config.min_period, config.max_period)).to(kv.device)
            x = x + dt * expert(x, emb, kv, prefix_mask)
    return image_tokens, kv, prefix_mask, x


# ---------------------------------------------------------------------------------------------

def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def write_tokenizer(tokenizer, tokenizer_json: Path, out: Path) -> None:
    data = json.loads(tokenizer_json.read_text())
    model = data["model"]
    assert model["type"] == "BPE" and not model.get("ignore_merges") and not model.get("byte_fallback")
    assert data["normalizer"] is None and data["post_processor"] is None
    pre = data["pre_tokenizer"]
    assert [p["type"] for p in pre["pretokenizers"]] == ["Digits", "ByteLevel"], pre
    assert pre["pretokenizers"][0]["individual_digits"] and not pre["pretokenizers"][1]["add_prefix_space"]
    out.mkdir(parents=True, exist_ok=True)
    vocab = sorted(model["vocab"].items(), key=lambda kv: kv[1])
    assert [i for _, i in vocab] == list(range(len(vocab)))
    with open(out / "vocab.txt", "w", encoding="utf-8") as f:
        for token, _ in vocab:
            assert "\n" not in token and "\r" not in token
            f.write(token + "\n")
    with open(out / "merges.txt", "w", encoding="utf-8") as f:
        for merge in model["merges"]:
            a, b = merge if isinstance(merge, list) else merge.split(" ")
            f.write(f"{a} {b}\n")
    with open(out / "added.txt", "w", encoding="utf-8") as f:
        for token in data["added_tokens"]:
            assert not (token["lstrip"] or token["rstrip"] or token["single_word"])
            f.write(f"{token['id']}\t{escape(token['content'])}\n")
    with open(out / "tests.txt", "w", encoding="utf-8") as f:
        for text in TOKENIZER_TESTS:
            ids = tokenizer(text)["input_ids"]
            f.write(f"{escape(text)}\t{' '.join(map(str, ids))}\n")


def write_meta(path: Path, config, stats_pre, stats_post, hidden, n_img, layers, kv_shape, chunk_dt, pad_id) -> None:
    def vec(t):
        return " ".join(f"{float(v):.9g}" for v in np.asarray(t).reshape(-1))
    lines = {
        "model": "smolvla",
        "vlm": config.vlm_model_name,
        "chunk_size": config.chunk_size,
        "chunk_dt_s": chunk_dt,
        "num_steps": config.num_steps,
        "max_state_dim": config.max_state_dim,
        "max_action_dim": config.max_action_dim,
        "state_names": "speed_mps steer",
        "action_names": "steer throttle",
        "image_size": f"{config.resize_imgs_with_padding[0]} {config.resize_imgs_with_padding[1]}",
        "num_image_tokens": n_img,
        "image_token_grid": int(round(math.sqrt(n_img))),
        "max_lang_tokens": config.tokenizer_max_length,
        "pad_token_id": pad_id,
        "lm_hidden": hidden,
        "lm_layers": layers,
        "kv_shape": " ".join(map(str, kv_shape)),
        "expert_hidden": int(config_expert_hidden(config, hidden)),
        "min_period": config.min_period,
        "max_period": config.max_period,
        "norm_eps": 1e-8,
        "state_mean": vec(stats_pre["observation.state"]["mean"]),
        "state_std": vec(stats_pre["observation.state"]["std"]),
        "action_mean": vec(stats_post["action"]["mean"]),
        "action_std": vec(stats_post["action"]["std"]),
        "default_task": ROOT_TASK,
    }
    path.write_text("".join(f"{k} = {v}\n" for k, v in lines.items()))


def config_expert_hidden(config, hidden):
    return hidden * config.expert_width_multiplier


def processor_stats(pipeline):
    for step in pipeline.steps:
        if hasattr(step, "_tensor_stats") and step._tensor_stats:
            return {k: {s: v.detach().cpu().numpy() for s, v in d.items()} for k, d in step._tensor_stats.items()}
        if hasattr(step, "stats") and step.stats:
            return {k: {s: np.asarray(v) for s, v in d.items()} for k, d in step.stats.items()}
    raise RuntimeError("no normalisation statistics in the processor pipeline")


def tokenizer_of(pipeline):
    for step in pipeline.steps:
        if hasattr(step, "input_tokenizer"):
            return step.input_tokenizer
    raise RuntimeError("no tokenizer step in the preprocessor")


def prepared(backend, image, state, task):
    """The batch LeRobotBackend.infer builds, through the checkpoint's own preprocessor."""
    import torch
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    batch = backend.preprocess({backend.camera: tensor, "observation.state": torch.from_numpy(state), "task": task})
    policy = backend.policy
    images, _ = policy.prepare_images(batch)
    return (images[0], batch["observation.language.tokens"], batch["observation.language.attention_mask"],
            policy.prepare_state(batch))


def sample_frames(episode: Path):
    """Hardest left, hardest right and straightest frame of an episode, by the teacher's first action."""
    from drive.learn.label import read_labels
    data = np.load(episode / "episode.npz")
    labels = read_labels(episode)
    steer = labels["plans"][:, 0, 0]
    moving = data["states"][:, 0] > 0.8
    picks = {"left": int(np.argmax(np.where(moving, steer, -9))),
             "right": int(np.argmin(np.where(moving, steer, 9))),
             "straight": int(np.argmin(np.where(moving, np.abs(steer), 9)))}
    return [(name, i, data["images"][i], data["states"][i].astype(np.float32), labels["plans"][i])
            for name, i in picks.items()]


SCORES: list[float] = []      # largest |q.k / sqrt(d)| of each attention call, while activation_report runs


def activation_report(model, run) -> dict:
    """The largest magnitudes an FP16 engine must hold: norm inputs, every Linear's output, attention scores."""
    import torch
    from torch import nn
    worst = {"norm input": (0.0, ""), "linear output": (0.0, ""), "attention score": (0.0, "")}

    def note(kind, value, name):
        if value > worst[kind][0]:
            worst[kind] = (value, name)

    hooks = []
    for name, module in model.named_modules():
        if type(module).__name__.endswith("RMSNorm") or isinstance(module, nn.LayerNorm):
            hooks.append(module.register_forward_pre_hook(
                lambda mod, args, name=name: note("norm input", float(args[0].detach().abs().max()), name)))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(
                lambda mod, args, out, name=name: note("linear output", float(out.detach().abs().max()), name)))
    import lerobot.policies.smolvla.smolvlm_with_expert as swe
    original = swe.SmolVLMWithExpertModel.eager_attention_forward

    def recording(self, mask, batch_size, head_dim, q, k, v):
        groups = self.num_attention_heads // self.num_key_value_heads
        kk = k.float().repeat_interleave(groups, dim=2)
        scores = torch.einsum("blhd,bshd->bhls", q.float(), kk) * head_dim ** -0.5
        note("attention score", float(scores.masked_fill(~mask[:, None], 0).abs().max()), "")
        return original(self, mask, batch_size, head_dim, q, k, v)

    swe.SmolVLMWithExpertModel.eager_attention_forward = recording
    try:
        run()
    finally:
        swe.SmolVLMWithExpertModel.eager_attention_forward = original
        for h in hooks:
            h.remove()
    return worst


def main():
    args = parse_args()
    import torch
    from drive.learn.project import Project
    from drive.policy.lerobot_backend import LeRobotBackend, chunk_dt_s
    from drive.policy.vla import BackendObservation

    project = Project(args.project)
    checkpoint = project.checkpoint(args.checkpoint)
    out = args.out or project.home / "vla" / f"{Path(str(args.checkpoint)).name}-trt"
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    print(f"checkpoint {checkpoint}\nout        {out}")

    backend = LeRobotBackend(checkpoint)
    policy, model, config = backend.policy, backend.policy.model, backend.policy.config
    vwe = model.vlm_with_expert
    hidden = vwe.config.text_config.hidden_size
    tokenizer = tokenizer_of(backend.preprocess)
    pad_id = tokenizer.pad_token_id
    rng = np.random.default_rng(args.seed)

    # 1. What LeRobot computes, as it runs (VLM and most of the expert in bf16).
    frames = sample_frames(project.episodes / args.episode)
    samples = []
    for name, index, image, state, plan in frames:
        noise = rng.standard_normal((1, config.chunk_size, config.max_action_dim)).astype(np.float32)
        chunk = backend.infer(BackendObservation(0, [image], state, ROOT_TASK, noise)).chunk
        samples.append(dict(name=f"{args.episode}-{index:03d}-{name}", image=image, state=state, plan=plan,
                            noise=noise, bf16=chunk))
    print(f"bf16 LeRobot chunks for {len(samples)} frames of {args.episode}")

    # 2. The same in float32, with LeRobot's code as it is.
    policy.float()
    policy.requires_grad_(False)
    for s in samples:
        s["fp32"] = backend.infer(BackendObservation(0, [s["image"]], s["state"], ROOT_TASK, s["noise"])).chunk

    # 3. The edits for export, then the split graphs against step 2.
    patched = patch_lerobot(model)
    print(f"edited for export: apply_rope, eager_attention_forward, {', '.join(patched)}.forward")
    vlm = vwe.get_vlm_model()
    vlm.vision_model.config._attn_implementation = "eager"
    device = next(policy.parameters()).device
    parts = tuple(part.to(device) for part in graphs(model))
    for s in samples:
        pixels, ids, mask, state = prepared(backend, s["image"], s["state"], ROOT_TASK)
        noise = torch.from_numpy(s["noise"]).to(device)
        with torch.no_grad():
            original = vwe.embed_image(pixels)
        tokens, kv, prefix_mask, x = run_split(parts, pixels, ids.int(), mask.int(), state, noise, config, int(config_expert_hidden(config, hidden)))
        split = backend.postprocess(x[:, :, :2]).float()[0].cpu().numpy()
        s.update(pixels=pixels.cpu().numpy(), ids=ids.int().cpu().numpy(), mask=mask.int().cpu().numpy(),
                 state_norm=state.cpu().numpy(), tokens=tokens.cpu().numpy(), split=split)
        print(f"  {s['name']}: split graphs vs LeRobot fp32 {np.abs(split - s['fp32']).max():.2e}, "
              f"vision vs embed_image {float((tokens - original).abs().max()):.2e}, "
              f"fp32 vs bf16 {np.abs(s['fp32'] - s['bf16']).max():.3f}")
        assert np.abs(split - s["fp32"]).max() < 1e-3, "the split graphs do not reproduce the policy"

    s0 = samples[0]
    inputs = [torch.from_numpy(s0[k]).to(device) for k in ("pixels", "ids", "mask", "state_norm")]
    noise0 = torch.from_numpy(s0["noise"]).to(device)
    worst = activation_report(model, lambda: run_split(parts, *inputs, noise0, config, int(config_expert_hidden(config, hidden))))
    print("largest magnitudes (FP16 holds 65504): " + ", ".join(
        f"{kind} {value:.0f}" + (f" ({name.replace('vlm_with_expert.', '')})" if name else "")
        for kind, (value, name) in worst.items()))

    # 4. ONNX.
    vision, prefix, expert = parts
    with torch.no_grad():
        tokens = vision(inputs[0])
        kv = prefix(tokens, inputs[1], inputs[2], inputs[3])
    prefix_mask = torch.cat([torch.ones(1, tokens.shape[1], dtype=torch.int32, device=device), inputs[2],
                             torch.ones(1, 1, dtype=torch.int32, device=device)], dim=1)
    emb = torch.from_numpy(time_embedding(1.0, int(config_expert_hidden(config, hidden)), config.min_period, config.max_period)).to(device)
    exports = [
        ("vision", vision, (inputs[0],), ["pixel_values"], ["image_tokens"]),
        ("prefix", prefix, (tokens, inputs[1], inputs[2], inputs[3]), ["image_tokens", "lang_ids", "lang_mask", "state"], ["kv_cache"]),
        ("expert", expert, (noise0, emb, kv, prefix_mask), ["x_t", "time_emb", "kv_cache", "prefix_mask"], ["v_t"]),
    ]
    for name, module, example, input_names, output_names in exports:
        started = time.perf_counter()
        path = out / f"{name}.onnx"
        with torch.no_grad():
            torch.onnx.export(module, example, str(path), input_names=input_names, output_names=output_names,
                              opset_version=args.opset, do_constant_folding=True, dynamo=False)
        print(f"  {path.name}: {path.stat().st_size / 1e6:.0f} MB in {time.perf_counter() - started:.0f} s")

    write_tokenizer(tokenizer, Path(tokenizer.name_or_path) / "tokenizer.json" if Path(tokenizer.name_or_path).is_dir()
                    else hf_file(config.vlm_model_name, "tokenizer.json"), out / "tokenizer")
    write_meta(out / "meta.txt", config, processor_stats(backend.preprocess), processor_stats(backend.postprocess),
               hidden, tokens.shape[1], vwe.num_vlm_layers, list(kv.shape), chunk_dt_s(checkpoint), pad_id)
    write_samples(out / "samples", samples)
    print(f"meta.txt, tokenizer/, samples/ written")

    # 5. The graphs in onnxruntime against PyTorch, with the PyTorch model gone from the GPU.
    reference = {"image_tokens": tokens.cpu().numpy(), "kv_cache": kv.cpu().numpy()}
    feeds = {"vision": {"pixel_values": s0["pixels"]},
             "prefix": {"image_tokens": reference["image_tokens"], "lang_ids": s0["ids"], "lang_mask": s0["mask"],
                        "state": s0["state_norm"]}}
    del policy, model, parts, vision, prefix, expert, backend, vwe, vlm
    torch.cuda.empty_cache()
    check_onnx(out, feeds, reference, samples[0])


def hf_file(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename))


def write_samples(directory: Path, samples) -> None:
    import cv2
    rows = []
    for s in samples:
        stem = directory / s["name"]
        cv2.imwrite(str(stem) + ".png", cv2.cvtColor(s["image"], cv2.COLOR_RGB2BGR))
        for key in ("pixels", "tokens", "noise", "fp32", "bf16", "plan"):
            np.asarray(s[key], np.float32).tofile(f"{stem}.{key}.f32")
        rows.append(f"{s['name']}\t{s['state'][0]:.6f} {s['state'][1]:.6f}\t{' '.join(map(str, s['ids'][0]))}")
    (directory / "samples.txt").write_text(
        "# name<TAB>speed steer<TAB>token ids. Beside each: .png (the camera frame), and float32 arrays:\n"
        "#   .pixels.f32 [1,3,512,512] model input   .tokens.f32 [1,64,960] image tokens   .noise.f32 [1,16,32]\n"
        "#   .fp32.f32 / .bf16.f32 [16,2] the chunk LeRobot computes   .plan.f32 [16,2] the teacher's plan\n"
        + "".join(r + "\n" for r in rows))


def check_onnx(out: Path, feeds, reference, sample) -> None:
    import onnxruntime as ort
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    results = {}
    for name in ("vision", "prefix"):
        session = ort.InferenceSession(str(out / f"{name}.onnx"), options, providers=providers)
        output = session.run(None, feeds[name])[0]
        key = "image_tokens" if name == "vision" else "kv_cache"
        results[key] = output
        print(f"  onnxruntime {name}: max |diff| vs PyTorch {np.abs(output - reference[key]).max():.2e}")
        del session
    session = ort.InferenceSession(str(out / "expert.onnx"), options, providers=providers)
    meta = dict(line.split(" = ", 1) for line in (out / "meta.txt").read_text().splitlines())
    steps, hidden = int(meta["num_steps"]), int(meta["expert_hidden"])
    mask = np.concatenate([np.ones((1, 64), np.int32), feeds["prefix"]["lang_mask"], np.ones((1, 1), np.int32)], 1)
    x, dt = sample["noise"], -1.0 / steps
    for step in range(steps):
        emb = time_embedding(1.0 + step * dt, hidden, float(meta["min_period"]), float(meta["max_period"]))
        x = x + dt * session.run(None, {"x_t": x, "time_emb": emb, "kv_cache": results["kv_cache"], "prefix_mask": mask})[0]
    mean = np.array(meta["action_mean"].split(), np.float32)
    std = np.array(meta["action_std"].split(), np.float32)         # LeRobot unnormalises without eps
    chunk = x[0, :, :2] * std + mean
    print(f"  onnxruntime, all three graphs and the Euler loop: max |diff| vs LeRobot fp32 "
          f"{np.abs(chunk - sample['fp32']).max():.2e}")


if __name__ == "__main__":
    main()
