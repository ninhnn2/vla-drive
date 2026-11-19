# smolvla_trt: SmolVLA in C++ with TensorRT, on the Jetson

`vla_probe` runs the fine-tuned SmolVLA (`smolvla-v1`, [Lab 15](../../docs/study/15-hands-on-labs.md))
on the Jetson Orin Nano with TensorRT, from C++, and prints what each stage does with its
inputs: how the task sentence is tokenised, how the camera frame becomes 64 image tokens,
how the state is normalised, how 10 denoising steps turn noise into a chunk of 16
`[steer, throttle]` actions, and how long each stage takes. Change the sentence, the image
or the state, and see the chunk move.

It is a probe, not the driver: nothing here sends commands to a car.

```
task text ──tokenizer (C++)──► 48 ids + mask ─────────────────────────┐
camera ──resize, pad, [-1,1] (C++)──► [3,512,512] ──vision.engine──► 64 tokens ──┤
[speed, steer] ──(x-mean)/std, pad to 32 (C++)─────────────────────────┼──prefix.engine──► KV cache
                                                                                  │
noise [16,32] ──10 × ( expert.engine: v = f(x, t, KV);  x += -0.1·v ) (C++ loop) ◄┘
              ──► x · std + mean ──► 16 × [steer, throttle], 0.1 s apart
```

| File | What |
|---|---|
| [`export.py`](export.py) | on the PC: cuts the policy into three ONNX graphs, writes the tokenizer, the statistics and reference samples |
| [`build_engines.sh`](build_engines.sh) | on the Jetson: `trtexec` builds `vision/prefix/expert.engine` |
| [`src/tokenizer.cpp`](src/tokenizer.cpp) | the SmolVLM2 byte-level BPE tokenizer, no Python |
| [`src/preprocess.cpp`](src/preprocess.cpp) | LeRobot's `resize_with_pad` and the [-1, 1] mapping |
| [`src/smolvla.cpp`](src/smolvla.cpp) | the pipeline: tokens, image, state, three engines, the Euler loop, unnormalising |
| [`src/trt_engine.cpp`](src/trt_engine.cpp) | one engine and its device buffers; outputs are handed to the next engine without copies |
| [`src/main.cpp`](src/main.cpp) | `vla_probe`: the report, `--interactive`, `--check`, `--dump` |

## 1. Export (PC, the LeRobot environment)

```bash
export HF_HOME=…/smolvla-local/hf PYTHONPATH=.        # as in Lab 15
$V jetson/smolvla_trt/export.py                        # -> runs/drive/vla/smolvla-v1-trt (1.6 GB)
```

It checks itself as it goes: the three graphs and a 10-step Euler loop against LeRobot's
own `predict_action_chunk` in fp32, then the ONNX files in onnxruntime against PyTorch.

Three edits make the graphs exportable and FP16-safe, none of which changes the fp32 result
(*measured* below 1e-6):

- **RMSNorm scaled by max|x|.** The residual stream reaches |x| ≈ 2600 in the last layers,
  and 2600² is far past FP16's 65 504: the plain RMSNorm becomes `inf` in an FP16 engine.
- **Finite mask value.** LeRobot masks attention scores with −3.4e38, which is −inf in FP16;
  a padding token's fully masked row would turn into NaN and spread through the KV cache.
- **RoPE without in-place slices**, and the vision patches' position ids as a constant
  (always 0…1023 for a 512×512 image), since `bucketize` has no ONNX form.

## 2. Copy, build the engines and the probe (Jetson)

```bash
rsync -a jetson/smolvla_trt jetson:vla-drive/jetson/
rsync -a runs/drive/vla/smolvla-v1-trt jetson:vla-drive/runs/drive/vla/
# on the Jetson:
cd ~/vla-drive
./jetson/smolvla_trt/build_engines.sh runs/drive/vla/smolvla-v1-trt          # fp16; add fp32 to compare
cd jetson/smolvla_trt && cmake -B build && cmake --build build -j4
```

Engines belong to one GPU and one TensorRT version: build them on the board that runs them.

## 3. Run

```bash
M=~/vla-drive/runs/drive/vla/smolvla-v1-trt
./build/vla_probe --model $M --check                                    # against Python, exit 1 on a mismatch
./build/vla_probe --model $M --image $M/samples/sim2-031-235-right.png --speed 1.35 --steer -0.65
./build/vla_probe --model $M --image frame.png --task "turn left" --dump out/   # model_input.png, trace.json, ...
./build/vla_probe --model $M --image frame.png --repeat 50              # timing
./build/vla_probe --model $M --tokens "rẽ trái ở góc tiếp theo"          # the tokenizer only
./build/vla_probe --model $M --image frame.png --interactive
```

In `--interactive` mode the starting noise stays fixed until `seed` changes it, so a change
in the chunk comes from what you changed:

```
vla> task stop the car
vla> state 0.5 0.0
vla> image another.png
vla> tokens lái xe theo làn đường
vla> seed 3
```

## What to expect

*Measured* on 2026-11-19, Jetson Orin Nano Super (JetPack 6.2, TensorRT 10.3, 25 W mode), FP16 engines.

**Building** (`build_engines.sh`): vision 1.5 min, prefix 2.5 min, expert 3 min; engines of
198, 394 and 201 MB.

**Agreement with Python** (`--check`, three held-out frames, the same starting noise):

| Stage | Max \|difference\| |
|---|---|
| model input (C++ resize and pad vs LeRobot) | 3.6e-7 |
| token ids (C++ tokenizer vs Python, 15 test strings) | identical |
| image tokens (FP16 engine vs fp32 PyTorch; values up to 76) | 0.56–0.75 |
| chunk, steer and throttle (vs LeRobot fp32) | **0.001–0.002** |
| chunk (vs LeRobot as it runs, in bf16) | 0.002–0.003 |

FP16 on the Jetson is as close to fp32 as LeRobot's own bf16 run is. On those frames the
first steering action is +0.527 (teacher +0.596) on a left bend, −0.547 (teacher −0.655) on
a right bend, −0.034 (teacher +0.002) on a straight.

**Time per inference** (`--repeat 30`, median):

| Stage | ms |
|---|---:|
| tokenise + image resize (CPU) | 0.1 + 1.8 |
| vision (SigLIP + connector) | 24.6 |
| prefix (16 LM layers, 113 tokens) | 8.9 |
| expert, 10 denoising steps | 43.0 |
| **total, with copies** | **82** |

That is 1.5 times faster than the PyTorch model on the RTX 4060 laptop of Lab 15 (124 ms).
The expert's 10 steps are half the time: `--steps 5` halves that part, at a cost in
accuracy that this probe can show.

**What the model responds to** (`--interactive`, the right-bend frame, fixed noise; mean
steering over the first 0.4 s):

| Change | Steer | Throttle |
|---|---:|---:|
| as trained: `drive along the track`, 1.35 m/s | −0.36 (right) | 0.16 |
| task `stop the car now` | −0.30 | 0.16 |
| task `turn left` | −0.31 (still right) | 0.16 |
| task `lái xe theo làn đường` (18 byte-level tokens) | −0.34 | 0.16 |
| speed 0.3 m/s instead of 1.35 | −0.30 | **0.88** |
| the left-bend frame | **+0.39 (left)** | 0.92 |

The image decides the steering and the state the throttle; the sentence barely matters.
Fine-tuned on one sentence, the model has learnt to ignore it: it does not stop when told,
and does not turn left on a right bend. Following instructions would need episodes with
different sentences and different behaviours.

## Limits

- The tokenizer's regex classes (`\p{L}`, `\p{N}`, `\s`) are exact for ASCII and
  approximated by Unicode ranges elsewhere. `--check` compares it with Python on 15 strings,
  Vietnamese and emoji among them.
- `--seed` noise comes from `std::mt19937_64`, not PyTorch's generator: the same seed gives
  another chunk than in Python. `--noise FILE` (and `--check`) use Python's noise.
- The model was fine-tuned on one sentence, `drive along the track`. Another sentence is
  something it has never been trained to follow; what it does with it is what this probe shows.
- The image must look like the simulator's renders: 320×240, the camera's height and pitch.
