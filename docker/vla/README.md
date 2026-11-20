# docker/vla: the LeRobot environment for SmolVLA

One image for the VLA work: fine-tuning and running `smolvla-v1`, the `vla` policy in the
simulator, `tools/` (`probe_vla`, `see_vla`, `vla_app`, `vla_video`, `check_env`), the ONNX
export for the Jetson, and the unit tests. Use it as a VS Code dev container
(`.devcontainer/`) or through `make vla-*`.

| | |
|---|---|
| base | `python:3.12-slim-bookworm` (Python 3.12.16) |
| packages | `requirements.lock`, the `pip freeze` of the environment the model runs in on the PC: torch 2.11.0 (CUDA 13.0), LeRobot 0.5.2 at commit `3dd19d0`, transformers 5.5.4, onnx, onnxruntime-gpu, gradio, av |
| apt | git, ffmpeg (torchcodec decodes LeRobot datasets with it), sudo |
| user | `dev`, UID 1000 (VS Code changes it to yours; compose runs as `HOST_UID`) |
| paths | the repository at `/work` (`PYTHONPATH`), the venv at `/opt/venv`, `HF_HOME=/hf` on the volume `vla-drive-hf` |

It is a separate image from `docker/Dockerfile` because LeRobot needs Python 3.12 and ROS 2
Humble is built for Python 3.10. ROS 2 stays in that image.

## What the host needs

- **Docker** with BuildKit (Docker 23 or newer). About 12 GB of disk for the image.
- **An NVIDIA driver 580 or newer**: torch 2.11 from PyPI is built for CUDA 13.0.
- **The NVIDIA container toolkit**, so that containers see the GPU. `nvidia-smi` working on the
  host says nothing about containers. Check:

  ```bash
  docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L
  ```

  If that fails, install the toolkit. On Docker 29, `--gpus` goes through CDI, so generate its spec too:

  ```bash
  sudo apt install -y nvidia-container-toolkit
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L
  ```

- **What is not in git**, under `runs/` (mounted with the repository): the checkpoint
  `runs/drive/vla/smolvla-v1/` and, for the check, `runs/drive/vla/smolvla-v1-trt/samples/`.
  The episodes in `runs/drive/episodes/` are needed for `probe_vla`, `see_vla` and the app's
  episode frames.
- **SmolVLM2** (about 1 GB), which LeRobot loads the processor and the VLM's config from. It is
  downloaded into the volume `vla-drive-hf` on first use. If the host already has it, copy it in
  instead:

  ```bash
  docker run --rm -v vla-drive-hf:/hf -v "$HF_HOME":/src:ro python:3.12-slim-bookworm \
      sh -c 'cp -a /src/. /hf/ && chown 1000:1000 /hf'
  ```

## Use it

**VS Code**: open the repository, then "Dev Containers: Reopen in Container". The first time it
builds the image. Terminals, the debugger and the Python extension then run inside, with
`/opt/venv/bin/python`.

VS Code adds `--gpus all` when `docker info` lists the `nvidia` runtime. If
`/etc/docker/daemon.json` declares that runtime but the toolkit is not installed, the container
fails to start ("failed to discover GPU vendor from CDI"). Install the toolkit, or set
`"dev.containers.gpuAvailability": "none"` in VS Code's settings and work on the CPU until you do.

**make** (from the repository root):

```bash
make vla-build                    # build the image
make vla-check                    # versions, GPU, the model on three known frames
make vla-test                     # the unit tests
make vla-shell                    # a shell inside
make vla-app                      # tools/vla_app.py at http://127.0.0.1:7860: an image or a video in, actions out
```

`make vla-*` asks for the GPU. Without one, run the container directly:

```bash
docker run --rm -it -u $(id -u):$(id -g) -e HOME=/tmp -v "$PWD":/work -v vla-drive-hf:/hf \
    vla-drive:lerobot0.5.2-torch2.11 python tools/check_env.py
```

Inside, everything runs from `/work` as on the host, without `PYTHONPATH=.` or `HF_HOME`:

```bash
python -m drive sim --policy vla --checkpoint smolvla-v1 --latency-ms 125 --duration-s 10 --video runs/debug/sim.mp4
python tools/vla_video.py --episode sim2-031 --seconds 10
python tools/probe_vla.py --episodes sim2-031 --every 15
```

To work offline once SmolVLM2 is in the volume, set `HF_HUB_OFFLINE=1`.

## Is it the right environment?

`tools/check_env.py` compares each package with `requirements.lock`, says what GPU torch sees,
and runs the model on the three frames `jetson/smolvla_trt/export.py` saved. Each frame has its
image, state, sentence and starting noise, and the chunk LeRobot computed on the PC. The same
inputs must give the same chunk.

Measured on the PC (RTX 2060, driver 595, i.e. the reference itself):

| where | steering left / right / straight | largest difference | per chunk |
|---|---|---|---|
| reference (`*.bf16.f32`) | +0.5253 / −0.5482 / −0.0371 | | |
| host env, GPU | +0.5247 / −0.5482 / −0.0368 | 5.5e-4 | 550 ms |
| host env, CPU | +0.5258 / −0.5480 / −0.0345 | 2.7e-3 | 5.0 s |
| this image, CPU | +0.5258 / −0.5480 / −0.0345 | 2.7e-3 | 4.8 s |

The image on the CPU gives the host's CPU numbers digit for digit. In it, the unit tests give
87 run, 5 skipped, as on the host. bf16 on a different GPU or on the CPU rounds differently, so
thousandths are expected. A
difference in the tenths means something else changed: a package, the checkpoint, the samples.

## What is not in here

- **ROS 2**: `docker/Dockerfile` (`make shell`, `make ros-*`).
- **Windows**: OpenCV is the headless build, as in the environment it copies, so `--show` opens
  nothing. Use `--video`, or the app in the browser.
- **TensorRT and the C++ probe**: they are built and run on the Jetson (`jetson/smolvla_trt/`).

## Changing a package

Change it in a working environment, check the model there (`python tools/check_env.py`), then
write the lock again from it and rebuild:

```bash
( head -5 docker/vla/requirements.lock; pip freeze | grep -v '^lerobot @' ) > /tmp/lock \
    && mv /tmp/lock docker/vla/requirements.lock && make vla-build && make vla-check
```
