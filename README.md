# vla-drive

A small self-driving stack for learning, built around the way a vision-language-action
(VLA) model answers: slowly, and with a chunk of future actions. It has a simulated
1:10 car, a 50 Hz control loop with a safety supervisor, a data pipeline that takes
recorded drives to a trained network, and ROS 2 nodes around all of it. No robot is
needed; everything runs in one Docker container on a PC.

The aim is a Jetson Orin Nano Super driving with a fine-tuned VLA. Half of that is
here: a SmolVLA fine-tuned on the simulator's episodes drives the simulated car from a
PC, and runs on the Jetson with TensorRT in a C++ probe, but not yet inside the control
loop there. The other policies are a scripted teacher, a classical lane follower and a
0.3 M-parameter network trained by the pipeline.

```
camera, speed ─► policy (slow, own thread) ─► chunk of [steer, throttle]
                                                  │
              control loop, 50 Hz: ChunkExecutor ◄┘   the action for *now*
                                        │
              range sensor ───► SafetySupervisor      has the last word
                                        │
                                     motors
```

## Use

Needs Docker with the NVIDIA container toolkit.

```bash
make build                  # once: ROS 2 Humble + PyTorch with CUDA, 9 GB
make sim                    # watch the simulated car drive; q quits, e is the emergency stop
make test
make help                   # every target
```

From drives to a network that drives:

```bash
make collect                # a teacher drives 40 times in the simulator, with disturbed steering
make validate               # is the dataset complete, consistent, varied enough?
make browse                 # every frame, with its label drawn on the road
make train V=v1             # about 20 s on an RTX 4060
make evaluate V=v1          # drive it on every track, both ways
make sim ARGS='--policy cnn --checkpoint v1 --track trefoil'
```

The same loop over ROS 2 (`make ros-build` once, then one terminal each):

```bash
make ros-sim                # the simulated car, with a window
make ros-drive POLICY=cnn V=v1
make ros-teleop             # or drive by hand with the mouse
make ros-record             # record the drive as an episode; Ctrl-C saves
```

`./drive.sh` also runs outside the container on a PC with numpy and OpenCV; training
and the learned policy need the container's PyTorch.

## The VLA

A SmolVLA (about 450 M parameters) fine-tuned on the same episodes. It runs in a
LeRobot environment (Python 3.12, so not the ROS 2 container): the image in
[docker/vla](docker/vla/README.md), also a VS Code dev container (`make vla-build`,
`make vla-check`, `make vla-shell`), or a local one with that environment's Python and
`PYTHONPATH=.`. The setup and every flag are in Lab 15 of
[docs/study/15-hands-on-labs.md](docs/study/15-hands-on-labs.md).

```bash
python -m drive export                  # the episodes as a LeRobotDataset, in runs/drive/lerobot
lerobot-train --policy.path=lerobot/smolvla_base ...   # 2000 steps of batch 16 on an 8 GB laptop GPU
python -m drive sim --policy vla --checkpoint smolvla-v1 --show
python -m drive evaluate smolvla-v1 --latency-ms 125
```

To study and debug it ([chapter 17](docs/study/17-debugging-the-model.md) of the [study guide](docs/study/README.md)):

```bash
python tools/probe_vla.py               # nine offline probes: accuracy per chunk step, noise, symmetry, timing
python tools/see_vla.py                 # where in the image it looks: its input, attention, occlusion
python tools/vla_app.py                 # a web app at http://127.0.0.1:7860: an image or a video in, what the model does out
python tools/vla_video.py --video x.mp4  # the model on every frame of a video (or --episode): an annotated mp4 and a csv
python tools/check_env.py               # is this environment right? versions, GPU, the model on three known frames
python3 tools/fetch_bdd100k.py --count 5  # a few real driving videos from BDD100K (no 1.97 TB download)
python3 tools/web_videos.py               # those videos upright, 640x360 H.264: examples in the app's Video tab
```

On the Jetson, in C++ with TensorRT ([jetson/smolvla_trt](jetson/smolvla_trt/README.md)):
`export.py` cuts the model into three ONNX graphs on the PC, `build_engines.sh` builds FP16
engines on the board, and `vla_probe` shows each stage, from the task's tokens to the
chunk, for any image, sentence and state.

```bash
python jetson/smolvla_trt/export.py     # on the PC -> runs/drive/vla/smolvla-v1-trt
./build/vla_probe --model $M --check    # on the Jetson: against Python's numbers
./build/vla_probe --model $M --image frame.png --interactive
```

## What it shows

Measured in the simulator; the details and the commands are in
[docs/driving.md](docs/driving.md).

- A network trained on 12 000 frames from one track drives all three tracks, in both
  directions, within 1.3 cm rms of the centerline.
- The same network trained on *perfect* driving only scores better on held-out frames
  (steer error 0.015 against 0.024) and leaves the road on every track. A policy is
  judged by driving it, not by its validation loss.
- Executing chunks by the time of their observation hides inference latency up to about
  200 ms at 1.5 m/s. Past about 250 ms the loop is unstable at that speed.
- The supervisor stops the car 0.30 m short of an obstacle the policy drives straight at.
- The fine-tuned SmolVLA passes all six evaluation drives at 1.44 m/s with 125 ms per
  inference, but weaves: 4.8 cm rms. Its teacher drives within 1.5 cm at about the same
  speed (1.47 m/s, 100 ms).
  Hiding parts of the image shows that it steers by the road from 0.4 m ahead to the
  horizon, where its teacher aims. So the weave more likely comes from how precisely it
  answers than from where it looks.
- On the Jetson Orin Nano Super, FP16 TensorRT engines compute the same chunk to within
  0.002 of fp32 PyTorch, in 82 ms per inference (vision 25, prefix 9, ten denoising steps
  43). Fine-tuned on one sentence, it ignores the sentence: "stop" and "turn left" barely
  change the chunk; the image sets the steering and the speed the throttle.

## Layout

```
drive/          the simulator, policies, control loop, safety supervisor
drive/learn/    the data pipeline: collect, label, validate, browse, train, evaluate, export, status
ros/            ROS 2 nodes: the simulated car, the driver, teleop, the recorder
tools/          probe_vla, see_vla, vla_app, vla_video, check_env: study and debug a fine-tuned VLA
jetson/         smolvla_trt: the fine-tuned SmolVLA in C++ with TensorRT, on the Jetson
docs/study/     a study guide: 18 chapters with labs, a glossary and draw.io diagrams
docker/         the containers: ROS 2 + PyTorch for drive/, and vla/ (LeRobot) for SmolVLA; Makefile has the commands
.devcontainer/  VS Code in the LeRobot container
tests/          make test
runs/           datasets, models and reports (not in git)
```

## Not here yet

A real car, a CARLA bridge, and a VLA driving on the Jetson. The fine-tuned SmolVLA runs
on the board with TensorRT, but only in the C++ probe: it is not yet a backend behind
`BackendPolicy`, so the control loop on the Jetson does not use it. The simulator
is flat and kinematic, so its numbers describe how the control loop behaves, not how a
car will.

## Thanks

To [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), which runs
VLA models on the 8 GB Orin Nano with TensorRT and is where this project started; its
backends are what `drive/policy/vla.py` is written to wrap.

## License

MIT.
