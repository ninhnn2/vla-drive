# vla-drive

[![tests](https://github.com/ninhnn2/vla-drive/actions/workflows/tests.yml/badge.svg)](https://github.com/ninhnn2/vla-drive/actions/workflows/tests.yml)

A small self-driving stack for learning, built around the way a vision-language-action
(VLA) model answers: slowly, and with a chunk of future actions. It has a simulated
1:10 car, a 50 Hz control loop with a safety supervisor, a data pipeline that takes
recorded drives to a trained network, and ROS 2 nodes around all of it. No robot is
needed; everything runs in one Docker container on a PC.

The aim is a Jetson Orin Nano Super driving with a fine-tuned VLA. Half of that is
here: a SmolVLA fine-tuned on the simulator's episodes drives the simulated car from a
PC; it has not run on a Jetson yet. The other policies are a scripted teacher, a
classical lane follower and a 0.3 M-parameter network trained by the pipeline.

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

On a Mac with Apple Silicon there is no NVIDIA GPU for Docker, so the same targets run
`drive/` natively in a `.venv` and train on the Mac's GPU (MPS). Only ROS 2 uses a
container, built for arm64 without CUDA, with its windows on XQuartz. Needs
[uv](https://docs.astral.sh/uv/) (`brew install uv`); for ROS 2 also Docker Desktop and
XQuartz (`make gui` says what to set).

```bash
make setup                  # once: Python 3.10, numpy, OpenCV and PyTorch in .venv
make sim
make test                   # without the ROS 2 tests; make ros-test runs those in the container
```

From drives to a network that drives:

```bash
make collect                # a teacher drives 40 times in the simulator, with disturbed steering
make validate               # is the dataset complete, consistent, varied enough?
make browse                 # every frame, with its label drawn on the road
make train V=v1             # about 20 s on an RTX 4060, 2 min on an M3 Mac
make evaluate V=v1          # drive it on every track, both ways
make sim ARGS='--policy cnn --checkpoint v1 --track trefoil'
./drive.sh probe v1 errors  # look inside it: layers, history, errors, activations, saliency, behaviour
```

A hands-on course in Vietnamese, from collecting data to judging and debugging the
network, every command with its real output:
[docs/thuc-hanh](docs/thuc-hanh/README.md). The theory, in English, is in
[docs/study](docs/study/README.md).

The same loop over ROS 2 (`make ros-build` once, then one terminal each):

```bash
make ros-sim                # the simulated car, with a window
make ros-drive POLICY=cnn V=v1
make ros-teleop             # or drive by hand with the mouse
make ros-record             # record the drive as an episode; Ctrl-C saves
```

`./drive.sh` also runs outside the container on a PC with numpy and OpenCV; training
and the learned policy need PyTorch, which the container and `make setup`'s `.venv` have.

## The VLA

A SmolVLA (about 450 M parameters) fine-tuned on the same episodes. It runs in a
LeRobot environment of its own, outside the container: `make vla-setup` makes it
(`.venv-vla`, Python 3.12, LeRobot 0.5.1 and gradio), and `./vla.sh` runs its Python with
this repository on the path. The same works on Linux and on a Mac, where the model runs on
the Apple GPU (MPS). The setup and every flag are in Lab 15 of
[docs/study/15-hands-on-labs.md](docs/study/15-hands-on-labs.md).

```bash
make vla-setup                          # once, with uv: about 1.6 GB on a Mac, several GB on Linux (CUDA)
./vla.sh -m drive export                # the episodes as a LeRobotDataset, in runs/drive/lerobot
./vla.sh lerobot-train --policy.path=lerobot/smolvla_base ...   # 2000 steps of batch 16 on an 8 GB laptop GPU
./vla.sh -m drive sim --policy vla --checkpoint smolvla-v1 --show
./vla.sh -m drive evaluate smolvla-v1 --latency-ms 125
```

To study and debug it ([chapter 17](docs/study/17-debugging-the-model.md) of the [study guide](docs/study/README.md)):

```bash
./vla.sh tools/probe_vla.py             # nine offline probes: accuracy per chunk step, noise, symmetry, timing
./vla.sh tools/see_vla.py               # where in the image it looks: its input, attention, occlusion
./vla.sh tools/vla_app.py --open        # a web app at http://127.0.0.1:7860: an image in, the chunk out
```

Each takes `--device cuda|mps|cpu` (or `VLA_DEVICE=…`): the same frame on two devices is
a quick check that a GPU computes what the CPU does. On a Mac,
[docs/thuc-hanh/08-vla-tren-mac.md](docs/thuc-hanh/08-vla-tren-mac.md) (in Vietnamese)
goes from an empty `runs/` to a fine-tuned SmolVLA and the three tools, with the
times an M3 takes.

- `VLA_VENV=/path/to/venv ./vla.sh …` uses another LeRobot environment, such as one built
  from a LeRobot checkout.
- A checkpoint fine-tuned with a newer LeRobot loads in 0.5.1: the config fields 0.5.1
  does not know are dropped, and named.
- If `~/.cache` is short of space, set `HF_HOME` (the model is about 1 GB) and
  `UV_CACHE_DIR`.
- CI runs every test on Linux and on a Mac, in both environments.
  `tests/test_vla_tools.py` runs the three tools on a tiny untrained SmolVLA, so
  `./vla.sh -m unittest tests.test_vla_tools` checks a setup in about 2 minutes without
  a checkpoint.

## CARLA

`drive/carla` makes CARLA a platform for the same loop and pipeline: a teacher drives
random routes through a town, is told where to turn ("take the next left"), and its
drives become episodes with a sentence per frame, for a SmolVLA that follows directions.
It runs against the stock `carlasim/carla:0.9.16` image; the client has its own
`.venv-carla`. Everything, with what was measured, is in [docs/carla.md](docs/carla.md).

```bash
make carla-setup                          # once: .venv-carla (Python 3.10, the carla wheel)
make carla-server TOWN=Town01             # the server in Docker, booted into one town
make carla-collect ARGS='--episodes 40 --show'   # -> runs/carla/episodes, watched in a window
make carla-down
```

A plan in those labels lands within 0.19 m of where the car really goes 1.5 s later
(median; 0.32 m at the 90th percentile), measured by `./drive.sh carla check`. Recorded
so far: Town01, Town03_Opt and Town05_Opt, 120 drives, 72 000 frames, every drive to its
end; it exports to LeRobot and SmolVLA trains on it. In the multi-lane towns the lane
still decides which way a car can turn (lane changes are next).

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

## Layout

```
drive/          the simulator, policies, control loop, safety supervisor
drive/learn/    the data pipeline: collect, label, validate, browse, train, evaluate, export, status
drive/carla/    CARLA as a platform: the car model, the route teacher, identify, check, collect
ros/            ROS 2 nodes: the simulated car, the driver, teleop, the recorder
tools/          probe_vla, see_vla, vla_app: study and debug a fine-tuned VLA
docs/study/     a study guide: 18 chapters with labs, a glossary and draw.io diagrams
docs/thuc-hanh/ a hands-on course in Vietnamese, lessons 1-8 (lesson 8: the VLA on a Mac)
docker/         the container everything runs in (compose.mac.yml: on a Mac); Makefile has the commands
vla.sh          runs the LeRobot environment's Python (make vla-setup) with this repository on the path
tests/          make test; ./vla.sh -m unittest discover -s tests -t . in the LeRobot environment
.github/        CI: every test on Linux and on a Mac
runs/           datasets, models and reports (not in git)
```

## Not here yet

A real car, a VLA trained to drive CARLA well (recording, training and `carla drive` to test it are in
docs/carla.md; only a 300-step check has been run through them), and a VLA on the Jetson. The fine-tuned SmolVLA runs in
PyTorch on a PC; it has not been exported to TensorRT or run on a board. The simulator
is flat and kinematic, so its numbers describe how the control loop behaves, not how a
car will.

## Thanks

To [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), which runs
VLA models on the 8 GB Orin Nano with TensorRT and is where this project started; its
backends are what `drive/policy/vla.py` is written to wrap.

## License

MIT.
