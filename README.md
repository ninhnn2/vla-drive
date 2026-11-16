# vla-drive

A small self-driving stack for learning, built around the way a vision-language-action
(VLA) model answers: slowly, and with a chunk of future actions. It has a simulated
1:10 car, a 50 Hz control loop with a safety supervisor, a data pipeline that takes
recorded drives to a trained network, and ROS 2 nodes around all of it. No robot is
needed; everything runs in one Docker container on a PC.

The aim is a Jetson Orin Nano Super driving with a fine-tuned VLA. That part is not
here yet: today the policies are a scripted teacher, a classical lane follower and a
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
```

The same loop over ROS 2 (`make ros-build` once, then one terminal each):

```bash
make ros-sim                # the simulated car, with a window
make ros-drive POLICY=cnn V=v1
make ros-teleop             # or drive by hand with the mouse
make ros-record             # record the drive as an episode; Ctrl-C saves
```

`./drive.sh` also runs outside the container on a PC with numpy and OpenCV; training
and the learned policy need PyTorch, which the container and `make setup`'s `.venv` have.

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

## Layout

```
drive/          the simulator, policies, control loop, safety supervisor
drive/learn/    the data pipeline: collect, label, validate, browse, train, evaluate, status
ros/            ROS 2 nodes: the simulated car, the driver, teleop, the recorder
docker/         the container everything runs in (compose.mac.yml: on a Mac); Makefile has the commands
tests/          make test
runs/           datasets, models and reports (not in git)
```

## Not here yet

A real car, a CARLA bridge, conversion of the episodes to a LeRobot dataset, and a
fine-tuned VLA. The simulator is flat and kinematic, so its numbers describe how the
control loop behaves, not how a car will.

## Thanks

To [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), which runs
VLA models on the 8 GB Orin Nano with TensorRT and is where this project started; its
backends are what `drive/policy/vla.py` is written to wrap.

## License

MIT.
