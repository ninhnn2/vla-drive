# Driving

`drive/` is a small driving stack built around a policy that answers slowly and in
chunks, which is what a VLA is: SmolVLA takes about 100 ms on the Orin Nano and returns
50 future actions. It runs on a PC with numpy and OpenCV, against a simulated 1:10 car.
On the Jetson only the policy changes.

```
camera, speed ─► policy (slow, ~10 Hz, own thread) ─► chunk of [steer, throttle]
                                                         │
                 control loop, 50 Hz:   ChunkExecutor ◄──┘   the action for *now*
                                             │
                 range sensor ─────► SafetySupervisor        has the last word
                                             │
                                          motors
```

## Try it

Everything runs in one container: ROS 2 Humble, PyTorch with CUDA, and windows on your
screen. It needs Docker with the NVIDIA container toolkit.

```bash
make build                     # once: the image (9 GB)
make sim                       # watch the default drive; q quits, e is the emergency stop
make sim ARGS='--policy lane-camera'          # steers from the camera image alone
make sim ARGS='--obstacle 8'                  # a cone the policy ignores: safety stops the car
make sim ARGS='--obstacle 8 --no-safety'      # ... and what happens without it
make shell                     # a shell in the container; ./drive.sh --help
make test
make help                      # every target
```

`./drive.sh` also runs straight on a PC that has numpy and OpenCV; only training and
the learned policy need the container's PyTorch.

```bash
./drive.sh sim --latency-ms 400 --max-plan-age-ms 1500 --video runs/slow.mp4
./drive.sh sweep                               # tracking error against inference latency
```

Tracks: `--track peanut` (default), `oval`, `trefoil`; `--reverse` drives one clockwise.

`--show` and `--realtime` run on the wall clock with inference on a real thread, as on
the car. Without them the run is as fast as the PC allows and repeats exactly.

## What is in it

| | |
|---|---|
| [`sim/`](../drive/sim) | the track, a kinematic bicycle model with servo lag, a forward camera rendered from a top-down map, a forward range sensor |
| [`policy/`](../drive/policy) | `pure-pursuit`: a teacher that reads the map and the car's pose (simulator only). `lane-camera`: finds the road in the image, no map. `vla.BackendPolicy`: wraps a TensorRT backend from [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla) |
| [`control/`](../drive/control) | `SimulatedWorker` / `ThreadedWorker`: one inference at a time, off the control loop. `ChunkExecutor`: the action for now from the newest chunk |
| [`safety.py`](../drive/safety.py) | emergency stop, stale-plan watchdog, NaN check, steering-rate and speed limits, braking for what the range sensor sees |
| [`loop.py`](../drive/loop.py) | one control tick |
| [`episode.py`](../drive/episode.py), [`cli.py`](../drive/cli.py) | a whole simulated drive and its measurements |
| [`recorder.py`](../drive/recorder.py), [`viewer.py`](../drive/viewer.py) | a drive to disk as an episode; the window and the video |
| [`learn/`](../drive/learn) | the data pipeline, one module per stage: collect, label, validate, browse, train, evaluate, status |
| [`policy/cnn.py`](../drive/policy/cnn.py) | drives with a network trained by `learn/` |
| [`platform.py`](../drive/platform.py) | what the loop needs from a car: the seam for ROS 2, CARLA or hardware |
| [`ros/src/drive_ros`](../ros/src/drive_ros) | ROS 2 nodes: the simulated car, the control loop, a recorder, a teleop window |
| [`docker/`](../docker), [`Makefile`](../Makefile) | the container and the commands that run in it |
| [`config.py`](../drive/config.py) | every number: the car, the camera, the limits |

An action is `[steer, throttle]`, both in [-1, 1]: steer +1 is full left, throttle -1 is
full brake. The state a policy gets is `[speed, steer]`. Units are SI.

## What the simulator shows

Measured with `./drive.sh`, 30 s runs, the teacher at 1.5 m/s on the 22 m track.

**Chunks hide the latency, up to a point.** A chunk is read at `now - t_obs`, so the
actions meant for the time spent computing are skipped (`ChunkExecutor`). From 0 to
200 ms the car stays within 2 cm rms of the centerline:

| latency | compensated: rms / max | played from the start: rms / max |
|---:|---:|---:|
| 0 ms | 1.8 / 3.5 cm | 1.6 / 3.0 cm |
| 100 ms | 1.8 / 3.4 cm | 0.7 / 2.2 cm |
| 200 ms | 2.0 / 3.6 cm | 3.8 / 8.8 cm |
| 250 ms | 3.1 / 6.6 cm | 10.0 / 23.3 cm |
| 400 ms | leaves the road | leaves the road |

(`./drive.sh sweep --max-plan-age-ms 1500 --latencies 0,100,200,250,400`.) Below
100 ms compensation does not matter here; the uncompensated numbers are lower only
because the delay happens to cancel pure pursuit's corner cutting.

**Past about 250 ms at 1.5 m/s the loop is unstable.** Each plan assumes it starts at
its observation, but the car spends the next latency on the previous plan, so every
correction arrives late and overshoots. It is speed-dependent: at 0.6 m/s, 400 ms holds
1.6 cm rms. A model that slow needs a plan conditioned on the actions already committed
(real-time chunking), or a slower car. That is not built here.

**The watchdog must exceed twice the latency.** The plan in use is between one and two
latencies old. With the default 600 ms limit a 400 ms model is braked half the time and
a 650 ms model never moves (`stale` in the result). Raising the limit is a decision
about how far the car may travel on old information, not a tuning knob.

**The supervisor does not need the policy to be right.** With a cone at 8 m the teacher
drives straight at it; the supervisor brakes from its own range sensor and stops 0.30 m
short. With `--no-safety` the car hits it.

## From drives to a network that drives

The whole imitation-learning loop, small enough to rerun in minutes. It is the loop a
VLA fine-tune goes through, with a 0.3 M-parameter network in place of SmolVLA.

```
collect ─┐
         ├─▶ label ─▶ validate ─▶ (browse) ─▶ train ─▶ evaluate
ROS 2 ───┘                                      ▲          │
  record            more or more varied drives  └──────────┘
```

```bash
make collect                # the teacher drives 40 times in the simulator: 12 000 frames, ~35 s
make validate               # is the dataset complete, consistent, varied enough?
make browse                 # every frame, with its label drawn on the road
make train V=v1             # ~20 s on an RTX 4060
make evaluate V=v1          # drive it on every track, both ways
make sim ARGS='--policy cnn --checkpoint v1 --track trefoil'
make status                 # what the project holds, and the next step
```

A project is one directory, `runs/drive` unless `--project` says otherwise:

```
runs/drive/episodes/<name>/episode.npz   t, images (N,240,320,3), states [speed, steer], actions [steer, throttle]
                           meta.json     the task, the rate, what the columns mean
                           labels.npz    plans (N,16,2), valid (N,), dt, source
           models/<version>.pt, .json    the network, and how its training went
           reports/<name>.json           an evaluation
```

| stage | what it does |
|---|---|
| [`collect`](../drive/learn/collect.py) | the pure-pursuit teacher drives with disturbed steering; each frame is stored with the teacher's plan from there |
| [`label`](../drive/learn/label.py) | for episodes without a plan (ROS 2 recordings, a person's drives): the plan is the 16 actions that followed the frame, 0.1 s apart. Frames in the last 1.5 s have no full future and are left out |
| [`validate`](../drive/learn/validate.py) | fails on what makes training wrong (unreadable or mismatched arrays, NaN, values out of range, timestamps that go backwards, mixed image sizes); warns on what makes it poor (gaps, frozen frames, a car standing still, too few frames, perfect driving only) |
| [`browse`](../drive/learn/browse.py) | a window (or `--video`) with each frame, its state and action, and the labelled plan rolled forward and drawn on the road. A good label's path follows the road |
| [`train`](../drive/learn/train.py) | regression on the plans; whole episodes held out for validation; mirrored images and brightness changes as augmentation. A version is never overwritten |
| [`evaluate`](../drive/learn/evaluate.py) | drives each model on every track, forwards and reversed, and compares them. A drive passes when the car never leaves the road, hits nothing and keeps moving |
| [`status`](../drive/learn/status.py) | episodes, labels, models, reports, and what to do next |

A stage is a module with `add_parser` and `run(project, args)`, listed in
[`learn/__init__.py`](../drive/learn/__init__.py); adding one is writing that module.

**The label is a plan, not a class.** Nobody draws boxes here: a driving label is what
to do next, and it comes for free from a teacher or from the recording itself. What
needs a person's eye is whether the labels make sense, which is what `browse` is for.

**The model** ([`learn/model.py`](../drive/learn/model.py)): five strided convolutions
on the 96x128 image, the state appended, an MLP out to 16 x `[steer, throttle]`. Like a
VLA it answers with a chunk, so the same executor and latency handling apply.

Measured (`make evaluate V="v1 v2 pure-pursuit"`, 40 s drives at 100 ms latency). `v1`
was trained on the 40 simulator episodes, all on the peanut track; `v2` on those plus
two drives recorded over ROS 2 on the oval and the trefoil:

| model | drives passed | mean speed | rms off the centerline | worst |
|---|---:|---:|---:|---:|
| v1 | 6/6 | 0.84 m/s | 1.3 cm | 4.0 cm |
| v2 | 6/6 | 0.90 m/s | 1.2 cm | 3.8 cm |
| pure-pursuit (the teacher) | 6/6 | 1.47 m/s | 1.5 cm | 3.4 cm |

The networks drive slower than the teacher: its target speed changed per episode and
is not in the image, so they settle on a cautious average.

**Why the teacher's steering is disturbed while recording.** A network shown only
perfect driving has never seen the car off the centerline; its first small error puts
it somewhere it has no answer for, and the error grows. So `collect` disturbs the
steering that is *applied* and records what the teacher *would* do from wherever the
car ends up (`--steer-noise`, default 0.25). The same pipeline with `--steer-noise 0`
(`--project runs/perfect`):

| training data | validation steer error | drives passed | on the peanut track |
|---|---:|---:|---|
| disturbed steering (`v1`) | 0.024 | 6/6 | 1.5 cm rms, never off the road |
| perfect driving only | 0.015 | 0/6 | off the road 75 % of the time |

The network trained on perfect driving scores *better* on held-out frames and cannot
drive. Offline error measures how well it copies the frames it was given, not what
happens once its own mistakes move the car. `validate` warns about such a dataset
("perfect driving only"), and `evaluate` is the test that counts.

## ROS 2

The same loop with the car behind topics. Build once with `make ros-build`, then each
in its own terminal:

```bash
make ros-sim                      # the simulated car, with a window     [TRACK=trefoil]
make ros-drive                    # the control loop drives it           [POLICY=cnn V=v1]
make ros-teleop                   # ... or you do: hold the left mouse button in the window
make ros-record NAME=ros          # record the drive; Ctrl-C saves runs/drive/episodes/ros-NNN
make ros-view                     # rqt_image_view on the camera
```

or the car, the driver and the recorder at once:
`make ros-launch ARGS='policy:=cnn checkpoint:=v1 record:=true'`. A recording comes
without labels; `make label`, then `make validate`, and it is training data.

| topic | type | |
|---|---|---|
| `/drive/image` | `sensor_msgs/Image` | the camera, rgb8, 25 Hz |
| `/drive/state` | `geometry_msgs/TwistStamped` | `linear.x`: speed in m/s; `angular.z`: steering position, -1..1 |
| `/drive/range` | `sensor_msgs/Range` | distance to the nearest thing ahead |
| `/drive/pose` | `geometry_msgs/Pose2D` | where the car is; only a simulator publishes it (the teacher needs it) |
| `/drive/cmd` | `geometry_msgs/Twist` | `linear.x`: throttle, `angular.z`: steer, both -1..1 |

| node | |
|---|---|
| [`sim`](../ros/src/drive_ros/drive_ros/sim_node.py) | `drive.sim.World` stepped at 50 Hz; brakes when no command has come for 0.5 s |
| [`driver`](../ros/src/drive_ros/drive_ros/drive_node.py) | `DriveLoop`, unchanged, on [`RosPlatform`](../ros/src/drive_ros/drive_ros/platform.py): the policy on its own thread, the supervisor last |
| [`teleop`](../ros/src/drive_ros/drive_ros/teleop_node.py) | a window with the camera; the pointer is the stick. Its commands go straight to the car, past the supervisor |
| [`recorder`](../ros/src/drive_ros/drive_ros/recorder_node.py) | the image, the state and the command, 10 times a second, into an episode |

The topics are the whole interface. A real car, or a CARLA bridge, that publishes the
sensors and obeys `/drive/cmd` takes the simulator's place, and the driver, the
recorder and the training pipeline do not change. Neither exists yet.

Each `make` target is its own container on the host's network, so the nodes find one
another; `ROS_LOCALHOST_ONLY=1` keeps them off the LAN. Windows (OpenCV, rqt, rviz2)
open on the host's X server.

## From here to a car

1. **Platform.** A node on the car that publishes the camera, the speed and the range
   and obeys `/drive/cmd`, as [`sim_node.py`](../ros/src/drive_ros/drive_ros/sim_node.py)
   does for the simulated one.
2. **Data.** Drive by hand with the recorder running, then label and validate. For a
   VLA, convert the episodes to a LeRobotDataset: state `[speed, steer]`, action
   `[steer, throttle]`.
3. **Fine-tune and export.** Fine-tune SmolVLA with LeRobot, then, with
   [jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla), `./export.sh`
   the checkpoint on the PC and `./run.sh` the bundle on the Jetson to get its real latency.
4. **Policy.** Load the bundle's backend and wrap it:
   `BackendPolicy(backend, chunk_size=50, noise_width=32, chunk_dt_s=1 / fps)` in a
   `ThreadedWorker`. This adapter is tested against a stub backend only; it has not
   run on a Jetson.

Not here yet: the real platform, a CARLA bridge, the LeRobot conversion, a fine-tuned
driving checkpoint. Driving by hand with `teleop` has been started and seen to open its
window, not driven for a recording. The simulator is flat and kinematic: no slip, no camera noise or exposure,
obstacles are painted on the ground. Treat its numbers as how the loop behaves, not as
how a car will.
