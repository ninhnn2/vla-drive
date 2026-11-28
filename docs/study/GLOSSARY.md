# Glossary

*[Tiếng Việt](GLOSSARY.vi.md)*

Terms as they are used in this repository. Each entry gives a definition, where the
idea lives in the code, and the chapter that teaches it.

## Driving stack

**Action** · A pair `[steer, throttle]`, both in $[-1, 1]$. Steer $+1$ is full left;
throttle $+1$ is full power and $-1$ full brake; there is no reverse.
[`drive/types.py`](../../drive/types.py) · [Ch. 1](01-system-architecture.md)

**Action chunk** · A policy's answer: $N$ future actions, a time step `dt`, and the
time of the observation they were computed from. `actions[i]` is meant for
`t_obs + i * dt`. [`ActionChunk`](../../drive/types.py#L41) · [Ch. 5](05-action-chunking-and-latency.md)

**Action chunking** · Having a policy predict a short sequence of future actions per
inference instead of one. It amortises a slow inference over many control ticks and
reduces compounding error in imitation learning. Introduced under this name by ACT
(Zhao et al. 2023). [Ch. 5](05-action-chunking-and-latency.md)

**Chunk executor** · The component that returns, every control tick, the action for
*now* from the newest chunk, fading it in over the previous one.
[`ChunkExecutor`](../../drive/control/executor.py#L11) · [Ch. 5](05-action-chunking-and-latency.md)

**Chunk exhaustion** · Reaching the end of a chunk before a newer one arrives. `sample`
returns `None` and the supervisor brakes. Avoided when the chunk's horizon is at least
twice the latency. [Ch. 5](05-action-chunking-and-latency.md)

**Chunk horizon** · The time a chunk covers: $(N - 1) \cdot dt$.
[`ActionChunk.horizon_s`](../../drive/types.py#L55)

**Control loop** · The code that runs at a fixed rate (50 Hz here) and sends a command
to the motors each time. [`DriveLoop.step`](../../drive/loop.py#L32) · [Ch. 2](02-data-flow-and-lifecycle.md)

**Control tick** · One iteration of the control loop: 20 ms.

**Decision** · The supervisor's output: the action that goes to the motors and the
reasons, if any, that it differs from the command. [`Decision`](../../drive/safety.py#L19)

**Episode** · One recorded drive: a directory with `episode.npz`, `meta.json` and,
once labelled, `labels.npz`. [Ch. 9](09-dataset-pipeline.md)

**Inference worker** · What runs the policy off the control loop, one inference at a
time. `SimulatedWorker` imposes a deterministic delay; `ThreadedWorker` uses a real
thread. [`drive/control/inference.py`](../../drive/control/inference.py) · [Ch. 2](02-data-flow-and-lifecycle.md)

**Latency** · The time from an observation to the arrival of the chunk computed from it.

**Latency compensation** · Reading a chunk at `now - t_obs` rather than from its first
action, so the actions meant for the time spent computing are skipped and every
action is played when it was meant to be. It corrects timing, not the age of the
information. [`ChunkExecutor.update`](../../drive/control/executor.py#L28) · [Ch. 5](05-action-chunking-and-latency.md)

**Observation** · What a policy sees at one instant: images, speed, steering position,
a task sentence and a timestamp (and, in the simulator only, the pose).
[`Observation`](../../drive/types.py#L27)

**Observation timestamp** · `obs.t`, the time the sensors were read; copied into the
chunk as `t_obs`. Over ROS 2 it is the image message's header stamp, so delivery time
counts as latency. [Ch. 2](02-data-flow-and-lifecycle.md)

**Plan** · In the data pipeline, the label for one frame: 16 future actions, 0.1 s
apart. [`drive/learn/label.py`](../../drive/learn/label.py#L18) · [Ch. 9](09-dataset-pipeline.md)

**Plan age** · `now - t_obs`: how old the information behind the action being played
is. With the policy running back to back it sweeps $[L, 2L)$.
[`ChunkExecutor.age_s`](../../drive/control/executor.py#L56) · [Ch. 5](05-action-chunking-and-latency.md)

**Platform** · What the control loop needs from a car: a clock, a state, `observe`,
`range_m` and `apply`. Implemented by the simulator and by the ROS 2 adapter.
[`Platform`](../../drive/platform.py#L14) · [Ch. 1](01-system-architecture.md)

**Policy** · Anything that turns an observation into an action chunk.
[`Policy`](../../drive/policy/base.py#L6) · [Ch. 4](04-driving-policies.md)

**Project** · A directory holding `episodes/`, `models/` and `reports/`; `runs/drive`
by default. [`Project`](../../drive/learn/project.py#L18)

**Safety supervisor** · The last component before the motors. It sees only the
command, the measured speed, its own range sensor and the plan's age, and may limit
or replace the command with a brake.
[`SafetySupervisor`](../../drive/safety.py#L24) · [Ch. 7](07-safety-supervisor.md)

**Stale plan** · A plan older than `max_plan_age_s` (0.6 s), or no plan at all. The
supervisor brakes, with the reason `stale`.

**Stale-plan watchdog** · The supervisor's rule that enforces the above. It must
exceed twice the latency of the model in use. [Ch. 5](05-action-chunking-and-latency.md)

**Teacher** · A policy with privileged information (here, the map and the car's exact
pose) used to produce demonstrations: the pure-pursuit policy. [Ch. 4](04-driving-policies.md)

## Vehicle and simulation

**Car frame** · Coordinates relative to the car: `forward` and `left` in metres from
the rear axle. [Ch. 3](03-vehicle-simulation.md)

**Curvature** · $\kappa = 1/R$, the reciprocal of the turning radius, in m⁻¹; positive
turns left here.

**Inverse perspective map** · For each image pixel, the point on the ground it looks
at. The camera's `forward_m` and `left_m` arrays.
[`Camera`](../../drive/sim/render.py#L48) · [Ch. 3](03-vehicle-simulation.md)

**Kinematic bicycle model** · A vehicle model that merges each axle into one wheel and
assumes no tyre slip, so the path follows from geometry alone:
$\dot\psi = \frac{v}{L}\tan\delta$. Accurate at low lateral acceleration.
[`step_vehicle`](../../drive/sim/vehicle.py#L10) · [Ch. 3](03-vehicle-simulation.md)

**Lateral offset** · Signed distance from the track's centreline, positive to the left
of the direction of travel. [`Track.project`](../../drive/sim/track.py#L69)

**Look-ahead distance** · How far along the path pure pursuit aims; here
$\text{clip}(0.4 + 0.35v, 0.5, 1.4)$ m.

**Pose** · Position and heading, $(x, y, \psi)$. Known exactly only in a simulator.

**Pure pursuit** · A path-tracking method: steer along the circle that passes through
the rear axle, tangent to the heading, and through a point a look-ahead distance along
the path. $\kappa = 2y/d^2$.
[`pursuit_steer`](../../drive/policy/pure_pursuit.py#L19) · [Ch. 4](04-driving-policies.md)

**Range sensor** · A forward distance sensor, the supervisor's own input.
[`World.range_m`](../../drive/sim/world.py#L53)

**Servo lag** · The steering does not reach a commanded position instantly. Modelled
as a first-order lag with time constant 0.08 s; the wheels' actual position is part of
the state. [Ch. 3](03-vehicle-simulation.md), [Ch. 6](06-control-theory.md)

**Wheelbase** · Distance between the front and rear axles, $L = 0.26$ m.

**Yaw** · Heading angle $\psi$, in radians, counter-clockwise from the world's $+x$ axis.

## Control

**Closed-loop control** · Control that measures the result and corrects for the
difference from what is wanted. Here the loop is closed at the inference rate (a new
plan from each observation) and, for the supervisor, at 50 Hz. [Ch. 6](06-control-theory.md)

**Open-loop control** · Applying inputs decided in advance without measuring the
result. Playing a chunk is open-loop until the next chunk arrives.

**Dead time (delay)** · A lag between a measurement and the action based on it. It
adds phase lag $\omega\tau$ and limits how aggressive a feedback loop can be.

**Delay margin** · The largest delay a feedback loop tolerates before becoming
unstable: phase margin divided by crossover frequency. [Ch. 6](06-control-theory.md)

**Feed-forward** · A control term computed from the target and a model of the plant
rather than from the error; the `0.1 * speed` in the throttle law.

**Overshoot** · Passing the target before settling.

**RMS tracking error** · $\sqrt{\frac1n\sum e_k^2}$ of the lateral offset over a
drive; `lateral_rms_m`. [`Metrics.summary`](../../drive/episode.py#L60) · [Ch. 11](11-model-evaluation.md)

**Stability** · A disturbance dies out instead of growing.

**Zero-order hold** · Holding a command constant between samples; what
`apply(action, dt)` does.

## Learning

**Behavior cloning (BC)** · Imitation learning by supervised learning: fit a model to
map the expert's observations to the expert's actions. [Ch. 8](08-imitation-learning.md)

**Closed-loop evaluation** · Judging a policy by letting it drive, so its own actions
determine what it sees next. `evaluate`. [Ch. 11](11-model-evaluation.md)

**Compounding error** · In sequential decisions, an error changes the next input,
making further errors more likely; cost can grow with the square of the horizon.

**Covariate shift** · A change in the distribution of a model's *inputs* between
training and use, with the correct output for each input unchanged. In driving: the
learner reaches states the expert never visited. [Ch. 8](08-imitation-learning.md)

**DAgger** · Dataset Aggregation: run the learner, have the expert label the states it
visits, add them to the data, retrain, repeat. Not implemented here.

**DART** · Disturbances for Augmenting Robot Trajectories: inject noise into the
expert's *executed* actions while recording, and keep the expert's *intended* actions
as labels. What `collect` does. [`collect`](../../drive/learn/collect.py#L120) · [Ch. 8](08-imitation-learning.md)

**Data augmentation** · Transforming recorded samples to enlarge the dataset: here,
mirroring (with the steering signs flipped) and brightness.
[`augment`](../../drive/learn/train.py#L20)

**Distribution shift** · Any difference between the data a model was trained on and
the data it meets in use.

**Expert demonstration** · A recorded example of the desired behaviour.

**Imitation learning** · Learning a policy from demonstrations rather than from a
reward. [Ch. 8](08-imitation-learning.md)

**Offline (open-loop) evaluation** · Comparing a model's outputs with the expert's on
recorded frames. `val_steer_error`. It does not predict driving ability.

**Ornstein–Uhlenbeck process** · Random noise that is correlated in time and reverts
to zero; used to disturb the steering smoothly.
[`Disturbance`](../../drive/learn/collect.py#L31)

**Recovery behaviour** · What to do when off the intended path. Absent from
demonstrations of perfect driving.

**Smooth L1 loss** · Quadratic for small errors, linear for large ones; the training
loss, with $\beta = 0.05$.

## VLA and deployment

**Action expert** · In VLAs with a continuous action head, the network that generates
the action chunk from the backbone's features, usually by iterative denoising.

**Attention sink** · A token that receives much attention without carrying information
about the input, e.g. padding or a corner token that is the same in every frame.
Transformers park attention there. It is why attention is not importance.
[Ch. 17](17-debugging-the-model.md#where-does-it-look-toolssee_vlapy)

**Backend** · In jetson-orin-nano-vla, an object that runs one model's inference:
`infer(observation)` returns a result with a `.chunk`. `BackendPolicy` wraps one.
[`drive/policy/vla.py`](../../drive/policy/vla.py) · [Ch. 13](13-vla-and-smolvla.md)

**Fine-tuning** · Continuing the training of a pre-trained model on data from the
target task.

**Flow matching** · A way of training a generative model to transform noise into data
by integrating a learned velocity field over a few steps; SmolVLA's action head.

**Image token** · One vector the vision encoder makes from a region of the image. In
SmolVLA a 512×512 input becomes 8×8 = 64 tokens. With this camera, 16 of them are
padding. [Ch. 17](17-debugging-the-model.md#how-the-image-becomes-tokens)

**Jetson Orin Nano** · NVIDIA's small embedded computer with an integrated GPU and
8 GB of memory shared between CPU and GPU; the intended on-car computer.
[Ch. 14](14-jetson-deployment.md)

**LeRobot** · Hugging Face's open-source robotics library: datasets, policies
(including SmolVLA), training and robot tooling.

**LeRobotDataset** · LeRobot's dataset format: Parquet tables, video for cameras, and
metadata. `./drive.sh export` writes the episodes in it.
[`drive/learn/export.py`](../../drive/learn/export.py) · [Ch. 13](13-vla-and-smolvla.md)

**Occlusion sensitivity** · Hiding one region of the input at a time and measuring how
far the output moves. It shows what the answer depends on, unlike attention.
[`tools/see_vla.py`](../../tools/see_vla.py) · [Ch. 17](17-debugging-the-model.md#two-ways-to-ask-where)

**ONNX** · An open file format for neural networks, used as the hand-over between
training frameworks and inference engines.

**Real-time chunking (RTC)** · Generating each new chunk conditioned on the actions
already committed for the time the inference will take, so consecutive chunks agree
(Black, Galliker and Levine 2025). Not implemented here.

**SmolVLA** · A compact (about 450 M parameters) open VLA from the LeRobot team, built
on a SmolVLM2 backbone with a flow-matching action expert, producing chunks of 50
actions. [Ch. 13](13-vla-and-smolvla.md)

**Temporal ensembling** · Averaging the overlapping predictions of successive chunks
(ACT). Not implemented here; the 0.1 s cross-fade is a simpler relative.

**TensorRT** · NVIDIA's inference optimiser and runtime. It compiles a network into
an engine specialised for one GPU, precision and input shape.
[Ch. 14](14-jetson-deployment.md)

**VLA (vision-language-action model)** · A model that takes camera images, a task in
natural language and the robot's state, and outputs robot actions; usually a
pre-trained vision-language model with an action output added. [Ch. 13](13-vla-and-smolvla.md)

**VLM (vision-language model)** · A model that takes images and text and produces text.

## ROS 2

**ROS 2** · Robot Operating System 2: middleware and tools for building a robot as
communicating processes. The repository uses the Humble distribution.
[Ch. 12](12-ros2-integration.md)

**Node** · A named participant in the ROS graph, doing one job.

**Topic** · A named, typed channel on which nodes publish and subscribe.

**Message** · The typed data sent on a topic, for example `sensor_msgs/Image`.

**Publisher / subscriber** · A node's handles for sending to and receiving from a topic.

**Parameter** · A named configuration value of a node.

**Launch file** · A script that starts a set of nodes with their parameters.

**QoS (quality of service)** · Delivery settings of a publisher or subscription;
here only a queue depth of 1.

**Command watchdog** · On the car's side: brake if no command has arrived for 0.5 s.
[`sim_node.py`](../../ros/src/drive_ros/drive_ros/sim_node.py#L20) · [Ch. 7](07-safety-supervisor.md), [Ch. 12](12-ros2-integration.md)

**`RosPlatform`** · The adapter that presents the `/drive/*` topics as a `Platform`.
[`platform.py`](../../ros/src/drive_ros/drive_ros/platform.py#L13)
