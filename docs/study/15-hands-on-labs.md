# 15 · Hands-on labs

Fifteen exercises, in order of dependence. Each takes 15 to 45 minutes, except Lab 15
(about an hour, most of it waiting).

## Before you start

```bash
make build        # once: the container (about 9 GB). Needs Docker and the NVIDIA container toolkit
make test         # 82 tests should pass
```

- `make <target>` runs in the container. Targets that open a window (`sim`, `browse`,
  `collect --show`, `ros-sim`, `ros-teleop`) need an X display.
- `./drive.sh …` runs the same commands directly on a PC with numpy and OpenCV, with
  no window unless you pass `--show`. Labs 1–9 can be done this way; labs 10–13 need
  the container.
- Results marked *expected* were measured for this guide on 2026-11-16. Pure-pursuit
  runs are deterministic: yours should match exactly. Runs that render the camera can
  differ in the last digit.
- Write your prediction **before** each run. The value of a lab is the gap between
  what you expected and what happened.
- Keep experiments out of `runs/drive`: use `--project runs/<name>`.

Record progress in the [study tracker](STUDY-TRACKER.md).

---

## Lab 1: Run the simulator

**Objective.** See the whole stack run, and learn to read the HUD and the result.
**Prerequisites.** [Chapter 1](01-system-architecture.md).
**Files.** [`drive/cli.py`](../../drive/cli.py), [`drive/episode.py`](../../drive/episode.py), [`drive/viewer.py`](../../drive/viewer.py).

```bash
make sim                                   # q quits, e toggles the emergency stop
./drive.sh sim --out runs/lab1/result.json # headless, with the result as JSON
./drive.sh sim --video runs/lab1/drive.mp4 # the HUD as a video
```

**Expected.**
```
pure-pursuit on peanut, 100.0 ms per inference (300 inferences), latency compensation on, safety on
  1.99 laps in 30.0 s at 1.46 m/s
  off the centerline: rms 1.8 cm, max 3.4 cm, off the road 0.0 % of the time
  safety overrode the policy: stale 0.1 s, steer_rate 0.06 s
```

**Observe.** Left: the car's camera. Right: the map, the car, and in cyan the plan in
the current chunk. Under them: "policy in control" in green, or the supervisor's
reasons in red. Press `e` and watch the line change.

The **debug panel** at the bottom shows what the policy outputs
([`Hud.debug_panel`](../../drive/viewer.py#L114)):

- the top line: the action the chunk gives for *now* (`model:`) and what the
  supervisor sent to the motors (`-> motors:`), in red with its reasons when they differ;
- left: the whole chunk in use, steer in cyan and throttle in orange, against time
  since its observation. The grey band is the part skipped because it was meant for
  the time the policy spent computing; the white line is the action being played;
- right: the last 5 s. Thick lines are what the model asked for, thin lines what the
  motors got; red ticks mark the ticks the supervisor overrode.

To keep every tick for later, add `--log runs/lab1/actions.csv`: one row per 20 ms tick
with the chunk's action, the motors' action, the plan's age and the supervisor's
reasons.

**Questions.**
1. Why 300 inferences in 30 s?
2. Where do `stale 0.1 s` and `steer_rate 0.06 s` come from?
3. With `--show`, `latency_ms_mean` is a little over 100. Why?

**Challenge.** Run `--track trefoil --reverse`. Does the teacher's tracking change? Should it?

---

## Lab 2: Understand the pure-pursuit policy

**Objective.** Connect the pursuit formula to the code and to the car's behaviour.
**Prerequisites.** [Chapter 4](04-driving-policies.md), section 1.
**Files.** [`drive/policy/pure_pursuit.py`](../../drive/policy/pure_pursuit.py).

```bash
python3 -c "
from drive.policy.pure_pursuit import pursuit_steer
from drive.config import VehicleParams
car = VehicleParams()
for target in [(1.0, 0.0), (1.0, 0.3), (1.0, -0.3), (0.5, 0.3), (0.1, 1.0)]:
    print(target, round(pursuit_steer(*target, car), 3))"

./drive.sh sim --speed 0.8
./drive.sh sim --speed 1.5
./drive.sh sim --speed 2.5
```

**Expected.** `(1.0, 0.3)` → 0.338 (worked in chapter 4); `(0.5, 0.3)` and `(0.1, 1.0)`
→ 1.0 (clipped). At `--speed 0.8`: 0.78 m/s, rms 0.9 cm. At `--speed 2.5`: mean speed
1.76 m/s, rms 3.5 cm, and `speed 5.5 s` among the supervisor's reasons.

**Observe.** How rms changes with speed, and that the mean speed is well under the target at 2.5.

**Questions.**
1. Compute `pursuit_steer(1.0, 0.3)` by hand.
2. Why is the mean speed 1.76 and not 2.5? Name both mechanisms that limit it.
3. Why does rms grow with speed for a controller with constant damping?

**Challenge.** In a scratch copy, change the look-ahead to a constant 0.5 m and to a
constant 1.4 m (construct `PurePursuitPolicy` yourself and drive it with
`episode.build`'s pattern). Relate what you see to $\omega_n = \sqrt2\,v/\ell$.

---

## Lab 3: Run the lane-camera policy

**Objective.** See a policy that uses only the image, and its limits.
**Prerequisites.** [Chapter 4](04-driving-policies.md), section 2.
**Files.** [`drive/policy/lane_camera.py`](../../drive/policy/lane_camera.py), [`drive/sim/render.py`](../../drive/sim/render.py).

```bash
make sim ARGS='--policy lane-camera'
./drive.sh sim --policy lane-camera
./drive.sh sim --policy lane-camera --track trefoil
python3 -c "
from drive.sim import World
from drive.policy.lane_camera import road_mask
import cv2, numpy as np
w = World(); img = w.image()
cv2.imwrite('runs/lab3_image.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imwrite('runs/lab3_mask.png', road_mask(img).astype(np.uint8) * 255)"
```

**Expected.** On the peanut: 1.2 laps at 0.86 m/s, rms 7.3 cm, max 19.2 cm, never off the road.

**Observe.** The cyan plan is a constant-curvature arc (one action repeated). Open the
two PNGs: the mask is the road, including the sky.

**Questions.**
1. The mask marks the sky as "road". Why does that not break the policy?
2. Why is its tracking four times worse than the teacher's?
3. What in `road_mask` would fail first on a real floor?

**Challenge.** Find the highest `--speed` at which it completes a lap without leaving the road.

---

## Lab 4: Inject inference latency

**Objective.** Feel what latency does, and meet the watchdog.
**Prerequisites.** [Chapters 2](02-data-flow-and-lifecycle.md) and [5](05-action-chunking-and-latency.md).
**Files.** [`drive/control/inference.py`](../../drive/control/inference.py), [`drive/safety.py`](../../drive/safety.py).

```bash
for L in 0 100 200 250 400 650; do ./drive.sh sim --latency-ms $L | sed -n '1,2p;4p'; done
make sim ARGS='--latency-ms 400'
```

**Expected.** `stale`: 0.0, 0.1, 0.2, 0.3, 13.9, 30.0 s. Laps: 2.00, 1.99, 1.98, 1.97, 0.14, 0.00.

**Observe.** At 400 ms the car lurches: it drives, brakes, drives. In the HUD, "plan
age" climbs past 600 ms and the status turns red each cycle.

**Questions.**
1. Why is the car braked for almost exactly half the time at 400 ms?
2. Why does it never move at 650 ms?
3. What is the largest latency the default watchdog permits without braking?

**Challenge.** Add `--jitter-ms 40` to the 250 ms run. Predict whether `stale`
appears, given that jitter makes some inferences slower than 300 ms.

---

## Lab 5: Compare compensated and uncompensated chunks

**Objective.** Reproduce the project's central measurement.
**Prerequisites.** [Chapter 5](05-action-chunking-and-latency.md).
**Files.** [`drive/control/executor.py`](../../drive/control/executor.py), [`drive/types.py`](../../drive/types.py).

```bash
./drive.sh sweep --max-plan-age-ms 1500 --latencies 0,50,100,200,250,300,400
make sim ARGS='--latency-ms 250 --max-plan-age-ms 1500 --no-latency-compensation'
make sim ARGS='--latency-ms 250 --max-plan-age-ms 1500'
```

**Expected** (rms / max cm):

| ms | compensated | not compensated |
|---:|---:|---:|
| 0 | 1.8 / 3.5 | 1.6 / 3.0 |
| 100 | 1.8 / 3.4 | 0.7 / 2.2 |
| 200 | 2.0 / 3.6 | 3.8 / 8.8 |
| 250 | 3.1 / 6.6 | 10.0 / 23.3 |
| 300 | 20.2 / 31.0 | 27.2 / 54.4 |
| 400 | 65.2 / 123.9 | 49.0 / 89.0 |

**Observe.** In the two windows at 250 ms: the uncompensated car turns in late on
every bend.

**Questions.**
1. Why is "not compensated" better at 100 ms?
2. Why is `--max-plan-age-ms 1500` needed for this experiment?
3. Compensation plays every action at its intended time. Why does the car still fail at 400 ms?

**Challenge.** Run the test `ExecutorTests` (`python3 -m unittest tests.test_drive.ExecutorTests -v`)
and, for each test, state the property of the executor it pins down.

---

## Lab 6: Test safety with an obstacle

**Objective.** Watch the supervisor overrule a policy that is wrong.
**Prerequisites.** [Chapter 7](07-safety-supervisor.md).
**Files.** [`drive/safety.py`](../../drive/safety.py), [`drive/sim/world.py`](../../drive/sim/world.py).

```bash
make sim ARGS='--obstacle 8'
./drive.sh sim --obstacle 8
./drive.sh sim --obstacle 8 --speed 0.8
```

**Expected.**
```
  0.33 laps in 30.0 s at 0.24 m/s
  obstacles: not hit, closest 0.3 m
  safety overrode the policy: obstacle_brake 0.22 s, obstacle_slow 0.22 s, obstacle_stop 24.48 s, stale 0.1 s, steer_rate 0.06 s
```

**Observe.** The cyan plan passes straight through the cone: the teacher does not
know it is there. The status line shows `obstacle_slow`, then `obstacle_brake`, then
`obstacle_stop`.

**Questions.**
1. At 1.5 m/s, at what range does `obstacle_slow` begin? `obstacle_brake`?
2. Why does the car stop at 0.30 m and not earlier?
3. Why do `slow` and `brake` alternate?

**Challenge.** Put two cones: `--obstacle 8 --obstacle 15`. What happens, and why can
the car never reach the second?

---

## Lab 7: Disable safety and analyse the result

**Objective.** Know what the supervisor was preventing.
**Prerequisites.** Lab 6.
**Files.** [`drive/loop.py`](../../drive/loop.py#L45).

```bash
./drive.sh sim --obstacle 8 --no-safety
./drive.sh sim --no-safety                       # no obstacle
./drive.sh sim --latency-ms 650 --no-safety
```

**Expected.** With the cone: `obstacles: HIT, closest 0.0 m`, 1.99 laps at 1.46 m/s
(the simulated car drives through). Without a cone: the same tracking as with safety.

**Observe.** The first line now says `safety OFF`, and there is no "safety overrode" line.

**Questions.**
1. In a clean run, safety on and off give the same tracking. What does that tell you about the supervisor's cost?
2. With `--no-safety`, what goes to the motors before the first chunk arrives? (Read the `else` branch in `DriveLoop.step`.)
3. At 650 ms without safety the car moves. Is that better?

**Challenge.** List every protection that `--no-safety` removes, from the rule table in chapter 7.

---

## Lab 8: Collect training episodes

**Objective.** Produce a dataset and understand what is in it.
**Prerequisites.** [Chapters 8](08-imitation-learning.md) and [9](09-dataset-pipeline.md).
**Files.** [`drive/learn/collect.py`](../../drive/learn/collect.py), [`drive/recorder.py`](../../drive/recorder.py).

```bash
make collect ARGS='--project runs/lab8 --episodes 2 --show'      # watch two drives being recorded
./drive.sh collect --project runs/lab8 --name more --seed 1 --episodes 6 --duration-s 20
./drive.sh status --project runs/lab8
```

Then read one episode with the snippet in [chapter 9](09-dataset-pipeline.md#reading-one-episode).

**Expected.** Two episodes of 300 frames, then six of 200. `status` reports 8
episodes, 1800 frames, all labelled by the teacher, and suggests validating and training.

**Observe.** In the window: the magenta trail wanders across the road while the cyan
plan always heads back to the centre. The bottom line shows what the teacher wanted
against what was applied.

**Questions.**
1. Which array holds the disturbed steering, and which the clean plan?
2. Why does the second command need `--name more --seed 1`?
3. How many frames is 6 episodes of 20 s, and why?

**Challenge.** Collect with `--steer-noise 0.5`. How far does the car stray? Does it
leave the road? Is such data still useful?

---

## Lab 9: Validate and browse a dataset

**Objective.** Judge a dataset before training on it.
**Prerequisites.** Lab 8.
**Files.** [`drive/learn/validate.py`](../../drive/learn/validate.py), [`drive/learn/browse.py`](../../drive/learn/browse.py).

```bash
./drive.sh validate --project runs/lab8; echo "exit $?"
make browse ARGS='--project runs/lab8'                       # d/a frame, n/p episode, space, q
make browse ARGS="--project runs/lab8 --match 'more-*'"
./drive.sh browse --project runs/lab8 --every 5 --video runs/lab8/browse.mp4

./drive.sh collect --project runs/lab9 --steer-noise 0 --episodes 3 --duration-s 10
./drive.sh validate --project runs/lab9
```

**Expected.** `runs/lab8`: `dataset: ok` with one warning (1800 frames is little).
`runs/lab9`: the "perfect driving only" warning as well.

**Observe.** In `browse`, the cyan path should follow the road ahead in every frame,
bending back towards the centre when the car is off it.

**Questions.**
1. What would a *bad* label look like in `browse`?
2. Why does `validate` say "perfect driving only" for `runs/lab9`? Which two arrays does it compare?
3. Which problems does `validate` FAIL on, and which does it only WARN about? Why that split?

**Challenge.** Do the "break a dataset" experiment of chapter 9.

---

## Lab 10: Train a CNN

**Objective.** Train a network and read its training report.
**Prerequisites.** [Chapter 10](10-model-training.md). Needs the container.
**Files.** [`drive/learn/train.py`](../../drive/learn/train.py), [`drive/learn/model.py`](../../drive/learn/model.py).

```bash
make collect  ARGS='--project runs/lab10'            # 40 episodes, about 35 s
make validate ARGS='--project runs/lab10'
make train    V=v1 ARGS='--project runs/lab10'       # about 20 s on a laptop GPU
make status   ARGS='--project runs/lab10'
python3 -c "
import json; r = json.load(open('runs/lab10/models/v1.json'))
print(r['frames'], r['validation_frames'], r['parameters'], r['device'], r['seconds'])
for h in r['history']: print(h)"
```

**Expected.** 10 800 training frames, 1 200 validation frames, 323 124 parameters.
With the default seed, a final validation steer error near 0.024 (*repo* value for the
same settings: 0.0242).

**Observe.** Both losses fall every epoch; the validation loss stays above the
training loss.

**Questions.**
1. Why 10 800 and 1 200?
2. Derive 323 124.
3. Run `make train V=v1 …` again. What happens, and why is that the right behaviour?

**Challenge.** Train with `--seed 1` as `v1s1`. How different is the final validation
error? Keep both for Lab 11.

---

## Lab 11: Evaluate on different tracks

**Objective.** Judge a model by driving it, and compare the verdict with the offline number.
**Prerequisites.** Lab 10, [chapter 11](11-model-evaluation.md).
**Files.** [`drive/learn/evaluate.py`](../../drive/learn/evaluate.py).

```bash
make evaluate V="v1 pure-pursuit" ARGS='--project runs/lab10'
make sim ARGS='--project runs/lab10 --policy cnn --checkpoint v1 --track trefoil'

# the counter-example
make collect  ARGS='--project runs/lab11 --steer-noise 0'
make train    V=v1 ARGS='--project runs/lab11'
make evaluate V=v1 ARGS='--project runs/lab11'
```

**Expected.** The disturbed-data model: 6/6 (*repo*: 0.84 m/s, 1.3 cm rms). The
perfect-data model: a lower validation steer error and 0/6 (*repo*: 0.015, off the
road 62–85 % of the time).

**Observe.** Watch the perfect-data model in a window:
`make sim ARGS='--project runs/lab11 --policy cnn --checkpoint v1'`. Note the moment
it first drifts, and that it never comes back.

**Questions.**
1. Which model has the better validation error? Which drives?
2. The network was trained on the peanut only. What does passing on the trefoil show, and what does it not show?
3. Why does the network drive slower than its teacher?

**Challenge.** Write the recovery test of chapter 11 and find the largest starting
offset from which your model returns to the centreline.

---

## Lab 12: Run the ROS 2 simulation

**Objective.** Run the same loop across processes and inspect it with ROS tools.
**Prerequisites.** [Chapter 12](12-ros2-integration.md).
**Files.** [`ros/src/drive_ros/drive_ros/`](../../ros/src/drive_ros/drive_ros).

```bash
make ros-build                         # once
# terminal 1
make ros-sim
# terminal 2
make ros-drive                         # or: make ros-drive POLICY=cnn V=v1
# terminal 3
make shell
  ros2 node list
  ros2 topic list -t
  ros2 topic hz /drive/image           # about 25
  ros2 topic hz /drive/cmd             # about 50
  ros2 topic echo /drive/state --once
  ros2 param list /drive_driver
```

**Expected.** Nodes `/drive_sim` and `/drive_driver`; five `/drive/*` topics; the
driver logs a status line every 2 s (speed, inference count, any overrides).

**Observe.** Stop the driver with Ctrl-C while the car is moving. The car brakes by
itself about half a second later.

**Questions.**
1. The driver logs nothing about braking when you stop it. Who stopped the car?
2. What happens if you start the driver with `TRACK=oval` against a simulator on the peanut? Why only for the teacher?
3. `make ros-launch ARGS='obstacles:=8'`: which node knows about the cone, and how does the driver find out?

**Challenge.** `make ros-drive ARGS='-p latency_ms:=300'`. What is the difference
between this `latency_ms` and the simulator's `--latency-ms`?

---

## Lab 13: Record teleoperation data

**Objective.** Record a drive over ROS 2 and turn it into training data.
**Prerequisites.** Lab 12.
**Files.** [`teleop_node.py`](../../ros/src/drive_ros/drive_ros/teleop_node.py), [`recorder_node.py`](../../ros/src/drive_ros/drive_ros/recorder_node.py), [`label.py`](../../drive/learn/label.py).

> **Status.** Recording a drive made by the *driver* node is tested
> (`NodeTests.test_teacher_drives_the_simulated_car_and_the_drive_is_recorded`).
> Recording a drive made *by hand* with teleop is **[UNTESTED]**: `docs/driving.md`
> says teleop "has been started and seen to open its window, not driven for a
> recording". Part B is therefore an experiment in the real sense. Note what you find.

**Part A: record the teacher.**
```bash
# terminal 1                         # terminal 2                    # terminal 3
make ros-sim TRACK=oval              make ros-drive TRACK=oval       make ros-record ARGS='-p project:=runs/lab13'
# after 40 s, Ctrl-C in terminal 3: "N frames (M s) saved to runs/lab13/episodes/ros-000"
./drive.sh status   --project runs/lab13        # 1 episode, unlabelled → next: label
./drive.sh label    --project runs/lab13
./drive.sh validate --project runs/lab13
make browse ARGS='--project runs/lab13'
```

**Part B: record yourself.** Replace terminal 2 with `make ros-teleop`. Hold the left
mouse button in the window: left/right of the middle steers, above the middle is
throttle, below is brake. Record a lap, then label, validate and browse it.

**Expected (A).** About 10 frames per second of driving; after `label`, the last 15
frames are not usable; `validate` reports one episode only as a failure for training
(a second episode is needed).

**Observe.** In `browse`, the plan is now the actions that *followed*, not a
teacher's intention. For your own drive, look at how it compares with the road.

**Questions.**
1. Why does the recorder wait for the first command?
2. Your hand-driven episode: is it "perfect driving", "disturbed driving", or neither? What would a network learn from it?
3. Teleop commands bypass the supervisor. What does that mean for the recorded `actions` compared with a driver-node recording?

**Challenge.** Record deliberate recoveries: drive towards an edge with the recorder
off, start it, and record only the return. Why might this be worth more than laps?

---

## Lab 14: Analyse the VLA deployment architecture

**Objective.** Separate what exists from what is planned, and plan the next step.
**Prerequisites.** [Chapters 13](13-vla-and-smolvla.md) and [14](14-jetson-deployment.md).
**Files.** [`drive/policy/vla.py`](../../drive/policy/vla.py), [`tests/test_drive.py`](../../tests/test_drive.py#L367).

A reading and design lab; no model is run.

```bash
sed -n 1,47p drive/policy/vla.py
python3 -m unittest tests.test_drive.PolicyTests.test_backend_policy_adapts_a_vla_backend -v
grep -n '"vla"\|BackendPolicy' -r drive ros --include=*.py
```

**Tasks.**
1. Draw the path episode → LeRobotDataset → fine-tune → export → TensorRT → Jetson →
   `BackendPolicy`. Colour each box: exists and has run / exists against a stub / not here.
   (Since Lab 15 the first two arrows have run on the PC; check what is left.)
2. Write a fake backend whose chunk steers 0.2 left at 30 % throttle, wrap it in
   `BackendPolicy`, and drive the simulator with it:
   ```python
   from types import SimpleNamespace
   import numpy as np
   from drive.control import SimulatedWorker
   from drive.loop import DriveLoop
   from drive.policy.vla import BackendPolicy
   from drive.safety import SafetySupervisor
   from drive.sim import World

   class Fake:
       def infer(self, obs):
           chunk = np.zeros((50, 32), np.float32); chunk[:, 0], chunk[:, 1] = 0.2, 0.3
           return SimpleNamespace(chunk=chunk)

   world = World()
   loop = DriveLoop(world, SimulatedWorker(BackendPolicy(Fake(), chunk_size=50, noise_width=32, chunk_dt_s=1 / 30), 0.108),
                    SafetySupervisor())
   for _ in range(250):
       tick = loop.step()
   print(world.state, tick.decision)
   ```
   What does the car do? What does this prove, and what does it not?
3. For a model with 108 ms latency: state the plan-age range, whether the default
   watchdog allows it, and how far the car travels on its oldest plan at 1.5 m/s.
4. Repeat task 3 for 335 ms and 643 ms. For each, choose a speed and a watchdog limit
   you would defend, citing the sweeps in chapter 5.
5. Decide what goes in the `action` column when converting `collect` episodes to a
   LeRobotDataset, and justify it (chapter 13).

**Questions.**
1. `--policy vla` now loads a LeRobot checkpoint. What would it take to make it load a TensorRT bundle instead, and where would that code go?
2. Why is running the fine-tuned model in the simulator, on the PC, the right step before the Jetson?
3. Which of the remaining tasks in chapter 14 can be done without hardware?

---

## Lab 15: Fine-tune SmolVLA on the driving data, on this laptop

**Objective.** Take the project's own episodes all the way to a fine-tuned SmolVLA
that drives the simulated car, on one laptop GPU, and judge it the way the CNN is
judged.
**Prerequisites.** Labs 8–11, [chapter 13](13-vla-and-smolvla.md). An NVIDIA GPU with
8 GB or more, about 15 GB of free disk **on a drive that has room**, and network for
the first download (about 4 GB of packages, 3 GB of model).
**Files.** [`drive/learn/export.py`](../../drive/learn/export.py),
[`drive/policy/lerobot_backend.py`](../../drive/policy/lerobot_backend.py),
[`drive/policy/vla.py`](../../drive/policy/vla.py).

**Status.** Every command below was run for this guide on 2026-11-17 on the laptop
this repository lives on (RTX 4060 Laptop GPU, 8 GB, driver 535) with LeRobot 0.5.2
(the local clone at `/home/fanning/data1/AI_SOURCE/VLA/lerobot`, commit `3dd19d043`).
The numbers under *expected* are from those runs. The Colab part at the end was **not**
run.

### Step 0: a LeRobot environment, once (about 30 minutes, mostly downloading)

The project's container has PyTorch but not LeRobot, and LeRobot needs Python 3.12. So
LeRobot gets its own virtual environment, outside the repository. Put it, and every
cache, on a disk with space: on this laptop `/` had only 21 GB free, and both uv and
Hugging Face cache to `~/.cache` (on `/`) by default.

```bash
D=/home/fanning/data1/AI_SOURCE/smolvla-local          # everything in one folder; delete it to undo
mkdir -p $D && cd $D
export UV_CACHE_DIR=$D/uv-cache UV_PYTHON_INSTALL_DIR=$D/python HF_HOME=$D/hf
uv venv --python 3.12 .venv
uv pip install --python .venv "/home/fanning/data1/AI_SOURCE/VLA/lerobot[smolvla,dataset,training]"
.venv/bin/python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
```

*Expected:* `2.11.0+cu128 True 0.5.2` and a 7.4 GB `.venv`. The CUDA 12.8 build of
PyTorch runs on driver 535 (CUDA 12.2) here.

The extras are `smolvla,dataset,training`. The Colab notebook writes `train`, which
does not exist in LeRobot 0.5.2.

LeRobot installs `opencv-python-headless`, which cannot open a window. For `--show`
(Step 3), swap it for the windowed build of the same version, once:

```bash
uv pip uninstall --python .venv opencv-python-headless
uv pip install --python .venv "opencv-python==4.13.0.92"     # the version the headless one was
```

**Every later shell** needs the cache location again, or the model is downloaded to
`/` a second time:

```bash
export D=/home/fanning/data1/AI_SOURCE/smolvla-local HF_HOME=/home/fanning/data1/AI_SOURCE/smolvla-local/hf
export V=$D/.venv/bin/python                           # the LeRobot environment's Python
cd /home/fanning/data1/AI_SOURCE/vla-drive && export PYTHONPATH=.
```

With `PYTHONPATH=.`, `$V -m drive …` runs this repository's CLI (the same commands as
`./drive.sh`) inside the LeRobot environment.

### Step 1: export the episodes (seconds)

```bash
./drive.sh validate                  # the dataset must be labelled and valid first
$V -m drive export                   # -> runs/drive/lerobot
```

*Expected* (82 episodes in `runs/drive`): `24876 frames in 82 episodes at 10 Hz`, a 35 MB
dataset, and the full `lerobot-train` command printed at the end. Run it again and it
refuses ("exists"): a dataset is written once.

Look at what was written:

```bash
cat runs/drive/lerobot/meta/info.json | head -40
$V - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/vla-drive", root="runs/drive/lerobot",
                    delta_timestamps={"action": [k / 10 for k in range(16)]})
item = ds[100]
print(len(ds), item["observation.images.front"].shape, item["observation.state"], item["action"].shape)
EOF
```

`item["action"]` is `(16, 2)`: the training target LeRobot builds for that frame from
the next 16 frames' actions.

### Step 2: fine-tune (about 13 minutes)

```bash
mkdir -p runs/drive/vla
HF_HUB_DISABLE_TELEMETRY=1 $D/.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/vla-drive --dataset.root=$PWD/runs/drive/lerobot \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --policy.chunk_size=16 --policy.n_action_steps=16 \
  --dataset.eval_split=0.1 --eval_steps=500 --max_eval_samples=512 \
  --batch_size=16 --steps=2000 --log_freq=50 --save_freq=1000 \
  --policy.push_to_hub=false --policy.device=cuda --wandb.enable=false \
  --output_dir=runs/drive/vla/smolvla-v1 --job_name=smolvla-v1 2>&1 | tee runs/drive/vla/smolvla-v1.log
```

What each choice does:

| Flag | Why |
|---|---|
| `--policy.path=lerobot/smolvla_base` | start from the pre-trained 450 M model. **Not** `--policy.type=smolvla` as in the Colab notebook: in LeRobot 0.5.2 that builds the VLM with untrained weights (`load_vlm_weights` defaults to false) and then freezes it |
| `--dataset.root=…` | a local dataset; nothing is downloaded or uploaded |
| `--rename_map` | `smolvla_base` expects cameras named `camera1`, `camera2`, `camera3`. Without it: "Feature mismatch between dataset/environment and policy config". One camera of three is accepted |
| `--policy.chunk_size=16` | 16 actions 0.1 s apart: the 1.5 s the CNN predicts. The default 50 would be 5 s at 10 Hz. The weights do not depend on it |
| `--dataset.eval_split=0.1 --eval_steps=500` | hold out 9 of the 82 episodes and report their loss every 500 steps |
| `--batch_size=16` | fits in 8 GB. 32 ran out of memory (with chunk 50; not retried with 16) |
| `--policy.push_to_hub=false` | the default is **true**; without this LeRobot wants a Hub repository to upload to |
| `--output_dir=runs/drive/vla/smolvla-v1` | where `Project.vla("smolvla-v1")` looks; `runs/` is not in git |

*Expected:* `73 train, 9 eval`, `num_learnable_params=99880992 (100M)` of 450 M,
2.5–2.7 steps per second, 2.9 GB of GPU memory, 13 minutes in all.

| Step | Training loss | Held-out loss |
|---:|---:|---:|
| 50 | 1.018 | |
| 500 | about 0.47 | 0.651 |
| 1000 | | 0.484 |
| 1500 | | 0.499 |
| 2000 | | 0.469 |

The checkpoint the next steps use is
`runs/drive/vla/smolvla-v1/checkpoints/last/pretrained_model/` (0.9 GB of weights, the
config, and the pre/post-processors with this dataset's normalisation statistics).

### Step 3: drive it (about a minute)

```bash
$V -m drive sim --policy vla --checkpoint smolvla-v1 --latency-ms 125 \
   --video runs/drive/vla/smolvla-v1-peanut.mp4
$V -m drive sim --policy vla --checkpoint smolvla-v1 --latency-ms 0 --show \
   --log runs/drive/vla/smolvla-v1-actions.csv                                   # a window, real time
```

With `--show` inference runs on a real thread, so `--latency-ms 0` adds nothing and
the latency is the model's own (*measured* 150–190 ms with the window open). The
debug panel under the HUD (Lab 1) shows each 16-action chunk SmolVLA outputs, the
part of it skipped while it was computed, and the model's command against the
motors'. `--log` writes the same per tick to CSV.

`--checkpoint smolvla-v1` is found under `runs/drive/vla/`; a path to any
`pretrained_model` directory works too. `--latency-ms 125` because that is what one
inference took on this laptop (*measured*: 124 ms mean, after a 460 ms first call; the
model loads in about 7 s).

*Expected:*
```
vla on peanut, 125.0 ms per inference (215 inferences), latency compensation on, safety on
  1.88 laps in 30.0 s at 1.39 m/s
  off the centerline: rms 4.4 cm, max 10.5 cm, off the road 0.0 % of the time
  safety overrode the policy: stale 0.14 s, steer_rate 0.06 s
```

### Step 4: evaluate it (about 5 minutes)

```bash
$V -m drive evaluate smolvla-v1 --latency-ms 125       # writes runs/drive/reports/smolvla-v1.json
```

*Expected:*

```
smolvla-v1, 125 ms per inference, 40 s per drive
track               laps   m/s  rms cm  max cm  off road  result
peanut              2.60  1.44     5.6    14.2     0.0 %  pass
peanut reversed     2.63  1.46     5.8    15.0     0.0 %  pass
oval                2.80  1.43     3.6    12.5     0.0 %  pass
oval reversed       2.78  1.43     4.2    12.5     0.0 %  pass
trefoil             2.40  1.44     3.8     9.3     0.0 %  pass
trefoil reversed    2.40  1.45     5.6    12.5     0.0 %  pass
6/6 drives passed
```

Summary: 1.44 m/s, 4.8 cm rms, 15.0 cm worst; the supervisor intervened only at
start-up (`stale 0.14 s`, `steer_rate` 0.02–0.06 s). Set beside the other models in
[chapter 13](13-vla-and-smolvla.md#the-pc-only-route-done-once): SmolVLA drives at the
teacher's speed, where the CNNs drive at 0.84–0.90 m/s, and tracks about three times
less tightly than either.

`evaluate` writes `runs/drive/reports/smolvla-v1.json`, and replaces it if you run it
again with other settings.

### Observe

- In the video or window, the cyan plan bends with the road, and the car drives at
  close to the teacher's speed.
- It weaves more than the CNN: 4–6 cm rms against 1–2 cm.
- The held-out loss stops improving after about 1000 steps, yet that says nothing
  about whether more steps would drive better.

### Questions

1. Why is `--action intended` the export's default, and what would `--action applied` teach?
2. The chunk target for frame $i$ is built from frames $i \dots i+15$. Why does that
   differ from the teacher's plan at frame $i$ for `collect` episodes, and which end
   of the chunk does it affect most? Which part of the chunk does the car actually
   execute at 125 ms latency?
3. The CNN's validation steer error was 0.024; on frames of `sim2-039`, SmolVLA's
   first action differs from the teacher's plan by about 0.1. Which one drives faster
   and still passes? What does that say about offline errors, again?
4. Why is `--latency-ms 125` the right setting here, and what would you set for a Jetson?

### Challenges

- Export with `--action applied` into another directory, fine-tune `smolvla-applied`,
  and evaluate both. Prediction first.
- Sweep latency: `$V -m drive sweep --policy vla --checkpoint smolvla-v1 --max-plan-age-ms 1500 --latencies 125,200,250,300`
  (not run for this guide; each row is a 30 s drive per mode, at about 1.5 s of compute per simulated second). Compare with the CNN's breakdown at 300 ms ([chapter 11](11-model-evaluation.md)).
- Train longer (`--steps=10000`, about an hour) and check whether driving improves
  while the held-out loss does not.

### Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `Feature mismatch … Missing features: ['observation.images.camera1', …]` | camera names differ from the pre-trained model's | `--rename_map` as above |
| `'repo_id' argument missing` | `push_to_hub` defaults to true | `--policy.push_to_hub=false` |
| `CUDA out of memory` | batch too large for 8 GB | `--batch_size=16` or 8 |
| the disk fills up | `HF_HOME` not set: the model went to `~/.cache` | export `HF_HOME` in every shell |
| `export needs LeRobot` / `the vla policy needs LeRobot` | the command ran with the system Python | use `$V -m drive …` |
| `no model 'smolvla-v1'` | the run's `--output_dir` is not `runs/drive/vla/smolvla-v1` | pass the `pretrained_model` path instead |
| loss falls but the car drives badly | the model learnt the wrong thing (check `--action`, the rename map, the chunk size) | evaluate early; do not trust the loss |

### Running the long fine-tune on Colab instead **[UNTESTED]**

The laptop is enough for the short run above. For many more steps or a batch of 64,
use the [Colab notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb)
with these changes, none of which were tried for this guide:

1. **Get the dataset there.** The export is 35 MB: upload the `runs/drive/lerobot`
   folder to Google Drive, or push it to a **private** Hugging Face dataset repository.
   Either sends your data off the machine; decide that deliberately.
2. **Install** with `pip install -e ".[smolvla,dataset,training]"` (not `train`).
3. **Train** with the Step 2 command, but `--dataset.root=/content/drive/MyDrive/…/lerobot`,
   a larger `--batch_size` if the GPU allows, and `--output_dir` on Google Drive so a
   disconnected session can continue with `--resume=true`.
4. **Bring the checkpoint back**: copy `checkpoints/last/pretrained_model/` into
   `runs/drive/vla/<version>/checkpoints/last/` and run Steps 3–4 on the laptop.

Colab's free GPUs and session limits change and are not guaranteed; check what you
are given before starting a long run.

---

## Answers to selected questions

<details><summary>Lab 1</summary>

1. One inference per 100 ms, back to back: 30 s × 10 Hz.
2. No plan exists for the first 100 ms (five ticks braked); the first steering command jumps from zero by more than 0.16 per tick for three ticks.
3. `--show` uses `ThreadedWorker`, whose latency is the 100 ms sleep plus the real compute time.
</details>

<details><summary>Lab 4</summary>

1. Plan age cycles between 0.4 and 0.8 s and is over 0.6 s for 9 ticks in 20.
2. Age is always above 0.65 s.
3. Just under 300 ms (age reaches twice the latency).
</details>

<details><summary>Lab 5</summary>

1. The delay happens to cancel pure pursuit's tendency to turn in early.
2. Otherwise the watchdog brakes at 400 ms and you measure the watchdog, not tracking.
3. The actions are timed correctly but computed from information 0.4–0.8 s old, on the assumption that the car followed this plan in the meantime; it followed the previous one.
</details>

<details><summary>Lab 6</summary>

1. Slow under 0.74 m, brake under 0.58 m.
2. `stop_margin_m = 0.30`: it is allowed to approach until the room is zero.
3. Braking takes the speed below the allowed curve, the brake is released, and the car, still closing, crosses the curve again.
</details>

<details><summary>Lab 7</summary>

1. On a well-behaved policy the supervisor is transparent.
2. `np.zeros(2)`: wheels straight, zero throttle. The car coasts instead of braking.
3. No. It is driving on plans up to 1.3 s old with nothing to stop it: *measured*, 150 cm rms and off the road half the time.
</details>

<details><summary>Lab 11</summary>

1. The perfect-data model has the better validation error; the disturbed-data model drives.
2. That the mapping it learned is local (bends, not a memorised circuit). Not that it would cope with a different-looking road.
3. The teacher's target speed varied per episode and is not visible in the image.
</details>

<details><summary>Lab 12</summary>

1. The simulated car's own 0.5 s command watchdog.
2. The teacher follows a map of the wrong road and drives off. Camera policies do not use the map.
3. Only the simulator. The driver learns of it through `/drive/range`.
</details>
