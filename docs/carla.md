# CARLA

`drive/carla/` makes CARLA a third platform for `drive/`, next to the 1:10 simulator and
ROS 2: the same `Platform` protocol, the same recorder, and the same pipeline from
episodes to a fine-tuned SmolVLA. A teacher drives random routes through a CARLA town and
is told, a sentence per frame, where to turn, so the dataset has what the simulator's
lacks: the same picture of an intersection with a different sentence, and a different
label, depending on where the route goes.

```
CARLA server (Docker, synchronous, 20 Hz)
   │  camera every tick, a frame kept every 2nd tick (10 Hz)
   ▼
CarlaPlatform ──► RouteTeacher on a random route ──► [steer, throttle] + DART push on the steering
   │                    │
   │                    └─ plan: 16 actions x 0.1 s, rolled out with the fitted CarModel  ─► labels.npz
   └─ image, [speed, steer], the sentence for where the car is                             ─► episode.npz
                                                       ./drive.sh validate | browse | export ─► LeRobotDataset
```

## Try it

Needs Docker with the NVIDIA container toolkit, and [uv](https://docs.astral.sh/uv/).
The client is a `.venv-carla` of its own (Python 3.10 and the `carla` wheel, no PyTorch).

```bash
make carla-setup                          # once: .venv-carla
make carla-server TOWN=Town01             # the stock carlasim/carla:0.9.16 image; minutes to load
make carla-collect ARGS='--episodes 40'   # waits for the server; -> runs/carla/episodes
make carla-down                           # one town per server: stop it before starting another
make carla-server TOWN=Town03_Opt && make carla-collect TOWN=Town03_Opt ARGS='--episodes 40 --seed 1'
```

Town03 and Town05 crash the stock server (at boot, and on Town05's elevated road); their
`_Opt` versions, the same roads, do not. Give each town its own `--seed`.

To watch it record, add `--show` (a window: each frame as it is recorded, the label's path
drawn on the road, the sentence, the car's distance from the lane centre; q stops, and
`--resume` carries on) and/or `--video runs/carla/videos/town01.mp4`. Afterwards, any
episode: `./drive.sh browse --project runs/carla --match 'carla-town05_opt-002'`, in a window
or `--video` to an .mp4.

Then the stages every dataset goes through, with the CARLA project:

```bash
PYTHON=.venv-carla/bin/python ./drive.sh validate --project runs/carla
PYTHON=.venv-carla/bin/python ./drive.sh browse --project runs/carla      # the label drawn on the road, and the sentence
./vla.sh -m drive export --project runs/carla                             # needs make vla-setup
```

and SmolVLA is fine-tuned as in [Lab 15](study/15-hands-on-labs.md), on that dataset:

```bash
./vla.sh lerobot-train --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/vla-drive --dataset.root=$PWD/runs/carla/lerobot \
    --rename_map='{"observation.images.front": "observation.images.camera1"}' \
    --policy.chunk_size=16 --policy.n_action_steps=16 --policy.push_to_hub=false \
    --batch_size=16 --steps=2000 --output_dir=runs/carla/vla/smolvla-carla-v1
```

The GPU is either CARLA's or the training's on an 8 GB card: `make carla-down` first.

If the server dies during a `collect`, start it again and add `--resume` with the same
`--seed`: the episodes already written are kept, and the rest are the ones an unbroken
run would have recorded (each episode draws from its own generator, seeded `[seed, index]`).

| command | what it does |
| --- | --- |
| `carla identify` | drives a car in CARLA and fits the car model the teacher plans with -> `drive/carla/vehicles/<blueprint>.json` |
| `carla check` | drives as `collect` does and measures how far that model's predictions land from the car: the labels' accuracy |
| `carla collect` | records the episodes (`--speeds LO,HI` the target speed drawn per episode, `--speed-noise` pushes it as DART pushes the steering) |
| `carla drive` | drives routes with a checkpoint (or `teacher`) and scores them; `--replan-ms`, `--ensemble` choose how its chunks are played |
| `carla dagger` | the checkpoint drives, the teacher labels every frame and takes over before the lane is left: episodes for the next round |

`make carla-identify` / `carla-check` / `carla-collect` run them with `--town $(TOWN)`, which refuses a
server in another town. A model for the Tesla Model 3 is in the repository; `identify` is
for another car, or another CARLA version.

## The episodes

The same files as every other episode ([chapter 9](study/09-dataset-pipeline.md)), so
validate, browse, export and the CNN's training read them unchanged, plus:

| file | field | |
| --- | --- | --- |
| `episode.npz` | `images` | 240 x 320 RGB, like the simulator's |
| | `states` | `[speed_mps, steer]`: speed is 0-9 m/s here, 1-2 m/s in the simulator |
| | `actions` | `[steer, throttle]` that were on the car, with the DART push in the steering |
| | `tasks` | the sentence at each frame; only in episodes where it changes |
| | `poses`, `route_s`, `lateral_m` | rear axle x, y, yaw; where on the route: for analysis, not trained on |
| `labels.npz` | `plans` | the teacher's plan at each frame: 16 actions, 0.1 s apart |
| `meta.json` | `carla` | town, weather, target speed, route and junctions, the car model and the camera it was recorded with |
| | `robot_type` | `carla:vehicle.tesla.model3`: export keeps CARLA and simulator episodes out of one dataset |

**The sentences.** From 30 m before a junction until the car has left it, the sentence is
that junction's maneuver; everywhere else it is to follow the lane. Each command has
three wordings (`drive/carla/route.py`, `PHRASES`); a junction keeps the one it was given,
and so does every stretch of lane-following in an episode. The route walker picks a
*maneuver* uniformly at each junction (left, right, straight: whichever exist), then a
branch that does it; picking a branch uniformly would favour going straight wherever two
lanes do. `validate` prints the share of frames per sentence.

**What an episode is not.** It ends early on a collision, off its lane, or stuck, and its
last 2 s are dropped then: they show a car heading into trouble with labels that pretend
otherwise. Every traffic light is held green while recording: the teacher does not
look at them, and a dataset of a car driving through red lights would teach that.

## Design, and what was measured

All on CARLA 0.9.16 (`carlasim/carla:0.9.16`), an RTX 4060 laptop GPU (8 GB) and an
i7-13650HX, Town01, `-quality-level=Low`.

### 20 Hz, a camera every tick, a frame every 2nd tick

`sensor_tick` does not give a steady 10 Hz: it accumulates rounding error. Over 600 ticks
of 0.02 s, `sensor_tick=0.1` gave 118 gaps of 5 ticks and one of 6; `0.099` gave 114 of 5
and six of 4. LeRobot places frame i at i / fps, so the camera renders every tick and a
frame is kept by counting ticks; the platform's clock counts ticks too.

| server tick | camera | simulated seconds per wall second |
| --- | --- | --- |
| 50 Hz | none | 5.16 |
| 50 Hz | every tick | 1.66 (1.71 again) |
| 50 Hz | `sensor_tick=0.1`: inexact | 3.34 |
| **20 Hz** | **every tick** | **4.29** |

20 Hz it is: 2.5 times faster with exact timing, physics sub-stepped at 100 Hz inside each
tick, and a control rate CARLA's own agents use. A whole `collect` (teacher, plans,
compression) runs at about 2.9 times real time. `CarlaPlatform.apply` refuses a step that
is not a whole number of ticks: the 50 Hz `DriveLoop` runs at `LoopConfig(control_hz=20)`
on it.

### The server's map

The stock image boots Town10HD_Opt, which held 5013 MiB of the GPU's 8188 before anything
ran; booted into Town01 the server holds 1750. The map on the command line is ignored, so
`make carla-server` edits the image's `GameDefaultMap`/`ServerDefaultMap` inside the
throwaway container. Booted that way, `Town03` segfaults the stock server within 10 s
(exit 139, every time); `Town03_Opt`, the same roads with the scenery in layers, boots,
and so do Town01, Town02 and Town05, at 1.7-1.9 GB each. Town01 answered within a few
minutes of a cold start; Town03_Opt took over ten, and a session waits up to twenty. A session refuses a server in the wrong town instead of loading
another map into it: loading does not free the first map, and repeated loads have crashed
the server (simulate_carlar/docs/VRAM_8GB.md). A session also puts the server back into
asynchronous mode whatever goes wrong; one left synchronous waits for a tick forever.

**Town05 crashes the server where its elevated road meets the city; use Town05_Opt.**
The same episode (seed 2, index 2) killed the server (exit 139) at the same place every
time, 54.6 s in, at (-239.9, -109.6, 10.0) in CARLA's coordinates, driving smoothly at
7.2 m/s with no collision. It did so with the default exposure, in ClearNoon instead of
MidRainyNoon, and with a lone camera and no car at all, moved along the road at the
car's speed: what kills it is rendering that view. In Town05_Opt the same camera went
130 m past the point, and the town has its buildings, trees and props. (The same seed
draws other routes in Town05_Opt: none of its 40 episodes came within 18 m of the point,
so that they finished says nothing about it; the camera does.) Before suspecting
the client after an exit 139, replay the episode with one thing changed at a time; the
first guess here (a camera destroyed with a frame in flight) was wrong.

`collect --resume` finishes a run the server died in, and a session's teardown gives a
dead server 10 s a call instead of 60 (three actors took four minutes to report it).

**A client started with the server can hang for good.** `docker run -p` accepts
connections on the host before CARLA listens inside, then closes them; a `carla.Client`
that connected in that window waited on a socket in CLOSE-WAIT for ten minutes, past
any timeout. `connect` now hands the port to CARLA only once a connection to it stays open
(`server.listening`), so `make carla-server && make carla-collect` is safe.

### The car and the frames

- The Tesla Model 3: one gear, so the throttle has no shift steps in it.
- CARLA is left-handed (y to the right, yaw clockwise, steer +1 to the right); `drive/` is
  not. y, yaw and steer change sign in `drive/carla/frames.py`, and nowhere else.
  `get_wheel_steer_angle` is positive turning right, like the command (measured: +0.5 gave
  the inner wheel 34.97 deg, half of the 70 deg lock, and the outer one 26.75 deg).
- The actor's origin is on the ground, mid-car; the rear axle is 1.386 m behind it. The
  state's (x, y), the model and the teacher all work from the rear axle.
- CARLA shrinks the lock with speed (`steering_curve`: 1.0 at rest, 0.9 at 20 km/h, 0.8 at
  60). The model takes a command as a share of the lock at the current speed, as CARLA does.

### The camera

320 x 240, 90 deg, 2.9 m ahead of the rear axle (over the front of the bonnet), 1.5 m up,
6 deg down. At the windscreen (2.4 m, 1.4 m, 8 deg) the bonnet filled the bottom quarter
of every frame. Checked by projecting where the car really went, from the poses of the
frames that follow, onto each frame: it lies on the lane the car drove, through straights
and right turns alike.

Exposed one stop down (`exposure_compensation = -1`, `collect --exposure`). CARLA's auto
exposure washed the road out at noon. Measured on the road below the horizon, Town05, one
spawn point, mean brightness and the variance of the Laplacian (how sharp the lane marks
and the road's texture are):

| | noon: brightness | noon: sharpness | sunset: brightness | sunset: sharpness |
| --- | --- | --- | --- | --- |
| default | 211 | 30 | 160 | 53 |
| **-1 stop** | **171** | **52** | **118** | **50** |

Bloom off changed nothing; manual exposure was worse (242, 6). The camera's settings are in
every episode's `meta.json`.

At `-quality-level=Low` every cloudy or rainy preset draws a near-black sky (the mean of
the top quarter: 44-51 of 255 at the default exposure, darker one stop down); clear ones
draw a blue one. The road, which the labels are about, is lit either way. It is a look no
real camera has: a reason to weigh the clear presets more, or to record at Epic, if the
VLA is to see real footage.

### The teacher

Pure pursuit from the rear axle at 3 m + 0.5 s x speed ahead (3-12 m), and no further
than half the radius of the sharpest bend within that reach; slower for the bends within
braking reach (2 m/s^2 sideways) and to a stop at the end of the route. The pedals come
from inverting the car model, and there is no integral term, so a plan depends on the
state it starts from only. The simulator's teacher (0.5-1.4 m ahead, +0.1 x speed of
throttle) is sized for a 1:10 car and does not carry over; and with a guessed car model
the first drive averaged 5.1 m/s against a 7 m/s target.

DART is the simulator's: an Ornstein-Uhlenbeck push on the applied steering (standard
deviation 0.06, correlation 0.5 s), held within two standard deviations; the label is
the clean plan.

The first Town01 recording had the lookahead at 4 m + 0.6 s x speed without the bend
limit, and an uncapped push: 4 of its 40 episodes ended early. Replayed (an episode is
fully determined by its seed), one had a push of +0.10 to +0.18 held for 2 s on a
straight, against a teacher pulling back with -0.15, and drifted off its lane; the
others were in junctions, where pure pursuit cuts corners (on average 0.13 m to the
inside of left turns and 0.18 m of right ones), one of them into a guardrail. Pure
pursuit pulls back in proportion to 1 / lookahead^2, and a teacher that comes back
weakly also teaches that. With the shorter, bend-limited lookahead and the cap, the same
four episodes drive their full minute, and on `carla check`'s drives:

| teacher | lateral rms | 99th percentile |
| --- | --- | --- |
| 4 m + 0.6 s, uncapped push | 0.36 m | 1.00 m |
| 3 m + 0.5 s, bend-limited, push within 2 sd | 0.22 m | 0.54 m |

The lane is 4 m and the car 2.16 m wide.

### The car model, and how good the labels are

A plan's first action is exactly what the teacher wanted; the later ones are as right as
the model's idea of where they take the car. `carla check` measures that: from the real
state every 0.1 s, the model is rolled forward with the actions that were really applied
and compared with where the car really was. With the model in the repository, on 4
minutes of collect-style driving it was not fitted on:

| ahead | position (median / p90) | heading | speed |
| --- | --- | --- | --- |
| 0.5 s | 0.03 / 0.06 m | 0.3 / 0.8 deg | 0.09 / 0.16 m/s |
| 1.0 s | 0.09 / 0.16 m | 0.5 / 1.3 deg | 0.15 / 0.28 m/s |
| 1.5 s | 0.19 / 0.32 m | 0.6 / 1.7 deg | 0.19 / 0.39 m/s |

At 5-8 m/s the car covers about 10 m in 1.5 s, so the end of a plan is off by 1-2 % of the
distance. Steering fits well: the wheels follow the command within a tick (0.014 s), the
mean of the two Ackermann wheels turns 0.94 of the inner one, and the effective wheelbase
is 3.13-3.17 m against 3.00 m measured between the axles.

The speed was the hard part, and what was learnt on the way is in
`drive/carla/identify.py`:

- **Judge the fit by its 1.5 s prediction, not its R^2.** The first fit's one-step R^2 was
  0.44 (the measured acceleration is noisy) while its speed was off by 0.19 m/s at 1.5 s.
- **Constrain it to physics.** Unconstrained, the drag came out as -0.20 + 0.47 v - 0.019 v^2
  (a car that creeps away from standstill, drag falling past 12 m/s) and the brake as
  8.3 - 0.92 v (none at 9 m/s). Fine where there was data, wrong elsewhere.
- **The brake is a table, fitted on its own.** CARLA brakes in jolts: at 0.3-0.45 of
  pedal the deceleration was 1.3 m/s^2 at the median and 22 m/s^2 at the 90th percentile in
  a single tick, and the brake answers a tick late. The table takes the mean per pedal bin
  (the speed lost is the jolts' sum) from separate drives with brake pulses. Fitted in the
  same least squares as the throttle, those pulses made the throttle fit worse.
- **Fit the throttle where the teacher drives.** CARLA's throttle is not linear (median
  acceleration 0.7, 2.3 and 3.9 m/s^2 at 0.3, 0.5 and 0.7 of pedal, at 0.5-3 m/s), so the
  best straight line depends on where it is drawn. Fitted on one collect-style log and
  tested on another, a fit on collect-style drives predicted the speed at 1.5 s within
  0.04-0.09 m/s; one on drives with the throttle pushed at random, 0.08-0.19 m/s. So
  `identify` drives as `collect` does for the throttle.
- **It does not settle.** The teacher presses the throttle according to the model it is
  given, so successive `identify` runs alternate between two fits (t0 near 6.5 and near
  12.7 m/s^2 per unit of throttle). Their labels are equally good on held-out drives
  (0.06-0.18 m at 1.5 s), which is why it was left there.
- **The teacher hardly brakes.** In collect-style driving it braked in 0.1 % of the ticks,
  at most 0.014 of pedal: the brake table does not move the labels' accuracy at all.

## The dataset as recorded

Three towns, 40 drives of 60 s each, every one to the end (none hit anything, left its
lane or stalled), on 2026-11-21:

| town | `--seed` | wall time (`--show --video`) | frames over a lane line | follow / left / right / straight |
| --- | --- | --- | --- | --- |
| Town01 | 0 | 1065 s | 0 | 63 / 13 / 12 / 11 % |
| Town03_Opt | 1 | 1334 s | 62 (0.3 %) | 35 / 12 / 23 / 30 % |
| Town05_Opt | 2 | 636 s (no window) | 62 (0.3 %) | 45 / 21 / 12 / 23 % |

`validate`: 120 episodes, 72000 frames, 119.8 min, speed 0-8.87 m/s (mean 5.24), steering
81 % straight; 12 sentences, 4.7-18.8 % of the frames each. Exported to LeRobot (581 MB),
it trains: 60 steps of SmolVLA at batch 16 on the 4060, 2.75 steps/s, loss 1.54 to 0.89.
Mind the laptop: it suspended during the export (`journalctl`), which took 10 minutes of
uptime and 101 of wall time; `systemd-inhibit --what=sleep:idle` around a long `lerobot-train`.

**Two things in it that a VLA could learn wrongly.** Both measured; both are next.

- *In a multi-lane town the lane, not the sentence, decides the turn.* The route never
  changes lane, and OpenDRIVE lets only the left lane turn left and only the right lane
  turn right. From the towns' OpenDRIVE (`tools/carla_lane_turns.py`; `carla.Map` builds
  the map client-side, no server): the lane
  allows fewer maneuvers than the road at 0 of Town01's 36 junction approaches, 0 of
  Town02's 24, 36 of Town03's 149 (24 %) and 132 of Town05's 154 (86 %). No Town05_Opt
  episode turns both left and right. Town01 is clean; Town03 and Town05 need lane changes
  before junctions, chosen by the command, so that a sentence can disagree with the lane
  and the car does what it says.
- *A tight right turn leaves the picture.* In junctions, the label's 1.5 s end point is
  outside the 90 deg frame in 15 % of the frames of right turns, none of left turns or
  straight on. A wider lens (110-120 deg) would keep it in.

## Training a VLA on it, and testing it

```bash
make carla-down                                       # the GPU is the training's
./vla.sh -m drive export --project runs/carla --out runs/carla/lerobot-train --exclude 'carla-*-03[5-9]'
HF_HOME=$HOME/data1/hf-cache systemd-inhibit --what=sleep:idle --who=vla-drive --why=SmolVLA \
  ./vla.sh lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/vla-drive --dataset.root=$PWD/runs/carla/lerobot-train \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --policy.chunk_size=16 --policy.n_action_steps=16 --batch_size=16 --steps=8000 \
  --log_freq=100 --save_freq=2000 --policy.push_to_hub=false --policy.device=cuda --wandb.enable=false \
  --output_dir=runs/carla/vla/smolvla-carla-v1 --job_name=smolvla-carla-v1
```

`--exclude` keeps the last five drives of each town (15, 9000 frames) out of training, for
the test below. 8000 steps at batch 16 are two passes over the 63 000 frames: 76 minutes
on the 4060 laptop for the whole run (0.57 s a step; a short test had run 0.36), a
checkpoint every 2000.

**On Google Colab** instead: `make carla-colab-pack` writes the training split as one tar
(509 MB) to upload to `MyDrive/vla-drive/`; `tools/colab_train_carla.ipynb` trains on the
Colab disk with the same arguments (`--dataset.video_backend=pyav`: PyAV decodes the AV1
videos with libdav1d, and torchcodec need not match Colab's torch), keeps only the newest
checkpoint on Drive (1.3 GB each: the model and its optimizer state) and resumes from it in a
new session; its last cell packs the model (about 900 MB) for `make carla-colab-unpack
TAR=... COLAB_RUN=smolvla-carla-v1`. Run here with a stand-in Drive: killed at step 160
with step 120 on Drive, a fresh session restored 120 and went on to 240 with the learning
rate where it had been; the packed model loaded with `make_vla_policy`.

On a free T4 it took 2.37 s a step, 4.2 times the 4060's 0.57: 8000 steps are 5.3 hours.
The one real run there lost its session at step 7700, and the Drive held only checkpoints
2000 and 3000, under a "storage full" banner, although the notebook had printed
`[drive] checkpoint` for 4000 to 7000. Two things add up to that. A file deleted through the
Drive mount goes to the Drive's Trash, which counts against the quota, so each replaced
checkpoint stayed there as 1.3 GB. And once the Drive was full, the mount went on accepting
the copies, which never reached Drive. Empty the Trash and check the free space before a run,
or train here when the 4060 is free: it does all 8000 steps in 76 minutes.

**Offline**, no server: `./vla.sh tools/probe_vla_carla.py --checkpoint smolvla-carla-v1`.
On the held-out drives it gives the steering error against the teacher's clean plan per
kind of sentence, and the test the dataset is for (`[listen]`): at the frames near or in a
junction, the same picture and noise with the sentence swapped for each maneuver. A model
that reads the sentence steers further left for "turn left" than for "turn right"; one
that ignores it, 50 % and no difference. After only 300 steps (a check of the commands,
not a model) it already did, at 100 % of 449 frames, but by little: mean steer over the
chunk +0.062 told left, +0.002 straight, -0.044 right; its throttle did not yet answer
the speed (+0.35 at rest, +0.40 at 9 m/s). One chunk: 164 ms on the 4060.

**One frame at a time, in the browser**: `./vla.sh tools/vla_app.py --project runs/carla
--checkpoint smolvla-carla-v3-dagger2`, then http://127.0.0.1:7860 (needs gradio in
`.venv-vla`). Any CARLA episode, `dagger*` included: a frame comes with the sentence the
driver was told then, the teacher's plan in cyan under the model's in orange (drawn with
that episode's camera and car), and for a DAgger frame whether the teacher was driving.
"Swap the sentence for" runs the same frame told another maneuver; paint, mirror, darken or
hide part of the image; the speed goes to 12 m/s. A model trained on the speed alone gets
only the speed, and the wheels then only start the drawn paths. On held-out
carla3-town01-036 frame 351, at a T-junction where the route goes left, DAgger2 told "turn
left" plays +0.17 over its first five steps (the teacher +0.13), told "turn right" -0.22.

**In CARLA**, closed loop, on routes it has never seen (`--seed 100`; collect used 0-2):

```bash
make carla-vla-setup                                  # once: the carla client into .venv-vla
make carla-server TOWN=Town01
make carla-drive TOWN=Town01 ARGS='--checkpoint smolvla-carla-v1 --episodes 10 --show --video runs/carla/videos/drive-v1-town01.mp4'
make carla-drive TOWN=Town01 ARGS='--checkpoint teacher --episodes 10'      # the same routes, the ceiling
```

The model drives through `DriveLoop` at 20 Hz, a chunk arriving 100 ms of simulated time
after its observation, behind a `SafetySupervisor` sized for a car; the server waits for
it, so a drive of 60 s takes about two minutes of wall time. Watched, that runs in fits:
between inferences the simulation races at three times real speed, then freezes for one
(150 ms). `--realtime` paces it to the wall clock and runs the model on its own thread, as
on the car: a chunk arrives when it is computed (160 ms on the 4060, the thread's first,
which sets CUDA up, done before the clock starts). DAgger2 with `--replan-ms 500`, ten new
routes (seed 200): never more than 2 ms behind the clock; four to the end, 2.47 km, 6
failures, junctions left 2/6, right 5/5, straight 7/7, where on the simulated clock five,
2.41 km, 5, left 1/4. A plan may age by `--replan-ms` past the supervisor's 0.6 s: asked
every 500 ms and computed in 190, it is 0.69 s old before the next, and 0.6 braked the car
hard two or three ticks in ten (it crawled 20 m in 8 s). The window shows the model's
plan in cyan over the riding teacher's in green. Scored: how far along each route, the
lateral rms, collisions, and each junction reached, passed or failed by maneuver; the
numbers go to `runs/carla/eval/<checkpoint>-<town>-seed<seed>.json`. Town02, which no
episode was recorded in, tests a town the model never saw.

The teacher through the same harness, two routes of Town01: both to the end, lateral rms
0.07 m, junctions left 2/2 and right 1/1. The 300-step check left the road in 7.8 s and
hit a sign in 4.2 s, steering right on a left bend: the harness is right, that model is not.

**smolvla-carla-v1**, the 8000 steps above. Probe: steering error at step 0 0.041 following
the lane, 0.053 left, 0.052 right, 0.042 straight (the 300-step check: 0.070, 0.078, 0.102,
0.062), correlation with the teacher 0.85 (0.53). It listens a little less: left beyond
right at 90 % of the junction frames, left > straight > right at 69 %, |left - right|
0.083, and where the route really goes straight "left" moves the mean steer by only
+0.036. The throttle still does not answer the speed: +0.31 at rest, +0.39 at 9 m/s.

In CARLA, Town01, the ten routes of seed 100, the teacher's numbers after the model's:
one drive of ten to the end (ten), 1.82 km (2.89), lateral rms 0.36 m (0.06), junctions
left 0/3 (7/7), right 2/4 (4/4), straight 3/3 (3/3); four drives hit something and five
left the route. `tools/failure_reel.py REPORT VIDEO OUT --sheets DIR` cuts the last 8 s of
every failed drive out of the `--video` into one reel. The nine are three kinds:

- **Sharp bends at the edge of the town** (four: three left, one right). The model's whole
  plan stays nearly straight while the teacher's curves, and the car runs into the
  guardrail or off. One came into the bend at 7.4 m/s; the teacher takes a bend of radius
  R at no more than sqrt(2 m/s² × R), braking for it in time (it did not get that far on
  this route in its 60 s, so how fast it would have been there is not measured).
- **Left turns at junctions** (three of three): in early and tight, across the corner,
  off the route on its inside.
- **Right turns** (two of four failed): once straight on, the sentence ignored; once too
  tight, into the kerb.

The speed is the data's fault, twice over. The teacher's target speed is drawn per episode,
4 to 8 m/s, and nothing the model sees says which, so at the same picture and speed the
label throttle goes up or down with a number it cannot know; the best it can learn is an
average. And the DART noise pushes only the steering: no frame shows the car too fast
before a bend and the teacher braking it back. Training longer changed neither: the 300-
and 8000-step throttles are equally flat.

The steering leans on the steer in the state, as behaviour cloning tends to. Same picture,
task and noise, the state's steer moved by ±0.3: the model's step 0 moves 0.49 times as
much, step 5 0.14 times, steps 10 and 15 not at all. In the data the label at step 0
correlates 0.83 with the state's steer (0.54 at step 15), so the shortcut pays offline. In
closed loop it costs: `DriveLoop` asks for a new chunk as soon as the last one is back, so
the car only ever executes a chunk's first step or two, the part that copies the steer it
is already holding. Offline, where the plan must change the steer by more than 0.1 within
1.5 s, the model's step 15 changes it by 0.155 against the teacher's 0.182, the same way
95 % of the time: the late plan knows the bend, the executed start of it hesitates.

**smolvla-carla-v2** tested that: the same episodes exported with `--state speed_mps`
(`LeRobotBackend` cuts the state to what a checkpoint was trained on, read from its
normalizer's statistics: `config.json` keeps smolvla_base's input features, an arm's six
joints and three cameras, after fine-tuning), trained the same 8000 steps. It listens more:
|left - right| 0.126 against 0.083, told left +0.071 against +0.039. Its first step is
rougher, as it must be without the steer it holds (step-0 error at turns 0.071 and 0.073
against 0.053 and 0.052; correlation 0.755 against 0.85); steps 5 to 15 are as before. On
the same ten routes it drove no better: one to the end, 1.86 km, lateral rms 0.40 m,
junctions left 1/4, right 3/3, straight 2/2. Drives 001 and 002 hit the same guardrails
within 2 s of where v1 did, the plan as straight. So the copied steer, though measured, was
not what kept the car straight into those bends. What is left is the speed: v2 was at
8.2 m/s six seconds before the bend of drive 001 and still at 5.1 one second before it.

**smolvla-carla-v3** takes the guessing out of the speed. The episodes were recorded again
(`carla3-*`, the same seeds, 40 a town) with one target speed for all, `--speeds 6,6`, and
the speed pushed as DART pushes the steering: `--speed-noise 0.3` scales the speed the
teacher drives towards by 1 + an Ornstein-Uhlenbeck noise (0.3 of it, 2 s, capped), while
the label stays the teacher's plan at the unscaled speed. So the car is sometimes too fast
for the bend ahead, and the label brakes it. In v1's episodes the label throttle rose with
the speed (correlation +0.44: a faster frame came from an episode with a higher target); in
v3's it falls (-0.54), and 16 % of the frames are above 7 m/s (v1 11 %), up to 11 m/s (8).
All 120 drives ran their 60 s; frames with a wheel over a lane line: Town01 0.0 %,
Town03_Opt 0.6 %, Town05_Opt 0.8 %. Exported with `--state speed_mps`, trained the same
8000 steps (70 minutes, 0.52 s a step).

Probe on carla3's held-out drives (885 frames), v2 on the same frames in brackets: throttle
error following the lane 0.011 (0.087); one frame at 0, 3, 6 and 9 m/s, throttle +0.31,
+0.45, +0.31, +0.16 (+0.30, +0.42, +0.34, +0.39): it now eases off as it goes faster.
Step-0 steering correlation 0.80 (0.72), left > straight > right at 91 % (62 %).

In CARLA every model now drives the ten routes at 6 m/s (`carla drive --speeds 6,6`; the
default, 4,8, draws a speed per route as `collect` did). v3, with v2 and the teacher in
brackets: three of ten to the end (one; ten), 2.16 km (1.90; 3.08), 7 failures (9; 0),
junctions left 2/6 (1/5; 7/7), right 3/4 (3/4; 4/4), straight 3/3 (2/2; 4/4). Route by
route it is better on three and no worse on any. Its mean speeds, 4.3 to 5.5 m/s, are the
teacher's (4.7 to 5.3); v2 averaged 7.2 m/s on route 001 and hit the guardrail, v3 drove
it to the end. But the rest failed where v2 did, and not fast: on routes 002 and 007 the
car was at 4.0 to 4.6 m/s a second before the guardrail, the plan bending too little and
too late. Speed was one
cause. The other is what no episode shows: the car already off the teacher's line.

**How the chunk is played.** `DriveLoop` asked for a chunk as soon as the last one was back,
about every 140 ms, so the car executed steps 1 and 2 of each. `--replan-ms N` asks no
sooner than N ms after the last ask, the chunk played meanwhile; `--ensemble` blends every
chunk that covers the present step (ACT's temporal ensembling; `--ensemble-k` weights the
older ones by exp(-k × rank), 0 = equally). v3 on the same ten routes:

| how the chunk is played | to the end | km | failures (hit / off) | km per failure | junctions L · R · S |
| --- | --- | --- | --- | --- | --- |
| a new chunk as soon as it is back | 3/10 | 2.16 | 7 (1/6) | 0.31 | 2/6 · 3/4 · 3/3 |
| `--replan-ms 300` | 4/10 | 1.99 | 6 (2/4) | 0.33 | 1/4 · 2/4 · 3/3 |
| `--replan-ms 500` | 6/10 | 2.30 | 4 (0/4) | 0.57 | 3/6 · 3/4 · 4/4 |
| `--ensemble` | 5/10 | 2.29 | 5 (1/4) | 0.46 | 4/7 · 4/4 · 3/3 |

Five steps of each chunk nearly halve the failures, with nothing trained; blending does as
well. Asked every frame, the model answers its own small wobble. Single routes flip between
them: route 002's guardrail bend passes only at 500 ms, route 007 only at 300 ms and
blended. The model takes these places with no margin. Two left turns (routes 005 and 008)
failed every way.

**DAgger with the riding teacher** (Ross, Gordon and Bagnell 2011): the model drives, and
the teacher labels every frame it reaches, as `collect` labels its own:

```bash
./vla.sh -m drive carla dagger --town Town01 --checkpoint smolvla-carla-v3 --seed 310 \
    --episodes 20 --name dagger1-town01 --speeds 6,6 --replan-ms 500 --show
```

The teacher takes the wheel before the car leaves the lane, as a safety driver would: when
the rear axle, carried on at its heading and speed for 0.5 s, would be 1 m from the route's
centre. It hands back once the car has been within 0.3 m and 0.1 rad for 1 s. Those frames
are recorded too: the recovery the dataset lacked. The episodes go to `runs/carla/episodes`
in `collect`'s format (plus `teacher_driving`), the totals to `runs/carla/eval/<name>.json`;
km per takeover is the measure, the number a safety driver gives. Two rounds
(`runs/carla/bc-chain.sh`), each 40 new routes (Town01 20, Town03_Opt 10, Town05_Opt 10),
aggregated with everything before and fine-tuned 4000 steps from the last model (35
minutes; `--policy.path` the last checkpoint):

| | round 1: v3 drives | round 2: DAgger1 drives |
| --- | --- | --- |
| km per takeover, Town01 | 0.170 (36 in 6.1 km) | 0.175 (35) |
| Town03_Opt | 0.049 (63 in 3.1 km) | 0.051 (58) |
| Town05_Opt | 0.053 (58 in 3.1 km) | 0.064 (48) |
| frames the teacher drove | 4287 of 23 922 | 4023 of 24 000 |
| junctions needing a takeover, Town01 L · R · S | 21/22 · 8/11 · 4/14 | 13/13 · 10/13 · 6/27 |
| Town03_Opt | 8/9 · 10/10 · 15/23 | 9/10 · 9/9 · 15/22 |

It needs the teacher at almost every turn: the turns it passes on the ten routes, where a
failure is 2.5 m off, it passes heading for more than 1 m off. In the multi-lane towns it
needs the teacher every 50 m. DAgger1 did not change that,
nor the ten routes (five to the end, 2.22 km, 5 failures). What it changed is the side: v3
drifted left (Town05_Opt 44 takeovers left of the centre, 14 right), DAgger1 in Town03_Opt
right (11 left, 47 right). Round 1's recoveries were mostly from the left, and the model
learnt to steer right; that is why DAgger trains on every round's data, not the last.
The recovery frames are few: 4287 of 86 922 in round 1's training set, seen 0.74 times each
in 4000 steps.

**smolvla-carla-v3-dagger2**, on round 1 and 2's data together (110 922 frames), on the ten
routes with `--replan-ms 500`: seven to the end, 2.87 km, 3 failures and no collision, 0.96
km per failure (v3 the same way: six, 2.30 km, 4, 0.57); junctions left 4/7, right 4/4,
straight 4/4. The two left turns that failed every way, routes 005 and 008, and route 007
now pass; 003 and 004 fail a left turn late (244 and 298 m) that v3 passed
(`tools/compare_drives.py` puts two runs of the same routes side by side). Its probe
hardly moved (step-0 correlation 0.76 against v3's 0.80, the same throttles): the held-out
drives are the teacher's own, on its line, where nothing was missing.

Ten routes are few, so DAgger2 also drove rounds 1 and 2's own routes again
(`runs/carla/measure-chain.sh`; episodes `check-dagger2-*`, not trained on). Where v3
needed the teacher 157 times in 12.2 km, DAgger2 needed it 115 (0.107 km per takeover
against 0.078); where DAgger1 needed it 141, DAgger2 needed 118 (0.104 against 0.086). Most
of it is Town05_Opt (58 to 32, 48 to 36) and junctions taken straight on (Town01 4/14 to
1/14, 6/27 to 2/26). Left turns still need the teacher at most (Town01 20/22, 9/13). The
side DAgger1 overshot to in Town03_Opt evened out (11 left and 47 right of the centre, then
27 and 26): round 2's data held the recoveries from the right. Two rounds bought about a
quarter fewer takeovers; the left turns are what is left.

## Not here yet

- **Lane changes**, for the reason above.
- **Traffic.** No other cars or people, and `CarlaPlatform.range_m` sees none: the
  supervisor's range is what a VLA would be judged against once there are some.
- **Slopes.** The model has no grade term, and Town01, where it was fitted, is flat. On
  Town03_Opt's hills it costs little: at 1.5 s the position is off by 0.20 m at the median
  (0.41 m at the 90th percentile) against 0.19 (0.32) on Town01, the speed by 0.20 (0.43)
  m/s against 0.19 (0.39). The teacher's lateral rms there is 0.25 m, 99th percentile 0.65.
- **Sim to real.** The camera and the car are a passenger car's, not the 1:10 car's.
