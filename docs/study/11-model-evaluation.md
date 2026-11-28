# 11 · Model evaluation

*[Tiếng Việt](11-model-evaluation.vi.md)*

Everything here is **[SIM-ONLY]**: the metrics need ground truth that only a
simulator has. **[TESTED]** by `EvaluateTests` and `DriveTests`.

## What problem are we solving?

Deciding whether a policy is good. [Chapter 8](08-imitation-learning.md) showed that
the obvious measure, error on held-out frames, can rank a model that cannot drive
above one that can. The measure that counts is **closed-loop**: put the policy in the
loop, let its own actions determine what it sees next, and watch what happens.

| | Offline (open-loop) | Closed-loop |
|---|---|---|
| Question | Does the model agree with the expert on the expert's states? | Does the model stay in states it can handle? |
| Data | recorded frames | a live drive |
| Errors | independent | compounding |
| Cost | milliseconds | a 40 s simulated drive per condition |
| In this repository | `val_steer_error` in `models/<v>.json` | `evaluate` → `reports/<v>.json` |

## The metrics

Computed every control tick by [`Metrics`](../../drive/episode.py#L37) from the
simulator's true state, and summarised by
[`Metrics.summary`](../../drive/episode.py#L60). Let $e_k$ be the lateral offset from
the centreline at tick $k$ ([`Track.project`](../../drive/sim/track.py#L69)), $n$ the
number of ticks.

| Key | Definition | Meaning |
|---|---|---|
| `lateral_rms_m` | $\sqrt{\frac1n\sum_k e_k^2}$ | typical distance from the centreline |
| `lateral_max_m` | $\max_k \lvert e_k \rvert$ | the worst moment |
| `off_track_pct` | $100 \cdot \frac1n \cdot$ (number of ticks with $\lvert e_k \rvert > 0.45$ m) | share of the time off the 0.9 m road |
| `mean_speed_mps` | $\frac1n\sum_k v_k$ | how briskly it drove |
| `laps` | progress along the track ÷ track length | distance covered |
| `collided` | true if the car's body ever overlapped an obstacle | |
| `min_range_m` | smallest range reading | how close it came |
| `safety_s` | seconds each supervisor rule was active | how often the policy had to be overruled |
| `inferences`, `latency_ms_mean` | from the worker | what the policy cost |

**Progress** is accumulated as the wrapped change in arc length each tick,
$\Delta s = \big((s_k - s_{k-1} + \tfrac{\ell_t}{2}) \bmod \ell_t\big) - \tfrac{\ell_t}{2}$
with $\ell_t$ the track length, so crossing the start line does not register as a jump
backwards, and driving backwards would count negative.

**Why RMS and max together?** RMS tells you about the whole drive and hides a single
bad corner; max shows that corner and is sensitive to one glitch. A policy with a low
RMS and a high max has a specific place where it struggles.

## Pass or fail

[`evaluate`](../../drive/learn/evaluate.py#L22), one line:

```python
ok = r["off_track_pct"] == 0 and not r["collided"] and r["mean_speed_mps"] > 0.3
```

A drive passes when the car **never leaves the road, hits nothing, and keeps moving**.

The third condition matters. A car that stands still never leaves the road. Without a
minimum speed, the safest policy would be the best. (`EvaluateTests.test_a_drive_that_hits_something_fails`
covers the second.)

## The evaluation protocol

Each model is driven on **every track, forwards and reversed**: six drives of 40 s
at 100 ms latency by default.

```bash
make evaluate V="v1 v2 pure-pursuit"
```

- **Tracks** `peanut`, `oval`, `trefoil` ([chapter 3](03-vehicle-simulation.md)).
  Models trained on the peanut alone have never seen the other two.
- **Reverse** swaps left and right bends: a check for directional bias.
- **Latency** is imposed by `SimulatedWorker`, so the comparison is not affected by how
  fast the PC happens to be.
- **Deterministic**: the same model gives the same report.

Output: a table per model, `reports/<model>.json`, a comparison table, and exit code 0
only if every drive of every model passed, so the command can gate a script.

## Results

*Repo* (`runs/drive/reports/*.json`; 40 s drives, 100 ms latency). `v1` was trained on
40 simulator episodes, all on the peanut. `v2` on those plus two ROS 2 recordings, one
on the oval and one on the trefoil.

| Model | Drives passed | Mean speed | rms off centreline | Worst |
|---|---:|---:|---:|---:|
| `v1` | 6/6 | 0.84 m/s | 1.3 cm | 4.0 cm |
| `v2` | 6/6 | 0.90 m/s | 1.2 cm | 3.8 cm |
| `pure-pursuit` (the teacher) | 6/6 | 1.47 m/s | 1.5 cm | 3.4 cm |

Per drive (rms / max in cm, mean speed in m/s):

| Track | `v1` | `v2` | `pure-pursuit` |
|---|---|---|---|
| peanut | 1.5 / 4.0 @ 0.84 | 1.1 / 2.7 @ 0.89 | 1.8 / 3.4 @ 1.46 |
| peanut reversed | 1.4 / 3.9 @ 0.83 | 1.3 / 3.5 @ 0.90 | 1.8 / 3.4 @ 1.46 |
| oval | 1.7 / 3.5 @ 0.86 | 0.8 / 2.0 @ 0.89 | 1.0 / 2.4 @ 1.47 |
| oval reversed | 1.3 / 3.5 @ 0.83 | 1.6 / 3.3 @ 0.92 | 1.0 / 2.4 @ 1.47 |
| trefoil | 1.0 / 2.9 @ 0.84 | 1.2 / 3.8 @ 0.91 | 1.7 / 2.8 @ 1.47 |
| trefoil reversed | 0.8 / 2.0 @ 0.82 | 1.0 / 3.2 @ 0.90 | 1.7 / 2.8 @ 1.47 |

In every drive the supervisor intervened only at start-up (`stale 0.1 s`, `steer_rate`
0.02–0.06 s).

### Reading the results

**1. `v1` generalises across tracks.** Trained on one shape, it passes on three, in
both directions, and its tracking on the unseen shapes (0.8–1.7 cm rms) is within a few
millimetres of its tracking on the trained one (1.4–1.5 cm). The teacher's own numbers
are identical forwards and reversed, as they should be for a symmetric algorithm;
the networks' are not, which shows a small directional asymmetry that mirroring did
not entirely remove.

**2. The networks track the centreline as closely as the teacher, or closer, while
driving 40 % slower.** These two facts are one fact. Pure pursuit's error is mostly
corner cutting, which grows with speed. A slower car cuts less. Comparing rms at
different speeds is not comparing like with like: the teacher at the networks' speed
would be the fair reference (*measured:* the teacher at 0.6 m/s holds 0.7 cm rms).

**3. Why slower?** `docs/driving.md`: "its target speed changed per episode and is not
in the image, so they settle on a cautious average." See
[chapter 8](08-imitation-learning.md), "what the image does not show".

**4. `v2` is not clearly better than `v1`.** 1.2 against 1.3 cm rms and 3.8 against
4.0 cm worst, from one training run each. `v2` is better on some drives and worse on
others. With a single seed per model, a difference this small is not evidence that the
two extra recordings helped. To claim it, train each configuration with several seeds
and compare the spread ([experiment E6](16-experiments.md)).

**5. And the model trained on perfect driving** (`runs/perfect/reports/v1.json`):
0/6, off the road 62 to 85 % of the time, at 1.7 m/s. Its validation steer error was
the best of the three.

| | `v1` | `v2` | perfect-only |
|---|---:|---:|---:|
| Validation steer error (offline) | 0.0242 | 0.0223 | **0.0149** |
| Drives passed (closed-loop) | 6/6 | 6/6 | **0/6** |

## Latency, for a learned policy

*Measured* for this guide: `./drive.sh sim --policy cnn --checkpoint v1 --latency-ms L
--max-plan-age-ms 1500 --duration-s 40` on the peanut, in the container.

| Latency | rms / max | Off the road | Mean speed |
|---:|---:|---:|---:|
| 0 ms | 1.3 / 2.8 cm | 0 % | 0.84 m/s |
| 100 ms | 1.5 / 4.0 cm | 0 % | 0.84 m/s |
| 200 ms | 2.5 / 5.9 cm | 0 % | 0.84 m/s |
| 300 ms | 22.6 / 54.5 cm | 6.2 % | 0.89 m/s |
| 400 ms | 259.8 / 825.5 cm | 75.2 % | 0.83 m/s |

Compare with the teacher at a similar speed ([chapter 5](05-action-chunking-and-latency.md)):
at 1.0 m/s it still holds 1.6 cm at 250 ms, and at 0.6 m/s it holds 1.6 cm at 400 ms.
The network fails at 300 ms while driving only 0.84 m/s. **A learned plan tolerates
less latency than the teacher's.** Plausible reasons, not separated here: its
predictions further into the chunk are less accurate than a model rollout, and its
plan has a coarser time step (0.1 s against 1/30 s). The practical consequence: do not
take a latency budget measured with the teacher and assume it for a learned policy.
Measure the policy you will run.

`evaluate` exposes this as `--latency-ms`.

> **Caution.** `evaluate` writes `reports/<model>.json` every time. Evaluating `v1`
> again with another `--latency-ms` or `--duration-s` **overwrites** the earlier
> report. Use `./drive.sh sim --policy cnn --checkpoint v1 …` for exploratory runs, as
> above, or a copy of the project.

## What the evaluation does not show

| Not tested | Why it matters |
|---|---|
| Obstacles | no evaluation drive has one; the networks were never shown a reason to stop |
| Starting off-centre, or recovering from a push | every drive starts on the centreline at rest |
| More than 40 s | a slow drift would not show |
| Seeds | one training run per model; no variance estimate |
| Any other appearance | same simulator, same colours, same camera as training |
| Jittery latency, real threads | `SimulatedWorker`, fixed latency |
| A real car | everything in [chapter 3](03-vehicle-simulation.md)'s limitations table |

"6/6" means: on these three shapes, in this simulator, from a standing start, for
40 s, at a fixed 100 ms.

## Common mistakes and misconceptions

- **"The networks beat the teacher on rms."** At a lower speed.
- **"6/6 means it is safe."** It means it met three conditions in six short drives.
- **"`v2` improved on `v1`."** Unproven with one seed.
- **"If validation loss went down, skip the evaluation."** The perfect-driving model.
- **"A pass at 100 ms means it will pass on the Jetson."** Only if the Jetson's
  latency is 100 ms and the model is the same.

## Practical experiment

Add what the standard protocol lacks: a recovery test.

```python
# recover.py: start 25 cm off the centreline, 0.3 rad askew, and see whether the policy returns
from drive.episode import EpisodeConfig, Metrics, build
from drive.learn.project import Project

config = EpisodeConfig(policy="cnn", checkpoint=str(Project().checkpoint("v1")), duration_s=10.0)
world, loop = build(config)
world.place(5.0, lateral_m=0.25, yaw_offset_rad=0.3)
metrics = Metrics(world)
while world.t < config.duration_s:
    metrics.add(loop.step())
loop.worker.close()
print(metrics.summary(loop.config.dt))
```

Run it in the container (`make shell`, then `python3 recover.py`). Repeat with
`Project("runs/perfect")`. Then vary the offset until `v1` fails: that offset is a
more honest description of its robustness than "6/6".

## Knowledge check

1. Why is a minimum mean speed part of the pass rule?
2. A policy has rms 1.0 cm and max 30 cm. What does that suggest?
3. Why is comparing the networks' rms with the teacher's misleading?
4. What evidence would show that `v2` is really better than `v1`?
5. What does the evaluation say about obstacles?

<details><summary>Answers</summary>

1. Otherwise a stationary car, which never leaves the road, would pass.
2. Good tracking almost everywhere and one place, or one moment, where it nearly failed.
3. They drive at different speeds, and pure pursuit's error grows with speed.
4. Several training runs of each with different seeds, and a difference larger than the run-to-run spread.
5. Nothing: no evaluation drive contains one.
</details>

## Further reading

- Codevilla et al. 2019, on the gap between offline metrics and driving performance.
- Ross and Bagnell 2010, for why the gap exists.
- The CARLA leaderboard's metrics (route completion, infractions), as an example of closed-loop evaluation at larger scale.
