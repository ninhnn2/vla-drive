# 16 · Experiments and engineering analysis

*[Tiếng Việt](16-experiments.vi.md)*

The labs of [chapter 15](15-hands-on-labs.md) show how the stack behaves. The
experiments here ask *why*, in a form you can reuse: a hypothesis that could turn out
false, one variable changed at a time, a command, a measurement, and what the result
would mean for an engineering decision.

## Ground rules

1. **Write the hypothesis and the prediction first.** An experiment whose result you
   cannot be surprised by teaches nothing.
2. **Change one variable.** Everything else stays at its default, and you say so.
3. **Never overwrite.** Use `--project runs/<experiment>`; note that `evaluate`
   replaces `reports/<model>.json` on every run.
4. **Record the command with the result.** A number without its command is a rumour.
5. **One training run is an anecdote.** Anything involving `train` needs several
   seeds before a small difference means something.

### What is and is not already measured

| Marker | Meaning |
|---|---|
| **Measured** | run for this guide on 2026-11-16; the numbers are in the table |
| **Repo** | reported in `docs/driving.md` or stored under `runs/` |
| **Open** | not run; the table is yours to fill. No expected numbers are given, on purpose |

A results table for your own runs is in the [study tracker](STUDY-TRACKER.md).

---

## E1 · Policy latency

| | |
|---|---|
| **Hypothesis** | Reading chunks by observation time hides inference latency up to a limit, beyond which tracking degrades sharply rather than gradually. |
| **Independent variable** | latency: 0–400 ms; compensation on / off |
| **Held fixed** | teacher, 1.5 m/s, peanut, 30 s, watchdog raised to 1500 ms |
| **Command** | `./drive.sh sweep --max-plan-age-ms 1500 --latencies 0,50,100,200,250,300,400` |
| **Measurement** | lateral rms and max (cm) |

**Measured.**

| ms | compensated | not compensated |
|---:|---:|---:|
| 0 | 1.8 / 3.5 | 1.6 / 3.0 |
| 50 | 1.8 / 3.5 | 1.0 / 2.4 |
| 100 | 1.8 / 3.4 | 0.7 / 2.2 |
| 200 | 2.0 / 3.6 | 3.8 / 8.8 |
| 250 | 3.1 / 6.6 | 10.0 / 23.3 |
| 300 | 20.2 / 31.0 | 27.2 / 54.4 |
| 400 | 65.2 / 123.9 | 49.0 / 89.0 |

**Interpretation.** Supported: flat to 200 ms, a cliff between 250 and 300 ms. The
shape matters for design: there is no graceful degradation to rely on, so a latency
budget needs a margin, and a model whose latency sits near the cliff will fall off it
when the board throttles. The uncompensated advantage at 50–100 ms is a property of
this teacher ([chapter 5](05-action-chunking-and-latency.md)).

**Follow-up (open).** Refine the cliff: `--latencies 250,260,270,280,290,300`.

---

## E2 · Vehicle speed against latency

| | |
|---|---|
| **Hypothesis** | The tolerable latency is not a fixed time. It scales with $\ell/v$ (look-ahead over speed), so what breaks the loop is roughly a distance travelled on old information. |
| **Independent variables** | speed and latency |
| **Command** | `./drive.sh sweep --speed S --max-plan-age-ms 1500 --latencies …` |
| **Measurement** | compensated rms (cm) |

**Measured.**

| Speed | 100 ms | 250 ms | 400 ms | 650 ms | $v \cdot L$ at last stable → first unstable |
|---:|---:|---:|---:|---:|---|
| 0.6 m/s | 0.7 | 0.8 | 1.6 | 35.1 | 0.24 → 0.39 m |
| 1.0 m/s | 1.1 | 1.6 | 27.7 | | 0.25 → 0.40 m |
| 1.5 m/s | 1.8 | 3.1 | 65.2 | | 0.375 → 0.45 m (300 ms: 20.2 cm) |

**Interpretation.** Consistent with a threshold near $v \cdot L \approx 0.38$ m for
this teacher, and with the estimate $\tau^* \approx 0.52\,\ell/v$ of
[chapter 6](06-control-theory.md) in trend. The sweeps are too coarse to tell a
constant $v \cdot L$ from $\ell/v$ scaling (they differ because $\ell$ grows with $v$).

**Follow-up (open).** The grid that would separate the two models:

```bash
for s in 0.6 0.8 1.0 1.2 1.5; do
  echo "== $s"; ./drive.sh sweep --speed $s --max-plan-age-ms 1500 --latencies 200,250,300,350,400,450,500,550,600
done
```

**Engineering use.** A rule for choosing the speed limit from the measured latency of
whatever model is deployed.

---

## E3 · Maximum plan age

| | |
|---|---|
| **Hypothesis** | The car is braked whenever plan age exceeds the limit; since age sweeps $[L, 2L)$, the fraction of time braked is $(2L - A)/L$ for $L < A < 2L$, zero below, one above. |
| **Independent variable** | latency, at the default limit $A = 0.6$ s; then the limit, at a fixed latency |
| **Command** | `./drive.sh sweep --latencies 0,100,200,250,400,650` and `./drive.sh sim --latency-ms 400 --max-plan-age-ms A` |
| **Measurement** | `stale` seconds out of 30; laps |

**Measured** (default limit).

| Latency | Predicted fraction braked | `stale` (s of 30) | Laps |
|---:|---:|---:|---:|
| 250 ms | 0 | 0.3 (start-up) | 1.97 |
| 400 ms | 0.5 (0.45 on a 20 ms grid) | 13.9 | 0.14 |
| 650 ms | 1 | 30.0 | 0.00 |

**Interpretation.** Supported. Note the trade the limit expresses: at 400 ms and
1500 ms the car is not braked and leaves the road (65 cm rms); at 600 ms it is braked
half the time and stays on it (the 14.3 cm rms in the default sweep is over 0.14 laps
of stop-and-go). The watchdog converts an unsafe failure into a safe one. Raising it
to silence the warning reverses that.

**Follow-up (open).** At 400 ms, sweep `--max-plan-age-ms 600,700,800,900` and record
both `stale` and `off_track_pct`. Is there any setting that is both moving and on the road at 1.5 m/s?

---

## E4 · Chunk length (exhaustion)

| | |
|---|---|
| **Hypothesis** | A policy whose chunk horizon $H$ is shorter than $2L$ is braked for lack of a plan, independently of the watchdog. |
| **Independent variable** | latency, for a policy with $H = 0.47$ s |
| **Command** | `./drive.sh sim --policy lane-camera --latency-ms L --max-plan-age-ms 1500` |
| **Measurement** | `stale` seconds with the watchdog out of the way |

**Measured.**

| Latency | $2L$ against $H = 0.47$ | `stale` | Mean speed |
|---:|---|---:|---:|
| 100 ms | 0.2 < 0.47 | 0.1 s | 0.86 m/s |
| 300 ms | 0.6 > 0.47 | 12.2 s | 0.18 m/s |

**Interpretation.** Supported. The requirement $H \ge 2L$ is a hard constraint on the
pairing of a model with a board, separate from stability.

**Follow-up (open).** Construct `LaneCameraPolicy(…, chunk_size=30)` in a script and
repeat at 300 ms. The braking should disappear. Does the tracking hold?

---

## E5 · Steering noise during collection

| | |
|---|---|
| **Hypothesis** | Closed-loop success needs recovery data. Offline validation error will *not* track closed-loop success, and may move the other way. |
| **Independent variable** | `--steer-noise` |
| **Held fixed** | 40 episodes, peanut, seed 0, 15 epochs |
| **Commands** | below |
| **Measurement** | final `val_steer_error`; drives passed; rms; `max_lateral_m` reported by `collect` |

```bash
for n in 0 0.05 0.1 0.25 0.5; do
  p=runs/noise-$n
  make collect  ARGS="--project $p --steer-noise $n"
  make validate ARGS="--project $p"
  make train    V=v1 ARGS="--project $p"
  make evaluate V=v1 ARGS="--project $p"
done
```

**Repo** (two of the five points).

| `--steer-noise` | Validation steer error | Drives passed | Peanut |
|---:|---:|---:|---|
| 0 | 0.015 | 0/6 | off the road 75 % of the time |
| 0.25 | 0.024 | 6/6 | 1.5 cm rms |

**Open:** 0.05, 0.1, 0.5.

**Interpretation of the two known points.** Supported, strongly: the ordering by
offline error is the reverse of the ordering by driving.

**Questions the open points would answer.** Is there a threshold of noise below which
the model cannot drive, or a gradual improvement? Does too much noise hurt (the car
leaves the road during collection; the views become unlike anything it will see when
driving well)? What is the least disturbance that suffices, which matters when the
"teacher" is a person who must endure it?

---

## E6 · Dataset size, diversity and seed

| | |
|---|---|
| **Hypothesis** | (a) More episodes on the same track improve tracking with diminishing returns. (b) Adding other tracks helps more than adding the same amount of the same track. (c) The difference between `v1` and `v2` is within seed-to-seed variation. |
| **Independent variables** | `--episodes`; `--tracks`; `--seed` of `train` |
| **Measurement** | drives passed; mean rms and worst max over the six drives |

**Repo** (one run each).

| Model | Training data | Passed | rms | Worst |
|---|---|---:|---:|---:|
| `v1` | 40 sim episodes, peanut | 6/6 | 1.3 cm | 4.0 cm |
| `v2` | the same + 2 ROS recordings (oval, trefoil) | 6/6 | 1.2 cm | 3.8 cm |

**Open.** The runs that would make these comparisons meaningful:

```bash
# (c) seed variation on identical data
for s in 0 1 2 3 4; do make train V=seed$s ARGS="--project runs/lab10 --seed $s"; done
make evaluate V="seed0 seed1 seed2 seed3 seed4" ARGS='--project runs/lab10'

# (a) size
for e in 5 10 20 40 80; do
  make collect ARGS="--project runs/size-$e --episodes $e"; make train V=v1 ARGS="--project runs/size-$e"
  make evaluate V=v1 ARGS="--project runs/size-$e"
done

# (b) diversity at equal size
make collect ARGS='--project runs/div-1 --episodes 42'
make collect ARGS='--project runs/div-3 --episodes 42 --tracks peanut,oval,trefoil'
```

**Caution for (b).** `evaluate` drives the same three shapes. A model trained on all
three is no longer tested on unseen tracks; to keep a held-out test, train on two and
read the third.

**Interpretation.** Until (c) is done, "v2 is better than v1" is not established
([chapter 11](11-model-evaluation.md)).

---

## E7 · Track generalisation

| | |
|---|---|
| **Hypothesis** | A network trained on one track has learned local road-following, not the circuit, and so drives shapes and directions it has not seen. |
| **Command** | `make evaluate V=v1` |
| **Measurement** | pass and rms per track and direction |

**Repo.** `v1` (peanut only): rms 1.5 / 1.4 cm on the peanut (forward / reversed),
1.7 / 1.3 on the oval, 1.0 / 0.8 on the trefoil. 6/6.

**Interpretation.** Supported, within this simulator. The unseen tracks are *easier*
than the training track (gentler bends: tightest radius 1.44 and 1.83 m against
1.05 m), so this shows generalisation to milder geometry, not to harder.

**Follow-up (open).** Add a harder shape to `SHAPES` (tightest radius just above 1 m,
with an S-bend) in a scratch copy and evaluate `v1` on it. And train on the oval
only, then evaluate on the peanut: does generalisation run from easy to hard?

---

## E8 · Latency for a learned policy

| | |
|---|---|
| **Hypothesis** | A learned plan tolerates less latency than the teacher's, because its prediction degrades further into the chunk. |
| **Command** | in the container: `./drive.sh sim --policy cnn --checkpoint v1 --latency-ms L --max-plan-age-ms 1500 --duration-s 40` |
| **Measurement** | rms / max, off-road %, at the network's own speed (about 0.84 m/s) |

**Measured.**

| Latency | rms / max | Off the road |
|---:|---:|---:|
| 0 ms | 1.3 / 2.8 cm | 0 % |
| 100 ms | 1.5 / 4.0 cm | 0 % |
| 200 ms | 2.5 / 5.9 cm | 0 % |
| 300 ms | 22.6 / 54.5 cm | 6.2 % |
| 400 ms | 259.8 / 825.5 cm | 75.2 % |

**Interpretation.** Supported in outcome: at 0.84 m/s the network fails at 300 ms,
where the teacher at 1.0 m/s holds 1.6 cm at 250 ms and at 0.6 m/s holds 1.6 cm at
400 ms. The cause is not isolated: prediction quality and the coarser plan step
(0.1 s against 1/30 s) are confounded.

**Engineering use.** The latency budget must be measured with the policy that will be
deployed. For a VLA this is the experiment to run in simulation, at the Jetson's
measured latency, before any hardware test.

**Follow-up (open).** Separate the two causes: give the teacher a 16 × 0.1 s chunk
(`PurePursuitPolicy(…, chunk_size=16, chunk_dt_s=0.1)`) and repeat its sweep at 0.84 m/s.

---

## E9 · Safety behaviour

| | |
|---|---|
| **Hypothesis** | The supervisor stops the car at the margin regardless of approach speed, because the allowed speed is derived from the remaining room. |
| **Independent variable** | approach speed |
| **Command** | `./drive.sh sim --obstacle 8 --speed S` |
| **Measurement** | `min_range_m`, `collided`, seconds per obstacle rule |

**Measured.**

| Speed | Closest | Hit | `slow` / `brake` |
|---:|---:|---|---:|
| 0.8 m/s | 0.30 m | no | 0.12 / 0.12 s |
| 1.5 m/s | 0.30 m | no | 0.22 / 0.22 s |
| 1.5 m/s, `--no-safety` | 0.00 m | **yes** | |

**Interpretation.** Supported at these two speeds.

**Follow-up (open).** What the simulator cannot tell you, and a real car must:
(1) reduce the *actual* brake below the planned 4 m/s² by constructing
`World(vehicle=VehicleParams(max_brake_mps2=3.0))` and repeat: where does the car
stop now? (2) Put the cone off-centre with `track.obstacle(8.0, lateral_m=…)`: at what
offset does the sensor's ±15° cone miss it while the car's body still hits it?
(3) Feed a NaN range ([chapter 7](07-safety-supervisor.md)).

---

## E10 · Recovery from an offset

| | |
|---|---|
| **Hypothesis** | A model trained with disturbed steering returns to the centreline from an off-centre, askew start; a model trained on perfect driving does not. |
| **Independent variable** | starting lateral offset and heading offset; the model |
| **Command** | the script in [chapter 11](11-model-evaluation.md), "Practical experiment", 10 s from 5 m along the peanut |
| **Measurement** | rms, max, off-road % over the 10 s |

**Measured** (in the container).

| Model | Start: offset, heading | rms | max | Off the road |
|---|---|---:|---:|---:|
| `v1` (disturbed data) | 25 cm, 0.3 rad | 19.3 cm | 31.3 cm | 0 % |
| `v1` | 35 cm, 0.5 rad | 33.6 cm | 50.6 cm | 24 % |
| perfect-only | 25 cm, 0.3 rad | 91.4 cm | 302 cm | 61.5 % |
| perfect-only | 35 cm, 0.5 rad | 132 cm | 429 cm | 57.5 % |

**Interpretation.** Supported, with a caveat that the standard evaluation hides:
`v1`'s recovery is real but **limited**. From 25 cm and 0.3 rad it stays on the road;
from 35 cm and 0.5 rad it leaves it. Its training data placed the car within ±15 cm
and ±0.2 rad at the start, with disturbance on top. "6/6" said nothing about this.

**Follow-up (open).** Map the boundary: a grid of offsets and headings, pass or fail.
Then collect with a wider start range (it is hard-coded in `collect`; change it in a
scratch copy) or more noise, retrain, and see whether the boundary moves.

---

## E11 · Latency jitter and real threads

| | |
|---|---|
| **Hypothesis** | Because the executor indexes chunks by observation time, variable latency needs no special handling, until the slow tail crosses the watchdog. |
| **Commands** | `./drive.sh sim --jitter-ms 40`; `./drive.sh sim --latency-ms 250 --jitter-ms 40`; `./drive.sh sim --duration-s 5 --realtime` |
| **Measurement** | rms; `stale`; inference count and mean latency |

**Measured.**

| Run | Result |
|---|---|
| 100 ms ± 40 ms | 1.8 / 3.5 cm, `stale 0.12 s`: indistinguishable from no jitter |
| 250 ms ± 40 ms | 3.6 / 7.9 cm, `stale 0.46 s` (0.3 s without jitter): the slow inferences push the age past 0.6 s now and then |
| `--realtime`, 5 s | 42 inferences at a mean of 105.6 ms, against 50 at exactly 100 ms simulated |

**Interpretation.** Supported. Jitter is free while $2 \times$ (worst latency) stays
under the limit; near the limit it shows up as occasional braking, a useful early
warning on a real board.

**Follow-up (open).** On the target computer: run the driver node against the
simulator node with the real policy and log the distribution of plan ages. The
quantity to budget for is the 99th percentile, not the mean.

---

## E12 · The "perfect driving" detector

| | |
|---|---|
| **Hypothesis** | `validate`'s warning (median difference between applied and intended steering under 0.02) separates datasets that produce drivable models from those that do not. |
| **Command** | E5's loop; read the warning and the evaluation together |
| **Measurement** | warning present or absent, against drives passed |

**Measured** (the warning only): present at `--steer-noise 0` (3 episodes of 10 s),
absent at the default 0.25. **Open:** the noise level at which the warning disappears,
and whether models trained just above it can drive.

**Engineering use.** If the detector's threshold and the driving threshold disagree,
the cheap check gives false comfort; then the threshold in
[`validate`](../../drive/learn/validate.py#L62) should move.

---

## Recommended next engineering experiments

In order of value for the project's stated aim.

| # | Experiment | Why first |
|---|---|---|
| 1 | **E8 with a fine-tuned SmolVLA in the simulator** (PyTorch backend, PC only) | *started*: one 2000-step fine-tune drives and passes at 125 ms ([Lab 15](15-hands-on-labs.md#lab-15-fine-tune-smolvla-on-the-driving-data-on-this-laptop)). Still open: its latency sweep, `--action applied` against `intended`, more steps, more seeds |
| 2 | **E6 (c): seed variance** | every later comparison between models depends on knowing the noise floor; costs minutes |
| 3 | **E10: recovery boundary**, added to `evaluate` | the standard protocol cannot distinguish a robust model from a fragile one that happens to start centred |
| 4 | **E5: the noise sweep** | sizes the disturbance for when the demonstrator is a person |
| 5 | **E2: the fine grid** | turns "about 250 ms at 1.5 m/s" into a speed rule usable by the supervisor |
| 6 | **E11 on the Jetson** | the first measurement that involves the real computer and no car |
| 7 | **E9 with degraded brakes and an off-centre cone** | finds the supervisor's real envelope before a real car does |
| 8 | **Real-time chunking** against E1/E8 at 300–650 ms | only if 1, 5 and 6 show the latency × speed product is too large |

## A template for your own experiment

```markdown
### E<n> · <title>
- Date, commit, machine:
- Hypothesis (falsifiable):
- Prediction (numbers, written before running):
- Independent variable(s) and values:
- Held fixed:
- Command(s):
- Raw result (paste):
- Supported / refuted / inconclusive, and why:
- What it changes about a design decision:
- Next experiment it suggests:
```
