# Study tracker

Copy this file, or edit it in place, and keep it honest. A chapter is done when you
can answer its knowledge check without looking and have run its experiment, not when
you have read it.

**Started:** ____________  **Path:** ☐ Beginner ☐ Intermediate ☐ Advanced
([roadmap](00-learning-roadmap.md))

## How to mark progress

| Column | Tick it when |
|---|---|
| **Status** | write `—` (not started), `…` (in progress) or `✓` (completed) |
| **Read** | you have read the chapter with the source open beside it |
| **Worked** | you did the numerical examples by hand and they matched |
| **Experiment** | you ran the chapter's practical experiment and wrote down prediction and result |
| **Check** | you answered the knowledge check without looking at the answers |
| **Understood** | you could explain the chapter's main idea to someone else in two minutes |

## Chapters

| # | Chapter | Status | Read | Worked | Experiment | Check | Understood | Questions remaining |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 00 | [Learning roadmap](00-learning-roadmap.md) | | ☐ | n/a | n/a | ☐ | ☐ | |
| 01 | [System architecture](01-system-architecture.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 02 | [Data flow and lifecycle](02-data-flow-and-lifecycle.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 03 | [Vehicle simulation](03-vehicle-simulation.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 04 | [Driving policies](04-driving-policies.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 05 | [Action chunking and latency](05-action-chunking-and-latency.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 06 | [Control theory](06-control-theory.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 07 | [Safety supervisor](07-safety-supervisor.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 08 | [Imitation learning](08-imitation-learning.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 09 | [Dataset pipeline](09-dataset-pipeline.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 10 | [Model training](10-model-training.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 11 | [Model evaluation](11-model-evaluation.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 12 | [ROS 2 integration](12-ros2-integration.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 13 | [VLA and SmolVLA](13-vla-and-smolvla.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 14 | [Jetson deployment](14-jetson-deployment.md) | | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 15 | [Hands-on labs](15-hands-on-labs.md) | | ☐ | n/a | see below | n/a | ☐ | |
| 16 | [Experiments](16-experiments.md) | | ☐ | n/a | see below | n/a | ☐ | |

## Concepts

Tick a concept when you can define it, point to where it lives in the code, and give
a number that illustrates it. Definitions are in the [glossary](GLOSSARY.md).

| Concept | Define | Find in code | Give a number | Notes |
|---|:---:|:---:|:---:|---|
| Action chunk | ☐ | ☐ | ☐ | |
| Latency compensation | ☐ | ☐ | ☐ | |
| Plan age | ☐ | ☐ | ☐ | |
| Chunk exhaustion | ☐ | ☐ | ☐ | |
| Stale-plan watchdog | ☐ | ☐ | ☐ | |
| Kinematic bicycle model | ☐ | ☐ | ☐ | |
| Servo lag | ☐ | ☐ | ☐ | |
| Pure pursuit | ☐ | ☐ | ☐ | |
| Closed-loop against open-loop control | ☐ | ☐ | ☐ | |
| Delay margin | ☐ | ☐ | ☐ | |
| RMS tracking error | ☐ | ☐ | ☐ | |
| Safety supervisor (all seven rules) | ☐ | ☐ | ☐ | |
| Behaviour cloning | ☐ | ☐ | ☐ | |
| Covariate shift and compounding error | ☐ | ☐ | ☐ | |
| DART (disturbed steering, clean labels) | ☐ | ☐ | ☐ | |
| Offline against closed-loop evaluation | ☐ | ☐ | ☐ | |
| Episode format (`episode.npz`, `labels.npz`) | ☐ | ☐ | ☐ | |
| `Platform` protocol and `RosPlatform` | ☐ | ☐ | ☐ | |
| The five `/drive/*` topics | ☐ | ☐ | ☐ | |
| VLA, action expert, flow matching | ☐ | n/a | ☐ | |
| LeRobotDataset | ☐ | n/a | ☐ | |
| TensorRT engine | ☐ | n/a | ☐ | |

## Labs

| # | Lab | Status | Ran the commands | Result matched the expected | Answered the questions | Challenge done | Notes |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| 1 | [Run the simulator](15-hands-on-labs.md#lab-1-run-the-simulator) | | ☐ | ☐ | ☐ | ☐ | |
| 2 | [Pure pursuit](15-hands-on-labs.md#lab-2-understand-the-pure-pursuit-policy) | | ☐ | ☐ | ☐ | ☐ | |
| 3 | [Lane camera](15-hands-on-labs.md#lab-3-run-the-lane-camera-policy) | | ☐ | ☐ | ☐ | ☐ | |
| 4 | [Inject latency](15-hands-on-labs.md#lab-4-inject-inference-latency) | | ☐ | ☐ | ☐ | ☐ | |
| 5 | [Compensated against uncompensated](15-hands-on-labs.md#lab-5-compare-compensated-and-uncompensated-chunks) | | ☐ | ☐ | ☐ | ☐ | |
| 6 | [Safety with an obstacle](15-hands-on-labs.md#lab-6-test-safety-with-an-obstacle) | | ☐ | ☐ | ☐ | ☐ | |
| 7 | [Safety disabled](15-hands-on-labs.md#lab-7-disable-safety-and-analyse-the-result) | | ☐ | ☐ | ☐ | ☐ | |
| 8 | [Collect episodes](15-hands-on-labs.md#lab-8-collect-training-episodes) | | ☐ | ☐ | ☐ | ☐ | |
| 9 | [Validate and browse](15-hands-on-labs.md#lab-9-validate-and-browse-a-dataset) | | ☐ | ☐ | ☐ | ☐ | |
| 10 | [Train a CNN](15-hands-on-labs.md#lab-10-train-a-cnn) | | ☐ | ☐ | ☐ | ☐ | |
| 11 | [Evaluate on tracks](15-hands-on-labs.md#lab-11-evaluate-on-different-tracks) | | ☐ | ☐ | ☐ | ☐ | |
| 12 | [ROS 2 simulation](15-hands-on-labs.md#lab-12-run-the-ros-2-simulation) | | ☐ | ☐ | ☐ | ☐ | |
| 13 | [Record teleoperation](15-hands-on-labs.md#lab-13-record-teleoperation-data) | | ☐ | ☐ | ☐ | ☐ | |
| 14 | [VLA deployment analysis](15-hands-on-labs.md#lab-14-analyse-the-vla-deployment-architecture) | | ☐ | n/a | ☐ | n/a | |
| 15 | [Fine-tune SmolVLA on the laptop](15-hands-on-labs.md#lab-15-fine-tune-smolvla-on-the-driving-data-on-this-laptop) | | ☐ | ☐ | ☐ | ☐ | |

## Experiments

For each: write the prediction before running. "Open" experiments have no reference
result; yours is the first.

| # | Experiment | Reference result | Status | My prediction | My result | Supported? |
|---|---|---|:---:|---|---|:---:|
| E1 | [Policy latency](16-experiments.md#e1--policy-latency) | measured | | | | |
| E2 | [Speed against latency](16-experiments.md#e2--vehicle-speed-against-latency) | measured (coarse); fine grid open | | | | |
| E3 | [Maximum plan age](16-experiments.md#e3--maximum-plan-age) | measured | | | | |
| E4 | [Chunk length](16-experiments.md#e4--chunk-length-exhaustion) | measured | | | | |
| E5 | [Steering noise](16-experiments.md#e5--steering-noise-during-collection) | repo (2 of 5 points) | | | | |
| E6 | [Size, diversity, seed](16-experiments.md#e6--dataset-size-diversity-and-seed) | open | | | | |
| E7 | [Track generalisation](16-experiments.md#e7--track-generalisation) | repo | | | | |
| E8 | [Latency for a learned policy](16-experiments.md#e8--latency-for-a-learned-policy) | measured | | | | |
| E9 | [Safety behaviour](16-experiments.md#e9--safety-behaviour) | measured; follow-ups open | | | | |
| E10 | [Recovery from an offset](16-experiments.md#e10--recovery-from-an-offset) | measured (4 points) | | | | |
| E11 | [Jitter and real threads](16-experiments.md#e11--latency-jitter-and-real-threads) | measured | | | | |
| E12 | [The perfect-driving detector](16-experiments.md#e12--the-perfect-driving-detector) | partly open | | | | |

## Acceptance checklist

The guide has done its job when you can tick all of these.

- ☐ I can describe the whole system without opening a source file.
- ☐ I can trace an observation from the camera to a motor command, naming each class it passes through.
- ☐ I can explain how a 100 ms policy and a 50 Hz loop work together, and what "plan age" is.
- ☐ Given a chunk, a latency and a time, I can compute which interpolated action is played, with and without compensation.
- ☐ I can state why the loop fails near 250 ms at 1.5 m/s, and why the threshold moves with speed.
- ☐ I can predict the supervisor's decision for any command, speed, range and plan age.
- ☐ I have run collect → validate → browse → train → evaluate on my own project directory.
- ☐ I can explain, with the repository's own numbers, why disturbed-steering data beats perfect driving, and why validation error did not show it.
- ☐ I can read one episode from disk and say what every array means.
- ☐ I have run every lab I have the hardware for.
- ☐ I can draw the ROS 2 node graph and say what a real car's node must provide.
- ☐ I can list what is implemented, what is tested only against a stub, and what does not exist on the path to a VLA on a Jetson.

## Open questions

Write them as they arise; revisit after each chapter. A question that survives three
chapters is a good candidate for an experiment in [chapter 16](16-experiments.md).

| Date | Chapter | Question | Resolved by |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

## My experiment log

Use the template at the end of [chapter 16](16-experiments.md).

| Date | Commit | Experiment | Command | Result | Conclusion |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
