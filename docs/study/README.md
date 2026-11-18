# Study guide: the vla-drive driving stack

A curriculum built from this repository's own code. Each chapter teaches one idea
(theory first, then the lines that implement it), and ends with an experiment you can
run and questions you should be able to answer.

It complements, and does not replace, [`README.md`](../../README.md) and
[`docs/driving.md`](../driving.md): those say what the project is and how to run it;
this says why it is built the way it is.

## Chapters

| # | Chapter | You will be able to |
|---|---|---|
| 00 | [Learning roadmap](00-learning-roadmap.md) | pick a path through the material for your background |
| 01 | [System architecture](01-system-architecture.md) | name every module and what it may and may not do |
| 02 | [Data flow and execution lifecycle](02-data-flow-and-lifecycle.md) | trace one observation from the camera to the motors |
| 03 | [Vehicle simulation](03-vehicle-simulation.md) | step the car model by hand and say where it differs from a real car |
| 04 | [Driving policies](04-driving-policies.md) | explain the four policies and choose between them |
| 05 | [Action chunking and latency compensation](05-action-chunking-and-latency.md) | compute which action of a chunk is played at any instant, and why the loop fails near 250 ms |
| 06 | [Control theory](06-control-theory.md) | read the stack as a set of feedback loops with delays |
| 07 | [Safety supervisor](07-safety-supervisor.md) | predict the supervisor's decision for any input |
| 08 | [Imitation learning fundamentals](08-imitation-learning.md) | explain covariate shift and why the steering is disturbed while recording |
| 09 | [Dataset pipeline](09-dataset-pipeline.md) | read an episode from disk and run collect → evaluate |
| 10 | [Model training](10-model-training.md) | derive the network's shapes and parameter count |
| 11 | [Model evaluation](11-model-evaluation.md) | read an evaluation report and say what it does not show |
| 12 | [ROS 2 integration](12-ros2-integration.md) | draw the node graph and replace the simulator with another car |
| 13 | [VLA and SmolVLA](13-vla-and-smolvla.md) | separate what is built from what is planned on the way to a VLA |
| 14 | [Jetson Orin Nano deployment](14-jetson-deployment.md) | list what remains before this drives a real car |
| 15 | [Hands-on labs](15-hands-on-labs.md) | run 14 practical exercises |
| 16 | [Experiments and engineering analysis](16-experiments.md) | design and interpret your own experiments |
| 17 | [Debugging the model and the algorithm](17-debugging-the-model.md) | localise a fault: data, training, the model offline, or the loop |
| | [Glossary](GLOSSARY.md) · [References](REFERENCES.md) · [Study tracker](STUDY-TRACKER.md) | |
| | [Thực hành (tiếng Việt)](../thuc-hanh/README.md): 7 bài | collect, read and judge data; name every layer; train and read the curves; analyse the model's output offline and in closed loop with `./drive.sh probe`; run 12 optimisation experiments; debug; the learning-rate range test, gradients and update ratios, input ablation, on-policy error and DAgger |
| | [Architecture diagrams](diagrams/vla-drive-architecture.drawio) (draw.io, 5 pages) | see the pipeline, the runtime loop and the SmolVLA model at a glance; find your way in the source; plan how to improve the model |

## How every statement is labelled

The project describes a destination (a fine-tuned VLA on a Jetson driving a car) that
it has not reached. So that the two are never confused, claims carry one of these tags:

| Tag | Meaning |
|---|---|
| **[TESTED]** | Implemented, and a test in [`tests/`](../../tests) exercises it |
| **[UNTESTED]** | Implemented, but not exercised by a test, or only against a stand-in |
| **[SIM-ONLY]** | True of the simulator; nothing says it holds on a real car |
| **[PROPOSED]** | Named as a next step in the project's documentation; no code exists |
| **[FUTURE]** | A direction suggested by this guide or the literature; not planned in the repo |

## Conventions

- **Source links** are relative to this directory, with a line anchor at the start of
  the class or function: [`DriveLoop.step`](../../drive/loop.py#L32). Line numbers were
  taken from the working tree on 2026-11-16 and will drift as the code changes; the
  class and function names are the stable reference.
- **Numbers.** A figure marked *measured* was produced for this guide by running the
  stated command on 2026-11-16 (host Python, numpy 2.2.6, OpenCV 4.11). A figure marked
  *repo* is quoted from [`docs/driving.md`](../driving.md) or from the reports under
  `runs/`. Nothing else is a measurement. Pure-pursuit runs in the simulator are
  deterministic, so you should reproduce them exactly; runs that render the camera
  (`lane-camera`, `cnn`) can differ slightly between OpenCV versions.
- **Units** are SI. An action is `[steer, throttle]`, both in $[-1, 1]$: steer $+1$ is
  full left, throttle $+1$ full power, $-1$ full brake. There is no reverse gear.
- **Commands.** `make …` runs in the project's Docker container; on a Mac it runs
  `drive/` natively in `.venv` (`make setup`) and only ROS 2 in a container.
  `./drive.sh …` also runs directly on a PC that has numpy and OpenCV; only `train` and
  the `cnn` policy need PyTorch, which the container and `.venv` have.
- **Diagrams** are Mermaid and **equations** are LaTeX; both render on GitHub and in
  VS Code's Markdown preview (with a Mermaid extension).

## State of the working tree this guide describes

Branch `smolvla-sim`: the initial commit (`34e83ed`) plus `collect --show/--video`,
`browse --match`, the `export` stage, the `vla` policy (`LeRobotBackend`), the
`vla/<version>` checkpoints of `Project`, the HUD's debug panel and `sim --log`.
82 tests pass in the container (`make test`); on the host Python, 5 of them are
skipped (PyTorch, ROS 2, LeRobot).
