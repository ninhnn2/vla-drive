"""Drive a simulated car with an action-chunking policy.

    ./drive.sh sim --show                          watch the default drive
    ./drive.sh sim --latency-ms 400 --video out.mp4
    ./drive.sh sim --obstacle 8 --show             a cone on the track: safety stops the car
    ./drive.sh sweep                               tracking error against inference latency

From drives to a network that drives (a project lives in runs/drive):

    ./drive.sh collect                 record the teacher, with disturbed steering
    ./drive.sh label                   label drives that came without a plan (ROS 2, a person)
    ./drive.sh validate                check the dataset
    ./drive.sh browse                  look at frames with their labels drawn on the road
    ./drive.sh train v1                train a network (needs PyTorch: make train)
    ./drive.sh evaluate v1             drive it on every track
    ./drive.sh sim --policy cnn --checkpoint v1 --show
    ./drive.sh status                  what there is, and the next step

Towards a VLA (needs an environment with lerobot; see docs/study/13-vla-and-smolvla.md):

    python -m drive export             the episodes as a LeRobotDataset (runs/drive/lerobot)
    lerobot-train ...                  fine-tune SmolVLA into runs/drive/vla/<version>
    python -m drive sim --policy vla --checkpoint smolvla-v1
    python -m drive evaluate smolvla-v1
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .episode import EpisodeConfig, run
from .learn import STAGES
from .learn.project import Project, add_project_argument
from .policy import POLICIES
from .sim import SHAPES


def episode_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", choices=POLICIES, default="pure-pursuit",
                   help="pure-pursuit: a teacher that knows the map; lane-camera: follows the road it sees; "
                        "cnn: a network trained with ./drive.sh train")
    p.add_argument("--checkpoint", help="for --policy cnn: a trained version (v1) or a checkpoint file; "
                                        "for --policy vla: a version in <project>/vla (smolvla-v1) or a "
                                        "LeRobot checkpoint directory")
    add_project_argument(p)
    p.add_argument("--track", choices=list(SHAPES), default="peanut")
    p.add_argument("--reverse", action="store_true", help="drive the track clockwise")
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--speed", type=float, help="target speed in m/s (default: the policy's own)")
    p.add_argument("--obstacle", type=float, action="append", default=[], metavar="S",
                   help="put a cone S metres along the track; repeatable")
    p.add_argument("--no-safety", action="store_true", help="remove the supervisor, to see what it prevents")
    p.add_argument("--max-plan-age-ms", type=float, default=600.0,
                   help="the supervisor brakes when the plan in use is older than this (default 600)")
    p.add_argument("--seed", type=int, default=0)


def config_from(args, **overrides) -> EpisodeConfig:
    checkpoint = args.checkpoint and str(Project(args.project).checkpoint(args.checkpoint))
    fields = dict(policy=args.policy, checkpoint=checkpoint,
                  track=args.track, reverse=args.reverse, duration_s=args.duration_s, target_speed_mps=args.speed,
                  obstacles_s=tuple(args.obstacle), safety=not args.no_safety,
                  max_plan_age_ms=args.max_plan_age_ms, seed=args.seed)
    return EpisodeConfig(**{**fields, **overrides})


def cmd_sim(args) -> int:
    config = config_from(args, latency_ms=args.latency_ms, jitter_ms=args.jitter_ms,
                         latency_compensation=not args.no_latency_compensation,
                         realtime=args.realtime or args.show)
    on_tick, closers = None, []
    if args.show or args.video or args.record:
        from .recorder import Recorder
        from .viewer import Hud, Output
        output = Output(args.show, args.video, fps=25) if args.show or args.video else None
        recorder = Recorder(args.record, task="drive along the track") if args.record else None
        hud, count = None, 0

        def on_tick(world, loop, tick):
            nonlocal hud, count
            if recorder is not None and recorder.due(tick.t):
                recorder.add(tick.t, world.image(), world.state.speed, world.state.steer, tick.decision.action)
            count += 1
            if output is None or count % 2:
                return True
            hud = hud or Hud(world)
            key = output.put(hud.frame(world, loop, tick))
            if key == "e" and loop.safety is not None:
                loop.safety.estop(not loop.safety.estopped)
            return key not in ("q", "\x1b")

        closers = [c for c in (output and output.close, recorder and recorder.save) if c]
    try:
        result = run(config, on_tick)
    finally:
        for close in closers:
            close()
    print(report(result))
    for label, path in (("video", args.video), ("recording", args.record)):
        if path:
            print(f"  {label}: {path}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  result: {args.out}")
    return 0


def report(r: dict) -> str:
    c = r["config"]
    lines = [f"{c['policy']} on {c['track']}{' reversed' if c['reverse'] else ''}, "
             f"{r['latency_ms_mean']} ms per inference ({r['inferences']} inferences), "
             f"latency compensation {'on' if c['latency_compensation'] else 'off'}, "
             f"safety {'on' if c['safety'] else 'OFF'}",
             f"  {r['laps']} laps in {r['duration_s']} s at {r['mean_speed_mps']} m/s",
             f"  off the centerline: rms {100 * r['lateral_rms_m']:.1f} cm, max {100 * r['lateral_max_m']:.1f} cm, "
             f"off the road {r['off_track_pct']} % of the time"]
    if c["obstacles_s"]:
        lines.append(f"  obstacles: {'HIT' if r['collided'] else 'not hit'}, closest {r['min_range_m']} m")
    if r["safety_s"]:
        lines.append("  safety overrode the policy: " + ", ".join(f"{k} {v} s" for k, v in r["safety_s"].items()))
    return "\n".join(lines)


def cmd_sweep(args) -> int:
    print(f"{args.policy}, {args.duration_s:.0f} s per run. rms/max: distance from the centerline; "
          "stale: time braked because the plan was too old\n")
    print(f"{'latency':>8} | {'latency compensated':^30} | {'not compensated':^30}")
    columns = f"{'rms cm':>7} {'max cm':>7} {'laps':>5} {'stale s':>8}"
    print(f"{'ms':>8} | {columns} | {columns}")
    for latency in args.latencies:
        cells = []
        for compensate in (True, False):
            r = run(config_from(args, latency_ms=latency, latency_compensation=compensate))
            cells.append(f"{100 * r['lateral_rms_m']:7.1f} {100 * r['lateral_max_m']:7.1f} "
                         f"{r['laps']:5.2f} {r['safety_s'].get('stale', 0.0):8.1f}")
        print(f"{latency:8.0f} | {cells[0]} | {cells[1]}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="drive.sh", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    sim = commands.add_parser("sim", help="run one simulated drive")
    episode_arguments(sim)
    sim.add_argument("--latency-ms", type=float, default=100.0, help="time one inference takes (default 100)")
    sim.add_argument("--jitter-ms", type=float, default=0.0, help="standard deviation of the latency")
    sim.add_argument("--no-latency-compensation", action="store_true",
                     help="play each chunk from its first action instead of skipping the stale ones")
    sim.add_argument("--show", action="store_true", help="open a window and run in real time")
    sim.add_argument("--realtime", action="store_true", help="real clock and inference thread, without a window")
    sim.add_argument("--video", type=Path, help="write the HUD to an .mp4")
    sim.add_argument("--record", type=Path, help="save frames, states and actions to this directory")
    sim.add_argument("--out", type=Path, help="write the result as JSON")
    sim.set_defaults(run=cmd_sim)

    sweep = commands.add_parser("sweep", help="tracking error across inference latencies")
    episode_arguments(sweep)
    sweep.add_argument("--latencies", type=lambda s: [float(x) for x in s.split(",")],
                       default=[0, 50, 100, 200, 400, 650], help="comma-separated ms")
    sweep.set_defaults(run=cmd_sweep)

    for stage in STAGES:
        stage.add_parser(commands)

    args = parser.parse_args(argv)
    if getattr(args, "duration_s", 1.0) <= 0:
        parser.error("--duration-s must be positive")
    try:
        if hasattr(args, "run"):                       # sim, sweep
            return args.run(args)
        stage = next(s for s in STAGES if s.__name__.rsplit(".", 1)[-1] == args.command)
        return stage.run(Project(args.project), args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"drive.sh: {error}\n")
