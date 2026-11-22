"""./drive.sh carla identify | check | collect   (needs the carla package: make carla-setup)
./vla.sh -m drive carla drive | dagger         (needs lerobot and carla: make vla-setup carla-vla-setup)

    identify   fit the car model to the car in CARLA -> drive/carla/vehicles/<blueprint>.json
    check      how far the model's predictions land from the real car: the labels' accuracy
    collect    record the teacher on random town routes, with a sentence per frame -> runs/carla
    drive      let a fine-tuned VLA drive new routes, told where to turn, and score it
    dagger     let it drive with the teacher labelling every frame and taking over: data to train on again
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .camera import CarlaCameraParams
from .collect import WEATHERS, CollectConfig

PROJECT = Path("runs/carla")
VEHICLE = "vehicle.tesla.model3"


def speeds(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(","))


def loop_arguments(q) -> None:
    """How the chunks are played (LoopConfig), for drive and dagger."""
    q.add_argument("--replan-ms", type=float, default=0.0,
                   help="at least this long between two observations given to the model; 0 (default): the "
                        "next as soon as a chunk is back, so only a chunk's first step or two is played")
    q.add_argument("--ensemble", action="store_true",
                   help="average every chunk that covers the moment (ACT's temporal ensembling) instead of "
                        "fading the newest in")
    q.add_argument("--ensemble-k", type=float, default=0.0,
                   help="with --ensemble, weights exp(-k x rank), rank 0 the oldest chunk (default 0: equal)")


def add_parser(commands) -> None:
    p = commands.add_parser("carla", help="CARLA: fit the car model, check the labels, record episodes",
                            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="carla_command", required=True)

    def server_arguments(q, seed=0):
        q.add_argument("--host", default="localhost")
        q.add_argument("--port", type=int, default=2000)
        q.add_argument("--town", help="refuse a server in another town (Town01 ... Town05)")
        q.add_argument("--vehicle", default=VEHICLE, help=f"the car's blueprint (default {VEHICLE})")
        q.add_argument("--seed", type=int, default=seed)

    q = sub.add_parser("identify", help="fit the car model to the car in CARLA")
    server_arguments(q)
    q.add_argument("--minutes", type=float, default=6.0, help="simulated minutes of driving to fit on (default 6)")
    q.add_argument("--out", type=Path, help="where to write the model (default drive/carla/vehicles/<vehicle>.json)")
    q.add_argument("--save-log", type=Path, metavar="NPZ",
                   help="also keep the raw drives, to fit again without a server: --from-log")
    q.add_argument("--from-log", type=Path, metavar="NPZ", help="fit on drives saved with --save-log; no server")

    q = sub.add_parser("check", help="measure the car model's predictions on collect-style drives")
    server_arguments(q)
    q.add_argument("--minutes", type=float, default=3.0)
    q.add_argument("--speeds", type=speeds, default=CollectConfig.speeds, metavar="LOW,HIGH")
    q.add_argument("--steer-noise", type=float, default=CollectConfig.steer_noise)
    q.add_argument("--save-log", type=Path, metavar="NPZ", help="keep the drives, to compare models without a server")

    q = sub.add_parser("collect", help="record the teacher on random routes, with a sentence per frame")
    server_arguments(q)
    q.add_argument("--project", type=Path, default=PROJECT, metavar="DIR", help=f"default {PROJECT}")
    q.add_argument("--name", default=None, help="episodes are <name>-000, ... (default carla-<town>)")
    q.add_argument("--episodes", type=int, default=CollectConfig.episodes)
    q.add_argument("--duration-s", type=float, default=CollectConfig.duration_s)
    q.add_argument("--speeds", type=speeds, default=CollectConfig.speeds, metavar="LOW,HIGH",
                   help="the range each episode's target speed is drawn from, m/s (default 4,8)")
    q.add_argument("--steer-noise", type=float, default=CollectConfig.steer_noise,
                   help="how hard the applied steering is pushed (DART); 0 records perfect driving only")
    q.add_argument("--speed-noise", type=float, default=CollectConfig.speed_noise,
                   help="drive at the target times 1 + this (standard deviation, wandering over seconds), bends "
                        "included, so the data shows the car too fast and the teacher braking it back (default 0)")
    q.add_argument("--announce-m", type=float, default=CollectConfig.announce_m,
                   help="a junction's command starts this many metres before it (default 30)")
    q.add_argument("--weathers", type=lambda s: tuple(s.split(",")), default=WEATHERS,
                   help="comma-separated carla.WeatherParameters presets to draw from")
    q.add_argument("--width", type=int, default=CarlaCameraParams.width)
    q.add_argument("--height", type=int, default=CarlaCameraParams.height)
    q.add_argument("--exposure", type=float, default=CarlaCameraParams.exposure_compensation, metavar="EV",
                   help=f"camera exposure compensation in stops (default {CarlaCameraParams.exposure_compensation:g})")
    q.add_argument("--show", action="store_true",
                   help="every recorded frame in a window as it is taken, with its label and sentence")
    q.add_argument("--video", type=Path, metavar="MP4", help="... or written to this video")
    q.add_argument("--resume", action="store_true",
                   help="keep the episodes already written under this name and seed, record the rest")

    q = sub.add_parser("drive", help="let a fine-tuned VLA drive new routes, told where to turn, and score it")
    server_arguments(q, seed=100)
    q.add_argument("--checkpoint", required=True,
                   help="a version in <project>/vla (smolvla-carla-v1), a LeRobot checkpoint directory, "
                        "or 'teacher': the teacher drives, through the same loop (the ceiling)")
    q.add_argument("--project", type=Path, default=PROJECT, metavar="DIR", help=f"default {PROJECT}")
    q.add_argument("--episodes", type=int, default=10)
    q.add_argument("--duration-s", type=float, default=60.0)
    q.add_argument("--latency-ms", type=float, default=100.0,
                   help="simulated time from an observation to its chunk (default 100)")
    q.add_argument("--off-route-m", type=float, default=2.5,
                   help="this far from the route ends a drive; a wrong turn ends there (default 2.5)")
    q.add_argument("--weathers", type=lambda s: tuple(s.split(",")), default=WEATHERS)
    q.add_argument("--speeds", type=speeds, default=(4.0, 8.0), metavar="LOW,HIGH",
                   help="the riding teacher's target speed range, as collect's --speeds (default 4,8)")
    loop_arguments(q)
    q.add_argument("--realtime", action="store_true",
                   help="run the simulation at the wall clock's pace, the model on its own thread: a chunk arrives "
                        "when it is computed (--latency-ms unused), as on the car, and the window plays smoothly "
                        "instead of racing between inferences and freezing in them")
    q.add_argument("--show", action="store_true", help="watch it drive: the model's plan in cyan, the teacher's in green")
    q.add_argument("--video", type=Path, metavar="MP4", help="... or write that to a video")
    q.add_argument("--report", type=Path, metavar="JSON",
                   help="where to write the scores (default <project>/eval/<checkpoint>-<town>-seed<seed>.json)")

    q = sub.add_parser("dagger", help="the model drives, the teacher labels every frame and takes over before "
                                      "it leaves the lane: episodes to train on again (DAgger)")
    server_arguments(q, seed=300)
    q.add_argument("--checkpoint", required=True, help="a version in <project>/vla, or a LeRobot checkpoint directory")
    q.add_argument("--project", type=Path, default=PROJECT, metavar="DIR", help=f"default {PROJECT}")
    q.add_argument("--name", default=None, help="episodes are <name>-000, ... (default dagger-<town>)")
    q.add_argument("--episodes", type=int, default=20)
    q.add_argument("--duration-s", type=float, default=60.0)
    q.add_argument("--latency-ms", type=float, default=100.0)
    q.add_argument("--speeds", type=speeds, default=(6.0, 6.0), metavar="LOW,HIGH",
                   help="the teacher's target speed: what the model was trained on (default 6,6)")
    loop_arguments(q)
    q.add_argument("--takeover-m", type=float, default=1.0,
                   help="the teacher takes over when the car, 0.5 s on, would be this far from the route (default 1)")
    q.add_argument("--weathers", type=lambda s: tuple(s.split(",")), default=WEATHERS)
    q.add_argument("--show", action="store_true", help="watch it: who drives, and the teacher's plan (the label)")
    q.add_argument("--video", type=Path, metavar="MP4", help="... or write that to a video")
    p.set_defaults(run=run)


def run(args) -> int:
    from .model import CarModel
    from .server import ServerConfig, session
    config = ServerConfig(args.host, args.port)
    rng = np.random.default_rng(args.seed)

    if args.carla_command == "identify":
        from .identify import fit, identify_drives, load_log, save_log
        if args.from_log:
            logged, nominal = load_log(args.from_log)
        else:
            # The teacher drives the identification drives with the model fitted last time, if
            # there is one: how it presses the throttle depends on the model, and the fit is
            # best where it drives. Running identify twice settles it.
            try:
                nominal = CarModel.load(args.vehicle)
            except ValueError:
                nominal = CarModel(blueprint=args.vehicle)
            with session(config, args.town) as s:
                nominal = _measured_geometry(s, nominal)
                print(f"identifying {args.vehicle} in {s.town}: {args.minutes:g} simulated minutes of excited driving")
                logged = identify_drives(s, nominal, args.minutes * 60, rng)
            if args.save_log:
                print(f"raw drives saved to {save_log(args.save_log, logged, nominal)}")
        model, report = fit(logged, nominal)
        path = model.save(args.out)
        print(json.dumps(report, indent=2))
        print(f"model written to {path}; next: ./drive.sh carla check")
        return 0

    model = CarModel.load(args.vehicle)
    if args.carla_command == "drive":
        return _drive(args, config, model)
    if args.carla_command == "dagger":
        return _dagger(args, config, model)
    if args.carla_command == "check":
        from .check import drive, prediction_errors, summary
        cfg = CollectConfig(speeds=args.speeds, steer_noise=args.steer_noise)
        with session(config, args.town) as s:
            print(f"checking {args.vehicle} in {s.town}: {args.minutes:g} simulated minutes, steer noise {cfg.steer_noise}")
            logs = drive(s, model, cfg, args.minutes * 60, rng)
            dt = s.tick_s
        if args.save_log:
            args.save_log.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_log, dt=dt, **{f"drive_{i:03d}": d for i, d in enumerate(logs)})
        print(json.dumps({"teacher": summary(logs, dt), "model_prediction_error": prediction_errors(logs, model, dt)},
                         indent=2))
        return 0

    from .collect import collect
    camera = CarlaCameraParams(width=args.width, height=args.height, exposure_compensation=args.exposure)
    with session(config, args.town) as s:
        cfg = CollectConfig(episodes=args.episodes, duration_s=args.duration_s, speeds=args.speeds,
                            steer_noise=args.steer_noise, speed_noise=args.speed_noise,
                            announce_m=args.announce_m, weathers=args.weathers,
                            name=args.name or f"carla-{s.town.lower()}", seed=args.seed, resume=args.resume)
        out = args.project / "episodes"
        print(f"recording {cfg.episodes} drives of up to {cfg.duration_s:.0f} s in {s.town} to {out}")
        done = collect(s, out, cfg, model, camera, show=args.show, video=args.video)
    frames = max(done["frames"], 1)
    print(f"{done['frames']} frames in {done['episodes']} episodes, {done['seconds']:.0f} s of wall time"
          + (f" ({done['skipped']} written before, kept)" if done["skipped"] else ""))
    print("  ended: " + ", ".join(f"{k} {v}" for k, v in sorted(done["ended"].items())))
    print("  commands: " + ", ".join(f"{k} {100 * v / frames:.0f} %" for k, v in sorted(done["commands"].items())))
    print(f"  {done['off_lane_frames']} frames ({100 * done['off_lane_frames'] / frames:.1f} %) with the car over a lane line")
    print(f"next: ./drive.sh validate --project {args.project}, then browse and export")
    return 0


def _dagger(args, config, model) -> int:
    from ..learn.project import Project
    from ..policy.lerobot_backend import make_vla_policy
    from .dagger import DaggerConfig, dagger
    from .server import session
    checkpoint = Project(args.project).checkpoint(args.checkpoint)
    policy = make_vla_policy(checkpoint, seed=args.seed)
    print(f"loaded {checkpoint} on {policy.backend.device}")
    out = args.project / "episodes"
    with session(config, args.town) as s:
        name = args.name or f"dagger-{s.town.lower()}"
        if any(out.glob(f"{name}-*")):
            raise ValueError(f"{out} has {name}-* episodes already; choose another --name")
        cfg = DaggerConfig(episodes=args.episodes, duration_s=args.duration_s, seed=args.seed, speeds=args.speeds,
                           latency_s=args.latency_ms / 1000, replan_s=args.replan_ms / 1000, ensemble=args.ensemble,
                           ensemble_k=args.ensemble_k, takeover_m=args.takeover_m, weathers=args.weathers, name=name)
        print(f"{cfg.episodes} drives of {args.checkpoint} in {s.town}, seed {cfg.seed}, the teacher labelling: {out}")
        done = dagger(s, policy, cfg, out, model, show=args.show, video=args.video, checkpoint=str(checkpoint))
    report = args.project / "eval" / f"{name}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"totals": done, "checkpoint": str(checkpoint), "seed": cfg.seed}, indent=2))
    print(f"{done['frames']} frames in {done['episodes']} episodes, the teacher driving {done['teacher_frames']}; "
          f"{done['takeovers']} takeovers in {done['driven_km']} km"
          + (f" ({done['km_per_takeover']} km each)" if done["km_per_takeover"] else "")
          + f"; {done['seconds']:.0f} s\n  ended: " + ", ".join(f"{k} {v}" for k, v in sorted(done["ended"].items()))
          + f"\n  totals in {report}")
    return 0


def _drive(args, config, model) -> int:
    from ..learn.project import Project
    from ..policy.lerobot_backend import make_vla_policy
    from .evaluate import EvalConfig, evaluate
    from .server import session
    if args.checkpoint == "teacher":                # the ceiling, and a check of the harness itself
        checkpoint, policy = "teacher", None
    else:
        checkpoint = Project(args.project).checkpoint(args.checkpoint)
        if not checkpoint.is_dir():
            raise ValueError(f"{checkpoint} is not a LeRobot checkpoint directory; carla drive runs a VLA")
        policy = make_vla_policy(checkpoint, seed=args.seed)
        print(f"loaded {checkpoint} on {policy.backend.device}")
    cfg = EvalConfig(episodes=args.episodes, duration_s=args.duration_s, seed=args.seed,
                     latency_s=args.latency_ms / 1000, off_route_m=args.off_route_m, weathers=args.weathers,
                     speeds=args.speeds, replan_s=args.replan_ms / 1000, ensemble=args.ensemble,
                     ensemble_k=args.ensemble_k, realtime=args.realtime)
    with session(config, args.town) as s:
        print(f"driving {cfg.episodes} new routes of up to {cfg.duration_s:.0f} s in {s.town}, seed {cfg.seed}")
        result = evaluate(s, policy, cfg, model, show=args.show, video=args.video, name=f"drive-{s.town.lower()}")
        town = s.town
    result["config"]["checkpoint"] = str(checkpoint)
    report = args.report or args.project / "eval" / f"{args.checkpoint.strip('/').replace('/', '_')}-{town.lower()}-seed{cfg.seed}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))
    m = result["summary"]
    print(f"{m['drives']} drives, {m['driven_km']} km, {m['wall_s']:.0f} s of wall time, "
          f"{m['inference_ms_mean']} ms per inference")
    print("  ended: " + ", ".join(f"{k} {v}" for k, v in sorted(m["ended"].items())))
    print(f"  {m['failures']} failures (a collision, off the route or stuck)"
          + (f", {m['km_per_failure']} km per failure" if m["failures"] else "") + f"; lateral rms {m['lateral_rms_m']} m")
    print("  junctions passed: " + ", ".join(f"{k} {v['passed']}/{v['reached']}" for k, v in m["junctions"].items()))
    print(f"scores written to {report}")
    return 0


def _measured_geometry(s, model):
    """The blueprint's axles and lock, read from CARLA, as the model's starting point."""
    import math
    from dataclasses import replace
    blueprint = s.world.get_blueprint_library().find(model.blueprint)
    vehicle = s.world.try_spawn_actor(blueprint, s.map.get_spawn_points()[0])
    if vehicle is None:
        raise RuntimeError("cannot spawn the vehicle to read its geometry: spawn point 0 is taken")
    try:
        s.world.tick()
        transform, physics = vehicle.get_transform(), vehicle.get_physics_control()
        forward = transform.get_forward_vector()

        def along(wheel):
            return ((wheel.position.x / 100 - transform.location.x) * forward.x
                    + (wheel.position.y / 100 - transform.location.y) * forward.y)
        wheels = physics.wheels
        front, rear = (along(wheels[0]) + along(wheels[1])) / 2, (along(wheels[2]) + along(wheels[3])) / 2
        extent = vehicle.bounding_box.extent
        return replace(model, wheelbase_m=front - rear, rear_axle_m=-rear,
                       max_steer_rad=math.radians(wheels[0].max_steer_angle),
                       steering_curve=tuple((round(p.x, 3), round(p.y, 3)) for p in physics.steering_curve),
                       length_m=2 * extent.x, width_m=2 * extent.y)
    finally:
        vehicle.destroy()
        s.world.tick()
