"""evaluate: judge models by driving them, on every track and in both directions.

    ./drive.sh evaluate v1 [v2 pure-pursuit ...]      reports/<model>.json, and a comparison

Low error on held-out frames does not mean a network can drive (see docs/driving.md).
This is the test that does. A drive passes when the car never leaves the road, hits
nothing and keeps moving.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ..episode import EpisodeConfig, run as drive
from ..sim import SHAPES
from .project import Project, add_project_argument

SCRIPTED = ("pure-pursuit", "lane-camera")


def evaluate(config: EpisodeConfig, tracks=tuple(SHAPES), log=print) -> dict:
    """Drive `config`'s policy on each track, forwards and reversed; print and return the results."""
    log(f"{'track':<18} {'laps':>5} {'m/s':>5} {'rms cm':>7} {'max cm':>7} {'off road':>9}  result")
    drives = []
    for track in tracks:
        for reverse in (False, True):
            r = drive(replace(config, track=track, reverse=reverse))
            ok = r["off_track_pct"] == 0 and not r["collided"] and r["mean_speed_mps"] > 0.3
            drives.append({"track": track, "reverse": reverse, "passed": ok,
                           **{k: r[k] for k in ("laps", "mean_speed_mps", "lateral_rms_m", "lateral_max_m",
                                                "off_track_pct", "collided", "safety_s")}})
            log(f"{track + (' reversed' if reverse else ''):<18} {r['laps']:5.2f} {r['mean_speed_mps']:5.2f} "
                f"{100 * r['lateral_rms_m']:7.1f} {100 * r['lateral_max_m']:7.1f} {r['off_track_pct']:7.1f} %  "
                f"{'pass' if ok else 'FAIL'}")
    passed = sum(d["passed"] for d in drives)
    log(f"{passed}/{len(drives)} drives passed")
    return {"policy": config.policy, "checkpoint": config.checkpoint, "latency_ms": config.latency_ms,
            "duration_s": config.duration_s, "passed": passed == len(drives),
            "summary": {"drives": len(drives), "drives_passed": passed,
                        "mean_speed_mps": round(sum(d["mean_speed_mps"] for d in drives) / len(drives), 2),
                        "lateral_rms_cm": round(100 * sum(d["lateral_rms_m"] for d in drives) / len(drives), 1),
                        "lateral_max_cm": round(100 * max(d["lateral_max_m"] for d in drives), 1)},
            "drives": drives}


def report_name(ref: str, checkpoint: Path) -> str:
    """What a report is called: the version, or for a path, the file's stem or the training run's directory."""
    if checkpoint.is_file():
        return Path(ref).stem
    if Path(ref) != checkpoint:                       # a VLA version, such as smolvla-v1
        return str(ref)
    parts = checkpoint.resolve().parts                # <run>/checkpoints/<step>/pretrained_model
    return parts[-4] if len(parts) >= 4 and parts[-3] == "checkpoints" else checkpoint.name


def add_parser(sub):
    p = sub.add_parser("evaluate", help="drive models on every track, both ways: the test that counts")
    add_project_argument(p)
    p.add_argument("models", nargs="+", metavar="model",
                   help=f"a trained version (v1, or a VLA's smolvla-v1), a checkpoint file or LeRobot "
                        f"checkpoint directory, or {' / '.join(SCRIPTED)}")
    p.add_argument("--tracks", type=lambda s: s.split(","), default=list(SHAPES))
    p.add_argument("--duration-s", type=float, default=40.0)
    p.add_argument("--latency-ms", type=float, default=100.0)


def run(project: Project, args) -> int:
    reports = {}
    for ref in args.models:
        if ref in SCRIPTED:
            policy, name = {"policy": ref}, ref
        else:
            checkpoint = project.checkpoint(ref)
            policy = {"policy": project.policy_for(checkpoint), "checkpoint": str(checkpoint)}
            name = report_name(ref, checkpoint)
        print(f"{ref}, {args.latency_ms:.0f} ms per inference, {args.duration_s:.0f} s per drive")
        reports[name] = evaluate(EpisodeConfig(duration_s=args.duration_s, latency_ms=args.latency_ms, **policy),
                                 tuple(args.tracks))
        path = project.report(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports[name], indent=2) + "\n")
        print(f"  report: {path}\n")
    if len(reports) > 1:
        print(f"{'model':<14} {'passed':>7} {'m/s':>5} {'rms cm':>7} {'worst cm':>9}")
        for name, report in reports.items():
            s = report["summary"]
            print(f"{name:<14} {s['drives_passed']:>3}/{s['drives']:<3} {s['mean_speed_mps']:5.2f} "
                  f"{s['lateral_rms_cm']:7.1f} {s['lateral_max_cm']:9.1f}")
    return 0 if all(report["passed"] for report in reports.values()) else 1
