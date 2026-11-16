"""status: where the project stands (episodes, labels, models, reports) and what to do next."""
from __future__ import annotations

import json
from collections import Counter

from .label import episodes, read_labels
from .project import Project, add_project_argument


def summary(project: Project) -> dict:
    found = episodes(project.episodes)
    frames = seconds = usable = 0
    sources: Counter = Counter()
    for episode in found:
        meta = json.loads((episode / "meta.json").read_text())
        frames += meta["frames"]
        seconds += meta["frames"] / meta["fps"]
        labels = read_labels(episode)
        sources[labels["source"] if labels else "unlabelled"] += 1
        usable += int(labels["valid"].sum()) if labels else 0
    models = []
    for version in project.versions():
        trained = json.loads(project.model(version).with_suffix(".json").read_text())
        report = project.report(version)
        models.append({"version": version, "frames": trained["frames"], "epochs": trained["epochs"],
                       "val_steer_error": trained["history"][-1]["val_steer_error"],
                       "evaluation": json.loads(report.read_text())["summary"] if report.is_file() else None})
    return {"project": str(project.home), "episodes": len(found), "frames": frames, "minutes": round(seconds / 60, 1),
            "labelled_frames": usable, "label_sources": dict(sources), "models": models}


def next_step(s: dict) -> str:
    if not s["episodes"]:
        return "record drives: ./drive.sh collect, or over ROS 2 (make ros-drive RECORD=1)"
    if s["label_sources"].get("unlabelled"):
        return "./drive.sh label"
    if not s["models"]:
        return "./drive.sh validate, then ./drive.sh train v1"
    unjudged = [m["version"] for m in s["models"] if m["evaluation"] is None]
    if unjudged:
        return f"./drive.sh evaluate {' '.join(unjudged)}"
    return "more or more varied episodes, then train the next version"


def add_parser(sub):
    add_project_argument(sub.add_parser("status", help="what the project holds, and the next step"))


def run(project: Project, args) -> int:
    s = summary(project)
    print(f"project {s['project']}")
    print(f"  episodes   {s['episodes']}: {s['frames']} frames, {s['minutes']} min")
    if s["episodes"]:
        kinds = ", ".join(f"{count} {source}" for source, count in sorted(s["label_sources"].items()))
        print(f"  labels     {s['labelled_frames']} usable frames ({kinds})")
    for m in s["models"]:
        e = m["evaluation"]
        judged = "not evaluated yet" if e is None else (
            f"drives {e['drives_passed']}/{e['drives']} passed, {e['lateral_rms_cm']} cm rms off the centerline")
        print(f"  model      {m['version']}: {m['frames']} frames, {m['epochs']} epochs, "
              f"validation steer error {m['val_steer_error']}; {judged}")
    print(f"  next       {next_step(s)}")
    return 0
