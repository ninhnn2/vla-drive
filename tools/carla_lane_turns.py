"""How often does the LANE, not the road, decide which way a car can turn at a junction?

    docker cp vla-carla:/workspace/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr runs/carla/xodr/
    .venv-carla/bin/python tools/carla_lane_turns.py runs/carla/xodr Town01 Town03 Town05

`collect`'s routes never change lane, and OpenDRIVE lets only the left lane turn left and
only the right lane turn right. Where the lane offers fewer maneuvers than the road, the
sentence is partly given away by the lane the car is in. The map is built client-side
from the town's OpenDRIVE file (`carla.Map`), so no server is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import carla  # noqa: E402

from drive.carla.route import maneuver  # noqa: E402
from drive.carla.routing import _exit, _heading  # noqa: E402


def maneuvers(waypoint) -> set:
    """What can be done at the junction right after `waypoint`, staying in its lane."""
    return {maneuver(_heading(waypoint), _heading(_exit(o, 1.0))) for o in waypoint.next(1.0) if o.is_junction}


def same_direction(waypoint) -> list:
    """`waypoint` and the driving lanes beside it that go the same way."""
    lanes, seen = [waypoint], {waypoint.lane_id}
    for side in ("get_left_lane", "get_right_lane"):
        lane = waypoint
        while True:
            lane = getattr(lane, side)()
            if (lane is None or lane.lane_type != carla.LaneType.Driving or lane.lane_id * waypoint.lane_id <= 0
                    or lane.lane_id in seen):
                break
            seen.add(lane.lane_id)
            lanes.append(lane)
    return lanes


def before_junction(waypoint, limit_m: int = 30):
    """The last waypoint of this lane before the junction ahead."""
    for _ in range(limit_m):
        following = waypoint.next(1.0)
        if not following or any(o.is_junction for o in following):
            break
        waypoint = following[0]
    return waypoint


def measure(world_map) -> dict:
    approaches = {}
    for waypoint in world_map.generate_waypoints(1.0):
        if waypoint.is_junction or waypoint.lane_type != carla.LaneType.Driving:
            continue
        following = waypoint.next(1.0)
        if any(o.is_junction for o in following):
            approaches.setdefault((waypoint.road_id, waypoint.lane_id, following[0].junction_id), waypoint)
    counts = {"approaches": 0, "multi_lane": 0, "fewer": 0, "only_one": 0}
    for waypoint in approaches.values():
        lane = maneuvers(waypoint)
        if not lane:
            continue
        lanes = same_direction(waypoint)
        road = set().union(*(maneuvers(before_junction(other)) for other in lanes))
        counts["approaches"] += 1
        counts["multi_lane"] += len(lanes) > 1
        counts["fewer"] += len(lane) < len(road)
        counts["only_one"] += len(lane) == 1 and len(road) > 1
    return counts


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    folder = Path(sys.argv[1])
    for town in sys.argv[2:]:
        c = measure(carla.Map(town, (folder / f"{town}.xodr").read_text()))
        n = max(c["approaches"], 1)
        print(f"{town}: {c['approaches']} junction approaches, {c['multi_lane']} on multi-lane roads; the lane allows "
              f"fewer maneuvers than the road at {c['fewer']} ({100 * c['fewer'] / n:.0f} %), only one where the road "
              f"allows more at {c['only_one']} ({100 * c['only_one'] / n:.0f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
