"""Random routes through a CARLA town, from its waypoints.

The pip wheel has no `agents` package (no GlobalRoutePlanner), so routes are walked on
the core API: `waypoint.next(step)` returns one waypoint along the lane, or several where
the lane branches, which is at the entrance of a junction. There each branch is followed
through the junction to see where it leads, and the walk picks a MANEUVER uniformly
(left, right, straight: whichever exist), then a branch with it. Picking a branch
uniformly instead would favour going straight wherever a junction has two lanes doing so.
"""
from __future__ import annotations

import numpy as np

from .frames import pose_from_carla
from .route import Junction, Route, maneuver

#: Followed through a junction at most this far to find where a branch leads.
JUNCTION_MAX_M = 120.0


def _heading(waypoint) -> float:
    return pose_from_carla(0.0, 0.0, waypoint.transform.rotation.yaw)[2]


def _exit(waypoint, step_m: float):
    """The first waypoint past the junction `waypoint` is in (or `waypoint` if it is not in one)."""
    travelled = 0.0
    while waypoint.is_junction and travelled < JUNCTION_MAX_M:
        following = waypoint.next(step_m)
        if not following:
            break
        waypoint, travelled = following[0], travelled + step_m
    return waypoint


def choose(options, heading_in: float, rng, step_m: float):
    """One of the branches `options`: a maneuver picked uniformly, then a branch that does it."""
    if not any(o.is_junction for o in options):          # a lane split outside a junction
        return options[int(rng.integers(len(options)))]
    by_maneuver: dict[str, list] = {}
    for option in options:
        by_maneuver.setdefault(maneuver(heading_in, _heading(_exit(option, step_m))), []).append(option)
    kinds = sorted(by_maneuver)
    picked = kinds[int(rng.integers(len(kinds)))]
    return by_maneuver[picked][int(rng.integers(len(by_maneuver[picked])))]


def random_route(start, length_m: float, rng, step_m: float = 1.0) -> Route:
    """A route of up to `length_m` metres from the waypoint `start`, branching at random.

    It ends early where the road does, and never inside a junction: a route that would end
    in one is cut before it, so every junction on it is driven through to the other side."""
    waypoint = start
    x, y, _ = pose_from_carla(start.transform.location.x, start.transform.location.y, 0.0)
    points, headings, widths = [(x, y)], [_heading(start)], [start.lane_width]
    along, junctions = 0.0, []
    entry: tuple[float, float] | None = None                  # where the route entered a junction, and its heading
    while along < length_m:
        options = waypoint.next(step_m)
        if not options:
            break
        following = choose(options, headings[-1], rng, step_m) if len(options) > 1 else options[0]
        x, y, _ = pose_from_carla(following.transform.location.x, following.transform.location.y, 0.0)
        along += float(np.hypot(x - points[-1][0], y - points[-1][1]))
        if following.is_junction and entry is None:
            entry = (along, headings[-1])
        elif not following.is_junction and entry is not None:
            s_entry, heading_in = entry
            junctions.append(Junction(s_entry, along, maneuver(heading_in, _heading(following))))
            entry = None
        points.append((x, y))
        headings.append(_heading(following))
        widths.append(following.lane_width)
        waypoint = following
    if entry is not None:                                  # ended inside a junction: stop short of it
        keep = int(np.searchsorted(np.cumsum(np.r_[0.0, np.hypot(*np.diff(np.array(points), axis=0).T)]),
                                   entry[0] - 2.0))
        points, headings = points[:max(keep, 2)], headings[:max(keep, 2)]
    return Route(points, headings, width_m=float(np.median(widths)), junctions=junctions)


def start_waypoint(world_map, rng, avoid_junction_m: float = 10.0):
    """A random lane-centre waypoint at a spawn point, not in or right before a junction."""
    import carla
    spawn_points = world_map.get_spawn_points()
    for _ in range(50):
        waypoint = world_map.get_waypoint(spawn_points[int(rng.integers(len(spawn_points)))].location,
                                          project_to_road=True, lane_type=carla.LaneType.Driving)
        ahead = waypoint.next(avoid_junction_m)
        if not waypoint.is_junction and ahead and not ahead[0].is_junction:
            return waypoint
    raise RuntimeError("no spawn point away from a junction after 50 tries")
