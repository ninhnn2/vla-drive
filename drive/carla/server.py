"""A session with a CARLA server: connect, take the clock, and leave the server as it was found.

Synchronous mode: the server advances only when we call `world.tick()`, by exactly
`tick_s` of simulated time, so a recording is the same however slow the PC is. Its
danger is the other side of that: a client that dies with the server in synchronous mode
leaves it waiting for a tick that never comes. `session()` restores the server's own
settings in a `finally`, whatever went wrong.

Two things learnt on an 8 GB GPU (simulate_carlar/docs/VRAM_8GB.md):
- `get_server_version()` answers while the map is still loading; the readiness probe is
  `get_world().get_map()`.
- Loading another map into a running server does not free the first one, and repeated
  loads have crashed it. A session refuses a server in the wrong town and says how to
  start one in the right town instead.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

#: Actors spawned by drive/ carry this role name, so a later session can clear what a
#: crashed one left behind.
ROLE = "vla-drive"


@dataclass(frozen=True)
class ServerConfig:
    host: str = "localhost"
    port: int = 2000
    tick_s: float = 0.05            # 20 Hz control; see docs/carla.md for why not 50
    ready_timeout_s: float = 1200.0  # a cold server takes minutes to load its map (Town03_Opt: over 10)


@dataclass
class Session:
    client: object
    world: object
    map: object
    town: str
    tick_s: float
    actors: list = field(default_factory=list)

    def keep(self, actor):
        """Remember `actor` so the session destroys it at the end."""
        self.actors.append(actor)
        return actor

    def destroy(self, actor) -> None:
        if actor in self.actors:
            self.actors.remove(actor)
        try:
            if hasattr(actor, "stop"):
                actor.stop()
            actor.destroy()
        except RuntimeError:
            pass


def listening(host: str, port: int, wait_s: float = 1.0) -> bool:
    """Whether a server really holds `port`, not just Docker's port forwarder.

    `docker run -p` accepts a connection on the host before the container listens, then
    closes it. A carla.Client that connects in that window never notices: it waits on a
    socket in CLOSE-WAIT forever, whatever its timeout (a client started with the server
    sat 10 minutes so). CARLA's RPC server says nothing first, so a connection that stays
    open and silent for `wait_s` is a server; one closed at once is the forwarder alone."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=2.0) as sock:
            sock.settimeout(wait_s)
            try:
                return sock.recv(1) != b""
            except socket.timeout:
                return True
    except OSError:
        return False


def connect(config: ServerConfig, log=print):
    """A client whose world is ready, waiting up to `config.ready_timeout_s`."""
    import carla
    deadline, said, client = time.monotonic() + config.ready_timeout_s, False, None
    while True:
        error = "nothing listening"
        if listening(config.host, config.port):
            try:
                client = client or carla.Client(config.host, config.port)
                client.set_timeout(10.0)
                client.get_world().get_map()
                client.set_timeout(60.0)
                return client
            except RuntimeError as failure:
                error = str(failure)
        if time.monotonic() > deadline:
            raise ValueError(f"no CARLA server ready at {config.host}:{config.port} after "
                             f"{config.ready_timeout_s:.0f} s ({error}); start one: make carla-server")
        if not said:
            log(f"waiting for the CARLA server at {config.host}:{config.port} ...")
            said = True
        time.sleep(3.0)


@contextmanager
def session(config: ServerConfig = ServerConfig(), town: str | None = None, log=print):
    """Yield a `Session` in synchronous mode; on exit destroy its actors and restore the server."""
    client = connect(config, log)
    world = client.get_world()
    world_map = world.get_map()
    current = world_map.name.rsplit("/", 1)[-1]
    if town and current != town:
        raise ValueError(f"the CARLA server runs {current}, not {town}. Start it in {town}: "
                         f"make carla-server TOWN={town} (loading another map into a running server "
                         "keeps the first one's GPU memory, and has crashed it)")
    leftovers = [a for a in world.get_actors() if a.attributes.get("role_name") == ROLE]
    for actor in leftovers:                         # from a session that died; their sensors are children
        for child in world.get_actors().filter("sensor.*"):
            if child.parent is not None and child.parent.id == actor.id:
                child.destroy()
        actor.destroy()
    if leftovers:
        log(f"cleared {len(leftovers)} actor(s) a previous session left behind")

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = config.tick_s
    # Physics at 100 Hz inside each 20 Hz tick.
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = max(int(round(config.tick_s / 0.01)), 1)
    world.apply_settings(settings)
    s = Session(client, world, world_map, current, config.tick_s)
    try:
        world.tick()
        yield s
    finally:
        # A dead server answers nothing: at 60 s a call, tearing down three actors took 4
        # minutes to say so. A live one does all of this in milliseconds.
        client.set_timeout(10.0)
        for actor in reversed(list(s.actors)):
            s.destroy(actor)
        try:
            world.tick()
        except RuntimeError:
            pass
        world.apply_settings(original)              # back to asynchronous, or the server waits forever
