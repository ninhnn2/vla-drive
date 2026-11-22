"""CARLA as a platform for drive/: the same loop, recorder and pipeline, a real town.

    ./drive.sh carla identify    fit the car model (model.py) to the car in CARLA
    ./drive.sh carla check       how well that model predicts the car: the labels' accuracy
    ./drive.sh carla collect     the teacher drives random routes, told where to turn -> episodes

Nothing here imports `carla` at module level: drive/ and its tests run without CARLA,
and only the modules that talk to a server (server, platform, routing, and the commands)
import it, inside their functions. docs/carla.md has the design and the measurements.
"""
