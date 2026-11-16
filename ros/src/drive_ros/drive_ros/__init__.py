"""ROS 2 nodes around drive/ (the `drive` package itself is found through PYTHONPATH).

    sim        the simulated car: publishes its sensors, obeys /drive/cmd
    driver     the control loop: a policy and the safety supervisor, publishes /drive/cmd
    teleop     a window to drive by hand with the mouse, publishes /drive/cmd
    recorder   writes what it hears to an episode, for labelling and training

The topics are the whole interface (topics.py). A real car, or a CARLA bridge, that
publishes and obeys the same ones takes the simulator's place; nothing else changes.
"""
