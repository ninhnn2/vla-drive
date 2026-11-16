"""A driving stack around an action-chunking policy: simulator, control loop, safety.

Everything here runs on a PC with numpy and OpenCV. The policy is the only part that
changes on the Jetson, where `drive.policy.vla.BackendPolicy` wraps a TensorRT backend
from jetson-orin-nano-vla.

    sim/        track, vehicle model, forward camera and range sensor
    policy/     what produces action chunks: scripted experts, or a VLA backend
    control/    asynchronous inference and time-indexed chunk execution
    safety.py   the supervisor between the policy and the motors
    loop.py     one control tick: observe -> infer -> execute -> supervise -> actuate
"""
