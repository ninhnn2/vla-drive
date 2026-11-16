#!/bin/bash
# Every command in the container sees ROS 2 and, once `make ros-build` has run, the
# drive_ros package.
set -e
source /opt/ros/humble/setup.bash
[[ -f /work/ros/install/setup.bash ]] && source /work/ros/install/setup.bash
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
exec "$@"
