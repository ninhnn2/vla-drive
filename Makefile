# drive/ in its container: ROS 2 Humble, PyTorch with CUDA, and windows on your screen.
# Needs Docker with the NVIDIA container toolkit.   make help
SHELL := /bin/bash
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)
# Without a terminal (a script, CI) docker must not ask for one.
RUN := docker compose run --rm $(shell [ -t 0 ] || echo -T) drive
ROS := $(RUN) ros2 run drive_ros

V ?= v1
TRACK ?= peanut
POLICY ?= pure-pursuit
NAME ?= ros
ARGS ?=

.PHONY: help build shell test gui sim collect label validate browse train evaluate status \
        ros-build ros-sim ros-drive ros-teleop ros-record ros-launch ros-view

help:
	@echo "container   make build | shell | test"
	@echo "simulator   make sim [ARGS='--policy cnn --checkpoint v1 --track trefoil']"
	@echo "data        make collect | label | validate | browse | status"
	@echo "            watch it:  make collect ARGS='--project runs/demo --episodes 2 --show'"
	@echo "                       make browse ARGS=\"--match 'sim2-*'\""
	@echo "model       make train V=v1 | evaluate V=v1"
	@echo "ROS 2       make ros-build, then each in its own terminal:"
	@echo "              make ros-sim [TRACK=trefoil]         the simulated car, with a window"
	@echo "              make ros-drive [POLICY=cnn V=v1]     the control loop drives it"
	@echo "              make ros-teleop                      or you do, with the mouse"
	@echo "              make ros-record [NAME=ros]           record the drive; Ctrl-C saves"
	@echo "              make ros-view                        rqt_image_view on the camera"
	@echo "            or all at once: make ros-launch [ARGS='policy:=cnn checkpoint:=v1 record:=true']"

build:
	docker compose build drive

shell: gui
	$(RUN) bash

test:
	$(RUN) python3 -m unittest discover -s tests -t .

# Let this user's containers open windows on the X server.
gui:
	@xhost +SI:localuser:$$USER >/dev/null 2>&1 || true

sim: gui
	$(RUN) ./drive.sh sim --show $(ARGS)

collect: gui
	$(RUN) ./drive.sh collect $(ARGS)

label validate status:
	$(RUN) ./drive.sh $@ $(ARGS)

browse: gui
	$(RUN) ./drive.sh browse $(ARGS)

train:
	$(RUN) ./drive.sh train $(V) $(ARGS)

evaluate:
	$(RUN) ./drive.sh evaluate $(V) $(ARGS)

ros-build:
	$(RUN) bash -c "cd ros && colcon build --symlink-install"

ros-sim: gui
	$(ROS) sim --ros-args -p track:=$(TRACK) -p show:=true $(ARGS)

ros-drive:
	$(ROS) driver --ros-args -p track:=$(TRACK) -p policy:=$(POLICY) $(if $(filter cnn,$(POLICY)),-p checkpoint:=$(V)) $(ARGS)

ros-teleop: gui
	$(ROS) teleop

ros-record:
	$(ROS) recorder --ros-args -p name:=$(NAME) $(ARGS)

ros-launch: gui
	$(RUN) ros2 launch drive_ros drive.launch.py track:=$(TRACK) $(ARGS)

ros-view: gui
	$(RUN) ros2 run rqt_image_view rqt_image_view /drive/image
