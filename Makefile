# drive/ in its container: ROS 2 Humble, PyTorch with CUDA, and windows on your screen.
# Needs Docker with the NVIDIA container toolkit.   make help
#
# On a Mac, drive/ runs natively in .venv instead (made by `make setup`, or by the first
# target that needs it) and trains on the Mac's GPU (MPS). ROS 2 still runs in a
# container, built for arm64 without CUDA (docker/compose.mac.yml), with its windows on
# XQuartz. NATIVE=0 runs drive/ in that container too; NATIVE=1 runs it in .venv on Linux.
SHELL := /bin/bash
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

ifeq ($(shell uname -s),Darwin)
MAC := 1
COMPOSE := docker compose -f docker-compose.yml -f docker/compose.mac.yml
NATIVE ?= 1
else
COMPOSE := docker compose
NATIVE ?= 0
endif

# Without a terminal (a script, CI) docker must not ask for one.
RUN := $(COMPOSE) run --rm $(shell [ -t 0 ] || echo -T) drive
ROS := $(RUN) ros2 run drive_ros

# drive/: in .venv, or in the container. Windows from .venv need no X server.
VENV := .venv
ifeq ($(NATIVE),1)
DRIVE_DEPS := $(VENV)/.installed
WINDOW_DEPS := $(DRIVE_DEPS)
PY := $(VENV)/bin/python
DRIVE := ./drive.sh
else
WINDOW_DEPS := gui
PY := $(RUN) python3
DRIVE := $(RUN) ./drive.sh
endif

V ?= v1
TRACK ?= peanut
POLICY ?= pure-pursuit
NAME ?= ros
ARGS ?=

.PHONY: help setup build shell test ros-test gui sim collect label validate browse train evaluate status \
        ros-build ros-sim ros-drive ros-teleop ros-record ros-launch ros-view

help:
ifeq ($(NATIVE),1)
	@echo "drive/      make setup | test                    in $(VENV); the targets below make it if need be"
	@echo "container   make build | shell | ros-test        for ROS 2; ros-test: the nodes' tests"
else
	@echo "container   make build | shell | test"
endif
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

# numpy, OpenCV and PyTorch in .venv, on the Python the container has (3.10). Needs uv.
setup: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	@command -v uv >/dev/null || { echo "make setup needs uv: brew install uv" >&2; exit 1; }
	uv venv --allow-existing --python 3.10 $(VENV)
	uv pip install --python $(VENV)/bin/python -r requirements.txt
	@touch $@

build:
	$(COMPOSE) build drive

shell: gui
	$(RUN) bash

test: $(DRIVE_DEPS)
	$(PY) -m unittest discover -s tests -t .

# The ROS 2 node tests; `make test` runs them too, except in .venv, which has no ROS 2.
ros-test:
	$(RUN) python3 -m unittest tests.test_drive_ros

# Let this user's containers open windows on the X server: on Linux the host's; on a Mac
# XQuartz, which the containers reach over TCP from Docker's VM.
gui:
ifdef MAC
	@[[ -x /opt/X11/bin/xhost ]] || { echo "windows from the container need XQuartz: brew install --cask xquartz" >&2; exit 1; }
	@[[ $$(defaults read org.xquartz.X11 nolisten_tcp 2>/dev/null) == 0 ]] || { \
	    echo "XQuartz must accept the container's windows over TCP. Once:" >&2; \
	    echo "  defaults write org.xquartz.X11 nolisten_tcp -bool false" >&2; \
	    echo "then quit XQuartz (it starts again by itself) and run make again." >&2; exit 1; }
	@open -g -a XQuartz; for _ in {1..20}; do /opt/X11/bin/xhost +localhost >/dev/null 2>&1 && exit 0; sleep 0.5; done; \
	    echo "xhost cannot reach XQuartz (DISPLAY=$$DISPLAY); log out and in once after installing it" >&2; exit 1
else
	@xhost +SI:localuser:$$USER >/dev/null 2>&1 || true
endif

sim: $(WINDOW_DEPS)
	$(DRIVE) sim --show $(ARGS)

collect: $(WINDOW_DEPS)
	$(DRIVE) collect $(ARGS)

label validate status: $(DRIVE_DEPS)
	$(DRIVE) $@ $(ARGS)

browse: $(WINDOW_DEPS)
	$(DRIVE) browse $(ARGS)

train: $(DRIVE_DEPS)
	$(DRIVE) train $(V) $(ARGS)

evaluate: $(DRIVE_DEPS)
	$(DRIVE) evaluate $(V) $(ARGS)

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
