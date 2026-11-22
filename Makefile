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
# The LeRobot environment for SmolVLA (docker/vla, also the VS Code dev container).
VLA := docker compose run --rm $(shell [ -t 0 ] || echo -T) vla

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

.PHONY: help setup vla-setup build shell test ros-test gui sim collect label validate browse train evaluate status \
        ros-build ros-sim ros-drive ros-teleop ros-record ros-launch ros-view \
        carla-setup carla-server carla-down carla-identify carla-check carla-collect carla-vla-setup carla-drive \
        carla-colab-pack carla-colab-unpack \
        vla-build vla-shell vla-check vla-test vla-app

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
	@echo "VLA         make vla-setup, then ./vla.sh tools/probe_vla.py | tools/see_vla.py | tools/vla_app.py --open"
	@echo "CARLA       make carla-setup, then make carla-server [TOWN=Town03] and, once it answers:"
	@echo "              make carla-identify | carla-check | carla-collect [ARGS='--episodes 40']   (docs/carla.md)"
	@echo "              make carla-drive ARGS='--checkpoint smolvla-carla-v1 --show'   a trained VLA drives"
	@echo "              make carla-colab-pack | carla-colab-unpack TAR=...   train on Colab: tools/colab_train_carla.ipynb"
	@echo "            make carla-down stops the server"
	@echo "SmolVLA     make vla-build, then: vla-check | vla-shell | vla-test | vla-app   (docker/vla/README.md)"

# numpy, OpenCV and PyTorch in .venv, on the Python the container has (3.10). Needs uv.
setup: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	@command -v uv >/dev/null || { echo "make setup needs uv: brew install uv" >&2; exit 1; }
	uv venv --allow-existing --python 3.10 $(VENV)
	uv pip install --python $(VENV)/bin/python -r requirements.txt
	@touch $@

# The fine-tuned VLA: LeRobot in its own environment, .venv-vla (Python 3.12), outside the
# container on every platform; on a Mac it runs on the Apple GPU. ./vla.sh runs its Python.
VLA_VENV := .venv-vla
vla-setup: $(VLA_VENV)/.installed

$(VLA_VENV)/.installed: requirements-vla.txt
	@command -v uv >/dev/null || { echo "make vla-setup needs uv: brew install uv" >&2; exit 1; }
	uv venv --allow-existing --python 3.12 $(VLA_VENV)
	uv pip install --python $(VLA_VENV)/bin/python -r requirements-vla.txt
	@# LeRobot brings OpenCV without windows; put in the same version with them, for sim --show.
	@# The two share the cv2 directory, so the one with windows is reinstalled after the other goes.
	@version=$$(uv pip show --python $(VLA_VENV)/bin/python opencv-python-headless 2>/dev/null | sed -n 's/^Version: //p'); \
	if [ -n "$$version" ]; then \
	    uv pip uninstall --python $(VLA_VENV)/bin/python opencv-python-headless && \
	    uv pip install --reinstall --no-deps --python $(VLA_VENV)/bin/python "opencv-python==$$version"; fi
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

# CARLA (drive/carla, docs/carla.md): the client in .venv-carla (Python 3.10 and the carla
# wheel; no PyTorch), the server in Docker. The wheel must come from the server's build:
# PyPI's for the stock image; CARLA_WHEEL=path/to/carla-*.whl for one built from source.
CARLA_VENV := .venv-carla
CARLA_VERSION := 0.9.16
CARLA_IMAGE ?= carlasim/carla:$(CARLA_VERSION)
CARLA_WHEEL ?=
TOWN ?= Town01
CARLA := PYTHON=$(CARLA_VENV)/bin/python ./drive.sh carla

carla-setup: $(CARLA_VENV)/.installed

$(CARLA_VENV)/.installed: requirements-carla.txt
	@command -v uv >/dev/null || { echo "make carla-setup needs uv: https://docs.astral.sh/uv/" >&2; exit 1; }
	uv venv --allow-existing --python 3.10 $(CARLA_VENV)
	uv pip install --python $(CARLA_VENV)/bin/python -r requirements-carla.txt "$(or $(CARLA_WHEEL),carla==$(CARLA_VERSION))"
	@touch $@

# Boots straight into $(TOWN): the image's map setting is edited inside the throwaway
# container (TransitionMap stays), because the packaged default, Town10HD_Opt, takes 5 GB
# of GPU memory, and loading another map into a running server keeps the first one.
# A cold start takes several minutes; the drive.sh commands wait for it. TOWN=Town03 segfaults
# the stock server at boot, and Town05 where its elevated road meets the city: use
# Town03_Opt and Town05_Opt (the same roads; docs/carla.md).
# No --rm: a server that crashes keeps its log for `docker logs vla-carla`.
carla-server:
	-@docker rm -f vla-carla >/dev/null 2>&1
	docker run -d --name vla-carla --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
	    -e NVIDIA_DRIVER_CAPABILITIES=all -p 2000-2002:2000-2002 $(CARLA_IMAGE) bash -c \
	    "sed -i -e 's,^GameDefaultMap=.*,GameDefaultMap=/Game/Carla/Maps/$(TOWN).$(TOWN),' \
	            -e 's,^ServerDefaultMap=.*,ServerDefaultMap=/Game/Carla/Maps/$(TOWN).$(TOWN),' \
	            CarlaUE4/Config/DefaultEngine.ini && \
	     exec ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000"
	@echo "CARLA is starting in $(TOWN) (several minutes); make carla-down stops it"

carla-down:
	-docker rm -f vla-carla

carla-identify carla-check carla-collect: $(CARLA_VENV)/.installed
	$(CARLA) $(@:carla-%=%) --town $(TOWN) $(ARGS)

# The VLA drives in CARLA from .venv-vla: the model needs LeRobot (Python 3.12), so the carla
# client goes in beside it. Its 3.12 wheel works with that environment's NumPy 2.
carla-vla-setup: $(VLA_VENV)/.carla
$(VLA_VENV)/.carla: $(VLA_VENV)/.installed
	uv pip install --python $(VLA_VENV)/bin/python "$(or $(CARLA_WHEEL_312),carla==$(CARLA_VERSION))"
	@touch $@

carla-drive: $(VLA_VENV)/.carla
	./vla.sh -m drive carla drive --town $(TOWN) $(ARGS)

# Training on Google Colab (tools/colab_train_carla.ipynb): the training split goes up as one
# tar (its videos are compressed already), the trained model comes back as another.
COLAB_RUN ?= smolvla-carla-v1
carla-colab-pack:
	@test -f runs/carla/lerobot-train/meta/info.json || { echo "no runs/carla/lerobot-train: ./vla.sh -m drive export --project runs/carla --out runs/carla/lerobot-train --exclude 'carla-*-03[5-9]'" >&2; exit 1; }
	tar -cf runs/carla/carla-lerobot-train.tar -C runs/carla lerobot-train
	@echo "upload runs/carla/carla-lerobot-train.tar to Google Drive, MyDrive/vla-drive/, and open tools/colab_train_carla.ipynb in Colab"

carla-colab-unpack:
	@test -n "$(TAR)" || { echo "make carla-colab-unpack TAR=path/to/<run>-pretrained_model.tar [COLAB_RUN=$(COLAB_RUN)]" >&2; exit 1; }
	@test ! -e runs/carla/vla/$(COLAB_RUN)/checkpoints/last/pretrained_model || { echo "runs/carla/vla/$(COLAB_RUN) has a model already; choose another COLAB_RUN" >&2; exit 1; }
	mkdir -p runs/carla/vla/$(COLAB_RUN)/checkpoints/last
	tar -xf $(TAR) -C runs/carla/vla/$(COLAB_RUN)/checkpoints/last
	@test -f runs/carla/vla/$(COLAB_RUN)/checkpoints/last/pretrained_model/config.json && echo "ready: --checkpoint $(COLAB_RUN)"

# ---- SmolVLA, in the LeRobot container (docker/vla) ----

vla-build:
	docker compose build vla

vla-shell:
	$(VLA) bash

# The versions, the GPU, and the model on three frames against what it computed on the PC.
vla-check:
	$(VLA) python tools/check_env.py $(ARGS)

vla-test:
	$(VLA) python -m unittest discover -s tests -t .

# Give the model an image or a video: http://127.0.0.1:7860
vla-app:
	$(VLA) python tools/vla_app.py $(ARGS)
