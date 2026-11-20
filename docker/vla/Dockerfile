# The LeRobot environment for the VLA work (SmolVLA, smolvla-v1): Python 3.12, PyTorch 2.11 for
# CUDA 13.0, LeRobot 0.5.2 at the commit smolvla-v1 was fine-tuned with, transformers 5.5.4, ONNX,
# Gradio, and the drive/ simulator. LeRobot needs Python 3.12, so this is not the ROS 2 Humble image
# of docker/Dockerfile (Python 3.10). Every package is pinned by requirements.lock, the pip freeze of
# the environment the model was checked in. docker/vla/README.md says how to use it.
#
#   docker compose build vla          or open the repository in VS Code: "Reopen in Container"
FROM python:3.12-slim-bookworm

# git for VS Code and pip, ffmpeg for torchcodec (LeRobot's default video decoder) and for looking
# at videos, sudo so the dev container's user can add what it needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg sudo ca-certificates curl less \
    && rm -rf /var/lib/apt/lists/*

# A user with the host's UID (VS Code updates it to yours; compose runs as HOST_UID), so what
# it writes to runs/ is yours. It owns the venv: pip install works without sudo.
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /opt/venv /hf /work && chown $USER_UID:$USER_GID /opt/venv /hf /work

USER $USERNAME
COPY requirements.lock /tmp/requirements.lock
RUN --mount=type=cache,target=/home/dev/.cache/pip,uid=1000,gid=1000 \
    python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /tmp/requirements.lock

# HF_HOME holds SmolVLM2, which LeRobot loads for the processor and the VLM's config: a volume
# (vla-drive-hf), so it is downloaded once. PYTHONPATH: run drive/ and tools/ from /work.
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    HF_HOME=/hf \
    PYTHONPATH=/work \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ANALYTICS_ENABLED=False
WORKDIR /work
CMD ["bash"]
