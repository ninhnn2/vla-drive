# One image for everything in drive/: ROS 2 Humble, PyTorch with CUDA, and windows on
# the host's screen (OpenCV, rqt, rviz2). Built and run through the Makefile.
# On a Mac, docker/compose.mac.yml builds it for arm64 instead: ROS 2's perception image
# (the desktop one is amd64 only) with rqt_image_view, and PyTorch for the CPU.
ARG BASE=osrf/ros:humble-desktop-full
FROM ${BASE}

# ROS packages the base lacks; none for the desktop-full image.
ARG APT_PACKAGES=""
RUN if [ -n "$APT_PACKAGES" ]; then \
        apt-get update && apt-get install -y --no-install-recommends $APT_PACKAGES \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# pip from its own installer rather than apt: the image has everything else already,
# and this keeps the build off the Ubuntu and ROS package mirrors.
RUN python3 -c "import urllib.request as u; u.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')" \
    && python3 /tmp/get-pip.py --no-setuptools --no-wheel && rm /tmp/get-pip.py

# CUDA 12.1 wheels from PyTorch's own index (host NVIDIA driver 525 or newer). numpy and
# OpenCV stay Ubuntu's: the ones ROS 2 is built against.
# Ubuntu's sympy cannot be uninstalled by pip, so PyTorch's is installed over it first.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121
RUN python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX} \
        --ignore-installed sympy==1.13.1 \
    && python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX} torch==2.5.1

# The container runs as the host user, who has no home directory in here.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/work \
    HOME=/tmp XDG_RUNTIME_DIR=/tmp/runtime
WORKDIR /work
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
