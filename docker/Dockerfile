# One image for everything in drive/: ROS 2 Humble, PyTorch with CUDA, and windows on
# the host's screen (OpenCV, rqt, rviz2). Built and run through the Makefile.
FROM osrf/ros:humble-desktop-full

# pip from its own installer rather than apt: the image has everything else already,
# and this keeps the build off the Ubuntu and ROS package mirrors.
RUN python3 -c "import urllib.request as u; u.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')" \
    && python3 /tmp/get-pip.py --no-setuptools --no-wheel && rm /tmp/get-pip.py

# CUDA 12.1 wheels from PyTorch's own index (host NVIDIA driver 525 or newer). numpy and
# OpenCV stay Ubuntu's: the ones ROS 2 is built against.
# Ubuntu's sympy cannot be uninstalled by pip, so PyTorch's is installed over it first.
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        --ignore-installed sympy==1.13.1 \
    && python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1

# The container runs as the host user, who has no home directory in here.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/work \
    HOME=/tmp XDG_RUNTIME_DIR=/tmp/runtime
WORKDIR /work
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
