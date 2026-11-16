#!/usr/bin/env bash
# Drive a simulated car, and train a network to.   ./drive.sh --help
# Needs numpy and OpenCV; `train` and the cnn policy also need PyTorch, which the
# container has: `make shell`, or the make targets (make help).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
python3 -c "import numpy, cv2" 2>/dev/null \
    || { echo "drive.sh needs numpy and OpenCV; the container has them: make shell" >&2; exit 1; }
PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" exec python3 -m drive "$@"
