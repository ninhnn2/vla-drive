#!/usr/bin/env bash
# Drive a simulated car, and train a network to.   ./drive.sh --help
# Needs numpy and OpenCV; `train` and the cnn policy also need PyTorch. The container
# has them (`make shell`, or the make targets: make help); so does the .venv that
# `make setup` makes, which is how drive/ runs on a Mac.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
# $PYTHON if set; else the repo's .venv, but not inside the container, where it is the host's.
if [[ -z ${PYTHON:-} ]]; then
    PYTHON=python3
    [[ ! -f /.dockerenv && -x $ROOT/.venv/bin/python ]] && PYTHON=$ROOT/.venv/bin/python
fi
"$PYTHON" -c "import numpy, cv2" 2>/dev/null \
    || { echo "drive.sh needs numpy and OpenCV: make setup (a .venv, on a Mac), or use the container: make shell" >&2
         exit 1; }
PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" exec "$PYTHON" -m drive "$@"
