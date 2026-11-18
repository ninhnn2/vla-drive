#!/usr/bin/env bash
# The fine-tuned VLA's side of the project. LeRobot needs Python 3.12 and its own PyTorch, so it
# lives in its own environment, .venv-vla, made by `make vla-setup` (on a Mac as on Linux). This
# runs that environment's Python with this repository on the path:
#
#   ./vla.sh -m drive export                               the episodes as a LeRobotDataset
#   ./vla.sh lerobot-train --policy.path=lerobot/smolvla_base ...   any program the environment has
#   ./vla.sh -m drive sim --policy vla --checkpoint smolvla-v1 --show
#   ./vla.sh tools/probe_vla.py | tools/see_vla.py | tools/vla_app.py --open
#
# The model runs on CUDA, else on a Mac's GPU (MPS), else on the CPU; VLA_DEVICE=cpu chooses.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
VENV=${VLA_VENV:-$ROOT/.venv-vla}
[[ -x $VENV/bin/python ]] || { echo "vla.sh needs the LeRobot environment: make vla-setup" >&2; exit 1; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
# A few operators have no Apple GPU version yet; run those on the CPU rather than fail.
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [[ $# -gt 0 && $1 != -* && $1 != */* && -x $VENV/bin/$1 ]]; then
    exec "$VENV/bin/$1" "${@:2}"
fi
exec "$VENV/bin/python" "$@"
