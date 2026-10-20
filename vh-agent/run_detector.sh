#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source vh-agent/training/env.sh
export PYTHONPATH="$PWD/vh-agent/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
exec python -u -m vh_agent.detector_server "$@"
