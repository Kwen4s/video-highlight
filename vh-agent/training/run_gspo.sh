#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source vh-agent/training/env.sh
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/vh-agent/training"
output="${1:-$VH_TRAIN_ROOT/runs/qwen35-9b-gspo-001}"
shift "$(( $# > 0 ? 1 : 0 ))"
mkdir -p "$output/metrics"
exec 9>"$output/.training.lock"
flock -n 9
export VERL_FILE_LOGGER_PATH="$output/metrics/$(date -u +%Y%m%dT%H%M%S)-$$.jsonl"
python vh-agent/training/gspo.py "$output" "$@" --prepare-only
rm -f "$output/failed.json"
python vh-agent/training/track.py "$output" > "$output/swanlab.log" 2>&1 &
tracker_pid=$!
trap 'kill "$tracker_pid" 2>/dev/null || true' EXIT
if python vh-agent/training/gspo.py "$output" "$@"; then
  wait "$tracker_pid"
else
  exit_code=$?
  python -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"exit_code":int(sys.argv[2])})+"\n")' "$output/failed.json" "$exit_code"
  wait "$tracker_pid" || true
  exit "$exit_code"
fi
check_steps="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["check_steps"])' "$output/run.json")"
if [[ "$check_steps" == "0" ]]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=2 vh-agent/training/predict.py "$output"
  python vh-agent/training/track.py "$output"
fi
