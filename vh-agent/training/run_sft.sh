#!/usr/bin/env bash
# Resume this SFT run on GPUs 0 and 1; retry only for CUDA memory exhaustion.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source vh-agent/training/env.sh

run_dir=$(realpath "${1:?用法：bash vh-agent/training/run_sft.sh 已存在的训练目录}")
test -d "$run_dir"
log_file="${run_dir}.log"
exec 9> "$run_dir/.run.lock"
flock -n 9 || { echo "该训练目录已有后台任务" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4
required_free_mib=36864

while [[ ! -f "$run_dir/completed.json" ]]; do
  free_mib=$(nvidia-smi --id=0,1 --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -n 1)
  if (( free_mib < required_free_mib )); then
    printf 'waiting_for_gpus=0,1 required_free_mib=%s min_free_mib=%s\n' "$required_free_mib" "$free_mib" >> "$log_file"
    while (( free_mib < required_free_mib )); do
      sleep 15
      free_mib=$(nvidia-smi --id=0,1 --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -n 1)
    done
  fi
  printf 'gpus_ready=0,1 min_free_mib=%s effective_batch=16\n' "$free_mib" >> "$log_file"
  start_offset=$(stat -c %s "$log_file")
  rm -f "${run_dir}.exitcode"
  if torchrun --standalone --nnodes=1 --nproc-per-node=2 vh-agent/training/sft.py \
      --output "$run_dir" --resume --effective-batch 16 \
      --full-attention-checkpoint-stride 1 >> "$log_file" 2>&1; then
    echo 0 > "${run_dir}.exitcode"
    exit 0
  else
    code=$?
    echo "$code" > "${run_dir}.exitcode"
  fi
  if ! tail -c "+$((start_offset + 1))" "$log_file" | \
      rg '^(\[rank[0-9]+\]: )?torch\.(cuda\.)?OutOfMemoryError: CUDA out of memory' > /dev/null; then
    exit "$code"
  fi
  printf 'cuda_oom_retry=1 resume=latest_checkpoint wait_sec=15\n' >> "$log_file"
  sleep 15
done
