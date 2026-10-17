#!/usr/bin/env bash
# Source this file before detector training.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "使用 source vh-agent/training/env.sh 激活训练环境"
  exit 1
fi

export VH_TRAIN_ROOT=/data0/wkw/video-highlight
export UV_PROJECT_ENVIRONMENT="${VH_TRAIN_ROOT}/envs/train"
export UV_CACHE_DIR="${VH_TRAIN_ROOT}/cache/uv"
export HF_HOME="${VH_TRAIN_ROOT}/cache/huggingface"
export TORCH_HOME="${VH_TRAIN_ROOT}/cache/torch"
export TRITON_CACHE_DIR="${VH_TRAIN_ROOT}/cache/triton"
export TMPDIR="${VH_TRAIN_ROOT}/cache/tmp"
export RAY_TMPDIR="${VH_TRAIN_ROOT}/cache/ray"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONNOUSERSITE=1
unset PYTHONPATH

source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
