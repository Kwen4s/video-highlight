#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=/data0/wkw/video-highlight
VERL_REVISION=6093e007cc341973c9d9a6fb3867a85976c7c458
VERL_ROOT="${TRAIN_ROOT}/envs/verl"
ENV_ROOT="${TRAIN_ROOT}/envs/train"
FLASH_WHEEL="${TRAIN_ROOT}/cache/wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${MODULE_ROOT}/.venv/bin/uv"

export UV_CACHE_DIR="${TRAIN_ROOT}/cache/uv"
export TMPDIR="${TRAIN_ROOT}/cache/tmp"
mkdir -p "${TRAIN_ROOT}/envs" "${UV_CACHE_DIR}" "${TMPDIR}" \
  "${TRAIN_ROOT}/cache/huggingface" "${TRAIN_ROOT}/cache/torch" \
  "${TRAIN_ROOT}/cache/triton" "${TRAIN_ROOT}/cache/ray"

# The upstream wheel URL is unavailable; use the verified local binary artifact.
echo "28f1a5ecce89158519ad906dd75ab608dcb7c2daeed225f6ecca3c654f40e528  ${FLASH_WHEEL}" | sha256sum --check

if [[ ! -d "${VERL_ROOT}/.git" ]]; then
  git init "${VERL_ROOT}"
  git -C "${VERL_ROOT}" remote add origin https://github.com/verl-project/verl.git
fi
if ! git -C "${VERL_ROOT}" cat-file -e "${VERL_REVISION}^{commit}" 2>/dev/null; then
  git -C "${VERL_ROOT}" fetch --depth 1 origin "${VERL_REVISION}"
fi
git -C "${VERL_ROOT}" checkout --detach "${VERL_REVISION}"

# Reuse the reference project's file mirror, keeping versions and hashes intact.
LOCK_BACKUP="$(mktemp "${TMPDIR}/verl-lock.XXXXXX")"
cp "${VERL_ROOT}/uv.lock" "${LOCK_BACKUP}"
restore_lock() {
  cp "${LOCK_BACKUP}" "${VERL_ROOT}/uv.lock"
  rm -f "${LOCK_BACKUP}"
}
trap restore_lock EXIT
sed 's#https://files.pythonhosted.org/packages/#https://pypi.tuna.tsinghua.edu.cn/packages/#g' \
  "${LOCK_BACKUP}" > "${VERL_ROOT}/uv.lock"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}pypi.tuna.tsinghua.edu.cn"
export no_proxy="${NO_PROXY}"

UV_PROJECT_ENVIRONMENT="${ENV_ROOT}" "${UV_BIN}" sync \
  --project "${VERL_ROOT}" --python /usr/bin/python3.12 \
  --frozen --extra fsdp --extra vllm --link-mode copy --no-install-package flash-attn
# Correct two dependency conflicts present in the reference environment.
"${UV_BIN}" pip install --python "${ENV_ROOT}/bin/python" --link-mode copy \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  "numpy==2.3.5" "nvidia-cusparselt-cu13==0.8.1" "${FLASH_WHEEL}"
"${UV_BIN}" pip check --python "${ENV_ROOT}/bin/python"

echo "环境已安装：${ENV_ROOT}"
echo "激活：source ${MODULE_ROOT}/training/env.sh"
