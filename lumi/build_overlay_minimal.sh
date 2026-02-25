#!/usr/bin/env bash
set -euo pipefail

# Minimal overlay builder for the LAIF ROCm 6.4.4 full container.
# This keeps the base container unchanged and only layers vLLM/Transformers
# (plus required Python dependencies) into a bind-mounted overlay directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs

: "${SIF:=$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}"
: "${OVERLAY_DIR:=$REPO_ROOT/overlay_vllm_minimal}"

: "${VLLM_REPO:=https://github.com/vllm-project/vllm.git}"
: "${VLLM_REF:=main}"

: "${TRANSFORMERS_FROM_SOURCE:=1}"
: "${TRANSFORMERS_REPO:=https://github.com/huggingface/transformers.git}"
: "${TRANSFORMERS_REF:=main}"
: "${TRANSFORMERS_VERSION:=5.1.0}"

: "${HF_HUB_VERSION:=1.4.1}"
: "${TOKENIZERS_VERSION:=0.22.2}"
: "${PYTORCH_ROCM_ARCH:=gfx90a}"
: "${MAX_JOBS:=64}"

if [[ ! -f "$SIF" ]]; then
  echo "FATAL: SIF not found: $SIF" >&2
  exit 1
fi

mkdir -p "$OVERLAY_DIR"/venv "$OVERLAY_DIR"/src "$OVERLAY_DIR"/cache
chmod 700 "$OVERLAY_DIR"

echo "+ SIF: $SIF"
echo "+ Overlay: $OVERLAY_DIR"
echo "+ vLLM ref: $VLLM_REF"
if [[ "$TRANSFORMERS_FROM_SOURCE" == "1" ]]; then
  echo "+ transformers ref: $TRANSFORMERS_REF (source)"
else
  echo "+ transformers version: $TRANSFORMERS_VERSION (wheel)"
fi

export VLLM_REPO VLLM_REF
export TRANSFORMERS_FROM_SOURCE TRANSFORMERS_REPO TRANSFORMERS_REF TRANSFORMERS_VERSION
export HF_HUB_VERSION TOKENIZERS_VERSION
export PYTORCH_ROCM_ARCH MAX_JOBS

singularity exec --rocm \
  -B "$BASE_DIR:$LAIFS_APPL_DIR" \
  -B "$OVERLAY_DIR:/overlay" \
  "$SIF" bash -eu -s <<'INSIDE'
source /opt/venv/bin/activate

if [[ ! -d /overlay/venv/vllm-min ]]; then
  python3 -m venv --system-site-packages /overlay/venv/vllm-min
fi
source /overlay/venv/vllm-min/bin/activate
OVERLAY_SITE=/overlay/venv/vllm-min/lib/python3.12/site-packages
export PYTHONPATH="${OVERLAY_SITE}${PYTHONPATH:+:$PYTHONPATH}"

export PIP_USER=0
unset PYTHONUSERBASE
export XDG_CACHE_HOME=/overlay/cache
export PIP_CACHE_DIR=/overlay/cache/pip
export TRITON_CACHE_DIR=/overlay/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/overlay/cache/torchinductor
export PYTORCH_KERNEL_CACHE_PATH=/overlay/cache/torch-kernels
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTORCH_KERNEL_CACHE_PATH"

python -m pip install --no-user -U pip 'setuptools>=79,<80' wheel ninja 'pybind11>=2.13,<3' 'cmake<4.0.0'

# Keep these explicit because transformers source currently needs newer hub
# than the base LAIF image provides.
python -m pip install --no-user -U \
  "huggingface_hub==${HF_HUB_VERSION}" \
  "tokenizers==${TOKENIZERS_VERSION}"

if [[ "${TRANSFORMERS_FROM_SOURCE}" == "1" ]]; then
  if [[ ! -d /overlay/src/transformers/.git ]]; then
    git clone "${TRANSFORMERS_REPO}" /overlay/src/transformers
  fi
  cd /overlay/src/transformers
  git fetch origin
  git checkout "${TRANSFORMERS_REF}"
  python -m pip install --no-user -U -e .
  export PYTHONPATH=/overlay/src/transformers/src:/overlay/src/vllm${PYTHONPATH:+:$PYTHONPATH}
else
  python -m pip install --no-user -U "transformers==${TRANSFORMERS_VERSION}"
fi

if [[ ! -d /overlay/src/vllm/.git ]]; then
  git clone "${VLLM_REPO}" /overlay/src/vllm
fi
cd /overlay/src/vllm
git fetch --tags origin
git checkout "${VLLM_REF}"

export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}"
export MAX_JOBS="${MAX_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"

# Minimal install approach:
# - build vLLM from source
# - do not force-reinstall the whole stack from requirements/rocm.txt
python -m pip install --no-user -e . --no-deps --no-build-isolation

python - <<'PY'
import torch
import transformers
import vllm
import vllm._C  # noqa: F401
from transformers import AutoConfig
from vllm.transformers_utils.config import get_config

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("transformers path:", transformers.__file__)
print("vllm:", vllm.__version__)
print("vllm path:", vllm.__file__)
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-397B-A17B")
print("AutoConfig model_type:", cfg.model_type)
v_cfg = get_config("Qwen/Qwen3.5-397B-A17B", trust_remote_code=False)
print("vllm.get_config model_type:", v_cfg.model_type)
PY

python -m pip freeze > /overlay/overlay-pip-freeze.txt
echo "+ Wrote /overlay/overlay-pip-freeze.txt"
INSIDE

echo "+ Build complete."
echo "+ Use OVERLAY_DIR=$OVERLAY_DIR with lumi/run_suite.sbatch"
