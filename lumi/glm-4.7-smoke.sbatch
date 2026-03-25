#!/bin/bash
# Smoke test for GLM-4.7 on the LUMI overlay/container stack.

#SBATCH --job-name=glm-47-smoke
#SBATCH --account=project_465002183
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals/logs/slurm/glm-4.7-smoke-%j.out
#SBATCH --error=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals/logs/slurm/glm-4.7-smoke-%j.err

set -euo pipefail

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif
REPO_ROOT=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal
HF_HOME=${HF_HOME:-/scratch/project_465002183/.cache/huggingface/}
LOG_DIR=$REPO_ROOT/logs/tmp-logs
MODEL_PATH=${MODEL_PATH:-zai-org/GLM-4.7}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-zai-org/GLM-4.7}
ENABLE_THINKING=${ENABLE_THINKING:-false}
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-1}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
TP=${TP:-16}
PP=${PP:-1}
DP=${DP:-1}
CTX=${CTX:-4096}
GPU_MEM=${GPU_MEM:-0.8}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-14400}
SERVER_START_STEP=${SERVER_START_STEP:-10}
MASTER_PORT=${MASTER_PORT:-29501}
FI_PROVIDER_NAME=${FI_PROVIDER_NAME:-cxi}
FI_CXI_DISABLE_HMEM_FLAG=${FI_CXI_DISABLE_HMEM_FLAG:-1}
GLOO_IFNAME=${GLOO_IFNAME:-hsn0}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
NCCL_DEBUG_LEVEL=${NCCL_DEBUG_LEVEL:-WARN}
NCCL_DEBUG_SUBSYS_LEVEL=${NCCL_DEBUG_SUBSYS_LEVEL:-INIT,NET}
NCCL_NET_BACKEND=${NCCL_NET_BACKEND:-}
NCCL_NET_PLUGIN_NAME=${NCCL_NET_PLUGIN_NAME:-}
AWS_OFI_RCCL_LIBDIR=${AWS_OFI_RCCL_LIBDIR:-}
export MODEL_PATH
export SERVED_MODEL_NAME
export ENABLE_THINKING
export ENABLE_EXPERT_PARALLEL
export ENFORCE_EAGER
export HF_HOME

mkdir -p "$LOG_DIR" "$HF_HOME" "${HF_HOME%/}/hub" "${HF_HOME%/}/transformers" "${HF_HOME%/}/datasets"

declare -a SING_BIND_ARGS=(
  -B "$BASE_DIR:$LAIFS_APPL_DIR"
  -B "$OVERLAY_DIR:/overlay"
  -B "$REPO_ROOT:$REPO_ROOT"
)

if [[ -d /scratch ]]; then
  SING_BIND_ARGS+=(-B /scratch:/scratch)
fi
if [[ -L /scratch/project_465002183 ]]; then
  SCRATCH_TARGET="$(readlink -f /scratch/project_465002183 || true)"
  if [[ -n "$SCRATCH_TARGET" && -d "$SCRATCH_TARGET" ]]; then
    SING_BIND_ARGS+=(-B "$SCRATCH_TARGET:$SCRATCH_TARGET")
  fi
fi

readarray -t NODE_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${MASTER_ADDR:-${NODE_HOSTS[0]}}
NODES_COUNT=${#NODE_HOSTS[@]}
HEADLESS=0
if [[ "$NODES_COUNT" -gt 1 ]]; then
  HEADLESS=1
fi

read -r -d '' SERVER_CMD <<'EOS' || true
set -euo pipefail
source /overlay/venv/vllm-min/bin/activate
OVERLAY_SITE=/overlay/venv/vllm-min/lib/python3.12/site-packages
export PYTHONPATH=/overlay/src/transformers/src:/overlay/src/vllm:${OVERLAY_SITE}${PYTHONPATH:+:$PYTHONPATH}
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export TORCHDYNAMO_DISABLE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PLUGINS=''
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi
unset ROCR_VISIBLE_DEVICES
export NODE_RANK=${SLURM_PROCID:-0}
export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export MODEL_PATH="${MODEL_PATH}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"
export ENABLE_THINKING="${ENABLE_THINKING}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL}"
export ENFORCE_EAGER="${ENFORCE_EAGER}"

DEFAULT_CHAT_TEMPLATE_KWARGS="$(python - <<'PY'
import json
import os

enable_thinking = os.environ.get("ENABLE_THINKING", "false").strip().lower() == "true"
print(json.dumps({"enable_thinking": enable_thinking}))
PY
)"

if [[ "${HEADLESS:-0}" == "1" ]]; then
  export VLLM_DISABLE_PYNCCL=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export FI_PROVIDER="${FI_PROVIDER_NAME}"
  export FI_CXI_DISABLE_HMEM="${FI_CXI_DISABLE_HMEM_FLAG}"
  export GLOO_SOCKET_IFNAME="${GLOO_IFNAME}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}"
  export NCCL_DEBUG="${NCCL_DEBUG_LEVEL}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS_LEVEL}"
  if [[ -d "${AWS_OFI_RCCL_LIBDIR:-}" ]]; then
    export LD_LIBRARY_PATH="${AWS_OFI_RCCL_LIBDIR}:${LD_LIBRARY_PATH:-}"
  fi
  if [[ -n "${NCCL_NET_PLUGIN_NAME:-}" ]]; then
    export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN_NAME}"
  else
    unset NCCL_NET_PLUGIN
  fi
  if [[ -n "${NCCL_NET_BACKEND:-}" ]]; then
    export NCCL_NET="${NCCL_NET_BACKEND}"
  else
    unset NCCL_NET
  fi
fi

EXTRA_VLLM_FLAGS=(
  --reasoning-parser glm45
  --default-chat-template-kwargs "${DEFAULT_CHAT_TEMPLATE_KWARGS}"
)
if [[ "${ENABLE_EXPERT_PARALLEL:-0}" == "1" ]]; then
  EXTRA_VLLM_FLAGS+=(--enable-expert-parallel)
fi
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  EXTRA_VLLM_FLAGS+=(--enforce-eager)
fi
if [[ "${HEADLESS:-0}" == "1" && "${NODE_RANK}" != "0" ]]; then
  EXTRA_VLLM_FLAGS+=(--headless)
fi
if [[ "${NODES:-1}" -gt 1 ]]; then
  EXTRA_VLLM_FLAGS+=(
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --nnodes "${NODES}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
  )
fi

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --dtype bfloat16 \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --data-parallel-size "${DP}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${CTX}" \
  "${EXTRA_VLLM_FLAGS[@]}"
EOS

JOB_ID="${SLURM_JOB_ID:-manual}"
SERVER_HOST=127.0.0.1
SERVER_PORT=8022
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
API_KEY=inspectai
VLLM_LOG="$LOG_DIR/glm-4.7-vllm-${JOB_ID}.log"
MODELS_JSON="$LOG_DIR/glm-4.7-models-${JOB_ID}.json"
CHAT_JSON="$LOG_DIR/glm-4.7-chat-${JOB_ID}.json"
RANK_LOG="$LOG_DIR/glm-4.7-rank-%t-${JOB_ID}.log"
RANK0_LOG="$LOG_DIR/glm-4.7-rank-0-${JOB_ID}.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting GLM-4.7 smoke on ${NODES_COUNT} node(s); master=${MASTER_ADDR}; tp=${TP}; pp=${PP}; dp=${DP}; HF_HOME=${HF_HOME}"

srun --overlap \
  --nodes="$NODES_COUNT" \
  --ntasks="$NODES_COUNT" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-56}" \
  --output="$RANK_LOG" \
  --error="$RANK_LOG" \
  singularity exec --rocm "${SING_BIND_ARGS[@]}" "$SIF" \
  env \
    MODEL_PATH="$MODEL_PATH" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    PORT="$SERVER_PORT" \
    API_KEY="$API_KEY" \
    TP="$TP" \
    PP="$PP" \
    DP="$DP" \
    NODES="$NODES_COUNT" \
    CTX="$CTX" \
    GPU_MEM="$GPU_MEM" \
    HF_HOME="$HF_HOME" \
    ENABLE_THINKING="$ENABLE_THINKING" \
    ENABLE_EXPERT_PARALLEL="$ENABLE_EXPERT_PARALLEL" \
    ENFORCE_EAGER="$ENFORCE_EAGER" \
    HEADLESS="$HEADLESS" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    FI_PROVIDER_NAME="$FI_PROVIDER_NAME" \
    FI_CXI_DISABLE_HMEM_FLAG="$FI_CXI_DISABLE_HMEM_FLAG" \
    GLOO_IFNAME="$GLOO_IFNAME" \
    NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
    NCCL_NET_GDR_LEVEL="$NCCL_NET_GDR_LEVEL" \
    NCCL_DEBUG_LEVEL="$NCCL_DEBUG_LEVEL" \
    NCCL_DEBUG_SUBSYS_LEVEL="$NCCL_DEBUG_SUBSYS_LEVEL" \
    NCCL_NET_BACKEND="$NCCL_NET_BACKEND" \
    NCCL_NET_PLUGIN_NAME="$NCCL_NET_PLUGIN_NAME" \
    AWS_OFI_RCCL_LIBDIR="$AWS_OFI_RCCL_LIBDIR" \
    bash -lc "$SERVER_CMD" \
    >"$VLLM_LOG" 2>&1 &
SERVER_PID=$!

for ((i=0; i<SERVER_START_TIMEOUT; i+=SERVER_START_STEP)); do
  if curl -fsS \
    -H "Authorization: Bearer ${API_KEY}" \
    "${SERVER_URL}/v1/models" >"$MODELS_JSON"; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before readiness." >&2
    tail -n 200 "$VLLM_LOG" >&2 || true
    tail -n 200 "$RANK0_LOG" >&2 || true
    exit 1
  fi
  sleep "$SERVER_START_STEP"
done

if [[ ! -s "$MODELS_JSON" ]]; then
  echo "Timed out waiting for vLLM readiness." >&2
  tail -n 200 "$VLLM_LOG" >&2 || true
  tail -n 200 "$RANK0_LOG" >&2 || true
  exit 1
fi

python - <<'PY'
import json
import os
from pathlib import Path

job_id = os.environ.get("SLURM_JOB_ID", "manual")
model_path = os.environ["MODEL_PATH"]
log_root = Path("/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals/logs/tmp-logs")
req_path = log_root / f"glm-4.7-chat-req-{job_id}.json"
payload = {
    "model": model_path,
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly OK.",
        }
    ],
    "temperature": 0,
    "max_tokens": 32,
}
req_path.write_text(json.dumps(payload))
PY

curl -fsS \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  "${SERVER_URL}/v1/chat/completions" \
  -d @"$LOG_DIR/glm-4.7-chat-req-${JOB_ID}.json" >"$CHAT_JSON"

python - <<'PY'
import json
import os
from pathlib import Path

job_id = os.environ.get("SLURM_JOB_ID", "manual")
log_root = Path("/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals/logs/tmp-logs")
models_path = log_root / f"glm-4.7-models-{job_id}.json"
chat_path = log_root / f"glm-4.7-chat-{job_id}.json"

models = json.loads(models_path.read_text())
chat = json.loads(chat_path.read_text())
print("MODELS_OK", [item["id"] for item in models.get("data", [])])
choice = chat["choices"][0]["message"]
print("CHAT_OK", json.dumps(choice, ensure_ascii=True))
usage = chat.get("usage")
if usage is not None:
    print("USAGE", json.dumps(usage, ensure_ascii=True))
PY
