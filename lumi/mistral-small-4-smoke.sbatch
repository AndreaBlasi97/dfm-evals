#!/bin/bash
#SBATCH --job-name=ms4-smoke
#SBATCH --account=project_465002183
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=00:45:00
#SBATCH --output=/pfs/lustrep4/scratch/project_465002183/rasmus/artifacts/evals/slurm/mistral-small-4-smoke-%j.out
#SBATCH --error=/pfs/lustrep4/scratch/project_465002183/rasmus/artifacts/evals/slurm/mistral-small-4-smoke-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
source "$SCRIPT_DIR/artifact_root.sh"
POST_ARTIFACT_ROOT="$(resolve_post_artifact_root "$REPO_ROOT")"
export POST_ARTIFACT_ROOT

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif
OVERLAY_DIR=${OVERLAY_DIR:-$REPO_ROOT/overlay_vllm_minimal}
HF_HOME=${HF_HOME:-/scratch/project_465002183/.cache/huggingface}
HF_HOME=${HF_HOME%/}
LOG_DIR=${LOG_DIR:-$POST_ARTIFACT_ROOT/evals/tmp-logs}
MODEL_PATH=${MODEL_PATH:-mistralai/Mistral-Small-4-119B-2603}
REASONING_EFFORT=${REASONING_EFFORT:-}
VLLM_MLA_DISABLE=${VLLM_MLA_DISABLE:-0}
export MODEL_PATH
export REASONING_EFFORT
export VLLM_MLA_DISABLE
export SINGULARITYENV_VLLM_MLA_DISABLE="${VLLM_MLA_DISABLE}"
export REPO_ROOT
export OVERLAY_DIR
export HF_HOME
export LOG_DIR

mkdir -p "$LOG_DIR"
mkdir -p "$POST_ARTIFACT_ROOT/evals/slurm"

declare -a SING_BIND_ARGS=(
  -B "$BASE_DIR:$LAIFS_APPL_DIR"
  -B "$OVERLAY_DIR:/overlay"
  -B "$REPO_ROOT:$REPO_ROOT"
  -B "$HF_HOME:$HF_HOME"
)

if [[ -d "$POST_ARTIFACT_ROOT" ]]; then
  SING_BIND_ARGS+=(-B "$POST_ARTIFACT_ROOT:$POST_ARTIFACT_ROOT")
fi

read -r -d '' INNER_SCRIPT <<'EOS' || true
set -euo pipefail
source /overlay/venv/vllm-min/bin/activate
OVERLAY_SITE=/overlay/venv/vllm-min/lib/python3.12/site-packages
export PYTHONPATH=/overlay/src/transformers/src:/overlay/src/vllm:${OVERLAY_SITE}${PYTHONPATH:+:$PYTHONPATH}
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export TORCHDYNAMO_DISABLE=1
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi
unset ROCR_VISIBLE_DEVICES
export HF_HOME="${HF_HOME%/}"
export HF_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export MODEL_PATH="${MODEL_PATH}"
export VLLM_MLA_DISABLE="${VLLM_MLA_DISABLE}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SERVER_HOST=127.0.0.1
SERVER_PORT=8020
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
API_KEY=inspectai
VLLM_LOG="$LOG_DIR/mistral-small-4-vllm-${JOB_ID}.log"
MODELS_JSON="$LOG_DIR/mistral-small-4-models-${JOB_ID}.json"
CHAT_JSON="$LOG_DIR/mistral-small-4-chat-${JOB_ID}.json"
CHAT_REQ_JSON="$LOG_DIR/mistral-small-4-chat-req-${JOB_ID}.json"

cd "$REPO_ROOT"

python - <<'PY'
import mistral_common
import transformers
import vllm

print("mistral_common", mistral_common.__version__, mistral_common.__file__)
print("transformers", transformers.__version__, transformers.__file__)
print("vllm", vllm.__version__, vllm.__file__)
PY

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

vllm serve "$MODEL_PATH" \
  --host "$SERVER_HOST" \
  --port "$SERVER_PORT" \
  --api-key "$API_KEY" \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --tokenizer-mode mistral \
  --reasoning-parser mistral \
  >"$VLLM_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  if curl -fsS \
    -H "Authorization: Bearer ${API_KEY}" \
    "${SERVER_URL}/v1/models" >"$MODELS_JSON"; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before readiness." >&2
    tail -n 200 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done

if [[ ! -s "$MODELS_JSON" ]]; then
  echo "Timed out waiting for vLLM readiness." >&2
  tail -n 200 "$VLLM_LOG" >&2 || true
  exit 1
fi

python - <<'PY'
import json
import os
from pathlib import Path

job_id = os.environ.get("SLURM_JOB_ID", "manual")
model_path = os.environ["MODEL_PATH"]
reasoning_effort = os.environ.get("REASONING_EFFORT", "")
req_path = Path(os.environ["LOG_DIR"]) / f"mistral-small-4-chat-req-{job_id}.json"
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
if reasoning_effort:
    payload["reasoning_effort"] = reasoning_effort
req_path.write_text(json.dumps(payload))
PY

curl -fsS \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  "${SERVER_URL}/v1/chat/completions" \
  -d @"$CHAT_REQ_JSON" >"$CHAT_JSON"

python - <<'PY'
import json
import os
from pathlib import Path

job_id = os.environ.get("SLURM_JOB_ID", "manual")
log_root = Path(os.environ["LOG_DIR"])
models_path = log_root / f"mistral-small-4-models-{job_id}.json"
chat_path = log_root / f"mistral-small-4-chat-{job_id}.json"

models = json.loads(models_path.read_text())
chat = json.loads(chat_path.read_text())
print("MODELS_OK", [item["id"] for item in models.get("data", [])])
choice = chat["choices"][0]["message"]
print("CHAT_OK", json.dumps(choice, ensure_ascii=True))
usage = chat.get("usage")
if usage is not None:
    print("USAGE", json.dumps(usage, ensure_ascii=True))
PY
EOS

singularity exec --rocm "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$INNER_SCRIPT"
