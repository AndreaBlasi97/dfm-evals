#!/bin/bash
# Smoke test for GPT-OSS 120B on the LUMI overlay/container stack.

#SBATCH --job-name=gpt-oss-120b-smoke
#SBATCH --account=project_465002183
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=00:45:00
#SBATCH --output=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals/logs/slurm/gpt-oss-120b-smoke-%j.out
#SBATCH --error=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals/logs/slurm/gpt-oss-120b-smoke-%j.err

set -euo pipefail

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif
REPO_ROOT=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal
HF_HOME=/flash/project_465002183/.cache/huggingface/
LOG_DIR=$REPO_ROOT/logs/tmp-logs
MODEL_PATH=${MODEL_PATH:-openai/gpt-oss-120b}
export MODEL_PATH

mkdir -p "$LOG_DIR"

declare -a SING_BIND_ARGS=(
  -B "$BASE_DIR:$LAIFS_APPL_DIR"
  -B "$OVERLAY_DIR:/overlay"
  -B "$REPO_ROOT:$REPO_ROOT"
)

if [[ -d /flash ]]; then
  SING_BIND_ARGS+=(-B /flash:/flash)
fi
if [[ -L /flash/project_465002183 ]]; then
  FLASH_TARGET="$(readlink -f /flash/project_465002183 || true)"
  if [[ -n "$FLASH_TARGET" && -d "$FLASH_TARGET" ]]; then
    SING_BIND_ARGS+=(-B "$FLASH_TARGET:$FLASH_TARGET")
  fi
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
export HF_HOME="/flash/project_465002183/.cache/huggingface/"
export HF_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
export MODEL_PATH="${MODEL_PATH}"

REPO_ROOT=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals
LOG_DIR="$REPO_ROOT/logs/tmp-logs"
JOB_ID="${SLURM_JOB_ID:-manual}"
SERVER_HOST=127.0.0.1
SERVER_PORT=8021
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
API_KEY=inspectai
VLLM_LOG="$LOG_DIR/gpt-oss-120b-vllm-${JOB_ID}.log"
MODELS_JSON="$LOG_DIR/gpt-oss-120b-models-${JOB_ID}.json"
CHAT_JSON="$LOG_DIR/gpt-oss-120b-chat-${JOB_ID}.json"
CHAT_REQ_JSON="$LOG_DIR/gpt-oss-120b-chat-req-${JOB_ID}.json"

cd "$REPO_ROOT"

python - <<'PY'
import openai_harmony
import transformers
import vllm

print("openai_harmony", getattr(openai_harmony, "__version__", "n/a"), openai_harmony.__file__)
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
req_path = Path("/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals/logs/tmp-logs") / f"gpt-oss-120b-chat-req-{job_id}.json"
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
    if reasoning_effort == "none":
        payload["include_reasoning"] = False
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
log_root = Path("/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals/logs/tmp-logs")
models_path = log_root / f"gpt-oss-120b-models-{job_id}.json"
chat_path = log_root / f"gpt-oss-120b-chat-{job_id}.json"

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
