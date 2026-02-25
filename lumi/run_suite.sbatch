#!/bin/bash
# Run dfm-evals smoke with Inspect/vLLM self-spawned model backend.

#SBATCH --job-name=dfm-evals-vllm-smoke
#SBATCH --account=project_465002183
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output=dfm-evals-vllm-smoke-%j.out
#SBATCH --error=dfm-evals-vllm-smoke-%j.err

set -euo pipefail

REPO_ROOT="${DFM_EVALS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_ROOT/lumi"

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=${SIF:-$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}
OVERLAY_DIR=${OVERLAY_DIR:-$REPO_ROOT/overlay_vllm_minimal}

MODEL=${MODEL:-google/gemma-3-4b-it}
DFM_EVALS_MODEL=${DFM_EVALS_MODEL:-vllm/$MODEL}
DFM_EVALS_TARGET_MODEL=${DFM_EVALS_TARGET_MODEL:-$DFM_EVALS_MODEL}
DFM_EVALS_JUDGE_MODEL=${DFM_EVALS_JUDGE_MODEL:-$DFM_EVALS_TARGET_MODEL}
DFM_EVALS_OPENAI_BASE_URL=${DFM_EVALS_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-}}
DFM_EVALS_SUITE=${DFM_EVALS_SUITE:-fundamentals}
DFM_EVALS_SUITES_FILE=${DFM_EVALS_SUITES_FILE:-}
DFM_EVALS_LIMIT=${DFM_EVALS_LIMIT:-20}
DFM_EVALS_TEMPERATURE=${DFM_EVALS_TEMPERATURE:-0}
DFM_EVALS_EXTRA_ARGS=${DFM_EVALS_EXTRA_ARGS:-}

TP=${TP:-8}
PP=${PP:-1}
CTX=${CTX:-4096}
GPU_MEM=${GPU_MEM:-0.92}
MAX_CONNECTIONS=${MAX_CONNECTIONS:-8}

HF_HOME=${HF_HOME:-/flash/project_465002183/.cache/huggingface/}
_raw_dfm_label=${DFM_EVALS_RUN_LABEL:-${DFM_EVALS_SUITE}__${DFM_EVALS_TARGET_MODEL}__job-${SLURM_JOB_ID:-manual}}
DFM_EVALS_RUN_LABEL=${_raw_dfm_label//[^[:alnum:]._-]/_}
DFM_EVALS_CACHE_DIR=${DFM_EVALS_CACHE_DIR:-/overlay/cache/dfm-evals-${DFM_EVALS_RUN_LABEL}}
DFM_EVALS_RUN_ROOT=${DFM_EVALS_RUN_ROOT:-$REPO_ROOT/logs/evals-runs}
DFM_EVALS_LOG_ROOT=${DFM_EVALS_LOG_ROOT:-$REPO_ROOT/logs/evals-logs}
DFM_EVALS_RUN_DIR=${DFM_EVALS_RUN_DIR:-${DFM_EVALS_RUN_ROOT}/${DFM_EVALS_RUN_LABEL}}
DFM_EVALS_LOG_DIR=${DFM_EVALS_LOG_DIR:-${DFM_EVALS_LOG_ROOT}/${DFM_EVALS_RUN_LABEL}}
DFM_EVALS_SERVER_LOG_DIR=${DFM_EVALS_SERVER_LOG_DIR:-${DFM_EVALS_LOG_DIR}/_vllm_server}

die() {
  echo "FATAL: $*" >&2
  exit 1
}

if [[ "$DFM_EVALS_TARGET_MODEL" == openai/* || "$DFM_EVALS_JUDGE_MODEL" == openai/* || "$DFM_EVALS_MODEL" == openai/* ]]; then
  [[ -n "${OPENAI_API_KEY:-}" ]] || die "OPENAI_API_KEY is required for openai/* models."
  [[ -n "$DFM_EVALS_OPENAI_BASE_URL" ]] || die "OPENAI_BASE_URL (or DFM_EVALS_OPENAI_BASE_URL) is required for openai/* models."
fi

if [[ -n "$DFM_EVALS_OPENAI_BASE_URL" ]]; then
  export OPENAI_BASE_URL="$DFM_EVALS_OPENAI_BASE_URL"
fi

build_evals_cli() {
  local -n out_cmd_ref=$1
  local -a evals_args

  evals_args=(
    suite
    "$DFM_EVALS_SUITE"
    --mode "run"
    --target-model "$DFM_EVALS_TARGET_MODEL"
    --judge-model "$DFM_EVALS_JUDGE_MODEL"
  )

  if [[ -n "$DFM_EVALS_SUITES_FILE" ]]; then
    evals_args+=(
      --file "$DFM_EVALS_SUITES_FILE"
    )
  fi

  evals_args+=(
    --
    --log-dir "$DFM_EVALS_LOG_DIR"
    --model "$DFM_EVALS_MODEL"
    --limit "$DFM_EVALS_LIMIT"
    --temperature "$DFM_EVALS_TEMPERATURE"
    --max-connections "$MAX_CONNECTIONS"
    -M "tensor_parallel_size=$TP"
    -M "pipeline_parallel_size=$PP"
    -M "max_model_len=$CTX"
    -M "gpu_memory_utilization=$GPU_MEM"
    # Use null so inspect emits a bare --enforce-eager flag (vLLM rejects "--enforce-eager True").
    -M "enforce_eager=null"
  )

  local escaped_args=""
  printf -v escaped_args '%q ' "${evals_args[@]}"
  out_cmd_ref="evals ${escaped_args}"
  if [[ -n "$DFM_EVALS_EXTRA_ARGS" ]]; then
    out_cmd_ref+="$DFM_EVALS_EXTRA_ARGS"
  fi
}

container_env_prefix() {
  cat <<'EOC'
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
export HF_HOME="__HF_HOME__"
export HF_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
EOC
}

render_env() {
  container_env_prefix | sed "s|__HF_HOME__|$HF_HOME|g"
}

[[ -f "$SIF" ]] || die "SIF not found: $SIF"
[[ -d "$OVERLAY_DIR/venv/vllm-min" ]] || die "overlay venv missing: $OVERLAY_DIR/venv/vllm-min"

echo "SIF: $SIF"
echo "Overlay: $OVERLAY_DIR"
echo "Model: $MODEL"
echo "Eval model: $DFM_EVALS_MODEL"
echo "Eval target model: $DFM_EVALS_TARGET_MODEL"
echo "Eval judge model: $DFM_EVALS_JUDGE_MODEL"
echo "Suite: $DFM_EVALS_SUITE"
echo "Run label: $DFM_EVALS_RUN_LABEL"
echo "Suites file: ${DFM_EVALS_SUITES_FILE:-<packaged default>}"
echo "Limit: $DFM_EVALS_LIMIT"
echo "Temperature: $DFM_EVALS_TEMPERATURE"
echo "TP: $TP"
echo "PP: $PP"
echo "CTX: $CTX"
echo "GPU_MEM: $GPU_MEM"
echo "MAX_CONNECTIONS: $MAX_CONNECTIONS"
echo "HF_HOME: $HF_HOME"
echo "Run dir: $DFM_EVALS_RUN_DIR"
echo "Log dir: $DFM_EVALS_LOG_DIR"
echo "vLLM server log dir: $DFM_EVALS_SERVER_LOG_DIR"
echo "OpenAI base URL: ${OPENAI_BASE_URL:-<unset>}"

SING_BIND_ARGS=(
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

EVALS_CMD=""
build_evals_cli EVALS_CMD

printf -v DFM_EVALS_CACHE_DIR_Q '%q' "$DFM_EVALS_CACHE_DIR"
printf -v DFM_EVALS_RUN_DIR_Q '%q' "$DFM_EVALS_RUN_DIR"
printf -v DFM_EVALS_LOG_DIR_Q '%q' "$DFM_EVALS_LOG_DIR"
printf -v DFM_EVALS_SERVER_LOG_DIR_Q '%q' "$DFM_EVALS_SERVER_LOG_DIR"
RUN_CMD="$(render_env)
export XDG_CACHE_HOME=${DFM_EVALS_CACHE_DIR_Q}
export INSPECT_LOCAL_SERVER_LOG_DIR=${DFM_EVALS_SERVER_LOG_DIR_Q}
export INSPECT_LOCAL_SERVER_LOG_PREFIX=vllm-server
mkdir -p ${DFM_EVALS_CACHE_DIR_Q}
mkdir -p ${DFM_EVALS_RUN_DIR_Q}
mkdir -p ${DFM_EVALS_LOG_DIR_Q}
mkdir -p ${DFM_EVALS_SERVER_LOG_DIR_Q}
cd ${DFM_EVALS_RUN_DIR_Q}
command -v evals >/dev/null 2>&1 || { echo 'FATAL: evals not installed in overlay venv'; exit 1; }
${EVALS_CMD}"

echo "Running dfm-evals suite with self-spawned vLLM backend..."
singularity exec --rocm "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$RUN_CMD"
echo "dfm-evals suite complete."
