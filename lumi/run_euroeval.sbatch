#!/bin/bash
# Two-node vLLM MP launch using a single srun step (one task per node),
# followed by EuroEval against the in-job OpenAI-compatible vLLM endpoint.

#SBATCH --job-name=vllm-q35-2n-euroeval
#SBATCH --account=project_465002183
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output=vllm-q35-mp1-%j.out
#SBATCH --error=vllm-q35-mp1-%j.err

set -euo pipefail

REPO_ROOT="${DFM_EVALS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_ROOT/lumi"

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=${SIF:-$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}
OVERLAY_DIR=${OVERLAY_DIR:-$REPO_ROOT/overlay_vllm_minimal}

MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
HF_HOME=${HF_HOME:-/flash/project_465002183/.cache/huggingface/}

PORT=${PORT:-8000}
TP=${TP:-4}
PP=${PP:-4}
CTX=${CTX:-8192}
GPU_MEM=${GPU_MEM:-0.92}

RUN_EUROEVAL=${RUN_EUROEVAL:-1}
EUROEVAL_MODEL=${EUROEVAL_MODEL:-$SERVED_MODEL_NAME}
EUROEVAL_API_BASE=${EUROEVAL_API_BASE:-http://127.0.0.1:$PORT/v1}
EUROEVAL_LANGUAGES=${EUROEVAL_LANGUAGES:-en}
EUROEVAL_TASKS=${EUROEVAL_TASKS:-}
EUROEVAL_NUM_ITERATIONS=${EUROEVAL_NUM_ITERATIONS:-10}
EUROEVAL_CACHE_DIR=${EUROEVAL_CACHE_DIR:-/overlay/euroeval-cache-${SLURM_JOB_ID:-manual}}
EUROEVAL_RUN_DIR=${EUROEVAL_RUN_DIR:-/overlay/euroeval-runs/${SLURM_JOB_ID:-manual}}
EUROEVAL_RAISE_ERRORS=${EUROEVAL_RAISE_ERRORS:-1}
EUROEVAL_EXTRA_ARGS=${EUROEVAL_EXTRA_ARGS:-}
SLURM_JOB_ID_SAFE=${SLURM_JOB_ID:-manual}
EUROEVAL_SLURM_LOG_DIR=${EUROEVAL_SLURM_LOG_DIR:-$REPO_ROOT/logs/slurm}
RANK_LOG_PATH="${EUROEVAL_SLURM_LOG_DIR}/vllm-q35-mp1-rank-%t-${SLURM_JOB_ID_SAFE}.log"
COMPLETION_FILE="${EUROEVAL_SLURM_LOG_DIR}/completion-qwen35-mp1-${SLURM_JOB_ID_SAFE}.json"

MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-29501}
LAUNCH_TIMEOUT=${LAUNCH_TIMEOUT:-7200}
LAUNCH_STEP=${LAUNCH_STEP:-10}

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
RCCL_NET_SO_SRC=${RCCL_NET_SO_SRC:-}
LIBFABRIC_EXTRACT_DIR=${LIBFABRIC_EXTRACT_DIR:-$SCRIPT_DIR/libfabric_extract_20260216}
LIBFABRIC_SO=${LIBFABRIC_SO:-$LIBFABRIC_EXTRACT_DIR/libfabric.so.1.27.0}
LIBCXI_SO=${LIBCXI_SO:-$LIBFABRIC_EXTRACT_DIR/libcxi.so.1.5.0}

if [[ ! -d "$LIBFABRIC_EXTRACT_DIR" && -d "$REPO_ROOT/../libfabric_extract_20260216" ]]; then
  LIBFABRIC_EXTRACT_DIR="$REPO_ROOT/../libfabric_extract_20260216"
  LIBFABRIC_SO="${LIBFABRIC_EXTRACT_DIR}/libfabric.so.1.27.0"
  LIBCXI_SO="${LIBFABRIC_EXTRACT_DIR}/libcxi.so.1.5.0"
fi

TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-}
ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE:-0}
TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
REASONING_PARSER=${REASONING_PARSER:-auto}
STRUCTURED_OUTPUTS_CONFIG=${STRUCTURED_OUTPUTS_CONFIG:-}

SRUN_PID=""
HEAD_NODE=""

die() {
  echo "FATAL: $*" >&2
  exit 1
}

build_euroeval_cli() {
  local -n out_cmd_ref=$1
  local -a euroeval_args

  euroeval_args=(
    --model "$EUROEVAL_MODEL"
    --api-base "$EUROEVAL_API_BASE"
    --cache-dir "$EUROEVAL_CACHE_DIR"
    --num-iterations "$EUROEVAL_NUM_ITERATIONS"
  )

  if [[ "$EUROEVAL_RAISE_ERRORS" == "1" ]]; then
    euroeval_args+=(--raise-errors)
  else
    euroeval_args+=(--no-raise-errors)
  fi

  local raw_languages="${EUROEVAL_LANGUAGES//,/ }"
  local lang
  local languages_added=0
  for lang in $raw_languages; do
    [[ -n "$lang" ]] || continue
    euroeval_args+=(--language "$lang")
    languages_added=1
  done
  if [[ "$languages_added" -eq 0 ]]; then
    euroeval_args+=(--language all)
  fi

  local raw_tasks="${EUROEVAL_TASKS//,/ }"
  local task
  for task in $raw_tasks; do
    [[ -n "$task" ]] || continue
    euroeval_args+=(--task "$task")
  done

  local escaped_args=""
  printf -v escaped_args '%q ' "${euroeval_args[@]}"
  out_cmd_ref="euroeval ${escaped_args}"
  if [[ -n "$EUROEVAL_EXTRA_ARGS" ]]; then
    out_cmd_ref+="$EUROEVAL_EXTRA_ARGS"
  fi
}

container_env_prefix() {
  cat <<'EOC'
set -euo pipefail
source /overlay/venv/vllm-min/bin/activate
OVERLAY_SITE=/overlay/venv/vllm-min/lib/python3.12/site-packages
if [[ -d /overlay/src/transformers/src ]]; then
  export PYTHONPATH=/overlay/src/transformers/src:/overlay/src/vllm:${OVERLAY_SITE}${PYTHONPATH:+:$PYTHONPATH}
else
  export PYTHONPATH=/overlay/src/vllm:${OVERLAY_SITE}${PYTHONPATH:+:$PYTHONPATH}
fi
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
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

cleanup() {
  set +e
  stop_srun
}
trap cleanup EXIT

stop_srun() {
  if [[ -z "${SRUN_PID}" ]]; then
    return
  fi
  if ! kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
    wait "${SRUN_PID}" >/dev/null 2>&1 || true
    SRUN_PID=""
    return
  fi

  kill "${SRUN_PID}" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    if ! kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
    scancel --signal=TERM "${SLURM_JOB_ID}.0" >/dev/null 2>&1 || true
    sleep 2
  fi
  if kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
    kill -9 "${SRUN_PID}" >/dev/null 2>&1 || true
  fi
  wait "${SRUN_PID}" >/dev/null 2>&1 || true
  SRUN_PID=""
}

[[ -f "$SIF" ]] || die "SIF not found: $SIF"
[[ -d "$OVERLAY_DIR/venv/vllm-min" ]] || die "overlay venv missing: $OVERLAY_DIR/venv/vllm-min"
[[ "${SLURM_NNODES:-0}" -ge 2 ]] || die "this script requires at least 2 nodes"
mkdir -p "$EUROEVAL_SLURM_LOG_DIR"

readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#NODES[@]}" -ge 2 ]] || die "expected 2 nodes, got ${#NODES[@]}"
HEAD_NODE="${NODES[0]}"

if [[ -z "$MASTER_ADDR" ]]; then
  MASTER_ADDR="$HEAD_NODE"
fi

echo "SIF: $SIF"
echo "Overlay: $OVERLAY_DIR"
echo "Model: $MODEL"
echo "HF_HOME: $HF_HOME"
echo "Nodes: ${NODES[*]}"
echo "Head node: $HEAD_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "NCCL_NET_GDR_LEVEL: $NCCL_NET_GDR_LEVEL"
echo "NCCL_NET_PLUGIN_NAME: ${NCCL_NET_PLUGIN_NAME:-<unset>}"
echo "AWS_OFI_RCCL_LIBDIR: $AWS_OFI_RCCL_LIBDIR"
echo "RCCL_NET_SO_SRC: $RCCL_NET_SO_SRC"
echo "LIBFABRIC_SO: $LIBFABRIC_SO"
echo "LIBCXI_SO: $LIBCXI_SO"
echo "TORCHDYNAMO_DISABLE: $TORCHDYNAMO_DISABLE"
echo "REASONING_PARSER: $REASONING_PARSER"
echo "STRUCTURED_OUTPUTS_CONFIG: ${STRUCTURED_OUTPUTS_CONFIG:-<default-auto>}"
echo "RUN_EUROEVAL: $RUN_EUROEVAL"
echo "EUROEVAL_MODEL: $EUROEVAL_MODEL"
echo "EUROEVAL_API_BASE: $EUROEVAL_API_BASE"
echo "EUROEVAL_LANGUAGES: $EUROEVAL_LANGUAGES"
echo "EUROEVAL_TASKS: ${EUROEVAL_TASKS:-<all-tasks>}"
echo "EUROEVAL_NUM_ITERATIONS: $EUROEVAL_NUM_ITERATIONS"
echo "EUROEVAL_CACHE_DIR: $EUROEVAL_CACHE_DIR"
echo "EUROEVAL_RUN_DIR: $EUROEVAL_RUN_DIR"
echo "Slurm log dir: $EUROEVAL_SLURM_LOG_DIR"
echo "Rank log path pattern: $RANK_LOG_PATH"

SING_BIND_ARGS=(
  -B "$BASE_DIR:$LAIFS_APPL_DIR"
  -B "$OVERLAY_DIR:/overlay"
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
if [[ -f "$RCCL_NET_SO_SRC" ]]; then
  SING_BIND_ARGS+=(
    -B "$RCCL_NET_SO_SRC:/lib/librccl-net.so"
    -B "$RCCL_NET_SO_SRC:/usr/lib/librccl-net.so"
  )
fi
if [[ -f "$LIBFABRIC_SO" ]]; then
  SING_BIND_ARGS+=(
    -B "$LIBFABRIC_SO:/lib/x86_64-linux-gnu/libfabric.so.1"
    -B "$LIBFABRIC_SO:/usr/lib/libfabric.so.1"
  )
fi
if [[ -f "$LIBCXI_SO" ]]; then
  SING_BIND_ARGS+=(
    -B "$LIBCXI_SO:/lib/libcxi.so.1"
    -B "$LIBCXI_SO:/usr/lib/libcxi.so.1"
  )
fi

RANK_CMD="$(render_env)"
RANK_CMD="${RANK_CMD}
export NODE_RANK=\${SLURM_PROCID}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PLUGINS=''
export VLLM_DISABLE_PYNCCL=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_PROVIDER='$FI_PROVIDER_NAME'
export FI_CXI_DISABLE_HMEM='$FI_CXI_DISABLE_HMEM_FLAG'
export GLOO_SOCKET_IFNAME='$GLOO_IFNAME'
export NCCL_SOCKET_IFNAME='$NCCL_SOCKET_IFNAME'
export NCCL_NET_GDR_LEVEL='$NCCL_NET_GDR_LEVEL'
export NCCL_DEBUG='$NCCL_DEBUG_LEVEL'
export NCCL_DEBUG_SUBSYS='$NCCL_DEBUG_SUBSYS_LEVEL'
export TORCHDYNAMO_DISABLE='$TORCHDYNAMO_DISABLE'
if [[ -d '$AWS_OFI_RCCL_LIBDIR' ]]; then
  export LD_LIBRARY_PATH='$AWS_OFI_RCCL_LIBDIR':\${LD_LIBRARY_PATH:-}
fi
if [[ -n '$NCCL_NET_PLUGIN_NAME' ]]; then
  export NCCL_NET_PLUGIN='$NCCL_NET_PLUGIN_NAME'
else
  unset NCCL_NET_PLUGIN
fi
if [[ -n '$NCCL_NET_BACKEND' ]]; then
  export NCCL_NET='$NCCL_NET_BACKEND'
else
  unset NCCL_NET
fi
EXTRA_VLLM_FLAGS=()
if [[ '$REASONING_PARSER' == 'auto' ]]; then
  case '$MODEL' in
    Qwen/Qwen3*|Qwen/Qwen3.5*) EXTRA_VLLM_FLAGS+=(--reasoning-parser qwen3) ;;
  esac
elif [[ -n '$REASONING_PARSER' && '$REASONING_PARSER' != 'none' ]]; then
  EXTRA_VLLM_FLAGS+=(--reasoning-parser '$REASONING_PARSER')
fi
if [[ -n '$STRUCTURED_OUTPUTS_CONFIG' ]]; then
  EXTRA_VLLM_FLAGS+=(--structured-outputs-config '$STRUCTURED_OUTPUTS_CONFIG')
fi
if [[ '$ENABLE_AUTO_TOOL_CHOICE' == '1' ]]; then
  EXTRA_VLLM_FLAGS+=(--enable-auto-tool-choice)
fi
if [[ -n '$TOOL_CALL_PARSER' ]]; then
  EXTRA_VLLM_FLAGS+=(--tool-call-parser '$TOOL_CALL_PARSER')
fi
if [[ \${NODE_RANK} -ne 0 ]]; then
  EXTRA_VLLM_FLAGS+=(--headless)
fi
vllm serve '$MODEL' \
  --served-model-name '$SERVED_MODEL_NAME' \
  --host 0.0.0.0 \
  --port '$PORT' \
  --dtype bfloat16 \
  --distributed-executor-backend mp \
  --data-parallel-backend mp \
  --tensor-parallel-size '$TP' \
  --pipeline-parallel-size '$PP' \
  --nnodes '${#NODES[@]}' \
  --node-rank \${NODE_RANK} \
  --master-addr '$MASTER_ADDR' \
  --master-port '$MASTER_PORT' \
  --gpu-memory-utilization '$GPU_MEM' \
  --max-model-len '$CTX' \
  --safetensors-load-strategy eager \
  --enforce-eager \
  \${EXTRA_VLLM_FLAGS[@]}"

echo "Starting single-step srun MP launch..."
srun --overlap \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-56}" \
  --output="$RANK_LOG_PATH" \
  --error="$RANK_LOG_PATH" \
  singularity exec --rocm "${SING_BIND_ARGS[@]}" "$SIF" \
  bash -lc "$RANK_CMD" &
SRUN_PID=$!

READY=0
for ((i=0; i<LAUNCH_TIMEOUT; i+=LAUNCH_STEP)); do
  if ! kill -0 "$SRUN_PID" >/dev/null 2>&1; then
    die "srun exited early; check ${EUROEVAL_SLURM_LOG_DIR}/vllm-q35-mp1-rank-*-${SLURM_JOB_ID_SAFE}.log"
  fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep "$LAUNCH_STEP"
done
[[ "$READY" -eq 1 ]] || die "vLLM server did not become ready"
echo "Server ready."

curl -sS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$SERVED_MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Write one short sentence about LUMI.\"}], \"temperature\": 0.2, \"max_tokens\": 64}" \
  > "$COMPLETION_FILE"

COMPLETION_FILE="$COMPLETION_FILE" python3 - <<'PY'
import json
import os
p = os.environ["COMPLETION_FILE"]
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)
print("Generated text:")
if "choices" in data and data["choices"]:
    print(data["choices"][0]["message"]["content"].strip())
else:
    print(json.dumps(data, indent=2))
    raise SystemExit(1)
PY

echo "Run complete."

if [[ "$RUN_EUROEVAL" == "1" ]]; then
  EUROEVAL_CMD=""
  build_euroeval_cli EUROEVAL_CMD

  printf -v EUROEVAL_CACHE_DIR_Q '%q' "$EUROEVAL_CACHE_DIR"
  printf -v EUROEVAL_RUN_DIR_Q '%q' "$EUROEVAL_RUN_DIR"
  EVAL_CMD="$(render_env)
export EUROEVAL_ALLOW_FLASH_ATTN=1
mkdir -p ${EUROEVAL_RUN_DIR_Q}
mkdir -p ${EUROEVAL_CACHE_DIR_Q}
cd ${EUROEVAL_RUN_DIR_Q}
command -v euroeval >/dev/null 2>&1 || { echo 'FATAL: euroeval not installed in overlay venv'; exit 1; }
${EUROEVAL_CMD}"

  echo "Starting EuroEval against ${EUROEVAL_API_BASE} ..."
  singularity exec --rocm "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$EVAL_CMD"
  echo "EuroEval results file: ${EUROEVAL_RUN_DIR}/euroeval_benchmark_results.jsonl"
  echo "EuroEval complete."
else
  echo "RUN_EUROEVAL=0, skipping EuroEval."
fi

stop_srun
