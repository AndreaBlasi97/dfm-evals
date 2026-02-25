#!/bin/bash
# Friendly wrapper for submitting 2-node vLLM + EuroEval jobs on LUMI.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_SUBMIT_SCRIPT="$SCRIPT_DIR/run_euroeval.sbatch"
ENV_FILE=${ENV_FILE:-$REPO_ROOT/.env}

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

SUBMIT_SCRIPT=${SUBMIT_SCRIPT:-$DEFAULT_SUBMIT_SCRIPT}
OVERLAY_DIR=${OVERLAY_DIR:-$REPO_ROOT/overlay_vllm_minimal}

MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
PORT=${PORT:-8000}
TP=${TP:-4}
PP=${PP:-4}
CTX=${CTX:-8192}
GPU_MEM=${GPU_MEM:-0.92}

RUN_EUROEVAL=${RUN_EUROEVAL:-1}
EUROEVAL_MODEL=${EUROEVAL_MODEL:-$SERVED_MODEL_NAME}
EUROEVAL_LANGUAGES=${EUROEVAL_LANGUAGES:-en}
EUROEVAL_TASKS=${EUROEVAL_TASKS:-}
EUROEVAL_NUM_ITERATIONS=${EUROEVAL_NUM_ITERATIONS:-10}
EUROEVAL_EXTRA_ARGS=${EUROEVAL_EXTRA_ARGS:-}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-$REPO_ROOT/logs/slurm}
TIME_LIMIT=${TIME_LIMIT:-}

DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./lumi/euroeval_submit.sh [options]

Options:
  --model <model>            Base served model (default: Qwen/Qwen3.5-397B-A17B)
  --served-model-name <id>   Served model name (default: <model>)
  --port <n>                 vLLM API port (default: 8000)
  --tp <n>                   Tensor parallel size (default: 4)
  --pp <n>                   Pipeline parallel size (default: 4)
  --ctx <n>                  Max model len (default: 8192)
  --gpu-mem <f>              GPU memory util (default: 0.92)
  --run-euroeval             Run EuroEval after server startup (default)
  --no-euroeval              Skip EuroEval and only launch/verify server
  --euroeval-model <id>      EuroEval model id (default: <served-model-name>)
  --languages <csv>          EuroEval languages (default: en)
  --tasks <csv>              EuroEval tasks (default: all)
  --iterations <n>           EuroEval num iterations (default: 10)
  --extra-args <string>      Extra args appended to EuroEval CLI
  --time <HH:MM:SS>          Slurm time limit override (default: from sbatch file)
  --slurm-log-dir <path>     Slurm stdout/err directory (default: ./logs/slurm)
  --script <path>            sbatch script path override
  --dry-run                  Print sbatch command/env and exit
  --help                     Show help

Examples:
  ./lumi/euroeval_submit.sh
  ./lumi/euroeval_submit.sh --languages en,da --iterations 1
  ./lumi/euroeval_submit.sh --tasks "knowledge,summarization"
  ./lumi/euroeval_submit.sh --time 12:00:00
  ./lumi/euroeval_submit.sh --slurm-log-dir /path/to/slurm-logs
  ./lumi/euroeval_submit.sh --no-euroeval
EOF
}

die() {
  echo "FATAL: $*" >&2
  exit 1
}

need_value() {
  local opt="$1"
  local remaining="$2"
  if [[ "$remaining" -lt 2 ]]; then
    die "missing value for $opt"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      need_value "$1" "$#"
      MODEL="$2"
      shift 2
      ;;
    --served-model-name)
      need_value "$1" "$#"
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --port)
      need_value "$1" "$#"
      PORT="$2"
      shift 2
      ;;
    --tp)
      need_value "$1" "$#"
      TP="$2"
      shift 2
      ;;
    --pp)
      need_value "$1" "$#"
      PP="$2"
      shift 2
      ;;
    --ctx)
      need_value "$1" "$#"
      CTX="$2"
      shift 2
      ;;
    --gpu-mem)
      need_value "$1" "$#"
      GPU_MEM="$2"
      shift 2
      ;;
    --run-euroeval)
      RUN_EUROEVAL=1
      shift
      ;;
    --no-euroeval)
      RUN_EUROEVAL=0
      shift
      ;;
    --euroeval-model)
      need_value "$1" "$#"
      EUROEVAL_MODEL="$2"
      shift 2
      ;;
    --languages)
      need_value "$1" "$#"
      EUROEVAL_LANGUAGES="$2"
      shift 2
      ;;
    --tasks)
      need_value "$1" "$#"
      EUROEVAL_TASKS="$2"
      shift 2
      ;;
    --iterations)
      need_value "$1" "$#"
      EUROEVAL_NUM_ITERATIONS="$2"
      shift 2
      ;;
    --extra-args)
      need_value "$1" "$#"
      EUROEVAL_EXTRA_ARGS="$2"
      shift 2
      ;;
    --time)
      need_value "$1" "$#"
      TIME_LIMIT="$2"
      shift 2
      ;;
    --slurm-log-dir)
      need_value "$1" "$#"
      SLURM_LOG_DIR="$2"
      shift 2
      ;;
    --script)
      need_value "$1" "$#"
      SUBMIT_SCRIPT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

[[ -f "$SUBMIT_SCRIPT" ]] || die "submit script not found: $SUBMIT_SCRIPT"
[[ -d "$OVERLAY_DIR" ]] || die "overlay dir not found: $OVERLAY_DIR"
mkdir -p "$SLURM_LOG_DIR"

raw_slurm_log_label="euroeval__${SERVED_MODEL_NAME}"
slurm_log_label="${raw_slurm_log_label//[^[:alnum:]._-]/_}"
slurm_out_path="${SLURM_LOG_DIR}/${slurm_log_label}-%j.out"
slurm_err_path="${SLURM_LOG_DIR}/${slurm_log_label}-%j.err"

env_kv=(
  "DFM_EVALS_REPO_ROOT=$REPO_ROOT"
  "OVERLAY_DIR=$OVERLAY_DIR"
  "MODEL=$MODEL"
  "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
  "PORT=$PORT"
  "TP=$TP"
  "PP=$PP"
  "CTX=$CTX"
  "GPU_MEM=$GPU_MEM"
  "RUN_EUROEVAL=$RUN_EUROEVAL"
  "EUROEVAL_MODEL=$EUROEVAL_MODEL"
  "EUROEVAL_LANGUAGES=$EUROEVAL_LANGUAGES"
  "EUROEVAL_NUM_ITERATIONS=$EUROEVAL_NUM_ITERATIONS"
  "EUROEVAL_SLURM_LOG_DIR=$SLURM_LOG_DIR"
)
if [[ -n "$EUROEVAL_TASKS" ]]; then
  env_kv+=("EUROEVAL_TASKS=$EUROEVAL_TASKS")
fi
if [[ -n "$EUROEVAL_EXTRA_ARGS" ]]; then
  env_kv+=("EUROEVAL_EXTRA_ARGS=$EUROEVAL_EXTRA_ARGS")
fi

echo "Submit script: $SUBMIT_SCRIPT"
echo "Overlay: $OVERLAY_DIR"
echo "Model: $MODEL"
echo "Served model name: $SERVED_MODEL_NAME"
echo "TP/PP: $TP/$PP"
echo "CTX: $CTX"
echo "GPU_MEM: $GPU_MEM"
echo "Port: $PORT"
echo "Run EuroEval: $RUN_EUROEVAL"
echo "EuroEval model: $EUROEVAL_MODEL"
echo "EuroEval languages: $EUROEVAL_LANGUAGES"
echo "EuroEval tasks: ${EUROEVAL_TASKS:-<all>}"
echo "EuroEval iterations: $EUROEVAL_NUM_ITERATIONS"
if [[ -n "$TIME_LIMIT" ]]; then
  echo "Slurm time limit override: $TIME_LIMIT"
fi
echo "Slurm stdout path pattern: $slurm_out_path"
echo "Slurm stderr path pattern: $slurm_err_path"
if [[ -n "$EUROEVAL_EXTRA_ARGS" ]]; then
  echo "EuroEval extra args: $EUROEVAL_EXTRA_ARGS"
fi

sbatch_args=(--output "$slurm_out_path" --error "$slurm_err_path")
if [[ -n "$TIME_LIMIT" ]]; then
  sbatch_args+=(--time "$TIME_LIMIT")
fi
cmd=(env "${env_kv[@]}" sbatch "${sbatch_args[@]}" "$SUBMIT_SCRIPT")
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Dry run command: '
  printf '(cd %q && ' "$REPO_ROOT"
  printf '%q ' "${cmd[@]}"
  printf ')'
  echo
  exit 0
fi

submit_out="$(cd "$REPO_ROOT" && "${cmd[@]}")"
echo "$submit_out"

job_id="$(awk '/Submitted batch job/{print $4}' <<<"$submit_out")"
if [[ -n "$job_id" ]]; then
  echo "Job id: $job_id"
  echo "Stdout: ${slurm_out_path//%j/$job_id}"
  echo "Stderr: ${slurm_err_path//%j/$job_id}"
  echo "Rank logs: $SLURM_LOG_DIR/vllm-q35-mp1-rank-*-${job_id}.log"
fi
