#!/bin/bash
# Friendly wrapper for submitting dfm-evals jobs without manual env var juggling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_SUBMIT_SCRIPT="$SCRIPT_DIR/run_suite.sbatch"
ENV_FILE=${ENV_FILE:-$REPO_ROOT/.env}

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

SUBMIT_SCRIPT=${SUBMIT_SCRIPT:-$DEFAULT_SUBMIT_SCRIPT}
OVERLAY_DIR=${OVERLAY_DIR:-$REPO_ROOT/overlay_vllm_minimal}
DFM_EVALS_RUN_ROOT=${DFM_EVALS_RUN_ROOT:-$REPO_ROOT/logs/evals-runs}
DFM_EVALS_LOG_ROOT=${DFM_EVALS_LOG_ROOT:-$REPO_ROOT/logs/evals-logs}

MODEL=${MODEL:-google/gemma-3-4b-it}
EVAL_MODEL=${EVAL_MODEL:-}
TARGET_MODEL=${TARGET_MODEL:-}
JUDGE_MODEL=${JUDGE_MODEL:-}
OPENAI_BASE_URL_OVERRIDE=${OPENAI_BASE_URL_OVERRIDE:-}
SUITE=${SUITE:-fundamentals}
LIMIT=${LIMIT:-100}
MAX_CONNECTIONS=${MAX_CONNECTIONS:-64}
RUN_LABEL=${RUN_LABEL:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-$REPO_ROOT/logs/slurm}
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./lumi/submit.sh [options]

Options:
  --model <model>            Base model id (default: google/gemma-3-4b-it)
  --eval-model <model>       Eval runtime model, default: vllm/<model>
  --target-model <model>     Suite target model, default: <eval-model>
  --judge-model <model>      Suite judge model, default: <target-model>
  --openai-base-url <url>    Override OPENAI_BASE_URL inside job/container
  --suite <name>             Eval suite (default: fundamentals)
  --limit <n>                Sample limit (default: 100)
  --max-connections <n>      Concurrency for inspect eval (default: 64)
  --run-label <label>        Optional DFM_EVALS_RUN_LABEL override
  --extra-args <string>      Extra args appended to evals CLI
  --slurm-log-dir <path>     Slurm stdout/err directory (default: ./logs/slurm)
  --script <path>            sbatch script to submit
  --dry-run                  Print sbatch command/env and exit
  --help                     Show help

Examples:
  ./lumi/submit.sh
  ./lumi/submit.sh --limit 100 --max-connections 64
  ./lumi/submit.sh --target-model vllm/google/gemma-3-4b-it --judge-model vllm/google/gemma-3-4b-it
  ./lumi/submit.sh --slurm-log-dir /path/to/slurm-logs
  ./lumi/submit.sh --run-label fundamentals_gemma64c --dry-run
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
    --eval-model)
      need_value "$1" "$#"
      EVAL_MODEL="$2"
      shift 2
      ;;
    --target-model)
      need_value "$1" "$#"
      TARGET_MODEL="$2"
      shift 2
      ;;
    --judge-model)
      need_value "$1" "$#"
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --openai-base-url)
      need_value "$1" "$#"
      OPENAI_BASE_URL_OVERRIDE="$2"
      shift 2
      ;;
    --suite)
      need_value "$1" "$#"
      SUITE="$2"
      shift 2
      ;;
    --limit)
      need_value "$1" "$#"
      LIMIT="$2"
      shift 2
      ;;
    --max-connections)
      need_value "$1" "$#"
      MAX_CONNECTIONS="$2"
      shift 2
      ;;
    --run-label)
      need_value "$1" "$#"
      RUN_LABEL="$2"
      shift 2
      ;;
    --extra-args)
      need_value "$1" "$#"
      EXTRA_ARGS="$2"
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

if [[ -z "$EVAL_MODEL" ]]; then
  EVAL_MODEL="vllm/$MODEL"
fi
if [[ -z "$TARGET_MODEL" ]]; then
  TARGET_MODEL="$EVAL_MODEL"
fi
if [[ -z "$JUDGE_MODEL" ]]; then
  JUDGE_MODEL="$TARGET_MODEL"
fi
if [[ -z "$OPENAI_BASE_URL_OVERRIDE" && -n "${OPENAI_BASE_URL:-}" ]]; then
  OPENAI_BASE_URL_OVERRIDE="$OPENAI_BASE_URL"
fi

[[ -f "$SUBMIT_SCRIPT" ]] || die "submit script not found: $SUBMIT_SCRIPT"
[[ -d "$OVERLAY_DIR" ]] || die "overlay dir not found: $OVERLAY_DIR"
mkdir -p "$SLURM_LOG_DIR"

if [[ -n "$RUN_LABEL" ]]; then
  raw_slurm_log_label="$RUN_LABEL"
else
  raw_slurm_log_label="${SUITE}__${TARGET_MODEL}"
fi
slurm_log_label="${raw_slurm_log_label//[^[:alnum:]._-]/_}"
slurm_out_path="${SLURM_LOG_DIR}/${slurm_log_label}-%j.out"
slurm_err_path="${SLURM_LOG_DIR}/${slurm_log_label}-%j.err"

env_kv=(
  "DFM_EVALS_REPO_ROOT=$REPO_ROOT"
  "DFM_EVALS_RUN_ROOT=$DFM_EVALS_RUN_ROOT"
  "DFM_EVALS_LOG_ROOT=$DFM_EVALS_LOG_ROOT"
  "MODEL=$MODEL"
  "DFM_EVALS_MODEL=$EVAL_MODEL"
  "DFM_EVALS_TARGET_MODEL=$TARGET_MODEL"
  "DFM_EVALS_JUDGE_MODEL=$JUDGE_MODEL"
  "DFM_EVALS_SUITE=$SUITE"
  "DFM_EVALS_LIMIT=$LIMIT"
  "MAX_CONNECTIONS=$MAX_CONNECTIONS"
)
if [[ -n "$RUN_LABEL" ]]; then
  env_kv+=("DFM_EVALS_RUN_LABEL=$RUN_LABEL")
fi
if [[ -n "$OPENAI_BASE_URL_OVERRIDE" ]]; then
  env_kv+=("DFM_EVALS_OPENAI_BASE_URL=$OPENAI_BASE_URL_OVERRIDE")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  env_kv+=("DFM_EVALS_EXTRA_ARGS=$EXTRA_ARGS")
fi

echo "Submit script: $SUBMIT_SCRIPT"
echo "Model: $MODEL"
echo "Eval model: $EVAL_MODEL"
echo "Target model: $TARGET_MODEL"
echo "Judge model: $JUDGE_MODEL"
echo "Suite: $SUITE"
echo "Limit: $LIMIT"
echo "Max connections: $MAX_CONNECTIONS"
echo "Eval run root: $DFM_EVALS_RUN_ROOT"
echo "Eval log root: $DFM_EVALS_LOG_ROOT"
echo "Slurm stdout path pattern: $slurm_out_path"
echo "Slurm stderr path pattern: $slurm_err_path"
if [[ -n "$RUN_LABEL" ]]; then
  echo "Run label override: $RUN_LABEL"
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  echo "Extra args: $EXTRA_ARGS"
fi
if [[ -n "$OPENAI_BASE_URL_OVERRIDE" ]]; then
  echo "OpenAI base URL override: $OPENAI_BASE_URL_OVERRIDE"
fi

cmd=(env "${env_kv[@]}" sbatch --output "$slurm_out_path" --error "$slurm_err_path" "$SUBMIT_SCRIPT")
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
  if [[ -n "$RUN_LABEL" ]]; then
    effective_label="$RUN_LABEL"
  else
    raw_label="${SUITE}__${TARGET_MODEL}__job-${job_id}"
    effective_label="${raw_label//[^[:alnum:]._-]/_}"
  fi
  echo "Job id: $job_id"
  echo "Expected run label: $effective_label"
  echo "Expected host run dir: $DFM_EVALS_RUN_ROOT/$effective_label"
  echo "Expected host log dir: $DFM_EVALS_LOG_ROOT/$effective_label"
  echo "Expected host vLLM server logs: $DFM_EVALS_LOG_ROOT/$effective_label/_vllm_server"
  echo "Slurm stdout: ${slurm_out_path//%j/$job_id}"
  echo "Slurm stderr: ${slurm_err_path//%j/$job_id}"
  echo "View this run: ./lumi/view.sh start --job-id $job_id"
fi
