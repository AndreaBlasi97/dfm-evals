#!/bin/bash
#SBATCH --job-name=ms4-upcast
#SBATCH --account=project_465002183
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=/pfs/lustrep4/scratch/project_465002183/rasmus/artifacts/evals/slurm/mistral-small-4-upcast-bf16-%j.out
#SBATCH --error=/pfs/lustrep4/scratch/project_465002183/rasmus/artifacts/evals/slurm/mistral-small-4-upcast-bf16-%j.err

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
SOURCE_DIR=${SOURCE_DIR:-$HF_HOME/hub/models--mistralai--Mistral-Small-4-119B-2603/snapshots/3b76d234c932e78dc989731bfc4c3b12c0a87918}
OUTPUT_DIR=${OUTPUT_DIR:-$POST_ARTIFACT_ROOT/evals/manual-vllm/local-upcasts/mistralai--Mistral-Small-4-119B-2603-bf16}
LOG_DIR=${LOG_DIR:-$POST_ARTIFACT_ROOT/evals/tmp-logs}
export REPO_ROOT
export OVERLAY_DIR
export HF_HOME
export SOURCE_DIR
export OUTPUT_DIR
export LOG_DIR

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")"
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
export HF_HOME="${HF_HOME%/}"
export HF_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
cd "$REPO_ROOT"

python lumi/upcast_fp8_checkpoint_to_bf16.py \
  --source "$SOURCE_DIR" \
  --output "$OUTPUT_DIR" \
  --skip-existing \
  --overwrite
EOS

singularity exec "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$INNER_SCRIPT"
