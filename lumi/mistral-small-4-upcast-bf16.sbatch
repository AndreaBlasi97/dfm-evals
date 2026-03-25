#!/bin/bash
#SBATCH --job-name=ms4-upcast
#SBATCH --account=project_465002183
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals/logs/slurm/mistral-small-4-upcast-bf16-%j.out
#SBATCH --error=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals/logs/slurm/mistral-small-4-upcast-bf16-%j.err

set -euo pipefail

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
SIF=$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif
REPO_ROOT=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal
SOURCE_DIR=/flash/project_465002183/.cache/huggingface/hub/models--mistralai--Mistral-Small-4-119B-2603/snapshots/3b76d234c932e78dc989731bfc4c3b12c0a87918
OUTPUT_DIR=/flash/project_465002183/.cache/huggingface/local-upcasts/mistralai--Mistral-Small-4-119B-2603-bf16
LOG_DIR=$REPO_ROOT/logs/tmp-logs

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")"

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
export HF_HOME="/flash/project_465002183/.cache/huggingface/"
export HF_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
export HF_DATASETS_CACHE="${HF_HOME%/}/datasets"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

cd /pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/dfm-evals

python lumi/upcast_fp8_checkpoint_to_bf16.py \
  --source /flash/project_465002183/.cache/huggingface/hub/models--mistralai--Mistral-Small-4-119B-2603/snapshots/3b76d234c932e78dc989731bfc4c3b12c0a87918 \
  --output /flash/project_465002183/.cache/huggingface/local-upcasts/mistralai--Mistral-Small-4-119B-2603-bf16 \
  --skip-existing \
  --overwrite
EOS

singularity exec "${SING_BIND_ARGS[@]}" "$SIF" bash -lc "$INNER_SCRIPT"
