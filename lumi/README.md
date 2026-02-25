# LUMI Toolkit

LUMI-specific helpers for running `dfm-evals` with inspect + vLLM in the LUMI container environment.

## Files

- `lumi/build_overlay_minimal.sh`: build/update overlay venv with vLLM + dependencies.
- `lumi/submit.sh`: submit suite runs via `sbatch`.
- `lumi/run_suite.sbatch`: batch job entrypoint used by `submit.sh`.
- `lumi/euroeval_submit.sh`: submit 2-node vLLM + EuroEval jobs.
- `lumi/run_euroeval.sbatch`: 2-node vLLM MP launcher with optional EuroEval.
- `lumi/view.sh`: inspect-view helper (default log root: `logs/evals-logs`).

## Quick Start

From repository root:

```bash
# 1) Configure credentials (if using openai/* models)
cat > .env <<'EOF'
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
EOF

# 2) Build overlay (first time or after runtime updates)
./lumi/build_overlay_minimal.sh

# 3) Submit fundamentals suite (default: gemma target + judge)
./lumi/submit.sh --limit 100
```

## Recommended Commands

Inspect smoke (fast validation):

```bash
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal \
./lumi/submit.sh --limit 1 --max-connections 2 --run-label inspect_smoke
```

EuroEval smoke on smaller model:

```bash
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal \
./lumi/euroeval_submit.sh \
  --model google/gemma-3-4b-it \
  --served-model-name google/gemma-3-4b-it \
  --euroeval-model google/gemma-3-4b-it \
  --tp 2 --pp 1 --ctx 4096 \
  --languages en --tasks knowledge --iterations 1
```

EuroEval on Qwen 3.5 397B, Danish, longer wall time:

```bash
OVERLAY_DIR=/pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal \
./lumi/euroeval_submit.sh \
  --model Qwen/Qwen3.5-397B-A17B \
  --served-model-name Qwen/Qwen3.5-397B-A17B \
  --euroeval-model Qwen/Qwen3.5-397B-A17B \
  --languages da \
  --iterations 10 \
  --time 12:00:00
```

## External Judge Example

```bash
./lumi/submit.sh \
  --target-model vllm/google/gemma-3-4b-it \
  --judge-model openai/<model> \
  --limit 100
```

For `openai/*`, `OPENAI_BASE_URL` must be configured (no fallback is used).

## EuroEval

Run the dedicated 2-node vLLM MP + EuroEval workflow:

```bash
./lumi/euroeval_submit.sh
```

Common examples:

```bash
./lumi/euroeval_submit.sh --languages en,da --iterations 1
./lumi/euroeval_submit.sh --tasks "knowledge,summarization"
./lumi/euroeval_submit.sh --time 12:00:00
./lumi/euroeval_submit.sh --no-euroeval
```

EuroEval artifacts are written under overlay paths from `run_euroeval.sbatch`, by default:

- `/overlay/euroeval-cache-<job_id>`
- `/overlay/euroeval-runs/<job_id>/euroeval_benchmark_results.jsonl`

These `/overlay/...` paths are container paths bind-mounted from host `OVERLAY_DIR`.

## Inspect View

```bash
./lumi/view.sh list
./lumi/view.sh start --latest
./lumi/view.sh start --job-id <job_id>
```

By default, `view.sh` reads from `logs/evals-logs/`.

## Log Locations

Default eval artifact roots:

- `logs/evals-runs/<run_label>`
- `logs/evals-logs/<run_label>`
- `logs/evals-logs/<run_label>/_vllm_server` (inspect-spawned vLLM raw logs)

Overlay still holds runtime environment assets (`venv`, source checkouts, cache).

Slurm stdout/stderr from `lumi/submit.sh` default to:

- `logs/slurm/<suite_or_run_label>-<job_id>.out`
- `logs/slurm/<suite_or_run_label>-<job_id>.err`

EuroEval submit/run logs default to:

- `logs/slurm/euroeval__<served_model_name>-<job_id>.out`
- `logs/slurm/euroeval__<served_model_name>-<job_id>.err`
- `logs/slurm/vllm-q35-mp1-rank-<rank>-<job_id>.log`
- `logs/slurm/completion-qwen35-mp1-<job_id>.json`

Override with:

- `--slurm-log-dir <path>` on `lumi/submit.sh`
- `--slurm-log-dir <path>` on `lumi/euroeval_submit.sh`
- or `SLURM_LOG_DIR=<path>` in the environment

## Monitoring

Use these during execution:

```bash
squeue -j <job_id> -o '%i %T %M %D %R %j'
tail -f logs/slurm/<logfile>.out
```

After completion:

```bash
sacct -j <job_id> --format=JobID,JobName%30,State,Elapsed,ExitCode -P
```

Inspect success checks:

```bash
find logs/evals-logs/<run_label> -maxdepth 1 -name '*.eval'
ls logs/evals-logs/<run_label>/_vllm_server/
```

EuroEval success checks:

```bash
grep -E 'Server ready|EuroEval complete' logs/slurm/euroeval__*.out
ls /pfs/lustrep4/scratch/project_465002183/rasmus/vllm-lumi/overlay_vllm_minimal/euroeval-runs/<job_id>/euroeval_benchmark_results.jsonl
```

`sacct` may show the `.0` step as cancelled during scripted shutdown while the top-level job is still `COMPLETED`; use the top-level job state/exit code as the source of truth.

## Common Failure Modes

- `overlay dir not found`: set `OVERLAY_DIR` explicitly to your existing overlay location.
- `openai/*` fails fast: set both `OPENAI_API_KEY` and `OPENAI_BASE_URL` (or pass `--openai-base-url`).
- vLLM startup fails with low free GPU memory: lower `GPU_MEM`, reduce `TP/PP`, or retry on a cleaner node allocation.
- `view.sh list` shows no runs: new default root is `logs/evals-logs`; for older overlay runs, use `EVAL_LOG_ROOT_HOST=<overlay>/dfm-evals-logs ./lumi/view.sh list`.
- Wrong EuroEval model due inherited env: pass `--euroeval-model` explicitly (recommended for reproducibility).
