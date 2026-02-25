# dfm-evals

Minimal `inspect_ai` companion package with:

- `evals` CLI wrapper (forwarding to `inspect`)
- local tasks: `dfm_evals/multi_wiki_qa`, `dfm_evals/bfcl-v1`, `dfm_evals/bfcl-v1-da`
- LUMI operational helpers under `lumi/`

## LUMI

For container/overlay/sbatch tooling on LUMI, see [`lumi/README.md`](lumi/README.md).
This includes both `dfm-evals` suite workflows and the dedicated 2-node EuroEval launcher.

Quick notes:

- Inspect smoke:
  `OVERLAY_DIR=/path/to/overlay_vllm_minimal ./lumi/submit.sh --limit 1 --max-connections 2 --run-label inspect_smoke`
- EuroEval (example: Danish + longer wall time):
  `OVERLAY_DIR=/path/to/overlay_vllm_minimal ./lumi/euroeval_submit.sh --model Qwen/Qwen3.5-397B-A17B --served-model-name Qwen/Qwen3.5-397B-A17B --euroeval-model Qwen/Qwen3.5-397B-A17B --languages da --iterations 10 --time 12:00:00`
- Job logs:
  `logs/slurm/`
- Inspect eval logs:
  `logs/evals-logs/<run_label>/`
- EuroEval results:
  `<OVERLAY_DIR>/euroeval-runs/<job_id>/euroeval_benchmark_results.jsonl`
- For `openai/*` models, set both:
  `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

## Install

```bash
uv sync
```

## CLI

```bash
uv run evals tasks
uv run evals run dfm_evals/multi_wiki_qa --model openai/gpt-5-mini
uv run evals run dfm_evals/bfcl-v1 --model openai/gpt-5-mini
```

Suites default to the packaged file at `dfm_evals/eval-sets.yaml`.
Use `--file <path>` for custom suite files.
Packaged suites are provider-agnostic.

```bash
uv run evals suite fundamentals \
  --target-model openai/gpt-5-mini \
  --judge-model openai/gpt-5-mini
```

Suites use placeholders in `args`, for example:

- `--model`, `"{{target_model}}"`
- `-T`, `grader_model={{judge_model}}`

Supported placeholders:

- `{{target_model}}`
- `{{target_base_url}}`
- `{{judge_model}}`
- `{{judge_base_url}}`

## Lint

```bash
uv run --group dev ruff check dfm_evals
```

## Task layout

Local tasks live under `dfm_evals/tasks/`.

Current task:

- `dfm_evals/tasks/multi_wiki_qa.py`
- `dfm_evals/tasks/bfcl/`
