# dfm-evals

`dfm-evals` is a small extension package around `inspect_ai`.

It provides:

- local tasks and scorers
- packaged eval suites
- tournament runs and a viewer for tournament results
- Every Eval Ever export helpers
- a built-in Prime sandbox provider
- LUMI-specific launch helpers under [`lumi/`](lumi/)

The package is exposed to Inspect through the `dfm_evals` registry entry point,
so local tasks can be run as `dfm_evals/<task-name>`.

## What Is In This Repo

Main areas:

- [`dfm_evals/`](dfm_evals/) contains tasks, scorers, sandboxes, the registry, the CLI, and tournament code.
- [`dfm_evals/eval-sets.yaml`](dfm_evals/eval-sets.yaml) defines packaged suites.
- [`configs/`](configs/) contains example tournament and sandbox configs.
- [`tests/`](tests/) contains the test suite.
- [`lumi/`](lumi/) contains Slurm and LUMI helpers.

Current local tasks include:

- `dfm_evals/multi_wiki_qa`
- `dfm_evals/gec_dala`
- `dfm_evals/bfcl-v1`
- `dfm_evals/bfcl-v1-da`
- `dfm_evals/ifeval-da`
- `dfm_evals/piqa`

## Install

Base install:

```bash
uv sync
```

Development tools:

```bash
uv sync --group dev
```

Optional extras:

- `uv sync --extra comet` for COMET-related dependencies
- `uv sync --extra harbor` for `inspect-harbor` tasks on Python 3.12+
- `uv sync --extra sandboxes` for compose-aware sandbox providers on Python 3.12+

## Common Commands

List available tasks:

```bash
uv run evals tasks
```

Run a local task:

```bash
uv run evals run dfm_evals/multi_wiki_qa --model openai/gpt-5-mini
```

Run a packaged suite:

```bash
uv run evals suite fundamentals \
  --target-model openai/gpt-5-mini \
  --judge-model openai/gpt-5-mini
```

Inspect tournament commands:

```bash
uv run evals tournament --help
```

View a completed tournament:

```bash
uv run evals tournament view logs/evals-logs/<tournament_run_label>/state \
  --host 127.0.0.1 \
  --port 7576
```

Export Inspect logs to Every Eval Ever format:

```bash
uv run evals eee inspect \
  --log-path logs/evals-logs/<run_label> \
  --output-dir out/eee/data
```

Run lint and tests:

```bash
uv run --group dev ruff check dfm_evals tests
uv run pytest
```

## Suites And Configs

Packaged suites live in [`dfm_evals/eval-sets.yaml`](dfm_evals/eval-sets.yaml).
They can reference both upstream Inspect tasks and local `dfm_evals/...` tasks.

The suite runner supports these placeholders:

- `{{target_model}}`
- `{{target_base_url}}`
- `{{judge_model}}`
- `{{judge_base_url}}`

Example configs for tournaments and sandboxes live under [`configs/`](configs/).

## Sandboxes

The built-in sandbox provider is `prime`.

Example:

```bash
PRIME_API_KEY=... \
uv run evals run dfm_evals/multi_wiki_qa \
  --model openai/gpt-5-mini \
  --sandbox prime
```

By default, the Prime provider looks for
[`configs/sandboxes/prime-sandbox.yaml`](configs/sandboxes/prime-sandbox.yaml).
You can also pass an explicit config path with
`--sandbox "prime:/path/to/prime-sandbox.yaml"`.

For LUMI-specific workflows, see [`lumi/README.md`](lumi/README.md).

## Adding New Components

### Add A Task

1. Create a module under [`dfm_evals/tasks/`](dfm_evals/tasks/).
2. Define the task with `@task(name="your-task-name")`.
3. Return an `inspect_ai.Task`.
4. If the task should be part of the public local registry, add it to [`dfm_evals/_exports.py`](dfm_evals/_exports.py) in `REGISTRY_EXPORTS`.
5. Add tests under [`tests/`](tests/).

Minimal shape:

```python
from inspect_ai import Task, task


@task(name="my-task")
def my_task() -> Task:
    ...
```

After that, the task becomes available as `dfm_evals/my-task` because the
package entry point already points to [`dfm_evals/_registry.py`](dfm_evals/_registry.py).

### Add A Scorer

1. Create a module under [`dfm_evals/scorers/`](dfm_evals/scorers/).
2. Define the scorer with `@scorer(...)`.
3. Export it from [`dfm_evals/_exports.py`](dfm_evals/_exports.py) if it should be reusable outside a single task.
4. Add focused tests under [`tests/`](tests/).

Minimal shape:

```python
from inspect_ai.scorer import Scorer, scorer


@scorer(...)
def my_scorer() -> Scorer:
    ...
```

If the scorer is only used inside one task module, it does not need to be added
to the public exports.

### Add A Suite

1. Add a new entry under `sets:` in [`dfm_evals/eval-sets.yaml`](dfm_evals/eval-sets.yaml).
2. Reference tasks as either plain task ids or `{name, args}` objects.
3. Put suite-wide CLI arguments in the suite-level `args` list.

Then run it with:

```bash
uv run evals suite <suite-name> --target-model ... --judge-model ...
```

### Add A Sandbox Provider

1. Create a module under [`dfm_evals/sandboxes/`](dfm_evals/sandboxes/).
2. Register the provider with `@sandboxenv(name="your-sandbox")`.
3. Ensure the module is imported by the registry. In this repo that means adding
   it to [`dfm_evals/_exports.py`](dfm_evals/_exports.py), usually via `REGISTRY_ONLY_MODULES`
   or `SANDBOX_EXPORTS`.
4. Add tests for config loading and registration behavior.

### Good Default Workflow

When adding any new component:

1. Implement the module.
2. Register it in [`dfm_evals/_exports.py`](dfm_evals/_exports.py) if needed.
3. Add or update tests.
4. Run `uv run evals tasks` to confirm the registry still loads.
5. Run the smallest relevant pytest subset before pushing.

## Development Notes

- The registry import path is centered on [`dfm_evals/_exports.py`](dfm_evals/_exports.py) and [`dfm_evals/_registry.py`](dfm_evals/_registry.py).
- Local task ids are exposed through Inspect as `dfm_evals/<task-name>`.
- Keep task-specific logic close to the task unless it is clearly reusable.
- Prefer small tests that exercise record normalization, scorer behavior, and registry loading.
