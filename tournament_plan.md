# Tournament Integration Plan

Date: 2026-03-07

## Decision

Tournament stays a separate workflow from `evals suite`.

We will integrate tournament with the LUMI work through:

- shared launcher conventions
- shared log and artifact locations
- shared Every Eval Ever reporting

We will not integrate tournament by forcing it into the suite abstraction.

## Current Status

- Phase 1 is complete.
- Phase 2 is complete.
- Phase 3 is complete.
- Phase 4 is complete.
- Real LUMI smoke validation is complete.
- Tournament generation and judge model construction now share the same model-arg resolution path.
- Judge execution now uses only the grader role model instead of mixing an ambient `model=...` with `model_roles={"grader": ...}`.
- Regression coverage was added for quoted `INSPECT_EVAL_MODEL_ARGS`, externally managed `vllm/...` endpoints, and the judge eval call shape.
- Tournament now exports into the shared Every Eval Ever schema through `evals eee tournament`.
- Tournament now has a dedicated LUMI submit wrapper, batch entrypoint, and separate launch-map helper for contestant and judge endpoint topology.
- Targeted launcher coverage now exists for runtime config emission, launch-map defaults, relative model paths, and stateful target resolution.
- Tournament now has a dedicated hosted viewer through `evals tournament view`.
- `inspect view` is still useful for raw generation and judge log debugging, but it is not the primary tournament report surface.
- Committed tournament definitions now live under `configs/tournaments/<name>/` instead of the repo root.
- Tournament definitions now use YAML and support `prompt_source` so default tournaments do not need large inline prompt arrays.
- LUMI tournament submission can now target a committed definition and override contestant/judge models at launch time.
- Remaining work is optional polish and more committed tournament definitions.

## Why Keep It Separate

- `lumi/run_suite.sbatch` is built around one target model plus one optional judge server.
- Tournament is multi-contestant, stateful, resumable, and supports `generate`, `run`, `resume`, `add-model`, `status`, and `export`.
- Tournament already has a dedicated CLI path through `evals tournament ...`.
- Folding tournament into `evals suite` would overload the suite model instead of reusing the launcher layer cleanly.

## Desired End State

- Tournament is launched on LUMI with its own wrapper scripts.
- Default committed tournaments live under `configs/tournaments/<name>/` with `tournament.yaml` and `launch-map.yaml`.
- Tournament reuses the same run-label and host-visible log conventions as the existing LUMI tooling.
- Tournament exposes a dedicated hosted viewer for standings, pairwise results, matches, prompts, and model drilldown.
- Tournament writes a single canonical run tree under `logs/evals-logs/<run_label>/`:
  - `config/`
  - `state/`
  - `inspect/`
  - `services/vllm/`
  - `exports/`
  - `traces/`
  - `launcher/`
- Slurm stdout/stderr still live under `logs/slurm/`, with links back from the run tree.
- Tournament exports into Every Eval Ever so the existing reporting flow remains the common interface.
- `inspect view` remains optional batch-log debugging, not the top-level tournament review surface.
- `lumi/results_table.sh` can consume tournament results via EEE instead of adding a third reporting format.

## Non-Goals

- Do not add tournament to `dfm_evals/eval-sets.yaml`.
- Do not force tournament into `evals suite`.
- Do not add raw tournament CSV parsing into `lumi/results_table.sh`.
- Do not make the core tournament config LUMI-specific.

## Architecture Direction

### 1. Keep The Existing CLI Boundary

Use the current `evals tournament ...` commands as the public interface for tournament operations.

This means the integration point is the LUMI orchestration layer, not the top-level CLI routing.

### 2. Add A Dedicated LUMI Tournament Launcher

Add:

- `lumi/tournament_submit.sh`
- `lumi/run_tournament.sbatch`

These should reuse the same operational patterns as the suite launcher:

- overlay and container setup
- explicit vLLM startup and teardown
- LoRA detection
- strict `openai/*` environment handling
- run-label generation
- host-visible log directories
- `services/vllm` raw logs

### 3. Run Tournament In Phases

The LUMI tournament launcher should be phase-based rather than pretending tournament is a single eval-set run.

Phases:

- `generate`
- `run`
- `export`

Suggested behavior:

- generation phase:
  start a contestant server, run `evals tournament generate --models <name>`, stop it, repeat per contestant when needed
- judge/run phase:
  start the judge server, run `evals tournament run` or `evals tournament resume`
- export phase:
  run `evals tournament export`, then `evals eee tournament`

### 4. Keep LUMI Runtime Details Out Of Core Tournament Config

Current tournament config is good for evaluation state and logic, but not for LUMI launch topology.

Do not overload the core config with cluster-specific launch details.

Instead, add a separate LUMI launch-map file keyed by contestant model name for things like:

- served model name
- model path or adapter path
- endpoint mode
- TP/PP/DP
- device selection
- context length
- tool-calling flags

This keeps tournament portable while still letting LUMI runs describe per-contestant serving requirements.

## Required Implementation Work

### Phase 1. Model Resolution Cleanup

Status: complete on 2026-03-07.

Before building the LUMI launcher, fix the model construction gap between generation and judging.

Resolved issue:

- generation honors `INSPECT_EVAL_MODEL_ARGS`
- judge construction does not
- judge execution currently mixes `model=...` and `model_roles={"grader": ...}`

Delivered:

- generation and judge paths resolve models the same way
- externally managed vLLM endpoints can be used consistently
- the judge path does not accidentally instantiate an extra backend
- created grader models are explicitly closed after judge batches
- regression tests cover the shared resolution path

Primary files:

- `dfm_evals/tournament/generation.py`
- `dfm_evals/tournament/judge_task.py`
- `dfm_evals/tournament/_model_args.py`
- `tests/test_tournament_model_resolution.py`

### Phase 2. EEE Export For Tournament

Status: complete on 2026-03-07.

Add `evals eee tournament`.

This exporter should translate tournament outputs into the same shared reporting layer used by suite runs and EuroEval.

Minimum inputs:

- tournament config or state dir
- tournament export artifacts or store state

Minimum outputs:

- EEE JSON written under the standard data root

Delivered:

- tournament uses the same reporting plane as the rest of the repo
- `lumi/results_table.sh` can stay focused on EEE input only
- one EEE record is written per contestant model with aggregate tournament standings metrics
- tournament judge metadata is included in exported evaluation metadata
- CLI routing and seeded export coverage exist for the tournament EEE path

Primary files:

- `dfm_evals/cli.py`
- `dfm_evals/eee_export.py`
- `dfm_evals/tournament/exports.py`
- `tests/test_tournament_eee_export.py`

### Phase 3. Dedicated LUMI Tournament Launcher

Status: complete on 2026-03-07.

Implement:

- `lumi/tournament_submit.sh`
- `lumi/run_tournament.sbatch`

Requirements:

- same overlay and bind handling as the suite launcher
- same run-label conventions
- same log-root conventions
- `services/vllm` logging for contestant and judge servers
- support for generate-only, run-only, resume, and export flows

Target outcome:

- tournament becomes a first-class LUMI run type without changing the suite path

Delivered:

- `lumi/tournament_submit.sh` now submits `all`, `generate`, `run`, `resume`, and `export` tournament phases through Slurm with the same host-visible run/log conventions as the suite launcher
- `lumi/run_tournament.sbatch` now materializes runtime tournament configs, initializes state, launches contestant and judge endpoints explicitly, and runs `evals tournament ...` plus `evals eee tournament`
- contestant and judge raw server logs now land under `logs/evals-logs/<run_label>/services/vllm`
- stateful `run` and `resume` now fail fast when generation is incomplete instead of letting the core tournament path self-generate against the wrong endpoint topology

Primary files:

- `lumi/tournament_submit.sh`
- `lumi/run_tournament.sbatch`
- `tests/test_tournament_launch.py`

### Phase 4. Tournament LUMI Launch Map

Status: complete on 2026-03-07.

Define the extra runtime configuration needed to serve multiple contestants cleanly on LUMI.

Requirements:

- separate from the core tournament config
- keyed by contestant model name
- minimal duplication with the suite launcher

Target outcome:

- contestants with different serving needs can be run reproducibly
- local model paths, LoRA adapters, and remote endpoints can be described cleanly

Delivered:

- `lumi/tournament_launch.py` now validates a separate launch-map file with shared defaults, judge defaults, contestant endpoint specs, and a dedicated judge spec
- launch-map entries support `local_vllm` and `external_openai` modes, including local model paths, served model names, TP/PP/DP/CTX/GPU settings, visible device selection, and tool-calling flags
- relative local model paths are resolved relative to the launch-map file instead of the shell working directory
- helper commands now emit shell-safe tournament target, status, and per-endpoint launch assignments for the batch script

Primary files:

- `lumi/tournament_launch.py`
- `tests/test_tournament_launch.py`

### Phase 5. Docs And Workflow Validation

Document the tournament path in:

- `README.md`
- `lumi/README.md`

Validation should cover:

- generate on LUMI
- run or resume on LUMI
- export to tournament artifacts
- export to EEE
- the hosted tournament viewer against a real tournament state dir

## Acceptance Criteria

- Tournament is still invoked as `evals tournament ...`.
- LUMI users get a dedicated tournament launcher instead of using suite wrappers.
- Tournament logs are visible in the same host paths as other LUMI runs.
- Tournament server logs are captured under `services/vllm`.
- Tournament exports to EEE.
- Tournament exposes a hosted viewer through `evals tournament view`.
- Existing reporting tools do not need a tournament-specific raw-file mode.

## Immediate Next Tasks

1. Add any remaining tournament notes to the top-level `README.md` if we want a non-LUMI entrypoint mention there.
2. Decide whether we want a dedicated LUMI shell wrapper for `evals tournament view`, or whether direct CLI invocation is enough.
3. Add any minimal viewer usage notes to `lumi/README.md` and confirm `lumi/results_table.sh` behavior against produced tournament EEE output.
