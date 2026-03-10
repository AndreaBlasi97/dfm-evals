# Tournament Review

Review target: `dfm_evals/tournament/`

## Findings

1. Resolved: `add-model` and `resume/status` config drift.

   This was fixed by separating pure config resolution from stateful resolution:
   - `resolve_tournament_config()` is pure again and loads the exact config object/file/state target it is given.
   - `resolve_stateful_tournament_config()` is now used by stateful operations that should honor persisted tournament state.
   - `resume_tournament()`, `add_models()`, `tournament_status()`, and `export_rankings()` now follow persisted `config_json` when it exists for the resolved `state_dir`, even if the caller passes the original config file path.

   This keeps ordinary config loading predictable while making resumed tournament operations consistent with prior state mutations such as `add-model`.

   Relevant code:
   - `dfm_evals/tournament/_resolve.py`
   - `dfm_evals/tournament/orchestrator.py`
   - `dfm_evals/tournament/exports.py`
   - `tests/test_tournament_resolve.py`

   Verification:
   - focused regression tests cover pure file resolution, stateful resolution, and `tournament_status()` via config-path target
   - `tests/test_tournament_resolve.py`

2. Resolved: response indexing provenance filter.

   This was fixed by introducing shared tournament provenance helpers and making generation and indexing agree on a stable tournament identity:
   - generation now writes metadata derived from a stable tournament `project_id`
   - the resolved `project_id` prefers persisted state when it exists, so operations such as `add-model` keep writing under the original tournament identity
   - indexing now requires all of the following to match before importing a log:
     - generation task name
     - tournament phase
     - tournament project id

   This prevents foreign eval logs in a shared `completion_log_dir` from being imported as tournament responses.

   Relevant code:
   - `dfm_evals/tournament/_provenance.py`
   - `dfm_evals/tournament/generation.py`
   - `dfm_evals/tournament/indexer.py`
   - `tests/test_tournament_provenance.py`

   Verification:
   - focused regression tests cover stable project-id resolution, generation metadata, and rejection of foreign logs during indexing
   - `tests/test_tournament_provenance.py`

3. Resolved: re-indexing can no longer mutate already judged matches.

   This was fixed by changing the response model from one mutable row per `(model_id, prompt_id)` to immutable response versions plus a separate `current` selection used for scheduling and coverage:
   - new indexed responses get versioned `response_id` values derived from log/sample/content identity
   - matches keep pointing at the exact response version they were scheduled with
   - scheduling and coverage only look at the `current = 1` response for each `(model_id, prompt_id)`
   - older logs can no longer overwrite the current response chosen from newer logs

   Old tournament state is not migrated. Older schema versions are rejected and must be deleted and re-initialized.

   Relevant code:
   - `dfm_evals/tournament/types.py`
   - `dfm_evals/tournament/store.py`
   - `dfm_evals/tournament/indexer.py`
   - `dfm_evals/tournament/scheduler.py`
   - `tests/test_tournament_responses.py`

   Verification:
   - focused regression tests cover immutable response versions, current-response selection, and rejection of old tournament schema
   - `tests/test_tournament_responses.py`

4. Medium: judge model construction is inconsistent with generation.

   Generation honors `INSPECT_EVAL_MODEL_ARGS`; judge resolution does not. `run_judge_batch()` also passes a top-level `model=parsed.judge_model` while the scorer actually uses the `"grader"` role, so this path can hit a different backend than generation and may instantiate an extra unused model/server.

   Relevant code:
   - `dfm_evals/tournament/generation.py:84`
   - `dfm_evals/tournament/generation.py:136`
   - `dfm_evals/tournament/judge_task.py:146`
   - `dfm_evals/tournament/judge_task.py:171`
   - `dfm_evals/tournament/scorer.py:73`

5. Medium: invalid/skip judgments still burn pair and prompt budget.

   Ratings skip `INVALID` outcomes under the default policy, but the scheduler's `max_pair_matches` and `max_prompt_uses_per_pair` are counted from every `matches` row. A flaky judge can therefore exhaust a pair without producing rating signal.

   Relevant code:
   - `dfm_evals/tournament/rating.py:93`
   - `dfm_evals/tournament/scheduler.py:95`
   - `dfm_evals/tournament/scheduler.py:198`

## Other Notes

Pairwise export disagrees with rating behavior when `invalid_policy="count_as_tie"`: ratings convert `INVALID` into `TIE`, but exports keep it as `INVALID`, so the matrix can diverge from standings/tie totals.

Relevant code:
- `dfm_evals/tournament/rating.py:93`
- `dfm_evals/tournament/exports.py:167`
- `dfm_evals/tournament/exports.py:192`

I did not run an end-to-end tournament here. Focused regression coverage now exists for config/state resolution, log provenance filtering, and immutable response storage. The highest-value remaining regression tests would cover the judge execution path and the scheduler/stopping behavior under noisy judgments.
