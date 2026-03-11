import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.scorer import Score
from pydantic import BaseModel, Field

from . import _run_state as run_state
from ._resolve import resolve_stateful_tournament_config
from .config import TournamentConfig, load_tournament_config
from .generation import run_generation
from .indexer import ResponseIndexReport, index_generation_responses
from .judge_task import JudgeMatch, run_judge_batch
from .rating import MatchOutcome, ModelStanding, apply_outcomes, summarize_ratings
from .scheduler import schedule_match_batch
from .scorer import canonicalize_side_decision, reconcile_side_swap
from .stopping import check_convergence, check_hard_stops_for_config
from .store import TournamentStore, initialize_tournament_store
from .types import Decision, deterministic_id

RUN_STATUS_RUNNING = "running"
RUN_STATUS_STOPPED = "stopped"
RUN_STATUS_COMPLETED = "completed"


class _ParsedJudgedSample(TypedDict):
    match_id: str
    side: str
    decision: str
    judge_model: str
    explanation: str | None
    raw_completion: str | None


class TournamentStatus(BaseModel):
    """Current state of a tournament run."""

    project_id: str | None
    run_status: str
    next_round_index: int
    pending_batch_id: str | None
    stable_batches: int
    converged: bool
    stop_reasons: list[str]
    total_models: int
    total_prompts: int
    response_count: int
    expected_responses: int
    missing_responses: int
    total_matches: int
    scheduled_matches: int
    judged_matches: int
    rated_matches: int
    min_adjacent_probability: float | None = None
    min_adjacent_margin: float | None = None
    standings: list[ModelStanding] = Field(default_factory=list)


class TournamentRunResult(BaseModel):
    """Result of running or resuming the tournament loop."""

    batches_completed: int
    matches_scheduled: int
    outcomes_processed: int
    outcomes_skipped: int
    status: TournamentStatus


class AddModelsResult(BaseModel):
    """Result of adding new models and resuming tournament execution."""

    requested_models: list[str]
    added_models: list[str]
    already_present_models: list[str]
    generated_models: list[str]
    run: TournamentRunResult

    @property
    def status(self) -> TournamentStatus:
        return self.run.status


class RegisterModelsResult(BaseModel):
    """Result of registering new models without generating or judging."""

    requested_models: list[str]
    added_models: list[str]
    already_present_models: list[str]
    status: TournamentStatus


class UpdateConfigResult(BaseModel):
    """Result of updating persisted tournament config for an existing run."""

    old_max_total_matches: int
    new_max_total_matches: int
    status: TournamentStatus


def run_tournament(
    config: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    max_batches: int | None = None,
) -> TournamentRunResult:
    """Run a tournament from config."""
    parsed = load_tournament_config(config)
    initialize_tournament_store(parsed)
    state_dir = _require_state_dir(parsed)

    with TournamentStore(state_dir) as store:
        store.initialize_from_config(parsed)
        _set_run_state_defaults(store, parsed)
        _ensure_response_coverage(parsed, store, force_regenerate=parsed.regenerate_completions)
        return _run_loop(parsed, store, max_batches=max_batches)


def resume_tournament(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    max_batches: int | None = None,
) -> TournamentRunResult:
    """Resume a tournament from config or existing state directory."""
    parsed = resolve_stateful_tournament_config(config_or_state)
    state_dir = _require_state_dir(parsed)
    with TournamentStore(state_dir) as store:
        store.initialize_from_config(parsed)
        _set_run_state_defaults(store, parsed)
        _ensure_response_coverage(parsed, store, force_regenerate=False)
        return _run_loop(parsed, store, max_batches=max_batches)


def add_models(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    models: Sequence[str],
    max_batches: int | None = None,
) -> AddModelsResult:
    """Add models to an existing tournament and continue the run loop."""
    requested_models, _parsed, updated_config, already_present, new_models = _register_models(
        config_or_state,
        models=models,
    )
    state_dir = _require_state_dir(updated_config)
    with TournamentStore(state_dir) as store:
        store.initialize_from_config(updated_config)
        _set_run_state_defaults(store, updated_config)

        if len(new_models) == 0:
            status = _status_from_store(updated_config, store)
            return AddModelsResult(
                requested_models=requested_models,
                added_models=[],
                already_present_models=already_present,
                generated_models=[],
                run=TournamentRunResult(
                    batches_completed=0,
                    matches_scheduled=0,
                    outcomes_processed=0,
                    outcomes_skipped=0,
                    status=status,
                ),
            )

        coverage_before = index_generation_responses(updated_config, store=store)
        generated_models = sorted(
            model_name
            for model_name, prompt_ids in coverage_before.missing_by_model.items()
            if len(prompt_ids) > 0
        )
        _ensure_response_coverage(updated_config, store, force_regenerate=False)

        run = _run_loop(updated_config, store, max_batches=max_batches)
        return AddModelsResult(
            requested_models=requested_models,
            added_models=[name for name in requested_models if name not in already_present],
            already_present_models=already_present,
            generated_models=generated_models,
            run=run,
        )


def register_models(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    models: Sequence[str],
) -> RegisterModelsResult:
    """Register new models in persisted tournament state without generation/judging."""
    requested_models, _parsed, updated_config, already_present, new_models = _register_models(
        config_or_state,
        models=models,
    )
    state_dir = _require_state_dir(updated_config)
    with TournamentStore(state_dir) as store:
        store.initialize_from_config(updated_config)
        _set_run_state_defaults(store, updated_config)
        status = _status_from_store(updated_config, store)

        if len(new_models) > 0:
            index_generation_responses(updated_config, store=store)
            status = _status_from_store(updated_config, store)

    return RegisterModelsResult(
        requested_models=requested_models,
        added_models=new_models,
        already_present_models=already_present,
        status=status,
    )


def tournament_status(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
) -> TournamentStatus:
    """Report current tournament status."""
    parsed = resolve_stateful_tournament_config(config_or_state)
    state_dir = _require_state_dir(parsed)
    with TournamentStore(state_dir) as store:
        return _status_from_store(parsed, store)


def update_tournament_config(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    max_total_matches: int | None = None,
) -> UpdateConfigResult:
    """Update persisted tournament config for an existing run."""
    if max_total_matches is None:
        raise ValueError("At least one config field update is required.")
    if max_total_matches <= 0:
        raise ValueError("max_total_matches must be greater than 0.")

    parsed = resolve_stateful_tournament_config(config_or_state)
    state_dir = _require_state_dir(parsed)
    with TournamentStore(state_dir) as store:
        persisted_config = _stored_config(store)
        base_config = persisted_config if persisted_config is not None else parsed
        store.initialize_from_config(base_config)
        _set_run_state_defaults(store, base_config)

        updated_config = base_config.model_copy(
            update={"max_total_matches": max_total_matches}
        )
        store.initialize_from_config(updated_config)

        total_matches = store.table_count("matches")
        state = run_state.load(store, run_status_default=RUN_STATUS_RUNNING)
        updated_stop_reasons = [
            reason
            for reason in state.stop_reasons
            if not (
                reason == "max_total_matches_reached"
                and total_matches < updated_config.max_total_matches
            )
        ]

        with store.transaction():
            run_state.set_config_json(store, updated_config.model_dump_json(), commit=False)
            if updated_stop_reasons != state.stop_reasons:
                run_state.set_stop_reasons(store, updated_stop_reasons, commit=False)
                if len(updated_stop_reasons) == 0 and state.run_status != RUN_STATUS_RUNNING:
                    run_state.set_run_status(store, RUN_STATUS_RUNNING, commit=False)

        status = _status_from_store(updated_config, store)

    return UpdateConfigResult(
        old_max_total_matches=base_config.max_total_matches,
        new_max_total_matches=updated_config.max_total_matches,
        status=status,
    )


def _register_models(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    models: Sequence[str],
) -> tuple[list[str], TournamentConfig, TournamentConfig, list[str], list[str]]:
    requested_models = _normalize_model_names(models)
    if len(requested_models) == 0:
        raise ValueError("At least one non-empty model name is required.")

    parsed = resolve_stateful_tournament_config(config_or_state)
    state_dir = _require_state_dir(parsed)
    with TournamentStore(state_dir) as store:
        persisted_config = _stored_config(store)
        base_config = persisted_config if persisted_config is not None else parsed
        store.initialize_from_config(base_config)
        _set_run_state_defaults(store, base_config)

        existing_models = set(base_config.contestant_models)
        already_present = [name for name in requested_models if name in existing_models]
        new_models = [name for name in requested_models if name not in existing_models]
        scaled_max_total_matches = _scaled_max_total_matches(
            current_max=base_config.max_total_matches,
            existing_model_count=len(base_config.contestant_models),
            added_model_count=len(new_models),
        )
        updated_config = (
            base_config.model_copy(
                update={
                    "contestant_models": [
                        *base_config.contestant_models,
                        *new_models,
                    ],
                    "max_total_matches": scaled_max_total_matches,
                }
            )
            if len(new_models) > 0
            else base_config
        )
        store.initialize_from_config(updated_config)

        if len(new_models) > 0:
            with store.transaction():
                run_state.set_config_json(store, updated_config.model_dump_json(), commit=False)
                run_state.set_run_status(store, RUN_STATUS_RUNNING, commit=False)
                run_state.set_stable_batches(store, 0, commit=False)
                run_state.set_converged(store, False, commit=False)
                run_state.set_stop_reasons(store, [], commit=False)

    return requested_models, parsed, updated_config, already_present, new_models


def _stored_config(store: TournamentStore) -> TournamentConfig | None:
    config_json = store.get_run_state(run_state.KEY_CONFIG_JSON)
    if config_json is None or config_json.strip() == "":
        return None
    try:
        return TournamentConfig.model_validate_json(config_json)
    except Exception as ex:
        raise ValueError(
            "Persisted tournament config in run_state[config_json] is invalid."
        ) from ex


def _normalize_model_names(models: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for model_name in models:
        value = model_name.strip()
        if value == "" or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _scaled_max_total_matches(
    *,
    current_max: int,
    existing_model_count: int,
    added_model_count: int,
) -> int:
    if added_model_count <= 0:
        return current_max

    old_pair_count = existing_model_count * (existing_model_count - 1) // 2
    new_model_count = existing_model_count + added_model_count
    new_pair_count = new_model_count * (new_model_count - 1) // 2
    if old_pair_count <= 0 or new_pair_count <= old_pair_count:
        return current_max

    return (current_max * new_pair_count + old_pair_count - 1) // old_pair_count


def _set_run_state_defaults(store: TournamentStore, config: TournamentConfig) -> None:
    run_state.ensure_defaults(
        store,
        config_json=config.model_dump_json(),
        run_status=RUN_STATUS_RUNNING,
    )


def _ensure_response_coverage(
    config: TournamentConfig,
    store: TournamentStore,
    *,
    force_regenerate: bool,
) -> ResponseIndexReport:
    report = index_generation_responses(config, store=store)
    if force_regenerate:
        run_generation(config, models=config.contestant_models)
        report = index_generation_responses(config, store=store)
    elif report.missing_count > 0:
        missing_models = sorted(
            model_name
            for model_name, missing_prompt_ids in report.missing_by_model.items()
            if len(missing_prompt_ids) > 0
        )
        if len(missing_models) > 0:
            run_generation(config, models=missing_models)
            report = index_generation_responses(config, store=store)

    if report.missing_count > 0:
        raise RuntimeError(
            "Missing prompt/model responses after generation: "
            + json.dumps(report.missing_by_model, sort_keys=True)
        )
    return report


def _run_loop(
    config: TournamentConfig,
    store: TournamentStore,
    *,
    max_batches: int | None,
) -> TournamentRunResult:
    batches_completed = 0
    matches_scheduled = 0
    outcomes_processed = 0
    outcomes_skipped = 0

    while True:
        state = run_state.load(store, run_status_default=RUN_STATUS_RUNNING)
        pending_batch_id = state.pending_batch_id
        pending_round_index = state.pending_round_index

        if pending_batch_id is None:
            total_matches = store.table_count("matches")
            if total_matches >= config.max_total_matches:
                _record_stop(
                    store,
                    run_status=RUN_STATUS_COMPLETED,
                    reasons=["max_total_matches_reached"],
                )
                break

            round_index = state.next_round_index
            batch_id = f"batch-{round_index:06d}"
            schedule_result = schedule_match_batch(
                config,
                store,
                batch_id=batch_id,
                round_index=round_index,
                seed=config.seed,
                persist=True,
            )
            if schedule_result.exhausted:
                _record_stop(
                    store,
                    run_status=RUN_STATUS_COMPLETED,
                    reasons=["no_eligible_pairs"],
                )
                break

            with store.transaction():
                run_state.set_pending_batch(
                    store,
                    batch_id=batch_id,
                    round_index=round_index,
                    commit=False,
                )
                run_state.set_run_status(store, RUN_STATUS_RUNNING, commit=False)
            pending_batch_id = batch_id
            pending_round_index = round_index
            matches_scheduled += len(schedule_result.scheduled)

        judged_now = _judge_pending_matches(config, store, pending_batch_id)
        del judged_now

        applied = _apply_pending_batch_outcomes(
            config,
            store,
            batch_id=pending_batch_id,
            round_index=pending_round_index,
        )
        if applied is None:
            _clear_pending_batch(store)
            continue

        outcomes_processed += applied["processed"]
        outcomes_skipped += applied["skipped"]

        if bool(applied.get("blocked", False)):
            break

        batches_completed += 1

        if max_batches is not None and batches_completed >= max_batches:
            break

        if applied["converged"]:
            _record_stop(
                store,
                run_status=RUN_STATUS_COMPLETED,
                reasons=["converged"],
            )
            break

        total_matches_after = store.table_count("matches")
        hard_stop = check_hard_stops_for_config(
            config,
            total_matches=total_matches_after,
            no_eligible_pairs=False,
        )
        if hard_stop.should_stop:
            _record_stop(
                store,
                run_status=RUN_STATUS_COMPLETED,
                reasons=hard_stop.reasons,
            )
            break

    status = _status_from_store(config, store)
    return TournamentRunResult(
        batches_completed=batches_completed,
        matches_scheduled=matches_scheduled,
        outcomes_processed=outcomes_processed,
        outcomes_skipped=outcomes_skipped,
        status=status,
    )


def _judge_pending_matches(
    config: TournamentConfig,
    store: TournamentStore,
    batch_id: str,
) -> int:
    rows = store.load_batch_matches(batch_id, statuses=["scheduled"])
    if len(rows) == 0:
        return 0

    matches = [
        JudgeMatch(
            match_id=str(row["match_id"]),
            prompt_id=str(row["prompt_id"]),
            prompt=str(row["prompt_text"]),
            model_a=str(row["model_a_name"]),
            model_b=str(row["model_b_name"]),
            model_a_id=str(row["model_a_id"]),
            model_b_id=str(row["model_b_id"]),
            response_a=str(row["response_a_text"]),
            response_b=str(row["response_b_text"]),
        )
        for row in rows
    ]

    side_count = 2 if config.side_swap else 1
    matches_per_judge_run = max(1, config.judge_max_samples // side_count)
    for chunk in _chunked(matches, size=matches_per_judge_run):
        judge_result = run_judge_batch(
            config,
            chunk,
            log_dir=config.judge_log_dir,
        )
        _ingest_judge_logs(config, store, judge_result.logs)

    with store.transaction():
        store.set_batch_match_status(
            batch_id=batch_id,
            status="judged",
            from_statuses=["scheduled"],
            commit=False,
        )
    return len(matches)


def _ingest_judge_logs(
    config: TournamentConfig,
    store: TournamentStore,
    logs: Sequence[EvalLog],
) -> None:
    with store.transaction():
        for log in logs:
            source_log = _relative_log_name(config.judge_log_dir, log.location)
            for sample in (log.samples or []):
                parsed = _parse_judged_sample(sample)
                if parsed is None:
                    continue
                judgment_id = deterministic_id(
                    "judgment",
                    parsed["match_id"],
                    parsed["side"],
                    length=20,
                )
                store.upsert_judgment(
                    judgment_id=judgment_id,
                    match_id=parsed["match_id"],
                    side=parsed["side"],
                    decision=parsed["decision"],
                    judge_model=parsed["judge_model"],
                    explanation=parsed["explanation"],
                    raw_completion=parsed["raw_completion"],
                    source_log=source_log,
                    sample_uuid=sample.uuid,
                    commit=False,
                )


def _parse_judged_sample(sample: EvalSample) -> _ParsedJudgedSample | None:
    metadata = sample.metadata if sample.metadata is not None else {}
    score = _extract_judge_score(sample.scores)
    if score is None:
        return None

    match_id = metadata.get("match_id")
    if not isinstance(match_id, str) or match_id.strip() == "":
        return None

    side = metadata.get("side")
    if not isinstance(side, str) or side not in ("ab", "ba"):
        side = "ab"

    score_metadata = score.metadata if isinstance(score.metadata, dict) else {}
    judge_model = score_metadata.get("judge_model")
    decision = _as_decision(score.value)
    return {
        "match_id": match_id,
        "side": side,
        "decision": decision,
        "judge_model": str(judge_model) if judge_model is not None else "",
        "explanation": score.explanation,
        "raw_completion": score.explanation,
    }


def _extract_judge_score(scores: dict[str, Score] | None) -> Score | None:
    if scores is None or len(scores) == 0:
        return None
    if "pairwise_judge" in scores:
        return scores["pairwise_judge"]
    return next(iter(scores.values()))


def _apply_pending_batch_outcomes(
    config: TournamentConfig,
    store: TournamentStore,
    *,
    batch_id: str,
    round_index: int,
) -> dict[str, int | bool] | None:
    rows_to_rate = store.load_batch_matches(batch_id, statuses=["judged", "scheduled"])
    if len(rows_to_rate) == 0:
        return None

    outcomes = _canonical_outcomes_for_batch(config, store, batch_id)
    active_ratings = store.load_active_model_ratings()
    active_model_ids = set(active_ratings.keys())
    active_outcomes = [
        outcome
        for outcome in outcomes
        if outcome.model_a in active_model_ids and outcome.model_b in active_model_ids
    ]
    inactive_outcomes_skipped = len(outcomes) - len(active_outcomes)
    stable_batches = run_state.load(
        store,
        run_status_default=RUN_STATUS_RUNNING,
    ).stable_batches
    rating_result = apply_outcomes(
        ratings=active_ratings,
        outcomes=active_outcomes,
        params=config.rating_params,
        conservative_k=config.conservative_k,
        elo_scale=config.elo_scale,
        invalid_policy=config.invalid_policy,
    )
    total_skipped = rating_result.skipped_outcomes + inactive_outcomes_skipped

    if rating_result.processed_outcomes == 0:
        # Preserve this batch for a retry instead of silently marking everything rated.
        with store.transaction():
            store.set_batch_match_status(
                batch_id=batch_id,
                status="scheduled",
                from_statuses=["judged"],
                commit=False,
            )
            run_state.set_run_status(store, RUN_STATUS_STOPPED, commit=False)
            run_state.set_stop_reasons(store, ["no_valid_judgments"], commit=False)
        return {
            "processed": rating_result.processed_outcomes,
            "skipped": total_skipped,
            "converged": False,
            "blocked": True,
        }

    with store.transaction():
        for rating in rating_result.ratings.values():
            store.upsert_model_rating(rating, commit=False)

        step_id = store.next_ratings_history_step()
        store.append_ratings_history(step_id, rating_result.ratings, commit=False)
        store.set_batch_match_status(
            batch_id=batch_id,
            status="rated",
            from_statuses=["scheduled", "judged"],
            commit=False,
        )

        convergence = check_convergence(
            rating_result.ratings,
            rating_params=config.rating_params,
            p_stop=config.p_stop,
            epsilon=config.epsilon,
            conservative_k=config.conservative_k,
            n_stable_batches=config.n_stable_batches,
            stable_batches=stable_batches,
            elo_scale=config.elo_scale,
        )
        run_state.set_stable_batches(store, convergence.stable_batches, commit=False)
        run_state.set_converged(store, convergence.converged, commit=False)
        run_state.clear_pending_batch(store, commit=False)
        run_state.set_next_round_index(store, round_index + 1, commit=False)
        run_state.set_last_batch_id(store, batch_id, commit=False)
        run_state.set_last_min_probability(store, convergence.min_probability, commit=False)
        run_state.set_last_min_margin(store, convergence.min_margin, commit=False)
        run_state.set_run_status(store, RUN_STATUS_RUNNING, commit=False)

    return {
        "processed": rating_result.processed_outcomes,
        "skipped": total_skipped,
        "converged": convergence.converged,
        "blocked": False,
    }


def _canonical_outcomes_for_batch(
    config: TournamentConfig,
    store: TournamentStore,
    batch_id: str,
) -> list[MatchOutcome]:
    rows = store.load_batch_judgments(batch_id)
    decisions_by_match: dict[str, dict[str, Decision]] = defaultdict(dict)
    model_pair_by_match: dict[str, tuple[str, str]] = {}

    for row in rows:
        match = str(row["match_id"])
        model_pair_by_match[match] = (str(row["model_a"]), str(row["model_b"]))

        side = row["side"]
        decision = row["decision"]
        if side is None or decision is None:
            continue
        side_value = str(side)
        if side_value not in ("ab", "ba"):
            continue
        decisions_by_match[match][side_value] = _as_decision(str(decision))

    outcomes: list[MatchOutcome] = []
    for match in sorted(model_pair_by_match):
        model_a, model_b = model_pair_by_match[match]
        side_decisions = decisions_by_match.get(match, {})
        ab = side_decisions.get("ab")
        ba = side_decisions.get("ba")
        if config.side_swap:
            if ab is not None and ba is not None:
                decision = reconcile_side_swap(ab, ba, invalid_policy=config.invalid_policy)
            else:
                decision = "INVALID"
        else:
            if ab is not None:
                decision = canonicalize_side_decision(ab, "ab")
            elif ba is not None:
                decision = canonicalize_side_decision(ba, "ba")
            else:
                decision = "INVALID"

        outcomes.append(
            MatchOutcome(
                model_a=model_a,
                model_b=model_b,
                decision=decision,
            )
        )
    return outcomes


def _status_from_store(config: TournamentConfig, store: TournamentStore) -> TournamentStatus:
    state = run_state.load(store, run_status_default=RUN_STATUS_RUNNING)
    ratings = store.load_active_model_ratings()
    names_by_id = store.active_model_names_by_id()
    standings = summarize_ratings(
        ratings,
        params=config.rating_params,
        conservative_k=config.conservative_k,
        elo_scale=config.elo_scale,
    )
    standings = [
        standing.model_copy(
            update={"model_name": names_by_id.get(standing.model_id, standing.model_id)}
        )
        for standing in standings
    ]
    expected_responses = len(config.contestant_models) * len(config.prompts)
    response_count = store.active_response_count()
    missing = max(0, expected_responses - response_count)
    status_counts = store.match_status_counts_for_active_models()
    total_matches = sum(status_counts.values())

    return TournamentStatus(
        project_id=state.project_id,
        run_status=state.run_status,
        next_round_index=state.next_round_index,
        pending_batch_id=state.pending_batch_id,
        stable_batches=state.stable_batches,
        converged=state.converged,
        stop_reasons=state.stop_reasons,
        total_models=len(config.contestant_models),
        total_prompts=len(config.prompts),
        response_count=response_count,
        expected_responses=expected_responses,
        missing_responses=missing,
        total_matches=total_matches,
        scheduled_matches=status_counts.get("scheduled", 0),
        judged_matches=status_counts.get("judged", 0),
        rated_matches=status_counts.get("rated", 0),
        min_adjacent_probability=state.last_min_probability,
        min_adjacent_margin=state.last_min_margin,
        standings=standings,
    )


def _record_stop(store: TournamentStore, *, run_status: str, reasons: list[str]) -> None:
    with store.transaction():
        run_state.set_run_status(store, run_status, commit=False)
        run_state.set_stop_reasons(store, reasons, commit=False)
        if "converged" in reasons:
            run_state.set_converged(store, True, commit=False)


def _clear_pending_batch(store: TournamentStore) -> None:
    with store.transaction():
        run_state.clear_pending_batch(store, commit=False)


def _as_decision(value: Any) -> Decision:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in ("A", "B", "TIE", "INVALID"):
            return normalized  # type: ignore[return-value]
    return "INVALID"


def _relative_log_name(base_dir: Path, log_name: str | None) -> str | None:
    if log_name is None or "://" in log_name:
        return log_name
    try:
        return Path(log_name).resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return log_name


def _chunked(items: Sequence[JudgeMatch], *, size: int) -> Sequence[Sequence[JudgeMatch]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def _require_state_dir(config: TournamentConfig) -> Path:
    return config.state_dir
