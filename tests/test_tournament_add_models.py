from pathlib import Path

from dfm_evals.tournament import _run_state as run_state
from dfm_evals.tournament.config import TournamentConfig, TournamentPrompt
from dfm_evals.tournament.orchestrator import register_models, update_tournament_config
from dfm_evals.tournament.store import TournamentStore


def _build_config(tmp_path: Path) -> TournamentConfig:
    return TournamentConfig(
        run_dir=tmp_path / "logs" / "evals-logs" / "tournament__demo__job-1",
        project_id="demo-tournament",
        contestant_models=["vllm/org/model-a", "vllm/org/model-b"],
        prompts=[TournamentPrompt(id="prompt-1", text="Write a short paragraph.")],
        max_total_matches=120,
        judge_model="openai/judge-model",
        judge_prompt_template="Judge:\n{prompt}\nA:{response_a}\nB:{response_b}",
    )


def test_register_models_updates_persisted_config_and_resets_run_state(
    tmp_path: Path,
) -> None:
    config = _build_config(tmp_path)

    with TournamentStore(config.state_dir) as store:
        store.initialize_from_config(config)
        run_state.ensure_defaults(
            store,
            config_json=config.model_dump_json(),
            run_status="completed",
        )
        run_state.set_run_status(store, "completed")
        run_state.set_stable_batches(store, 5)
        run_state.set_converged(store, True)
        run_state.set_stop_reasons(store, ["converged"])

    result = register_models(
        config.run_dir,
        models=[
            "vllm/org/model-c",
            "vllm/org/model-a",
            "vllm/org/model-c",
        ],
    )

    assert result.requested_models == ["vllm/org/model-c", "vllm/org/model-a"]
    assert result.added_models == ["vllm/org/model-c"]
    assert result.already_present_models == ["vllm/org/model-a"]
    assert result.status.run_status == "running"
    assert result.status.converged is False
    assert result.status.stable_batches == 0
    assert result.status.stop_reasons == []
    assert result.status.total_models == 3
    assert result.status.expected_responses == 3
    assert result.status.response_count == 0
    assert result.status.missing_responses == 3

    with TournamentStore(config.state_dir) as store:
        persisted = TournamentConfig.model_validate_json(
            store.get_run_state("config_json") or ""
        )

    assert persisted.contestant_models == [
        "vllm/org/model-a",
        "vllm/org/model-b",
        "vllm/org/model-c",
    ]
    assert persisted.max_total_matches == 360


def test_update_tournament_config_can_raise_max_total_matches_and_clear_stop_reason(
    tmp_path: Path,
) -> None:
    config = _build_config(tmp_path)

    with TournamentStore(config.state_dir) as store:
        store.initialize_from_config(config)
        run_state.ensure_defaults(
            store,
            config_json=config.model_dump_json(),
            run_status="completed",
        )
        run_state.set_run_status(store, "completed")
        run_state.set_stop_reasons(store, ["max_total_matches_reached"])

    result = update_tournament_config(
        config.run_dir,
        max_total_matches=240,
    )

    assert result.old_max_total_matches == 120
    assert result.new_max_total_matches == 240
    assert result.status.run_status == "running"
    assert result.status.stop_reasons == []

    with TournamentStore(config.state_dir) as store:
        persisted = TournamentConfig.model_validate_json(
            store.get_run_state("config_json") or ""
        )

    assert persisted.max_total_matches == 240
