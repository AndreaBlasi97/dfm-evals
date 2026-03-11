import sys
import types
from functools import lru_cache
from pathlib import Path


def _install_trueskill_stub() -> None:
    if "trueskill" in sys.modules:
        return

    trueskill = types.ModuleType("trueskill")

    class Rating:
        def __init__(self, mu: float = 25.0, sigma: float = 25.0 / 3.0) -> None:
            self.mu = mu
            self.sigma = sigma

    class TrueSkill:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def cdf(self, value: float) -> float:
            del value
            return 0.5

    def rate_1vs1(
        first: Rating,
        second: Rating,
        drawn: bool = False,
        env: TrueSkill | None = None,
    ) -> tuple[Rating, Rating]:
        del drawn, env
        return first, second

    trueskill.Rating = Rating
    trueskill.TrueSkill = TrueSkill
    trueskill.rate_1vs1 = rate_1vs1
    sys.modules["trueskill"] = trueskill


_install_trueskill_stub()


@lru_cache(maxsize=1)
def _modules() -> dict[str, object]:
    from dfm_evals.tournament import _run_state as run_state
    from dfm_evals.tournament.config import TournamentConfig, TournamentPrompt
    from dfm_evals.tournament.orchestrator import _apply_pending_batch_outcomes
    from dfm_evals.tournament.store import TournamentStore
    from dfm_evals.tournament.types import match_id, model_id, response_id

    return {
        "TournamentConfig": TournamentConfig,
        "TournamentPrompt": TournamentPrompt,
        "TournamentStore": TournamentStore,
        "_apply_pending_batch_outcomes": _apply_pending_batch_outcomes,
        "match_id": match_id,
        "model_id": model_id,
        "response_id": response_id,
        "run_state": run_state,
    }


def _config(tmp_path: Path, *, judge_max_samples: int = 8) -> object:
    modules = _modules()
    TournamentConfig = modules["TournamentConfig"]
    TournamentPrompt = modules["TournamentPrompt"]
    return TournamentConfig(
        run_dir=tmp_path / "logs",
        contestant_models=["model/A", "model/B"],
        prompts=[TournamentPrompt(id="prompt-1", text="Hello")],
        judge_model="judge/model",
        judge_max_samples=judge_max_samples,
        judge_prompt_template="{prompt}\nA:{response_a}\nB:{response_b}",
        side_swap=True,
    )


def test_side_swap_requires_at_least_two_judge_samples(tmp_path: Path) -> None:
    try:
        _config(tmp_path, judge_max_samples=1)
    except ValueError as exc:
        assert "judge_max_samples must be at least 2" in str(exc)
    else:
        raise AssertionError("expected side-swapped config to reject judge_max_samples=1")


def test_one_sided_side_swap_judgment_blocks_rating(tmp_path: Path) -> None:
    modules = _modules()
    config = _config(tmp_path, judge_max_samples=2)
    run_state = modules["run_state"]
    TournamentStore = modules["TournamentStore"]
    model_a = modules["model_id"]("model/A")
    model_b = modules["model_id"]("model/B")
    response_a = modules["response_id"](
        model_a,
        "prompt-1",
        source_log="a.eval",
        sample_id="1",
        sample_uuid="uuid-a",
        response_text="response A",
    )
    response_b = modules["response_id"](
        model_b,
        "prompt-1",
        source_log="b.eval",
        sample_id="1",
        sample_uuid="uuid-b",
        response_text="response B",
    )
    scheduled_match_id = modules["match_id"](
        model_a,
        model_b,
        "prompt-1",
        1,
        "batch-000001",
    )

    with TournamentStore(config.state_dir) as store:
        store.initialize_from_config(config)
        run_state.ensure_defaults(
            store,
            config_json=config.model_dump_json(),
            run_status="running",
        )

        store.upsert_response(
            response_id=response_a,
            model_id=model_a,
            prompt_id="prompt-1",
            response_text="response A",
            source_log="a.eval",
            source_log_mtime=100.0,
            sample_id="1",
            sample_uuid="uuid-a",
        )
        store.upsert_response(
            response_id=response_b,
            model_id=model_b,
            prompt_id="prompt-1",
            response_text="response B",
            source_log="b.eval",
            source_log_mtime=100.0,
            sample_id="1",
            sample_uuid="uuid-b",
        )
        store.upsert_match(
            match_id=scheduled_match_id,
            model_a=model_a,
            model_b=model_b,
            prompt_id="prompt-1",
            response_a_id=response_a,
            response_b_id=response_b,
            batch_id="batch-000001",
            round_index=1,
            status="judged",
        )
        store.upsert_judgment(
            judgment_id="judge-ab",
            match_id=scheduled_match_id,
            side="ab",
            decision="A",
            judge_model=config.judge_model,
            explanation="A wins on one side only.",
            raw_completion="DECISION: A",
            source_log="judge.eval",
            sample_uuid="judge-ab",
        )

        result = modules["_apply_pending_batch_outcomes"](
            config,
            store,
            batch_id="batch-000001",
            round_index=1,
        )
        state = run_state.load(store, run_status_default="running")
        rows = store.load_batch_matches("batch-000001", statuses=["scheduled"])

    assert result == {
        "processed": 0,
        "skipped": 1,
        "converged": False,
        "blocked": True,
    }
    assert state.run_status == "stopped"
    assert state.stop_reasons == ["no_valid_judgments"]
    assert len(rows) == 1
