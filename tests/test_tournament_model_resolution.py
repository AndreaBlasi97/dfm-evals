import logging
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
def _tournament_modules() -> dict[str, object]:
    from dfm_evals.tournament.config import TournamentConfig, TournamentPrompt
    from dfm_evals.tournament.generation import run_generation
    from dfm_evals.tournament.judge_task import JudgeMatch, run_judge_batch

    return {
        "JudgeMatch": JudgeMatch,
        "TournamentConfig": TournamentConfig,
        "TournamentPrompt": TournamentPrompt,
        "run_generation": run_generation,
        "run_judge_batch": run_judge_batch,
    }


def _config(
    tmp_path: Path,
    *,
    contestant_model: str = "vllm/model/A",
    judge_model: str = "vllm/judge/model",
) -> object:
    modules = _tournament_modules()
    TournamentConfig = modules["TournamentConfig"]
    TournamentPrompt = modules["TournamentPrompt"]
    return TournamentConfig(
        run_dir=tmp_path / "logs",
        contestant_models=[contestant_model, "model/B"],
        prompts=[TournamentPrompt(id="prompt-1", text="Hello")],
        contestant_generate_config={"temperature": 0.1},
        judge_model=judge_model,
        judge_generate_config={"temperature": 0.2},
        judge_prompt_template="{prompt}\nA:{response_a}\nB:{response_b}",
    )


def test_run_generation_uses_shared_env_model_args(tmp_path: Path, monkeypatch) -> None:
    modules = _tournament_modules()
    config = _config(tmp_path)
    generation_module = sys.modules["dfm_evals.tournament.generation"]

    captured_get_model_calls: list[tuple[str, dict[str, object]]] = []
    closed_models: list[object] = []
    fake_model = object()

    def fake_get_model(model_name: str, **kwargs: object) -> object:
        captured_get_model_calls.append((model_name, kwargs))
        return fake_model

    def fake_eval_set(**kwargs: object) -> tuple[bool, list[object]]:
        del kwargs
        return True, []

    monkeypatch.setenv(
        "INSPECT_EVAL_MODEL_ARGS",
        'base-url=https://contestant.example/v1 api-key="abc 123"',
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(generation_module, "get_model", fake_get_model)
    monkeypatch.setattr(generation_module, "eval_set", fake_eval_set)
    monkeypatch.setattr(generation_module, "_close_model", lambda model: closed_models.append(model))

    modules["run_generation"](config, models=["vllm/model/A"])

    assert len(captured_get_model_calls) == 1
    model_name, kwargs = captured_get_model_calls[0]
    assert model_name == "vllm/model/A"
    assert kwargs["base_url"] == "https://contestant.example/v1"
    assert kwargs["api_key"] == "abc 123"
    assert kwargs["memoize"] is False
    assert "device" not in kwargs
    assert closed_models == [fake_model]


def test_run_judge_batch_uses_grader_role_model_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modules = _tournament_modules()
    config = _config(tmp_path)
    JudgeMatch = modules["JudgeMatch"]
    judge_module = sys.modules["dfm_evals.tournament.judge_task"]

    captured_get_model_calls: list[tuple[str, dict[str, object]]] = []
    captured_eval_calls: list[dict[str, object]] = []
    closed_models: list[object] = []
    fake_grader = object()

    def fake_get_model(model_name: str, **kwargs: object) -> object:
        captured_get_model_calls.append((model_name, kwargs))
        return fake_grader

    def fake_eval(**kwargs: object) -> list[object]:
        captured_eval_calls.append(kwargs)
        return []

    monkeypatch.setenv("INSPECT_EVAL_MODEL_ARGS", "base-url=https://judge.example/v1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(judge_module, "get_model", fake_get_model)
    monkeypatch.setattr(judge_module, "eval", fake_eval)
    monkeypatch.setattr(judge_module, "close_model", lambda model: closed_models.append(model))

    result = modules["run_judge_batch"](
        config,
        matches=[
            JudgeMatch(
                match_id="match-1",
                prompt_id="prompt-1",
                prompt="Hello",
                model_a="model/A",
                model_b="model/B",
                response_a="response A",
                response_b="response B",
            )
        ],
    )

    assert len(captured_get_model_calls) == 1
    model_name, kwargs = captured_get_model_calls[0]
    assert model_name == "vllm/judge/model"
    assert kwargs["base_url"] == "https://judge.example/v1"
    assert kwargs["memoize"] is False
    assert kwargs["config"].model_dump(exclude_none=True) == {"temperature": 0.2}
    assert "device" not in kwargs

    assert len(captured_eval_calls) == 1
    eval_kwargs = captured_eval_calls[0]
    assert eval_kwargs["model"] is None
    assert eval_kwargs["model_roles"] == {"grader": fake_grader}
    assert closed_models == [fake_grader]
    assert result.log_count == 0


def test_close_model_ignores_closed_event_loop_error(caplog) -> None:
    from dfm_evals.tournament._model_args import close_model

    class FakeModel:
        def __init__(self) -> None:
            self._closed = False

        def __str__(self) -> str:
            return "openai/qwen-235b"

        def __exit__(self, exc_type, exc, exc_tb) -> None:
            del exc_type, exc, exc_tb
            raise RuntimeError("OpenAIAPI models require an async close / context manager.")

        async def __aexit__(self, exc_type, exc, exc_tb) -> None:
            del exc_type, exc, exc_tb
            raise RuntimeError("Event loop is closed")

    model = FakeModel()

    with caplog.at_level(logging.WARNING):
        close_model(model)

    assert model._closed is True
    assert caplog.records == []
