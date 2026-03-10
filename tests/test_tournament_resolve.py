import json
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
    from dfm_evals.tournament import _run_state as run_state
    from dfm_evals.tournament._resolve import (
        resolve_stateful_tournament_config,
        resolve_tournament_config,
    )
    from dfm_evals.tournament.config import (
        TournamentConfig,
        TournamentPrompt,
        load_tournament_config,
    )
    from dfm_evals.tournament.orchestrator import tournament_status
    from dfm_evals.tournament.store import TournamentStore, initialize_tournament_store

    return {
        "TournamentConfig": TournamentConfig,
        "TournamentPrompt": TournamentPrompt,
        "TournamentStore": TournamentStore,
        "initialize_tournament_store": initialize_tournament_store,
        "load_tournament_config": load_tournament_config,
        "resolve_stateful_tournament_config": resolve_stateful_tournament_config,
        "resolve_tournament_config": resolve_tournament_config,
        "run_state": run_state,
        "tournament_status": tournament_status,
    }


def _write_config(path: Path, contestant_models: list[str]) -> object:
    modules = _tournament_modules()
    TournamentConfig = modules["TournamentConfig"]
    TournamentPrompt = modules["TournamentPrompt"]

    config = TournamentConfig(
        run_dir=Path("logs"),
        contestant_models=contestant_models,
        prompts=[TournamentPrompt(id="prompt-1", text="Hello")],
        judge_model="judge/model",
        judge_prompt_template="{prompt}\nA:{response_a}\nB:{response_b}",
    )
    path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return modules["load_tournament_config"](path)


def test_resolve_tournament_config_falls_back_to_file_when_state_has_no_config_json(
    tmp_path: Path,
) -> None:
    modules = _tournament_modules()
    config_path = tmp_path / "tournament.json"
    parsed = _write_config(config_path, ["model/A", "model/B"])

    modules["initialize_tournament_store"](config_path)

    resolved = modules["resolve_tournament_config"](config_path)

    assert resolved.contestant_models == parsed.contestant_models


def test_initialize_tournament_store_persists_config_json(tmp_path: Path) -> None:
    modules = _tournament_modules()
    config_path = tmp_path / "tournament.json"
    parsed = _write_config(config_path, ["model/A", "model/B"])

    modules["initialize_tournament_store"](config_path)

    with modules["TournamentStore"](parsed.state_dir) as store:
        persisted = store.get_run_state(modules["run_state"].KEY_CONFIG_JSON)

    assert persisted is not None
    assert (
        modules["TournamentConfig"].model_validate_json(persisted).contestant_models
        == ["model/A", "model/B"]
    )


def test_resolve_stateful_tournament_config_falls_back_to_runtime_config_for_run_dir(
    tmp_path: Path,
) -> None:
    modules = _tournament_modules()
    config_path = tmp_path / "tournament.json"
    parsed = _write_config(config_path, ["model/A", "model/B"])
    runtime_path = parsed.run_dir / "config" / "runtime.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(
        json.dumps(parsed.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    modules["initialize_tournament_store"](config_path)
    with modules["TournamentStore"](parsed.state_dir) as store:
        store.set_run_state(modules["run_state"].KEY_CONFIG_JSON, "", commit=True)

    resolved = modules["resolve_stateful_tournament_config"](parsed.run_dir)

    assert resolved.contestant_models == parsed.contestant_models


def test_resolve_tournament_config_uses_file_config_even_when_state_differs(
    tmp_path: Path,
) -> None:
    modules = _tournament_modules()
    config_path = tmp_path / "tournament.json"
    parsed = _write_config(config_path, ["model/A", "model/B"])
    updated = parsed.model_copy(
        update={"contestant_models": ["model/A", "model/B", "model/C"]}
    )

    modules["initialize_tournament_store"](config_path)
    with modules["TournamentStore"](parsed.state_dir) as store:
        store.initialize_from_config(updated)
        modules["run_state"].set_config_json(store, updated.model_dump_json())

    resolved = modules["resolve_tournament_config"](config_path)

    assert resolved.contestant_models == parsed.contestant_models


def test_resolve_stateful_tournament_config_prefers_persisted_state_for_config_path(
    tmp_path: Path,
) -> None:
    modules = _tournament_modules()
    config_path = tmp_path / "tournament.json"
    parsed = _write_config(config_path, ["model/A", "model/B"])
    updated = parsed.model_copy(
        update={"contestant_models": ["model/A", "model/B", "model/C"]}
    )

    modules["initialize_tournament_store"](config_path)
    with modules["TournamentStore"](parsed.state_dir) as store:
        store.initialize_from_config(updated)
        modules["run_state"].set_config_json(store, updated.model_dump_json())

    resolved = modules["resolve_stateful_tournament_config"](config_path)

    assert resolved.contestant_models == updated.contestant_models


def test_tournament_status_uses_persisted_config_for_file_target(tmp_path: Path) -> None:
    modules = _tournament_modules()
    config_path = tmp_path / "tournament.json"
    parsed = _write_config(config_path, ["model/A", "model/B"])
    updated = parsed.model_copy(
        update={"contestant_models": ["model/A", "model/B", "model/C"]}
    )

    modules["initialize_tournament_store"](config_path)
    with modules["TournamentStore"](parsed.state_dir) as store:
        store.initialize_from_config(updated)
        modules["run_state"].set_config_json(store, updated.model_dump_json())

    status = modules["tournament_status"](config_path)

    assert status.total_models == 3
    assert {standing.model_name for standing in status.standings} == {
        "model/A",
        "model/B",
        "model/C",
    }
