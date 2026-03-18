import json
import os
import sys
import threading
import types
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen


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
    from dfm_evals.tournament.cli import main as tournament_main
    from dfm_evals.tournament.config import TournamentConfig, TournamentPrompt
    from dfm_evals.tournament.store import TournamentStore
    from dfm_evals.tournament.types import ModelRating, match_id, model_id, response_id
    from dfm_evals.tournament.viewer import (
        TournamentViewDataSource,
        create_tournament_view_server,
        export_tournament_view_html,
        list_tournament_view_runs,
        resolve_tournament_view_target,
    )

    return {
        "ModelRating": ModelRating,
        "TournamentConfig": TournamentConfig,
        "TournamentPrompt": TournamentPrompt,
        "TournamentStore": TournamentStore,
        "TournamentViewDataSource": TournamentViewDataSource,
        "export_tournament_view_html": export_tournament_view_html,
        "create_tournament_view_server": create_tournament_view_server,
        "list_tournament_view_runs": list_tournament_view_runs,
        "match_id": match_id,
        "model_id": model_id,
        "response_id": response_id,
        "resolve_tournament_view_target": resolve_tournament_view_target,
        "run_state": run_state,
        "tournament_main": tournament_main,
    }


def _config(tmp_path: Path, *, run_dir: Path | None = None) -> object:
    modules = _modules()
    TournamentConfig = modules["TournamentConfig"]
    TournamentPrompt = modules["TournamentPrompt"]
    resolved_run_dir = run_dir if run_dir is not None else (tmp_path / "logs")
    return TournamentConfig(
        run_dir=resolved_run_dir,
        project_id="demo-tournament",
        contestant_models=["org/model-a", "org/model-b"],
        prompts=[
            TournamentPrompt(
                id="prompt-1",
                text="Write a short poem about winter light.",
                metadata={"title": "Winter Light", "category": "creative", "source": "unit"},
            ),
            TournamentPrompt(
                id="prompt-2",
                text="Summarize the moral of a story about stubbornness.",
                metadata={"title": "Stubborn Story", "category": "summary", "source": "unit"},
            ),
        ],
        judge_model="openai/qwen-judge",
        judge_prompt_template="Judge:\n{prompt}\nA:{response_a}\nB:{response_b}",
    )


def _seed_state(config: object) -> dict[str, str]:
    modules = _modules()
    TournamentStore = modules["TournamentStore"]
    ModelRating = modules["ModelRating"]
    run_state = modules["run_state"]

    model_a_id = modules["model_id"]("org/model-a")
    model_b_id = modules["model_id"]("org/model-b")
    response_a_1 = modules["response_id"](
        model_a_id,
        "prompt-1",
        source_log="a-1.eval",
        sample_id="1",
        sample_uuid="uuid-a-1",
        response_text="The snow keeps its own counsel.",
    )
    response_b_1 = modules["response_id"](
        model_b_id,
        "prompt-1",
        source_log="b-1.eval",
        sample_id="1",
        sample_uuid="uuid-b-1",
        response_text="Winter glows and then is gone.",
    )
    response_a_2 = modules["response_id"](
        model_a_id,
        "prompt-2",
        source_log="a-2.eval",
        sample_id="1",
        sample_uuid="uuid-a-2",
        response_text="Stubbornness blocks learning until the cost becomes obvious.",
    )
    response_b_2 = modules["response_id"](
        model_b_id,
        "prompt-2",
        source_log="b-2.eval",
        sample_id="1",
        sample_uuid="uuid-b-2",
        response_text="Refusing to bend makes avoidable pain last longer.",
    )

    match_1 = modules["match_id"](model_a_id, model_b_id, "prompt-1", 1, "batch-000001")
    match_2 = modules["match_id"](model_a_id, model_b_id, "prompt-2", 2, "batch-000002")

    with TournamentStore(config.state_dir) as store:
        store.initialize_from_config(config)
        run_state.ensure_defaults(
            store,
            config_json=config.model_dump_json(),
            run_status="completed",
        )
        run_state.set_run_status(store, "completed")
        run_state.set_converged(store, False)
        run_state.set_stop_reasons(store, ["max_total_matches_reached"])
        run_state.set_last_batch_id(store, "batch-000002")
        run_state.set_next_round_index(store, 3)

        store.upsert_response(
            response_id=response_a_1,
            model_id=model_a_id,
            prompt_id="prompt-1",
            response_text="The snow keeps its own counsel.",
            source_log="a-1.eval",
            source_log_mtime=100.0,
            sample_id="1",
            sample_uuid="uuid-a-1",
        )
        store.upsert_response(
            response_id=response_b_1,
            model_id=model_b_id,
            prompt_id="prompt-1",
            response_text="Winter glows and then is gone.",
            source_log="b-1.eval",
            source_log_mtime=100.0,
            sample_id="1",
            sample_uuid="uuid-b-1",
        )
        store.upsert_response(
            response_id=response_a_2,
            model_id=model_a_id,
            prompt_id="prompt-2",
            response_text="Stubbornness blocks learning until the cost becomes obvious.",
            source_log="a-2.eval",
            source_log_mtime=110.0,
            sample_id="1",
            sample_uuid="uuid-a-2",
        )
        store.upsert_response(
            response_id=response_b_2,
            model_id=model_b_id,
            prompt_id="prompt-2",
            response_text="Refusing to bend makes avoidable pain last longer.",
            source_log="b-2.eval",
            source_log_mtime=110.0,
            sample_id="1",
            sample_uuid="uuid-b-2",
        )

        store.upsert_match(
            match_id=match_1,
            model_a=model_a_id,
            model_b=model_b_id,
            prompt_id="prompt-1",
            response_a_id=response_a_1,
            response_b_id=response_b_1,
            batch_id="batch-000001",
            round_index=1,
            status="rated",
        )
        store.upsert_match(
            match_id=match_2,
            model_a=model_a_id,
            model_b=model_b_id,
            prompt_id="prompt-2",
            response_a_id=response_a_2,
            response_b_id=response_b_2,
            batch_id="batch-000002",
            round_index=2,
            status="rated",
        )

        store.upsert_judgment(
            judgment_id="judge-1-ab",
            match_id=match_1,
            side="ab",
            decision="A",
            judge_model=config.judge_model,
            explanation="A has stronger imagery.",
            raw_completion="DECISION: A",
            source_log="judge-1.eval",
            sample_uuid="judge-1-ab",
        )
        store.upsert_judgment(
            judgment_id="judge-1-ba",
            match_id=match_1,
            side="ba",
            decision="B",
            judge_model=config.judge_model,
            explanation="A still wins after the swap.",
            raw_completion="DECISION: B",
            source_log="judge-1.eval",
            sample_uuid="judge-1-ba",
        )
        store.upsert_judgment(
            judgment_id="judge-2-ab",
            match_id=match_2,
            side="ab",
            decision="TIE",
            judge_model=config.judge_model,
            explanation="Both summaries are equally clear.",
            raw_completion="DECISION: TIE",
            source_log="judge-2.eval",
            sample_uuid="judge-2-ab",
        )
        store.upsert_judgment(
            judgment_id="judge-2-ba",
            match_id=match_2,
            side="ba",
            decision="TIE",
            judge_model=config.judge_model,
            explanation="No meaningful difference after swap.",
            raw_completion="DECISION: TIE",
            source_log="judge-2.eval",
            sample_uuid="judge-2-ba",
        )

        store.upsert_model_rating(
            ModelRating(
                model_id=model_a_id,
                mu=27.0,
                sigma=7.0,
                games=2,
                wins=1,
                losses=0,
                ties=1,
            )
        )
        store.upsert_model_rating(
            ModelRating(
                model_id=model_b_id,
                mu=23.0,
                sigma=7.5,
                games=2,
                wins=0,
                losses=1,
                ties=1,
            )
        )

    return {
        "model_a_id": model_a_id,
        "model_b_id": model_b_id,
        "match_1": match_1,
        "match_2": match_2,
    }


def _get_json(url: str) -> dict[str, object]:
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def test_view_data_source_summary_and_pairwise(tmp_path: Path) -> None:
    modules = _modules()
    config = _config(tmp_path)
    ids = _seed_state(config)
    source = modules["TournamentViewDataSource"](config)

    summary = source.summary()
    assert summary["project_id"] == "demo-tournament"
    assert summary["response_count"] == 4
    assert summary["expected_responses"] == 4
    assert summary["rated_matches"] == 2
    assert summary["batch_count"] == 2
    assert summary["judgment_count"] == 4

    pairwise = source.pairwise()
    assert [model["model_id"] for model in pairwise["models"]] == [
        ids["model_a_id"],
        ids["model_b_id"],
    ]
    cell = pairwise["rows"][0]["cells"][1]
    assert cell["wins"] == 1
    assert cell["losses"] == 0
    assert cell["ties"] == 1
    assert cell["rated_games"] == 2
    assert cell["score"] == 0.75


def test_view_data_source_match_filters_and_details(tmp_path: Path) -> None:
    modules = _modules()
    config = _config(tmp_path)
    ids = _seed_state(config)
    source = modules["TournamentViewDataSource"](config)

    all_matches = source.list_matches()
    assert all_matches["total"] == 2

    tie_matches = source.list_matches({"outcome": "tie"})
    assert tie_matches["total"] == 1
    assert tie_matches["items"][0]["match_id"] == ids["match_2"]

    detail = source.match_detail(ids["match_1"])
    assert detail["winner_model_id"] == ids["model_a_id"]
    assert "stronger imagery" in detail["judgments"]["ab"]["explanation"]
    assert "snow keeps" in detail["response_a_text"]

    prompt = source.prompt_detail("prompt-1")
    assert prompt["response_count"] == 2
    assert prompt["rated_match_count"] == 1


def test_view_http_server_serves_assets_and_api(tmp_path: Path) -> None:
    modules = _modules()
    config = _config(tmp_path)
    ids = _seed_state(config)
    server = modules["create_tournament_view_server"](config, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"{server.url}/") as response:
            html = response.read().decode("utf-8")
            assert response.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate"
        assert "Tournament Viewer" in html
        assert "Auto refresh every 10s" in html

        with urlopen(f"{server.url}/api/summary") as response:
            assert response.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate"
            summary = json.loads(response.read().decode("utf-8"))
        assert summary["project_id"] == "demo-tournament"

        match = _get_json(
            f"{server.url}/api/match?match_id={ids['match_1']}"
        )
        assert match["winner_model_id"] == ids["model_a_id"]

        prompt = _get_json(f"{server.url}/api/prompt?prompt_id=prompt-1")
        assert prompt["responses"][0]["model_name"] == "org/model-a"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_static_view_export_writes_single_html(tmp_path: Path) -> None:
    modules = _modules()
    config = _config(tmp_path)
    ids = _seed_state(config)

    result = modules["export_tournament_view_html"](config)

    assert result.output_html == config.exports_dir / "viewer.html"
    assert result.output_html.is_file()
    html = result.output_html.read_text(encoding="utf-8")
    assert "window.__TOURNAMENT_STATIC_DATA__" in html
    assert "<style>" in html
    assert '<script src="/app.js"></script>' not in html
    assert "The snow keeps its own counsel." in html
    assert "A has stronger imagery." in html
    assert ids["match_1"] in html


def test_cli_export_html_writes_static_view(tmp_path: Path) -> None:
    modules = _modules()
    config = _config(tmp_path)
    _seed_state(config)
    output_html = tmp_path / "report" / "creative-writing-viewer.html"

    result = modules["tournament_main"](
        [
            "export-html",
            config.run_dir.as_posix(),
            "--output",
            output_html.as_posix(),
        ]
    )

    assert result == 0
    assert output_html.is_file()
    html = output_html.read_text(encoding="utf-8")
    assert "window.__TOURNAMENT_STATIC_DATA__" in html
    assert "Both summaries are equally clear." in html
    assert "Static export" in html


def test_view_target_resolution_supports_latest_job_id_and_label(tmp_path: Path) -> None:
    modules = _modules()
    log_root = tmp_path / "logs" / "evals-logs"
    older_run = log_root / "tournament__older__job-111"
    newer_run = log_root / "tournament__newer__job-222"

    older_config = _config(tmp_path, run_dir=older_run)
    newer_config = _config(tmp_path, run_dir=newer_run)
    _seed_state(older_config)
    _seed_state(newer_config)

    os.utime(older_run / "state" / "tournament.db", (100.0, 100.0))
    os.utime(newer_run / "state" / "tournament.db", (200.0, 200.0))

    runs = modules["list_tournament_view_runs"](log_root=log_root)
    assert [run.run_label for run in runs] == [
        "tournament__newer__job-222",
        "tournament__older__job-111",
    ]

    assert modules["resolve_tournament_view_target"](log_root=log_root) == newer_run
    assert (
        modules["resolve_tournament_view_target"]("222", log_root=log_root)
        == newer_run
    )
    assert (
        modules["resolve_tournament_view_target"](
            run_label="tournament__older__job-111",
            log_root=log_root,
        )
        == older_run
    )


def test_cli_view_dispatches(tmp_path: Path, monkeypatch) -> None:
    modules = _modules()
    called: list[dict[str, object]] = []

    def fake_serve(target: object, *, host: str, port: int) -> int:
        called.append({"target": target, "host": host, "port": port})
        return 0

    cli_module = sys.modules["dfm_evals.tournament.cli"]
    monkeypatch.setattr(cli_module, "serve_tournament_view", fake_serve)

    result = modules["tournament_main"](
        ["view", tmp_path.as_posix(), "--host", "0.0.0.0", "--port", "8123"]
    )

    assert result == 0
    assert called == [
        {"target": tmp_path, "host": "0.0.0.0", "port": 8123}
    ]


def test_cli_view_defaults_to_latest_run(tmp_path: Path, monkeypatch) -> None:
    modules = _modules()
    log_root = tmp_path / "logs" / "evals-logs"
    latest_run = log_root / "tournament__latest__job-333"
    config = _config(tmp_path, run_dir=latest_run)
    _seed_state(config)

    called: list[dict[str, object]] = []

    def fake_serve(target: object, *, host: str, port: int) -> int:
        called.append({"target": target, "host": host, "port": port})
        return 0

    cli_module = sys.modules["dfm_evals.tournament.cli"]
    monkeypatch.setattr(cli_module, "serve_tournament_view", fake_serve)

    result = modules["tournament_main"](
        [
            "view",
            "--log-root",
            log_root.as_posix(),
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ]
    )

    assert result == 0
    assert called == [
        {
            "target": latest_run,
            "host": "127.0.0.1",
            "port": 9000,
        }
    ]
