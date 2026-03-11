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
def _modules() -> dict[str, object]:
    from dfm_evals import cli as cli_module
    from dfm_evals import eee_export as eee_export_module
    from dfm_evals.tournament import _run_state as run_state
    from dfm_evals.tournament.config import TournamentConfig, TournamentPrompt
    from dfm_evals.tournament.store import TournamentStore
    from dfm_evals.tournament.types import ModelRating, match_id, model_id, response_id

    return {
        "ModelRating": ModelRating,
        "TournamentConfig": TournamentConfig,
        "TournamentPrompt": TournamentPrompt,
        "TournamentStore": TournamentStore,
        "cli_module": cli_module,
        "eee_export_module": eee_export_module,
        "match_id": match_id,
        "model_id": model_id,
        "response_id": response_id,
        "run_state": run_state,
    }


def _config(tmp_path: Path) -> object:
    modules = _modules()
    TournamentConfig = modules["TournamentConfig"]
    TournamentPrompt = modules["TournamentPrompt"]
    return TournamentConfig(
        run_dir=tmp_path / "logs",
        project_id="demo-tournament",
        contestant_models=["org/model-a", "org/model-b"],
        prompts=[TournamentPrompt(id="prompt-1", text="Hello")],
        contestant_generate_config={"temperature": 0.1, "max_tokens": 32},
        judge_model="openai/judge-model",
        judge_generate_config={"temperature": 0.0},
        judge_prompt_template="Judge:\n{prompt}\nA:{response_a}\nB:{response_b}",
    )


def _seed_state(config: object) -> None:
    modules = _modules()
    TournamentStore = modules["TournamentStore"]
    ModelRating = modules["ModelRating"]
    run_state = modules["run_state"]

    model_a_id = modules["model_id"]("org/model-a")
    model_b_id = modules["model_id"]("org/model-b")
    response_a_id = modules["response_id"](
        model_a_id,
        "prompt-1",
        source_log="a.eval",
        sample_id="1",
        sample_uuid="uuid-a",
        response_text="response A",
    )
    response_b_id = modules["response_id"](
        model_b_id,
        "prompt-1",
        source_log="b.eval",
        sample_id="1",
        sample_uuid="uuid-b",
        response_text="response B",
    )
    scheduled_match_id = modules["match_id"](
        model_a_id,
        model_b_id,
        "prompt-1",
        1,
        "batch-000001",
    )

    with TournamentStore(config.state_dir) as store:
        store.initialize_from_config(config)
        run_state.ensure_defaults(
            store,
            config_json=config.model_dump_json(),
            run_status="completed",
        )
        run_state.set_run_status(store, "completed")
        run_state.set_converged(store, True)
        run_state.set_stop_reasons(store, ["converged"])

        store.upsert_response(
            response_id=response_a_id,
            model_id=model_a_id,
            prompt_id="prompt-1",
            response_text="response A",
            source_log="a.eval",
            source_log_mtime=100.0,
            sample_id="1",
            sample_uuid="uuid-a",
        )
        store.upsert_response(
            response_id=response_b_id,
            model_id=model_b_id,
            prompt_id="prompt-1",
            response_text="response B",
            source_log="b.eval",
            source_log_mtime=100.0,
            sample_id="1",
            sample_uuid="uuid-b",
        )
        store.upsert_match(
            match_id=scheduled_match_id,
            model_a=model_a_id,
            model_b=model_b_id,
            prompt_id="prompt-1",
            response_a_id=response_a_id,
            response_b_id=response_b_id,
            batch_id="batch-000001",
            round_index=1,
            status="rated",
        )
        store.upsert_judgment(
            judgment_id="judge-ab",
            match_id=scheduled_match_id,
            side="ab",
            decision="A",
            judge_model=config.judge_model,
            explanation="A wins",
            raw_completion="DECISION: A",
            source_log="judge.eval",
            sample_uuid="judge-ab",
        )
        store.upsert_judgment(
            judgment_id="judge-ba",
            match_id=scheduled_match_id,
            side="ba",
            decision="B",
            judge_model=config.judge_model,
            explanation="A wins",
            raw_completion="DECISION: B",
            source_log="judge.eval",
            sample_uuid="judge-ba",
        )
        store.upsert_model_rating(
            ModelRating(
                model_id=model_a_id,
                mu=27.0,
                sigma=7.0,
                games=1,
                wins=1,
                losses=0,
                ties=0,
            )
        )
        store.upsert_model_rating(
            ModelRating(
                model_id=model_b_id,
                mu=23.0,
                sigma=7.5,
                games=1,
                wins=0,
                losses=1,
                ties=0,
            )
        )


def _write_euroeval_results_file(path: Path) -> Path:
    entry = {
        "dataset": "nlu-danish",
        "task": "sentiment",
        "model": "org/model-a",
        "languages": ["da"],
        "results": {
            "total": {
                "test_accuracy": 0.75,
                "test_accuracy_se": 0.05,
            },
            "raw": [
                {"correct": True},
                {"correct": False},
            ],
        },
        "euroeval_version": "1.2.3",
        "vllm_version": "0.8.5",
    }
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return path


def _records_by_model(paths: list[Path]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["model_info"]["id"]] = record
    return records


def test_export_tournament_results_writes_eee_records(tmp_path: Path) -> None:
    modules = _modules()
    config = _config(tmp_path)
    _seed_state(config)

    written = modules["eee_export_module"].export_tournament_results(
        target=config,
        output_dir=tmp_path / "eee",
    )

    assert len(written) == 2

    records_by_model = _records_by_model(written)

    assert set(records_by_model.keys()) == {"org/model-a", "org/model-b"}

    model_a_record = records_by_model["org/model-a"]
    assert model_a_record["source_metadata"]["source_name"] == "tournament"
    assert model_a_record["eval_library"]["name"] == "dfm_evals.tournament"

    metrics = {
        result["score_details"]["details"]["metric"]: result
        for result in model_a_record["evaluation_results"]
    }
    assert metrics["conservative"]["score_details"]["details"]["task"] == "demo-tournament"
    assert metrics["conservative"]["score_details"]["details"]["rank"] == "1"
    assert metrics["conservative"]["score_details"]["score"] == 6.0
    assert metrics["conservative"]["source_data"]["additional_details"]["rated_matches"] == "1"
    assert metrics["conservative"]["metric_config"]["llm_scoring"]["judges"][0]["model_info"]["name"] == "openai/judge-model"
    assert metrics["conservative"]["generation_config"]["additional_details"]["judge_model"] == "openai/judge-model"
    assert "win_rate" in metrics

    model_b_metrics = {
        result["score_details"]["details"]["metric"]: result
        for result in records_by_model["org/model-b"]["evaluation_results"]
    }
    assert model_b_metrics["conservative"]["score_details"]["details"]["rank"] == "2"


def test_export_tournament_results_keeps_stable_evaluation_ids_across_reruns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modules = _modules()
    config = _config(tmp_path)
    _seed_state(config)

    timestamps = iter(["1000.0", "2000.0"])
    monkeypatch.setattr(
        modules["eee_export_module"],
        "_now_unix_timestamp",
        lambda: next(timestamps),
    )

    first_written = modules["eee_export_module"].export_tournament_results(
        target=config,
        output_dir=tmp_path / "eee",
    )
    first_records = _records_by_model(first_written)
    second_written = modules["eee_export_module"].export_tournament_results(
        target=config,
        output_dir=tmp_path / "eee",
    )
    second_records = _records_by_model(second_written)

    assert first_records.keys() == second_records.keys()
    for model_id in first_records:
        assert first_written[0].parent == second_written[0].parent
        assert (
            first_records[model_id]["evaluation_id"]
            == second_records[model_id]["evaluation_id"]
        )
        assert (
            first_records[model_id]["retrieved_timestamp"]
            != second_records[model_id]["retrieved_timestamp"]
        )
    assert set(first_written) == set(second_written)


def test_export_euroeval_results_keeps_stable_evaluation_ids_across_reruns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modules = _modules()
    results_file = _write_euroeval_results_file(tmp_path / "euroeval.jsonl")

    timestamps = iter(["1000.0", "2000.0"])
    monkeypatch.setattr(
        modules["eee_export_module"],
        "_now_unix_timestamp",
        lambda: next(timestamps),
    )

    first_written = modules["eee_export_module"].export_euroeval_results(
        results_file=results_file,
        output_dir=tmp_path / "eee",
    )
    first_record = json.loads(first_written[0].read_text(encoding="utf-8"))
    second_written = modules["eee_export_module"].export_euroeval_results(
        results_file=results_file,
        output_dir=tmp_path / "eee",
    )

    assert len(first_written) == 1
    assert len(second_written) == 1

    second_record = json.loads(second_written[0].read_text(encoding="utf-8"))

    assert first_written == second_written
    assert first_record["evaluation_id"] == second_record["evaluation_id"]
    assert first_record["retrieved_timestamp"] != second_record["retrieved_timestamp"]


def test_export_inspect_logs_overwrites_stable_paths_on_rerun(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modules = _modules()
    eee_export_module = modules["eee_export_module"]
    log_path = tmp_path / "inspect" / "run.eval"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("{}", encoding="utf-8")

    inspect_pkg = types.ModuleType("inspect_ai")
    inspect_log_module = types.ModuleType("inspect_ai.log")

    sample = types.SimpleNamespace(
        id="sample-1",
        input="Question?",
        target="Answer",
        choices=None,
        messages=None,
        scores={},
        output=types.SimpleNamespace(
            completion="Answer",
            model="org/model-a",
            usage=None,
            choices=None,
        ),
        error=None,
        metadata=None,
    )
    scorer = types.SimpleNamespace(
        name="accuracy",
        params={},
        metrics={"accuracy": types.SimpleNamespace(name="accuracy", value=1.0)},
    )
    eval_log = types.SimpleNamespace(
        eval=types.SimpleNamespace(
            task="suite/demo-task",
            dataset={"name": "demo-dataset", "location": "local"},
            created="2026-03-10T00:00:00+00:00",
            model="org/model-a",
            packages={"inspect_ai": "0.1.0"},
            model_generate_config={},
            task_args={},
            config={},
            model_args={},
            model_base_url=None,
        ),
        stats=types.SimpleNamespace(started_at="2026-03-10T00:00:00+00:00"),
        results=types.SimpleNamespace(scores=[scorer]),
        samples=[sample],
    )

    inspect_log_module.list_eval_logs = lambda path: [types.SimpleNamespace(name=log_path.as_posix())]
    inspect_log_module.read_eval_log = lambda path, header_only=False: eval_log
    inspect_pkg.log = inspect_log_module

    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_pkg)
    monkeypatch.setitem(sys.modules, "inspect_ai.log", inspect_log_module)

    timestamps = iter(["1000.0", "2000.0"])
    monkeypatch.setattr(eee_export_module, "_now_unix_timestamp", lambda: next(timestamps))

    first_written = eee_export_module.export_inspect_logs(
        log_path=log_path,
        output_dir=tmp_path / "eee",
    )
    first_record = json.loads(first_written[0].read_text(encoding="utf-8"))
    first_instance_path = first_written[0].with_suffix(".jsonl")
    first_instance_rows = first_instance_path.read_text(encoding="utf-8")

    second_written = eee_export_module.export_inspect_logs(
        log_path=log_path,
        output_dir=tmp_path / "eee",
    )
    second_record = json.loads(second_written[0].read_text(encoding="utf-8"))
    second_instance_path = second_written[0].with_suffix(".jsonl")
    second_instance_rows = second_instance_path.read_text(encoding="utf-8")

    assert first_written == second_written
    assert first_instance_path == second_instance_path
    assert first_record["evaluation_id"] == second_record["evaluation_id"]
    assert first_record["retrieved_timestamp"] != second_record["retrieved_timestamp"]
    assert json.loads(second_instance_rows.splitlines()[0])["evaluation_id"] == second_record["evaluation_id"]
    assert first_instance_rows == second_instance_rows


def test_cli_eee_tournament_dispatches(tmp_path: Path, monkeypatch, capsys) -> None:
    modules = _modules()
    cli_module = modules["cli_module"]
    eee_export_module = modules["eee_export_module"]
    output_path = tmp_path / "exported.json"
    output_path.write_text("{}\n", encoding="utf-8")
    captured: list[dict[str, object]] = []

    def fake_export_tournament_results(**kwargs: object) -> list[Path]:
        captured.append(dict(kwargs))
        return [output_path]

    monkeypatch.setattr(
        eee_export_module,
        "export_tournament_results",
        fake_export_tournament_results,
    )

    exit_code = cli_module.main(
        [
            "eee",
            "tournament",
            "--target",
            "state-dir",
            "--output-dir",
            tmp_path.as_posix(),
        ]
    )

    assert exit_code == 0
    assert captured == [
        {
            "target": "state-dir",
            "output_dir": tmp_path.as_posix(),
            "source_organization_name": "unknown",
            "evaluator_relationship": "third_party",
            "source_organization_url": None,
            "source_organization_logo_url": None,
            "eval_library_name": "dfm_evals.tournament",
            "eval_library_version": None,
        }
    ]

    stdout = capsys.readouterr().out
    assert output_path.as_posix() in stdout
    assert "Exported 1 file(s)." in stdout
