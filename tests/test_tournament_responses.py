import sqlite3
import sys
import types
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


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
    from dfm_evals.tournament._provenance import (
        GENERATION_PHASE,
        GENERATION_TASK_NAME,
        TOURNAMENT_PHASE_KEY,
        TOURNAMENT_PROJECT_KEY,
        resolve_tournament_project_id,
    )
    from dfm_evals.tournament.config import TournamentConfig, TournamentPrompt
    from dfm_evals.tournament.indexer import index_generation_responses
    from dfm_evals.tournament.scheduler import schedule_match_batch
    from dfm_evals.tournament.store import TournamentStore
    from dfm_evals.tournament.types import match_id, model_id, response_id

    return {
        "GENERATION_PHASE": GENERATION_PHASE,
        "GENERATION_TASK_NAME": GENERATION_TASK_NAME,
        "TOURNAMENT_PHASE_KEY": TOURNAMENT_PHASE_KEY,
        "TOURNAMENT_PROJECT_KEY": TOURNAMENT_PROJECT_KEY,
        "TournamentConfig": TournamentConfig,
        "TournamentPrompt": TournamentPrompt,
        "TournamentStore": TournamentStore,
        "index_generation_responses": index_generation_responses,
        "match_id": match_id,
        "model_id": model_id,
        "resolve_tournament_project_id": resolve_tournament_project_id,
        "response_id": response_id,
        "schedule_match_batch": schedule_match_batch,
    }


def _config(tmp_path: Path) -> object:
    modules = _tournament_modules()
    TournamentConfig = modules["TournamentConfig"]
    TournamentPrompt = modules["TournamentPrompt"]
    return TournamentConfig(
        run_dir=tmp_path / "logs",
        contestant_models=["model/A", "model/B"],
        prompts=[TournamentPrompt(id="prompt-1", text="Hello")],
        judge_model="judge/model",
        judge_prompt_template="{prompt}\nA:{response_a}\nB:{response_b}",
    )


def test_opening_legacy_tournament_state_requires_reinit(tmp_path: Path) -> None:
    modules = _tournament_modules()
    db_path = tmp_path / "state" / "tournament.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE responses (
              response_id TEXT PRIMARY KEY,
              model_id TEXT NOT NULL,
              prompt_id TEXT NOT NULL,
              response_text TEXT NOT NULL,
              source_log TEXT,
              sample_id TEXT,
              sample_uuid TEXT,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()

    try:
        with modules["TournamentStore"](db_path):
            pass
    except ValueError as ex:
        assert "Delete the old tournament state" in str(ex)
    else:
        raise AssertionError("expected legacy tournament state to be rejected")


def test_new_response_versions_do_not_mutate_existing_matches(tmp_path: Path) -> None:
    modules = _tournament_modules()
    config = _config(tmp_path)
    model_a = modules["model_id"]("model/A")
    model_b = modules["model_id"]("model/B")

    with modules["TournamentStore"](config.state_dir) as store:
        store.initialize_from_config(config)

        old_a = modules["response_id"](
            model_a,
            "prompt-1",
            source_log="old-a.eval",
            sample_id="1",
            sample_uuid="uuid-old-a",
            response_text="old A",
        )
        old_b = modules["response_id"](
            model_b,
            "prompt-1",
            source_log="old-b.eval",
            sample_id="1",
            sample_uuid="uuid-old-b",
            response_text="old B",
        )
        store.upsert_response(
            response_id=old_a,
            model_id=model_a,
            prompt_id="prompt-1",
            response_text="old A",
            source_log="old-a.eval",
            source_log_mtime=100.0,
            sample_id="1",
            sample_uuid="uuid-old-a",
        )
        store.upsert_response(
            response_id=old_b,
            model_id=model_b,
            prompt_id="prompt-1",
            response_text="old B",
            source_log="old-b.eval",
            source_log_mtime=100.0,
            sample_id="1",
            sample_uuid="uuid-old-b",
        )

        first_match_id = modules["match_id"](model_a, model_b, "prompt-1", 1, "batch-000001")
        store.upsert_match(
            match_id=first_match_id,
            model_a=model_a,
            model_b=model_b,
            prompt_id="prompt-1",
            response_a_id=old_a,
            response_b_id=old_b,
            batch_id="batch-000001",
            round_index=1,
        )

        new_a = modules["response_id"](
            model_a,
            "prompt-1",
            source_log="new-a.eval",
            sample_id="2",
            sample_uuid="uuid-new-a",
            response_text="new A",
        )
        new_b = modules["response_id"](
            model_b,
            "prompt-1",
            source_log="new-b.eval",
            sample_id="2",
            sample_uuid="uuid-new-b",
            response_text="new B",
        )
        store.upsert_response(
            response_id=new_a,
            model_id=model_a,
            prompt_id="prompt-1",
            response_text="new A",
            source_log="new-a.eval",
            source_log_mtime=200.0,
            sample_id="2",
            sample_uuid="uuid-new-a",
        )
        store.upsert_response(
            response_id=new_b,
            model_id=model_b,
            prompt_id="prompt-1",
            response_text="new B",
            source_log="new-b.eval",
            source_log_mtime=200.0,
            sample_id="2",
            sample_uuid="uuid-new-b",
        )

        existing_match = store.load_batch_matches("batch-000001")[0]
        assert existing_match["response_a_text"] == "old A"
        assert existing_match["response_b_text"] == "old B"

        scheduled = modules["schedule_match_batch"](
            config,
            store,
            batch_id="batch-000002",
            round_index=2,
            persist=False,
        )

    assert len(scheduled.scheduled) == 1
    assert {
        scheduled.scheduled[0].response_a_id,
        scheduled.scheduled[0].response_b_id,
    } == {new_a, new_b}


def test_indexer_keeps_newest_response_current_when_older_logs_reappear(
    tmp_path: Path,
    monkeypatch,
) -> None:
    modules = _tournament_modules()
    config = _config(tmp_path)
    indexer_module = sys.modules["dfm_evals.tournament.indexer"]
    project_id = modules["resolve_tournament_project_id"](config)
    model_identifier = modules["model_id"]("model/A")

    newer_log = SimpleNamespace(
        name=(config.generation_log_dir / "newer.eval").as_posix(),
        mtime=200.0,
    )
    older_log = SimpleNamespace(
        name=(config.generation_log_dir / "older.eval").as_posix(),
        mtime=100.0,
    )
    headers = {
        newer_log.name: SimpleNamespace(
            eval=SimpleNamespace(
                task=modules["GENERATION_TASK_NAME"],
                model="model/A",
                metadata={
                    modules["TOURNAMENT_PHASE_KEY"]: modules["GENERATION_PHASE"],
                    modules["TOURNAMENT_PROJECT_KEY"]: project_id,
                },
            )
        ),
        older_log.name: SimpleNamespace(
            eval=SimpleNamespace(
                task=modules["GENERATION_TASK_NAME"],
                model="model/A",
                metadata={
                    modules["TOURNAMENT_PHASE_KEY"]: modules["GENERATION_PHASE"],
                    modules["TOURNAMENT_PROJECT_KEY"]: project_id,
                },
            )
        ),
    }
    samples = {
        newer_log.name: [
            SimpleNamespace(
                metadata={"prompt_id": "prompt-1"},
                id="sample-new",
                output=SimpleNamespace(completion="newest response"),
                uuid="uuid-new",
            )
        ],
        older_log.name: [
            SimpleNamespace(
                metadata={"prompt_id": "prompt-1"},
                id="sample-old",
                output=SimpleNamespace(completion="older response"),
                uuid="uuid-old",
            )
        ],
    }

    monkeypatch.setattr(indexer_module, "list_eval_logs", lambda path: [newer_log, older_log])
    monkeypatch.setattr(
        indexer_module,
        "read_eval_log",
        lambda log_info, header_only=True: headers[log_info.name],
    )
    monkeypatch.setattr(
        indexer_module,
        "read_eval_log_samples",
        lambda log_info, all_samples_required=False: samples[log_info.name],
    )

    report = modules["index_generation_responses"](config)

    assert report.logs_processed == 2
    assert report.responses_inserted == 2

    with modules["TournamentStore"](config.state_dir) as store:
        current_row = store.connection().execute(
            """
            SELECT response_text
            FROM responses
            WHERE model_id = ?
              AND prompt_id = ?
              AND current = 1
            """,
            (model_identifier, "prompt-1"),
        ).fetchone()
        version_count = store.connection().execute(
            """
            SELECT COUNT(*) AS count
            FROM responses
            WHERE model_id = ?
              AND prompt_id = ?
            """,
            (model_identifier, "prompt-1"),
        ).fetchone()

    assert current_row is not None
    assert current_row["response_text"] == "newest response"
    assert version_count is not None
    assert int(version_count["count"]) == 2
