import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from ._resolve import resolve_stateful_tournament_config
from .config import TournamentConfig
from .orchestrator import TournamentStatus, tournament_status
from .scorer import canonicalize_side_decision, reconcile_side_swap
from .store import TournamentStore
from .types import Decision, InvalidPolicy


class ExportResult(BaseModel):
    """Paths for generated tournament export artifacts."""

    output_dir: Path
    rankings_json: Path
    rankings_csv: Path
    pairwise_matrix_csv: Path | None = None


class PromptResponsesExportResult(BaseModel):
    """Path and summary for prompt/response export artifacts."""

    output_path: Path
    prompt_count: int
    model_count: int


@dataclass
class TournamentExportSnapshot:
    """Resolved tournament state used by export backends."""

    config: TournamentConfig
    status: TournamentStatus
    names_by_id: dict[str, str]
    pairwise_stats: dict[tuple[str, str], "_PairStats"]


def load_export_snapshot(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    include_pairwise_matrix: bool = False,
) -> TournamentExportSnapshot:
    """Load resolved tournament state for export backends."""
    config = resolve_stateful_tournament_config(config_or_state)
    status = tournament_status(config)
    state_dir = _require_state_dir(config)

    with TournamentStore(state_dir) as store:
        names_by_id = _model_names_by_id(store)
        pairwise_stats = (
            _pairwise_stats(config, store) if include_pairwise_matrix else {}
        )

    return TournamentExportSnapshot(
        config=config,
        status=status,
        names_by_id=names_by_id,
        pairwise_stats=pairwise_stats,
    )


def export_rankings(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    output_dir: str | Path | None = None,
    include_pairwise_matrix: bool = True,
) -> ExportResult:
    """Export standings to JSON/CSV plus optional pairwise matrix CSV."""
    snapshot = load_export_snapshot(
        config_or_state,
        include_pairwise_matrix=include_pairwise_matrix,
    )
    config = snapshot.config
    status = snapshot.status

    export_dir = Path(output_dir) if output_dir is not None else config.exports_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    rankings_json = export_dir / "rankings.json"
    rankings_csv = export_dir / "rankings.csv"
    pairwise_matrix_csv = (
        export_dir / "pairwise_matrix.csv" if include_pairwise_matrix else None
    )

    ranking_rows = [
        {
            "rank": rank,
            "model_id": standing.model_id,
            "model_name": snapshot.names_by_id.get(standing.model_id, standing.model_id),
            "mu": standing.mu,
            "sigma": standing.sigma,
            "conservative": standing.conservative,
            "elo_like": standing.elo_like,
            "games": standing.games,
            "wins": standing.wins,
            "losses": standing.losses,
            "ties": standing.ties,
        }
        for rank, standing in enumerate(status.standings, start=1)
    ]

    rankings_json.write_text(
        json.dumps(
            {
                "project_id": status.project_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_status": status.run_status,
                "converged": status.converged,
                "stop_reasons": status.stop_reasons,
                "total_matches": status.total_matches,
                "rated_matches": status.rated_matches,
                "models": ranking_rows,
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with rankings_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "model_id",
                "model_name",
                "mu",
                "sigma",
                "conservative",
                "elo_like",
                "games",
                "wins",
                "losses",
                "ties",
            ],
        )
        writer.writeheader()
        writer.writerows(ranking_rows)

    if pairwise_matrix_csv is not None:
        _write_pairwise_matrix_csv(
            pairwise_matrix_csv,
            standings=[standing.model_id for standing in status.standings],
            names_by_id=snapshot.names_by_id,
            stats=snapshot.pairwise_stats,
        )

    return ExportResult(
        output_dir=export_dir,
        rankings_json=rankings_json,
        rankings_csv=rankings_csv,
        pairwise_matrix_csv=pairwise_matrix_csv,
    )


def export_prompt_responses(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    output_path: str | Path | None = None,
) -> PromptResponsesExportResult:
    """Export prompts with current responses and no judge outputs."""
    config = resolve_stateful_tournament_config(config_or_state)
    status = tournament_status(config)
    export_path = (
        Path(output_path)
        if output_path is not None
        else config.exports_dir / "prompt_responses.json"
    )
    export_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_order = {prompt.id: index for index, prompt in enumerate(config.prompts)}
    model_order = {
        model_name: index for index, model_name in enumerate(config.contestant_models)
    }

    with TournamentStore(_require_state_dir(config)) as store:
        conn = store.connection()
        prompt_rows = conn.execute(
            """
            SELECT prompt_id, prompt_text, metadata_json
            FROM prompts
            """
        ).fetchall()
        model_rows = conn.execute(
            """
            SELECT model_id, model_name
            FROM models
            WHERE active = 1
            """
        ).fetchall()
        response_rows = conn.execute(
            """
            SELECT model_id, prompt_id, response_id, response_text, source_log
            FROM responses
            WHERE current = 1
            """
        ).fetchall()

    ordered_prompts = sorted(
        prompt_rows,
        key=lambda row: (
            prompt_order.get(str(row["prompt_id"]), len(prompt_order)),
            str(row["prompt_id"]),
        ),
    )
    ordered_models = sorted(
        model_rows,
        key=lambda row: (
            model_order.get(str(row["model_name"]), len(model_order)),
            str(row["model_name"]),
        ),
    )
    responses_by_key = {
        (str(row["prompt_id"]), str(row["model_id"])): row for row in response_rows
    }

    prompts_payload: list[dict[str, Any]] = []
    for prompt_row in ordered_prompts:
        metadata = _parse_metadata(prompt_row["metadata_json"])
        prompt_id = str(prompt_row["prompt_id"])
        responses_payload = []
        response_count = 0
        for model_row in ordered_models:
            response_row = responses_by_key.get(
                (prompt_id, str(model_row["model_id"])),
            )
            if response_row is not None:
                response_count += 1
            responses_payload.append(
                {
                    "model_id": str(model_row["model_id"]),
                    "model_name": str(model_row["model_name"]),
                    "response_id": (
                        str(response_row["response_id"])
                        if response_row is not None and response_row["response_id"] is not None
                        else None
                    ),
                    "response_text": (
                        str(response_row["response_text"])
                        if response_row is not None and response_row["response_text"] is not None
                        else None
                    ),
                    "source_log": (
                        str(response_row["source_log"])
                        if response_row is not None and response_row["source_log"] is not None
                        else None
                    ),
                }
            )

        prompts_payload.append(
            {
                "prompt_id": prompt_id,
                "prompt_text": str(prompt_row["prompt_text"]),
                "title": _first_metadata_value(metadata, ("title", "name")) or prompt_id,
                "category": _first_metadata_value(
                    metadata,
                    ("category", "task", "domain"),
                ),
                "source": _first_metadata_value(
                    metadata,
                    ("source", "source_file", "dataset"),
                ),
                "metadata": metadata,
                "response_count": response_count,
                "expected_responses": len(ordered_models),
                "responses": responses_payload,
            }
        )

    export_path.write_text(
        json.dumps(
            {
                "project_id": status.project_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_status": status.run_status,
                "total_prompts": len(prompts_payload),
                "total_models": len(ordered_models),
                "prompts": prompts_payload,
            },
            indent=2,
            sort_keys=False,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return PromptResponsesExportResult(
        output_path=export_path,
        prompt_count=len(prompts_payload),
        model_count=len(ordered_models),
    )


@dataclass
class _PairStats:
    wins_low: int = 0
    wins_high: int = 0
    ties: int = 0
    invalid: int = 0


def _pairwise_stats(
    config: TournamentConfig,
    store: TournamentStore,
) -> dict[tuple[str, str], _PairStats]:
    conn = store.connection()
    match_rows = conn.execute(
        """
        SELECT match_id, model_a, model_b
        FROM matches
        WHERE status = 'rated'
        ORDER BY match_id
        """
    ).fetchall()
    judgment_rows = conn.execute(
        """
        SELECT match_id, side, decision
        FROM judgments
        ORDER BY match_id, side
        """
    ).fetchall()

    side_decisions: dict[str, dict[str, Decision]] = defaultdict(dict)
    for row in judgment_rows:
        match_id = str(row["match_id"])
        side = str(row["side"])
        if side not in ("ab", "ba"):
            continue
        side_decisions[match_id][side] = _as_decision(str(row["decision"]))

    pair_stats: dict[tuple[str, str], _PairStats] = {}
    for row in match_rows:
        match_id = str(row["match_id"])
        model_a = str(row["model_a"])
        model_b = str(row["model_b"])
        decision = _canonical_decision_for_match(
            config.side_swap,
            config.invalid_policy,
            side_decisions.get(match_id, {}),
        )

        low, high = (model_a, model_b) if model_a <= model_b else (model_b, model_a)
        stats = pair_stats.get((low, high), _PairStats())

        if decision == "TIE":
            stats.ties += 1
        elif decision == "INVALID":
            stats.invalid += 1
        else:
            winner = model_a if decision == "A" else model_b
            if winner == low:
                stats.wins_low += 1
            else:
                stats.wins_high += 1

        pair_stats[(low, high)] = stats

    return pair_stats


def _canonical_decision_for_match(
    side_swap: bool,
    invalid_policy: InvalidPolicy,
    decisions: dict[str, Decision],
) -> Decision:
    ab = decisions.get("ab")
    ba = decisions.get("ba")
    if side_swap:
        if ab is not None and ba is not None:
            return reconcile_side_swap(ab, ba, invalid_policy=invalid_policy)
        return "INVALID"

    if ab is not None:
        return canonicalize_side_decision(ab, "ab")
    if ba is not None:
        return canonicalize_side_decision(ba, "ba")
    return "INVALID"


def _write_pairwise_matrix_csv(
    path: Path,
    *,
    standings: list[str],
    names_by_id: dict[str, str],
    stats: dict[tuple[str, str], _PairStats],
) -> None:
    columns = [names_by_id.get(model_id, model_id) for model_id in standings]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model_id", "model_name", *columns])
        for row_model in standings:
            row_values: list[str] = []
            for column_model in standings:
                if row_model == column_model:
                    row_values.append("")
                    continue
                low, high = (
                    (row_model, column_model)
                    if row_model <= column_model
                    else (column_model, row_model)
                )
                pair = stats.get((low, high))
                if pair is None:
                    row_values.append("")
                    continue

                total = pair.wins_low + pair.wins_high + pair.ties
                if total == 0:
                    row_values.append("")
                    continue
                if row_model == low:
                    score = (pair.wins_low + (0.5 * pair.ties)) / total
                else:
                    score = (pair.wins_high + (0.5 * pair.ties)) / total
                row_values.append(f"{score:.6f}")

            writer.writerow(
                [row_model, names_by_id.get(row_model, row_model), *row_values]
            )


def _model_names_by_id(store: TournamentStore) -> dict[str, str]:
    return store.active_model_names_by_id()


def _require_state_dir(config: TournamentConfig) -> Path:
    return config.state_dir


def _as_decision(value: str) -> Decision:
    normalized = value.strip().upper()
    if normalized in ("A", "B", "TIE", "INVALID"):
        return normalized  # type: ignore[return-value]
    return "INVALID"


def _parse_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or value.strip() == "":
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _first_metadata_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text != "":
            return text
    return None
