from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from ._resolve import resolve_stateful_tournament_config
from ._viewer_assets import VIEWER_CSS, VIEWER_HTML, VIEWER_JS
from .config import TournamentConfig
from .exports import load_export_snapshot
from .orchestrator import tournament_status
from .scorer import canonicalize_side_decision, reconcile_side_swap
from .store import TournamentStore
from .types import Decision, InvalidPolicy

DEFAULT_TOURNAMENT_VIEW_LOG_ROOT = Path("logs/evals-logs")


@dataclass(frozen=True)
class TournamentViewRun:
    """Discovered tournament run available for viewing."""

    run_label: str
    run_dir: Path
    state_dir: Path
    updated_at: float


@dataclass(frozen=True)
class TournamentViewExportResult:
    """Generated standalone HTML viewer artifact."""

    output_html: Path
    project_id: str | None
    run_label: str
    total_models: int
    total_prompts: int
    total_matches: int


class TournamentViewDataSource:
    """Read-only payload builder for the hosted tournament viewer."""

    def __init__(self, target: TournamentConfig | Mapping[str, Any] | str | Path):
        self._target = target

    def summary(self) -> dict[str, Any]:
        config = self._resolve_config()
        status = tournament_status(config)
        with TournamentStore(_require_state_dir(config)) as store:
            conn = store.connection()
            row = conn.execute(
                """
                SELECT
                  (SELECT COUNT(DISTINCT batch_id)
                   FROM matches m
                   JOIN models ma ON ma.model_id = m.model_a
                   JOIN models mb ON mb.model_id = m.model_b
                   WHERE ma.active = 1
                     AND mb.active = 1) AS batch_count,
                  (SELECT COUNT(*)
                   FROM judgments j
                   JOIN matches m ON m.match_id = j.match_id
                   JOIN models ma ON ma.model_id = m.model_a
                   JOIN models mb ON mb.model_id = m.model_b
                   WHERE ma.active = 1
                     AND mb.active = 1) AS judgment_count,
                  (SELECT MAX(event_time)
                   FROM (
                     SELECT MAX(updated_at) AS event_time FROM run_state
                     UNION ALL
                     SELECT MAX(created_at) AS event_time FROM responses
                     UNION ALL
                     SELECT MAX(scheduled_at) AS event_time FROM matches
                     UNION ALL
                     SELECT MAX(judged_at) AS event_time FROM judgments
                     UNION ALL
                     SELECT MAX(updated_at) AS event_time FROM ratings
                   )) AS last_updated_at
                """
            ).fetchone()

        coverage = (
            status.response_count / status.expected_responses
            if status.expected_responses > 0
            else None
        )
        return {
            "project_id": status.project_id,
            "run_label": config.run_label,
            "run_status": status.run_status,
            "converged": status.converged,
            "stop_reasons": status.stop_reasons,
            "judge_model": config.judge_model,
            "total_models": status.total_models,
            "total_prompts": status.total_prompts,
            "response_count": status.response_count,
            "expected_responses": status.expected_responses,
            "response_coverage": coverage,
            "total_matches": status.total_matches,
            "rated_matches": status.rated_matches,
            "judged_matches": status.judged_matches,
            "scheduled_matches": status.scheduled_matches,
            "batch_count": int(row["batch_count"] or 0) if row is not None else 0,
            "judgment_count": int(row["judgment_count"] or 0) if row is not None else 0,
            "last_updated_at": (
                str(row["last_updated_at"]) if row is not None and row["last_updated_at"] else None
            ),
            "next_round_index": status.next_round_index,
            "pending_batch_id": status.pending_batch_id,
            "stable_batches": status.stable_batches,
            "min_adjacent_probability": status.min_adjacent_probability,
            "min_adjacent_margin": status.min_adjacent_margin,
            "run_dir": config.run_dir.as_posix(),
            "state_dir": _require_state_dir(config).as_posix(),
        }

    def standings(self) -> dict[str, Any]:
        status = tournament_status(self._resolve_config())
        items = [
            {
                "rank": rank,
                "model_id": standing.model_id,
                "model_name": standing.model_name or standing.model_id,
                "mu": standing.mu,
                "sigma": standing.sigma,
                "conservative": standing.conservative,
                "elo_like": standing.elo_like,
                "games": standing.games,
                "wins": standing.wins,
                "losses": standing.losses,
                "ties": standing.ties,
                "win_rate": (
                    (standing.wins + (0.5 * standing.ties)) / standing.games
                    if standing.games > 0
                    else None
                ),
            }
            for rank, standing in enumerate(status.standings, start=1)
        ]
        return {"items": items}

    def pairwise(self) -> dict[str, Any]:
        snapshot = load_export_snapshot(self._resolve_config(), include_pairwise_matrix=True)
        models = [
            {
                "rank": rank,
                "model_id": standing.model_id,
                "model_name": snapshot.names_by_id.get(standing.model_id, standing.model_id),
                "conservative": standing.conservative,
                "elo_like": standing.elo_like,
                "games": standing.games,
            }
            for rank, standing in enumerate(snapshot.status.standings, start=1)
        ]

        rows: list[dict[str, Any]] = []
        for row_model in models:
            cells: list[dict[str, Any] | None] = []
            for column_model in models:
                if row_model["model_id"] == column_model["model_id"]:
                    cells.append(None)
                    continue

                low, high = _ordered_pair(
                    str(row_model["model_id"]),
                    str(column_model["model_id"]),
                )
                stats = snapshot.pairwise_stats.get((low, high))
                if stats is None:
                    cells.append(
                        {
                            "opponent_model_id": column_model["model_id"],
                            "opponent_model_name": column_model["model_name"],
                            "has_data": False,
                            "wins": 0,
                            "losses": 0,
                            "ties": 0,
                            "invalid": 0,
                            "rated_games": 0,
                            "total_games": 0,
                            "score": None,
                        }
                    )
                    continue

                rated_games = stats.wins_low + stats.wins_high + stats.ties
                total_games = rated_games + stats.invalid
                if row_model["model_id"] == low:
                    wins = stats.wins_low
                    losses = stats.wins_high
                else:
                    wins = stats.wins_high
                    losses = stats.wins_low
                score = ((wins + (0.5 * stats.ties)) / rated_games) if rated_games > 0 else None
                cells.append(
                    {
                        "opponent_model_id": column_model["model_id"],
                        "opponent_model_name": column_model["model_name"],
                        "has_data": total_games > 0,
                        "wins": wins,
                        "losses": losses,
                        "ties": stats.ties,
                        "invalid": stats.invalid,
                        "rated_games": rated_games,
                        "total_games": total_games,
                        "score": score,
                    }
                )

            rows.append(
                {
                    "model_id": row_model["model_id"],
                    "model_name": row_model["model_name"],
                    "rank": row_model["rank"],
                    "cells": cells,
                }
            )

        return {"models": models, "rows": rows}

    def list_matches(self, filters: Mapping[str, str] | None = None) -> dict[str, Any]:
        config = self._resolve_config()
        filters = filters or {}
        limit = _parse_positive_int(filters.get("limit"), default=100)
        offset = _parse_non_negative_int(filters.get("offset"), default=0)
        records = self._load_match_summaries(config)
        filtered = [
            record
            for record in records
            if _match_filters(record, filters)
        ]
        return {
            "items": filtered[offset : offset + limit],
            "total": len(filtered),
            "limit": limit,
            "offset": offset,
        }

    def match_detail(self, match_id: str) -> dict[str, Any]:
        config = self._resolve_config()
        with TournamentStore(_require_state_dir(config)) as store:
            conn = store.connection()
            row = conn.execute(
                """
                SELECT
                  m.match_id,
                  m.batch_id,
                  m.round_index,
                  m.status,
                  p.prompt_id,
                  p.prompt_text,
                  p.metadata_json,
                  ma.model_id AS model_a_id,
                  ma.model_name AS model_a_name,
                  mb.model_id AS model_b_id,
                  mb.model_name AS model_b_name,
                  ra.response_text AS response_a_text,
                  ra.source_log AS response_a_log,
                  rb.response_text AS response_b_text,
                  rb.source_log AS response_b_log,
                  jab.decision AS decision_ab,
                  jab.explanation AS explanation_ab,
                  jab.raw_completion AS raw_ab,
                  jab.judge_model AS judge_model_ab,
                  jab.source_log AS judge_log_ab,
                  jba.decision AS decision_ba,
                  jba.explanation AS explanation_ba,
                  jba.raw_completion AS raw_ba,
                  jba.judge_model AS judge_model_ba,
                  jba.source_log AS judge_log_ba
                FROM matches m
                JOIN prompts p ON p.prompt_id = m.prompt_id
                JOIN models ma ON ma.model_id = m.model_a
                JOIN models mb ON mb.model_id = m.model_b
                JOIN responses ra ON ra.response_id = m.response_a_id
                JOIN responses rb ON rb.response_id = m.response_b_id
                LEFT JOIN judgments jab ON jab.match_id = m.match_id AND jab.side = 'ab'
                LEFT JOIN judgments jba ON jba.match_id = m.match_id AND jba.side = 'ba'
                WHERE m.match_id = ?
                """,
                (match_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"Unknown match id: {match_id}")

        record = _serialize_match_row(config, row, include_responses=True)
        return record

    def list_prompts(self) -> dict[str, Any]:
        config = self._resolve_config()
        with TournamentStore(_require_state_dir(config)) as store:
            conn = store.connection()
            rows = conn.execute(
                """
                SELECT
                  p.prompt_id,
                  p.prompt_text,
                  p.metadata_json,
                  COUNT(DISTINCT CASE WHEN r.current = 1 AND m.active = 1 THEN r.response_id END)
                    AS response_count,
                  COUNT(DISTINCT CASE WHEN ma.active = 1 AND mb.active = 1 THEN mt.match_id END)
                    AS match_count,
                  COUNT(DISTINCT CASE
                    WHEN ma.active = 1 AND mb.active = 1 AND mt.status = 'rated'
                    THEN mt.match_id
                  END) AS rated_match_count
                FROM prompts p
                LEFT JOIN responses r ON r.prompt_id = p.prompt_id
                LEFT JOIN models m ON m.model_id = r.model_id
                LEFT JOIN matches mt ON mt.prompt_id = p.prompt_id
                LEFT JOIN models ma ON ma.model_id = mt.model_a
                LEFT JOIN models mb ON mb.model_id = mt.model_b
                GROUP BY p.prompt_id, p.prompt_text, p.metadata_json
                ORDER BY p.prompt_id
                """
            ).fetchall()

        items = [_serialize_prompt_row(row, expected_responses=len(config.contestant_models)) for row in rows]
        items.sort(key=lambda item: (item["category"] or "", item["title"] or item["prompt_id"]))
        return {"items": items}

    def prompt_detail(self, prompt_id: str) -> dict[str, Any]:
        config = self._resolve_config()
        with TournamentStore(_require_state_dir(config)) as store:
            conn = store.connection()
            prompt_row = conn.execute(
                """
                SELECT prompt_id, prompt_text, metadata_json
                FROM prompts
                WHERE prompt_id = ?
                """,
                (prompt_id,),
            ).fetchone()
            if prompt_row is None:
                raise KeyError(f"Unknown prompt id: {prompt_id}")

            response_rows = conn.execute(
                """
                SELECT
                  m.model_id,
                  m.model_name,
                  r.response_id,
                  r.response_text,
                  r.source_log
                FROM models m
                LEFT JOIN responses r
                  ON r.model_id = m.model_id
                 AND r.prompt_id = ?
                 AND r.current = 1
                WHERE m.active = 1
                ORDER BY m.model_name
                """,
                (prompt_id,),
            ).fetchall()

        prompt = next(
            (
                item
                for item in self.list_prompts()["items"]
                if item["prompt_id"] == prompt_id
            ),
            _serialize_prompt_row(
                prompt_row,
                expected_responses=len(config.contestant_models),
            ),
        )
        prompt_matches = self.list_matches({"prompt_id": prompt_id, "limit": "1000"})["items"]
        prompt["matches"] = prompt_matches
        prompt["responses"] = [
            {
                "model_id": str(row["model_id"]),
                "model_name": str(row["model_name"]),
                "response_id": str(row["response_id"]) if row["response_id"] is not None else None,
                "response_text": str(row["response_text"]) if row["response_text"] is not None else None,
                "source_log": str(row["source_log"]) if row["source_log"] is not None else None,
            }
            for row in response_rows
        ]
        return prompt

    def list_models(self) -> dict[str, Any]:
        config = self._resolve_config()
        standings = self.standings()["items"]
        with TournamentStore(_require_state_dir(config)) as store:
            rows = store.connection().execute(
                """
                SELECT
                  m.model_id,
                  COUNT(CASE WHEN r.current = 1 THEN r.response_id END) AS response_count
                FROM models m
                LEFT JOIN responses r ON r.model_id = m.model_id
                WHERE m.active = 1
                GROUP BY m.model_id
                """
            ).fetchall()
        response_counts = {
            str(row["model_id"]): int(row["response_count"] or 0)
            for row in rows
        }
        items = []
        for standing in standings:
            count = response_counts.get(str(standing["model_id"]), 0)
            items.append(
                {
                    **standing,
                    "response_count": count,
                    "expected_responses": len(config.prompts),
                    "response_coverage": (count / len(config.prompts)) if len(config.prompts) > 0 else None,
                }
            )
        return {"items": items}

    def model_detail(self, model_id: str) -> dict[str, Any]:
        config = self._resolve_config()
        models = self.list_models()["items"]
        details = next((item for item in models if item["model_id"] == model_id), None)
        if details is None:
            raise KeyError(f"Unknown model id: {model_id}")

        with TournamentStore(_require_state_dir(config)) as store:
            conn = store.connection()
            response_rows = conn.execute(
                """
                SELECT
                  p.prompt_id,
                  p.prompt_text,
                  p.metadata_json,
                  r.response_text,
                  r.source_log
                FROM prompts p
                LEFT JOIN responses r
                  ON r.prompt_id = p.prompt_id
                 AND r.model_id = ?
                 AND r.current = 1
                ORDER BY p.prompt_id
                """,
                (model_id,),
            ).fetchall()

        pairwise = self.pairwise()
        row = next((item for item in pairwise["rows"] if item["model_id"] == model_id), None)
        details["pairwise"] = []
        if row is not None:
            for cell in row["cells"]:
                if cell is None:
                    continue
                details["pairwise"].append(cell)

        details["responses"] = []
        for row_data in response_rows:
            prompt = _serialize_prompt_row(row_data, expected_responses=len(config.contestant_models))
            details["responses"].append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "title": prompt["title"],
                    "category": prompt["category"],
                    "source": prompt["source"],
                    "prompt_text": prompt["prompt_text"],
                    "response_text": (
                        str(row_data["response_text"])
                        if row_data["response_text"] is not None
                        else None
                    ),
                    "source_log": (
                        str(row_data["source_log"]) if row_data["source_log"] is not None else None
                    ),
                }
            )
        return details

    def _resolve_config(self) -> TournamentConfig:
        return resolve_stateful_tournament_config(self._target)

    def _load_match_summaries(self, config: TournamentConfig) -> list[dict[str, Any]]:
        with TournamentStore(_require_state_dir(config)) as store:
            rows = store.connection().execute(
                """
                SELECT
                  m.match_id,
                  m.batch_id,
                  m.round_index,
                  m.status,
                  p.prompt_id,
                  p.prompt_text,
                  p.metadata_json,
                  ma.model_id AS model_a_id,
                  ma.model_name AS model_a_name,
                  mb.model_id AS model_b_id,
                  mb.model_name AS model_b_name,
                  jab.decision AS decision_ab,
                  jab.explanation AS explanation_ab,
                  jab.raw_completion AS raw_ab,
                  jab.judge_model AS judge_model_ab,
                  jab.source_log AS judge_log_ab,
                  jba.decision AS decision_ba,
                  jba.explanation AS explanation_ba,
                  jba.raw_completion AS raw_ba,
                  jba.judge_model AS judge_model_ba,
                  jba.source_log AS judge_log_ba
                FROM matches m
                JOIN prompts p ON p.prompt_id = m.prompt_id
                JOIN models ma ON ma.model_id = m.model_a
                JOIN models mb ON mb.model_id = m.model_b
                LEFT JOIN judgments jab ON jab.match_id = m.match_id AND jab.side = 'ab'
                LEFT JOIN judgments jba ON jba.match_id = m.match_id AND jba.side = 'ba'
                WHERE ma.active = 1
                  AND mb.active = 1
                ORDER BY m.round_index DESC, m.match_id DESC
                """
            ).fetchall()
        return [_serialize_match_row(config, row, include_responses=False) for row in rows]


@dataclass
class TournamentViewServer:
    """Configured HTTP server for a tournament viewer instance."""

    httpd: ThreadingHTTPServer
    host: str

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def list_tournament_view_runs(
    *,
    log_root: str | Path | None = None,
) -> list[TournamentViewRun]:
    """Discover tournament runs under a log root, newest first."""
    root = Path(log_root) if log_root is not None else DEFAULT_TOURNAMENT_VIEW_LOG_ROOT
    if not root.exists() or not root.is_dir():
        return []

    discovered: list[TournamentViewRun] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        state_dir = entry / "state"
        db_path = state_dir / "tournament.db"
        if not db_path.is_file():
            continue
        updated_at = max(
            entry.stat().st_mtime,
            state_dir.stat().st_mtime if state_dir.exists() else 0.0,
            db_path.stat().st_mtime,
        )
        discovered.append(
            TournamentViewRun(
                run_label=entry.name,
                run_dir=entry,
                state_dir=state_dir,
                updated_at=updated_at,
            )
        )

    discovered.sort(key=lambda run: (run.updated_at, run.run_label), reverse=True)
    return discovered


def format_tournament_view_runs(
    runs: Sequence[TournamentViewRun],
    *,
    log_root: str | Path | None = None,
) -> str:
    """Format discovered tournament runs for terminal output."""
    root = Path(log_root) if log_root is not None else DEFAULT_TOURNAMENT_VIEW_LOG_ROOT
    lines = [f"Tournament runs under {root.as_posix()}:"]
    if len(runs) == 0:
        lines.append("(none)")
        return "\n".join(lines)

    for run in runs:
        lines.append(f"- {run.run_label}")
        lines.append(f"  run: {run.run_dir.as_posix()}")
        lines.append(f"  state: {run.state_dir.as_posix()}")
    return "\n".join(lines)


def resolve_tournament_view_target(
    target: TournamentConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    latest: bool = False,
    run_label: str | None = None,
    job_id: str | int | None = None,
    log_root: str | Path | None = None,
) -> TournamentConfig | Mapping[str, Any] | str | Path:
    """Resolve a convenient viewer target into config/run input."""
    if isinstance(target, TournamentConfig):
        return target
    if isinstance(target, Mapping):
        return target

    target_text = str(target).strip() if target is not None else ""
    selector_count = sum(
        1
        for present in (
            bool(target_text),
            latest,
            run_label is not None and str(run_label).strip() != "",
            job_id is not None and str(job_id).strip() != "",
        )
        if present
    )
    if selector_count > 1:
        raise ValueError(
            "Specify only one of target, --latest, --label, or --job-id for tournament view."
        )

    if target_text != "":
        candidate_path = Path(target_text)
        if candidate_path.exists():
            return candidate_path

        discovered = list_tournament_view_runs(log_root=log_root)
        if target_text.isdigit():
            return _resolve_view_run_by_job_id(discovered, target_text).run_dir

        matched_label = _maybe_run_by_label(discovered, target_text)
        if matched_label is not None:
            return matched_label.run_dir

        return target_text

    discovered = list_tournament_view_runs(log_root=log_root)
    if run_label is not None and str(run_label).strip() != "":
        return _resolve_view_run_by_label(discovered, str(run_label).strip()).run_dir

    if job_id is not None and str(job_id).strip() != "":
        return _resolve_view_run_by_job_id(discovered, str(job_id).strip()).run_dir

    if latest or target is None:
        if len(discovered) == 0:
            root = Path(log_root) if log_root is not None else DEFAULT_TOURNAMENT_VIEW_LOG_ROOT
            raise FileNotFoundError(
                f"No tournament runs found under {root.as_posix()}."
            )
        return discovered[0].run_dir

    raise FileNotFoundError("Unable to resolve tournament view target.")


def create_tournament_view_server(
    target: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7576,
) -> TournamentViewServer:
    source = TournamentViewDataSource(target)
    handler = _make_handler(source)
    httpd = ThreadingHTTPServer((host, port), handler)
    return TournamentViewServer(httpd=httpd, host=host)


def serve_tournament_view(
    target: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7576,
) -> int:
    server = create_tournament_view_server(target, host=host, port=port)
    print(f"Tournament viewer: {server.url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


def export_tournament_view_html(
    target: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    output_path: str | Path | None = None,
) -> TournamentViewExportResult:
    """Export the hosted tournament viewer as a single standalone HTML file."""
    source = TournamentViewDataSource(target)
    payload = _build_static_view_payload(source)
    summary = payload["summary"]
    config = source._resolve_config()

    output_html = (
        Path(output_path)
        if output_path is not None
        else config.exports_dir / "viewer.html"
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_render_static_view_html(payload), encoding="utf-8")

    return TournamentViewExportResult(
        output_html=output_html,
        project_id=summary.get("project_id"),
        run_label=str(summary.get("run_label") or config.run_label),
        total_models=int(summary.get("total_models") or 0),
        total_prompts=int(summary.get("total_prompts") or 0),
        total_matches=int(summary.get("total_matches") or 0),
    )


def _make_handler(source: TournamentViewDataSource) -> type[BaseHTTPRequestHandler]:
    class TournamentViewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            params = {
                key: values[-1]
                for key, values in parse_qs(parsed.query, keep_blank_values=False).items()
                if len(values) > 0
            }

            try:
                if path == "/":
                    self._send_text(VIEWER_HTML, content_type="text/html; charset=utf-8")
                    return
                if path == "/app.css":
                    self._send_text(VIEWER_CSS, content_type="text/css; charset=utf-8")
                    return
                if path == "/app.js":
                    self._send_text(
                        VIEWER_JS,
                        content_type="application/javascript; charset=utf-8",
                    )
                    return
                if path == "/api/summary":
                    self._send_json(source.summary())
                    return
                if path == "/api/standings":
                    self._send_json(source.standings())
                    return
                if path == "/api/pairwise":
                    self._send_json(source.pairwise())
                    return
                if path == "/api/matches":
                    self._send_json(source.list_matches(params))
                    return
                if path == "/api/match":
                    match_id = _require_query_param(params, "match_id")
                    self._send_json(source.match_detail(match_id))
                    return
                if path == "/api/prompts":
                    self._send_json(source.list_prompts())
                    return
                if path == "/api/prompt":
                    prompt_id = _require_query_param(params, "prompt_id")
                    self._send_json(source.prompt_detail(prompt_id))
                    return
                if path == "/api/models":
                    self._send_json(source.list_models())
                    return
                if path == "/api/model":
                    model_id = _require_query_param(params, "model_id")
                    self._send_json(source.model_detail(model_id))
                    return

                self._send_json(
                    {"error": f"Unknown route: {path}"},
                    status=HTTPStatus.NOT_FOUND,
                )
            except KeyError as ex:
                self._send_json({"error": str(ex)}, status=HTTPStatus.NOT_FOUND)
            except ValueError as ex:
                self._send_json({"error": str(ex)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as ex:  # pragma: no cover - defensive fallback
                self._send_json(
                    {"error": f"Internal viewer error: {ex}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(body)

        def _send_text(
            self,
            payload: str,
            *,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(body)

    return TournamentViewHandler


def _build_static_view_payload(source: TournamentViewDataSource) -> dict[str, Any]:
    summary = source.summary()
    standings = source.standings()
    pairwise = source.pairwise()
    prompts = source.list_prompts()
    models = source.list_models()
    matches = source.list_matches({"limit": "1000000000"})

    return {
        "summary": summary,
        "standings": standings,
        "pairwise": pairwise,
        "prompts": prompts,
        "models": models,
        "matches": matches,
        "prompt_details": {
            str(prompt["prompt_id"]): source.prompt_detail(str(prompt["prompt_id"]))
            for prompt in prompts["items"]
        },
        "model_details": {
            str(model["model_id"]): source.model_detail(str(model["model_id"]))
            for model in models["items"]
        },
        "match_details": {
            str(match["match_id"]): source.match_detail(str(match["match_id"]))
            for match in matches["items"]
        },
    }


def _render_static_view_html(payload: Mapping[str, Any]) -> str:
    data_script = (
        "<script>\n"
        f"window.__TOURNAMENT_STATIC_DATA__ = {_html_safe_json(payload)};\n"
        "</script>"
    )
    inline_script = f"<script>\n{VIEWER_JS}\n</script>"
    return (
        VIEWER_HTML.replace(
            '<link rel="stylesheet" href="/app.css">',
            f"<style>\n{VIEWER_CSS}\n</style>",
        )
        .replace(
            '<script src="/app.js"></script>',
            f"{data_script}\n{inline_script}",
        )
    )


def _html_safe_json(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _maybe_run_by_label(
    runs: Sequence[TournamentViewRun],
    run_label: str,
) -> TournamentViewRun | None:
    for run in runs:
        if run.run_label == run_label:
            return run
    return None


def _resolve_view_run_by_label(
    runs: Sequence[TournamentViewRun],
    run_label: str,
) -> TournamentViewRun:
    matched = _maybe_run_by_label(runs, run_label)
    if matched is None:
        raise FileNotFoundError(f"No tournament run found for label `{run_label}`.")
    return matched


def _resolve_view_run_by_job_id(
    runs: Sequence[TournamentViewRun],
    job_id: str,
) -> TournamentViewRun:
    suffix = f"__job-{job_id}"
    for run in runs:
        if run.run_label.endswith(suffix):
            return run
    raise FileNotFoundError(f"No tournament run found for job id `{job_id}`.")


def _serialize_prompt_row(row: Any, *, expected_responses: int) -> dict[str, Any]:
    metadata = _parse_metadata(row["metadata_json"] if "metadata_json" in row.keys() else None)
    prompt_id = str(row["prompt_id"])
    prompt_text = str(row["prompt_text"])
    return {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "title": _first_metadata_value(metadata, ("title", "name")) or prompt_id,
        "category": _first_metadata_value(metadata, ("category", "task", "domain")),
        "source": _first_metadata_value(metadata, ("source", "source_file", "dataset")),
        "metadata": metadata,
        "response_count": int(row["response_count"] or 0) if "response_count" in row.keys() else 0,
        "expected_responses": expected_responses,
        "match_count": int(row["match_count"] or 0) if "match_count" in row.keys() else 0,
        "rated_match_count": (
            int(row["rated_match_count"] or 0)
            if "rated_match_count" in row.keys()
            else 0
        ),
    }


def _serialize_match_row(
    config: TournamentConfig,
    row: Any,
    *,
    include_responses: bool,
) -> dict[str, Any]:
    metadata = _parse_metadata(row["metadata_json"])
    decision_ab = _as_decision(row["decision_ab"])
    decision_ba = _as_decision(row["decision_ba"])
    canonical = _canonical_decision(
        side_swap=config.side_swap,
        invalid_policy=config.invalid_policy,
        ab=decision_ab,
        ba=decision_ba,
    )

    winner_model_id: str | None
    winner_model_name: str | None
    if canonical == "A":
        winner_model_id = str(row["model_a_id"])
        winner_model_name = str(row["model_a_name"])
    elif canonical == "B":
        winner_model_id = str(row["model_b_id"])
        winner_model_name = str(row["model_b_name"])
    else:
        winner_model_id = None
        winner_model_name = None

    explanation_preview = _truncate(
        str(row["explanation_ab"] or row["explanation_ba"] or row["raw_ab"] or row["raw_ba"] or ""),
        length=220,
    )
    record = {
        "match_id": str(row["match_id"]),
        "batch_id": str(row["batch_id"]),
        "round_index": int(row["round_index"]),
        "status": str(row["status"]),
        "prompt_id": str(row["prompt_id"]),
        "prompt_title": _first_metadata_value(metadata, ("title", "name")) or str(row["prompt_id"]),
        "prompt_text": str(row["prompt_text"]),
        "category": _first_metadata_value(metadata, ("category", "task", "domain")),
        "source": _first_metadata_value(metadata, ("source", "source_file", "dataset")),
        "metadata": metadata,
        "model_a_id": str(row["model_a_id"]),
        "model_a_name": str(row["model_a_name"]),
        "model_b_id": str(row["model_b_id"]),
        "model_b_name": str(row["model_b_name"]),
        "canonical_decision": canonical,
        "winner_model_id": winner_model_id,
        "winner_model_name": winner_model_name,
        "judge_model": (
            str(row["judge_model_ab"])
            if row["judge_model_ab"] is not None
            else (str(row["judge_model_ba"]) if row["judge_model_ba"] is not None else None)
        ),
        "explanation_preview": explanation_preview if explanation_preview != "" else None,
        "judgments": {
            "ab": _serialize_judgment_side(
                decision=decision_ab,
                judge_model=row["judge_model_ab"],
                explanation=row["explanation_ab"],
                raw_completion=row["raw_ab"],
                source_log=row["judge_log_ab"],
            ),
            "ba": _serialize_judgment_side(
                decision=decision_ba,
                judge_model=row["judge_model_ba"],
                explanation=row["explanation_ba"],
                raw_completion=row["raw_ba"],
                source_log=row["judge_log_ba"],
            ),
        },
    }
    if include_responses:
        record["response_a_text"] = str(row["response_a_text"])
        record["response_b_text"] = str(row["response_b_text"])
        record["response_a_log"] = (
            str(row["response_a_log"]) if row["response_a_log"] is not None else None
        )
        record["response_b_log"] = (
            str(row["response_b_log"]) if row["response_b_log"] is not None else None
        )
    return record


def _serialize_judgment_side(
    *,
    decision: Decision | None,
    judge_model: Any,
    explanation: Any,
    raw_completion: Any,
    source_log: Any,
) -> dict[str, Any] | None:
    if decision is None and judge_model is None and explanation is None and raw_completion is None:
        return None
    return {
        "decision": decision,
        "judge_model": str(judge_model) if judge_model is not None else None,
        "explanation": str(explanation) if explanation is not None else None,
        "raw_completion": str(raw_completion) if raw_completion is not None else None,
        "source_log": str(source_log) if source_log is not None else None,
    }


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


def _first_metadata_value(metadata: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text != "":
            return text
    return None


def _canonical_decision(
    *,
    side_swap: bool,
    invalid_policy: InvalidPolicy,
    ab: Decision | None,
    ba: Decision | None,
) -> Decision:
    if side_swap:
        if ab is not None and ba is not None:
            return reconcile_side_swap(ab, ba, invalid_policy=invalid_policy)
        if ab is not None:
            return canonicalize_side_decision(ab, "ab")
        if ba is not None:
            return canonicalize_side_decision(ba, "ba")
        return "INVALID"

    if ab is not None:
        return canonicalize_side_decision(ab, "ab")
    if ba is not None:
        return canonicalize_side_decision(ba, "ba")
    return "INVALID"


def _as_decision(value: Any) -> Decision | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized in ("A", "B", "TIE", "INVALID"):
        return normalized  # type: ignore[return-value]
    return "INVALID"


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _match_filters(record: Mapping[str, Any], filters: Mapping[str, str]) -> bool:
    model_filter = filters.get("model", "").strip()
    if model_filter and model_filter not in (
        record["model_a_id"],
        record["model_b_id"],
        record["model_a_name"],
        record["model_b_name"],
    ):
        return False

    opponent_filter = filters.get("opponent", "").strip()
    if opponent_filter and opponent_filter not in (
        record["model_a_id"],
        record["model_b_id"],
        record["model_a_name"],
        record["model_b_name"],
    ):
        return False

    prompt_filter = filters.get("prompt_id", "").strip()
    if prompt_filter and prompt_filter != record["prompt_id"]:
        return False

    category_filter = filters.get("category", "").strip()
    if category_filter and category_filter != (record.get("category") or ""):
        return False

    winner_filter = filters.get("winner", "").strip()
    if winner_filter:
        if winner_filter in ("TIE", "INVALID"):
            if record["canonical_decision"] != winner_filter:
                return False
        elif winner_filter not in (
            record.get("winner_model_id"),
            record.get("winner_model_name"),
        ):
            return False

    outcome_filter = filters.get("outcome", "").strip()
    if outcome_filter == "decisive" and record["canonical_decision"] not in ("A", "B"):
        return False
    if outcome_filter == "tie" and record["canonical_decision"] != "TIE":
        return False
    if outcome_filter == "invalid" and record["canonical_decision"] != "INVALID":
        return False

    batch_filter = filters.get("batch_id", "").strip()
    if batch_filter and batch_filter != record["batch_id"]:
        return False

    return True


def _truncate(value: str, *, length: int) -> str:
    if len(value) <= length:
        return value
    return value[: max(0, length - 1)].rstrip() + "..."


def _parse_positive_int(value: str | None, *, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _parse_non_negative_int(value: str | None, *, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _require_query_param(params: Mapping[str, str], key: str) -> str:
    value = params.get(key)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required query parameter: {key}")
    return value


def _require_state_dir(config: TournamentConfig) -> Path:
    return config.state_dir
