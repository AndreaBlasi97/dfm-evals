import sqlite3
from pathlib import Path
from typing import Any, Mapping

from ._definitions import resolve_tournament_definition
from .config import TournamentConfig, load_tournament_config


def resolve_tournament_config(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
) -> TournamentConfig:
    """Resolve config from config object, run path, state path, or config file.

    Resolution is explicit:
    - config mappings / TournamentConfig instances are validated directly
    - run/state paths (`*.db` or existing directories) load persisted `config_json`
    - regular files load via `load_tournament_config`
    """
    if isinstance(config_or_state, TournamentConfig):
        return config_or_state
    if isinstance(config_or_state, Mapping):
        return TournamentConfig.model_validate(dict(config_or_state))

    path = Path(config_or_state)
    if path.exists() and path.is_dir():
        config = _maybe_load_definition_config(path)
        if config is not None:
            return config
        return _load_state_config(path)

    if path.suffix == ".db":
        return _load_state_config(path)

    if path.exists() and path.is_file():
        return load_tournament_config(path)

    return load_tournament_config(path)


def resolve_stateful_tournament_config(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
) -> TournamentConfig:
    """Resolve config for operations that should honor persisted tournament state.

    Stateful operations such as resume/status/export/add-model should use the
    persisted `config_json` when it exists for the resolved run/state directory, even if
    the caller passed the original config file path.
    """
    if isinstance(config_or_state, TournamentConfig):
        return _prefer_persisted_state_config(config_or_state)
    if isinstance(config_or_state, Mapping):
        return _prefer_persisted_state_config(
            TournamentConfig.model_validate(dict(config_or_state))
        )

    path = Path(config_or_state)
    if path.exists() and path.is_dir():
        config = _maybe_load_definition_config(path)
        if config is not None:
            return _prefer_persisted_state_config(config)
        return _load_state_config(path)

    if path.suffix == ".db":
        return _load_state_config(path)

    if path.exists() and path.is_file():
        return _prefer_persisted_state_config(load_tournament_config(path))

    return _prefer_persisted_state_config(load_tournament_config(path))


def _load_state_config(path: Path) -> TournamentConfig:
    config_json = _state_config_json(path)
    if config_json is None or config_json.strip() == "":
        runtime_config = _maybe_load_runtime_config(path)
        if runtime_config is not None:
            return runtime_config
        raise ValueError(
            f"No persisted config_json found in tournament state at: {path}"
        )
    try:
        return TournamentConfig.model_validate_json(config_json)
    except Exception as state_error:
        raise ValueError(
            f"Persisted config_json in tournament state is invalid at: {path}"
        ) from state_error


def _prefer_persisted_state_config(config: TournamentConfig) -> TournamentConfig:
    persisted = _maybe_load_state_config(config.run_dir)
    return persisted if persisted is not None else config


def _maybe_load_state_config(path: Path) -> TournamentConfig | None:
    config_json = _state_config_json(path)
    if config_json is None or config_json.strip() == "":
        return _maybe_load_runtime_config(path)
    try:
        return TournamentConfig.model_validate_json(config_json)
    except Exception as state_error:
        raise ValueError(
            f"Persisted config_json in tournament state is invalid at: {path}"
        ) from state_error


def _state_config_json(path: Path) -> str | None:
    db_path = _resolve_state_db_path(path)
    if not db_path.exists() or not db_path.is_file():
        return None

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT value FROM run_state WHERE key = ?",
            ("config_json",),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    if row is None:
        return None
    return str(row[0])


def _maybe_load_runtime_config(path: Path) -> TournamentConfig | None:
    runtime_path = _runtime_config_path(path)
    if runtime_path is None or not runtime_path.is_file():
        return None
    return load_tournament_config(runtime_path)


def _runtime_config_path(path: Path) -> Path | None:
    if path.is_file():
        if path.name == "runtime.json" and path.parent.name == "config":
            return path
        if path.suffix == ".db" and path.parent.name == "state":
            candidate = path.parent.parent / "config" / "runtime.json"
            return candidate if candidate.is_file() else None
        return None

    if path.name == "state":
        candidate = path.parent / "config" / "runtime.json"
        return candidate if candidate.is_file() else None

    candidate = path / "config" / "runtime.json"
    return candidate if candidate.is_file() else None


def _maybe_load_definition_config(path: Path) -> TournamentConfig | None:
    try:
        definition_path = resolve_tournament_definition(path, kind="config")
    except FileNotFoundError:
        return None
    return load_tournament_config(definition_path)


def _resolve_state_db_path(path: Path) -> Path:
    if path.suffix == ".db":
        return path
    run_db_path = path / "state" / "tournament.db"
    if run_db_path.exists() or (path.exists() and (path / "state").is_dir()):
        return run_db_path
    return path / "tournament.db"
