from __future__ import annotations

from pathlib import Path
from typing import Literal

DefinitionKind = Literal["config", "launch_map"]

_CONFIG_FILENAMES = (
    "tournament.yaml",
    "tournament.yml",
    "tournament.json",
)
_LAUNCH_MAP_FILENAMES = (
    "launch-map.yaml",
    "launch-map.yml",
    "launch-map.json",
    "lumi.yaml",
    "lumi.yml",
    "lumi.json",
)


def tournament_definitions_root() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "tournaments"


def list_tournament_definitions(root: str | Path | None = None) -> list[Path]:
    definitions_root = _definitions_root(root)
    if not definitions_root.is_dir():
        return []

    definitions: list[Path] = []
    for candidate in definitions_root.iterdir():
        if not candidate.is_dir():
            continue
        if _find_definition_file(candidate, kind="config") is None:
            continue
        definitions.append(candidate.resolve())
    return sorted(definitions, key=lambda path: path.name)


def resolve_tournament_definition(
    value: str | Path,
    *,
    kind: DefinitionKind,
    root: str | Path | None = None,
) -> Path:
    raw_path = Path(value)
    candidates: list[Path] = []

    if raw_path.exists():
        candidates.append(raw_path)

    if not raw_path.is_absolute():
        definitions_root = _definitions_root(root)
        candidates.append(definitions_root / raw_path)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve() if candidate.exists() else candidate
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)

        resolved = _resolve_candidate(candidate, kind=kind)
        if resolved is not None:
            return resolved

    label = "config" if kind == "config" else "launch-map"
    raise FileNotFoundError(
        f"Tournament {label} definition not found for `{value}`"
    )


def _definitions_root(root: str | Path | None) -> Path:
    if root is None:
        return tournament_definitions_root()
    return Path(root).resolve()


def _resolve_candidate(candidate: Path, *, kind: DefinitionKind) -> Path | None:
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_dir():
        return None

    definition_file = _find_definition_file(candidate, kind=kind)
    if definition_file is None:
        return None
    return definition_file.resolve()


def _find_definition_file(directory: Path, *, kind: DefinitionKind) -> Path | None:
    filenames = _CONFIG_FILENAMES if kind == "config" else _LAUNCH_MAP_FILENAMES
    for filename in filenames:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None
