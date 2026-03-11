import json
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, Field, JsonValue, model_validator
from typing_extensions import Self

from ._definitions import resolve_tournament_definition
from .types import InvalidPolicy


class TournamentPrompt(BaseModel):
    """Prompt used in the tournament."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, JsonValue] | None = Field(default=None)


class TrueSkillRatingParams(BaseModel):
    """TrueSkill hyperparameters."""

    mu: float = 25.0
    sigma: float = 25.0 / 3.0
    beta: float = 25.0 / 6.0
    tau: float = 25.0 / 300.0
    draw_probability: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_positive(self) -> Self:
        if self.sigma <= 0.0:
            raise ValueError("rating_params.sigma must be greater than 0")
        if self.beta <= 0.0:
            raise ValueError("rating_params.beta must be greater than 0")
        if self.tau <= 0.0:
            raise ValueError("rating_params.tau must be greater than 0")
        return self


class TournamentPromptSource(BaseModel):
    """Prompt source used to materialize prompts for the tournament."""

    path: Path
    format: str | None = None
    text_field: str = Field(default="prompt", min_length=1)
    id_field: str | None = Field(default="id", min_length=1)
    id_template: str | None = None
    metadata_fields: list[str] = Field(default_factory=list)
    metadata_rename: dict[str, str] = Field(default_factory=dict)
    static_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    limit: int | None = Field(default=None, gt=0)

    model_config = {
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if self.id_field is None and self.id_template is None:
            raise ValueError("prompt_source requires `id_field` or `id_template`")
        return self


class TournamentConfig(BaseModel):
    """Tournament configuration schema."""

    run_dir: Path
    project_id: str | None = Field(default=None, min_length=1)

    contestant_models: list[str] = Field(min_length=2)
    prompts: list[TournamentPrompt] = Field(min_length=1)
    regenerate_completions: bool = False
    contestant_generate_config: dict[str, JsonValue] = Field(default_factory=dict)

    rating_params: TrueSkillRatingParams = Field(default_factory=TrueSkillRatingParams)
    conservative_k: float = Field(default=3.0, gt=0.0)
    elo_scale: float = Field(default=173.7178, gt=0.0)

    batch_size: int = Field(default=8, gt=0)
    max_total_matches: int = Field(default=128, gt=0)
    p_stop: float = Field(default=0.98, gt=0.0, le=1.0)
    epsilon: float = Field(default=0.15, ge=0.0)
    min_pair_matches: int = Field(default=2, ge=0)
    max_pair_matches: int = Field(default=24, gt=0)
    max_prompt_uses_per_pair: int = Field(default=3, gt=0)
    n_stable_batches: int = Field(default=3, gt=0)
    seed: int = 42

    judge_model: str = Field(min_length=1)
    judge_max_samples: int = Field(default=8, gt=0)
    judge_generate_config: dict[str, JsonValue] = Field(default_factory=dict)
    judge_prompt_template: str = Field(min_length=1)
    side_swap: bool = True
    prompt_id_field: str = Field(default="prompt_id", min_length=1)

    invalid_policy: InvalidPolicy = "skip"

    model_config = {
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if self.min_pair_matches > self.max_pair_matches:
            raise ValueError("min_pair_matches cannot exceed max_pair_matches")
        if self.side_swap and self.judge_max_samples < 2:
            raise ValueError(
                "judge_max_samples must be at least 2 when side_swap is enabled"
            )

        duplicate_models = _duplicates(self.contestant_models)
        if len(duplicate_models) > 0:
            raise ValueError(
                f"contestant_models contains duplicates: {', '.join(duplicate_models)}"
            )

        prompt_ids = [prompt.id for prompt in self.prompts]
        duplicate_prompt_ids = _duplicates(prompt_ids)
        if len(duplicate_prompt_ids) > 0:
            raise ValueError(
                f"prompts contains duplicate ids: {', '.join(duplicate_prompt_ids)}"
            )

        return self

    def with_resolved_paths(self, base_dir: Path) -> "TournamentConfig":
        """Resolve run_dir against a base directory."""
        resolved_run_dir = _resolve_path(base_dir, self.run_dir)
        return self.model_copy(
            update={
                "run_dir": resolved_run_dir,
            }
        )

    @property
    def run_label(self) -> str:
        return self.run_dir.name

    @property
    def config_dir(self) -> Path:
        return self.run_dir / "config"

    @property
    def runtime_config_path(self) -> Path:
        return self.config_dir / "runtime.json"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def state_dir(self) -> Path:
        return self.run_dir / "state"

    @property
    def state_db_path(self) -> Path:
        return self.state_dir / "tournament.db"

    @property
    def inspect_dir(self) -> Path:
        return self.run_dir / "inspect"

    @property
    def generation_log_dir(self) -> Path:
        return self.inspect_dir / "generation"

    @property
    def judge_log_dir(self) -> Path:
        return self.inspect_dir / "judge"

    @property
    def traces_dir(self) -> Path:
        return self.run_dir / "traces"

    @property
    def exports_dir(self) -> Path:
        return self.run_dir / "exports"

    @property
    def services_dir(self) -> Path:
        return self.run_dir / "services"

    @property
    def vllm_log_dir(self) -> Path:
        return self.services_dir / "vllm"


def load_tournament_config(
    config: TournamentConfig | Mapping[str, Any] | str | Path,
) -> TournamentConfig:
    """Load and validate a tournament config from object or file."""
    if isinstance(config, TournamentConfig):
        return config

    if isinstance(config, Mapping):
        config_dict = _expand_prompt_source(dict(config), base_dir=Path.cwd())
        return TournamentConfig.model_validate(config_dict)

    config_path = resolve_tournament_definition(config, kind="config")
    if not config_path.exists():
        raise FileNotFoundError(f"Tournament config file not found: {config_path}")

    config_text = config_path.read_text(encoding="utf-8")
    config_dict = _read_config_object(config_text)
    config_dict = _expand_prompt_source(config_dict, base_dir=config_path.parent)
    parsed = TournamentConfig.model_validate(config_dict)
    return parsed.with_resolved_paths(config_path.parent)


def _resolve_path(base_dir: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base_dir / value).resolve()


def _duplicates(values: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted([value for value, count in counts.items() if count > 1])


def _read_config_object(config_text: str) -> dict[str, Any]:
    """Read a JSON or YAML config object."""
    stripped = config_text.strip()
    if stripped == "":
        raise ValueError("Tournament config is empty")

    # Try JSON first for strictness, then YAML.
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(stripped)

    if not isinstance(parsed, dict):
        raise ValueError("Tournament config must be a JSON/YAML object")
    return parsed


def _expand_prompt_source(
    config_dict: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    if "prompt_source" not in config_dict:
        return config_dict
    if "prompts" in config_dict:
        raise ValueError("Tournament config cannot define both `prompts` and `prompt_source`")

    expanded = dict(config_dict)
    prompt_source = TournamentPromptSource.model_validate(expanded.pop("prompt_source"))
    expanded["prompts"] = _load_prompt_source(prompt_source, base_dir=base_dir)
    return expanded


def _load_prompt_source(
    prompt_source: TournamentPromptSource,
    *,
    base_dir: Path,
) -> list[dict[str, Any]]:
    source_path = _resolve_path(base_dir, prompt_source.path)
    file_format = _prompt_source_format(prompt_source, source_path)

    if file_format != "jsonl":
        raise ValueError(
            f"Unsupported prompt_source format `{file_format}` for {source_path}"
        )

    prompts: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if stripped == "":
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in prompt_source at {source_path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"prompt_source rows must be JSON objects at {source_path}:{line_number}"
                )
            prompts.append(_build_prompt_from_row(prompt_source, row))
            if prompt_source.limit is not None and len(prompts) >= prompt_source.limit:
                break

    if len(prompts) == 0:
        raise ValueError(f"prompt_source produced no prompts from {source_path}")
    return prompts


def _prompt_source_format(
    prompt_source: TournamentPromptSource,
    source_path: Path,
) -> str:
    if prompt_source.format is not None:
        return prompt_source.format.strip().lower()
    if source_path.suffix == ".jsonl":
        return "jsonl"
    raise ValueError(
        f"Could not infer prompt_source format from path: {source_path}"
    )


def _build_prompt_from_row(
    prompt_source: TournamentPromptSource,
    row: dict[str, Any],
) -> dict[str, Any]:
    text_value = row.get(prompt_source.text_field)
    if text_value is None:
        raise ValueError(
            f"prompt_source row is missing text field `{prompt_source.text_field}`"
        )

    prompt_id = _prompt_id_from_row(prompt_source, row)
    metadata = dict(prompt_source.static_metadata)
    for field_name in prompt_source.metadata_fields:
        if field_name not in row:
            raise ValueError(
                f"prompt_source row is missing metadata field `{field_name}`"
            )
        metadata[prompt_source.metadata_rename.get(field_name, field_name)] = row[field_name]

    prompt: dict[str, Any] = {
        "id": prompt_id,
        "text": str(text_value),
    }
    if metadata:
        prompt["metadata"] = metadata
    return prompt


def _prompt_id_from_row(
    prompt_source: TournamentPromptSource,
    row: dict[str, Any],
) -> str:
    if prompt_source.id_template is not None:
        try:
            prompt_id = prompt_source.id_template.format_map(row)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Failed to render prompt_source id_template `{prompt_source.id_template}`"
            ) from exc
        return _stringify_prompt_source_value(prompt_id, field_name="id_template")

    if prompt_source.id_field is None:
        raise ValueError("prompt_source id_field is required when id_template is unset")
    if prompt_source.id_field not in row:
        raise ValueError(
            f"prompt_source row is missing id field `{prompt_source.id_field}`"
        )
    return _stringify_prompt_source_value(
        row[prompt_source.id_field],
        field_name=prompt_source.id_field,
    )


def _stringify_prompt_source_value(value: Any, *, field_name: str) -> str:
    string_value = str(value).strip()
    if string_value == "":
        raise ValueError(f"prompt_source field `{field_name}` resolved to an empty string")
    return string_value
