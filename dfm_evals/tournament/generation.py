from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from inspect_ai import Task, eval_set
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, get_model
from pydantic import BaseModel, Field

from ._model_args import (
    close_model as _close_model,
)
from ._model_args import (
    resolve_env_model_args,
    resolve_tournament_model_args,
)
from ._provenance import GENERATION_TASK_NAME, generation_log_metadata
from ._trace import tournament_trace_file
from .config import TournamentConfig, load_tournament_config
from .indexer import index_generation_responses

GENERATION_PHASE = "generation"


class GenerationRunResult(BaseModel):
    """Result summary for a generation run."""

    models: list[str]
    prompt_count: int
    generation_log_dir: Path
    log_count: int
    logs: list[EvalLog] = Field(default_factory=list)


def build_generation_task(config: TournamentConfig) -> Task:
    """Create generation task from tournament prompts."""
    samples: list[Sample] = [
        Sample(
            id=prompt.id,
            input=prompt.text,
            metadata={config.prompt_id_field: prompt.id},
        )
        for prompt in config.prompts
    ]
    generate_config = GenerateConfig.model_validate(config.contestant_generate_config)
    return Task(
        name=GENERATION_TASK_NAME,
        dataset=samples,
        config=generate_config,
    )


def run_generation(
    config: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    models: Sequence[str] | None = None,
) -> GenerationRunResult:
    """Run model generation for tournament prompts."""
    parsed = load_tournament_config(config)
    selected_models = list(models) if models is not None else parsed.contestant_models
    if len(selected_models) == 0:
        raise ValueError("At least one model is required for generation")

    unknown_models = sorted(set(selected_models) - set(parsed.contestant_models))
    if len(unknown_models) > 0:
        raise ValueError(
            "Unknown model(s) requested for generation: "
            + ", ".join(unknown_models)
            + "."
        )

    task = build_generation_task(parsed)
    metadata = generation_log_metadata(parsed)
    model_args = resolve_env_model_args()
    logs: list[EvalLog] = []
    with tournament_trace_file(parsed.traces_dir, GENERATION_PHASE):
        for model_name in selected_models:
            _purge_generation_logs_for_model(parsed, model_name)
            resolved_model_args = resolve_tournament_model_args(model_name, model_args)
            model = get_model(model_name, memoize=False, **resolved_model_args)
            try:
                success, model_logs = eval_set(
                    tasks=task,
                    model=model,
                    log_dir=parsed.generation_log_dir.as_posix(),
                    metadata=metadata,
                    score=False,
                    log_dir_allow_dirty=True,
                )
                logs.extend(model_logs)
                if not success:
                    raise RuntimeError(
                        "Generation run did not complete successfully for model "
                        + f"'{model_name}'"
                    )
                # Keep tournament state in sync as contestant generation progresses.
                index_generation_responses(parsed)
            finally:
                _close_model(model)

    return GenerationRunResult(
        models=selected_models,
        prompt_count=len(parsed.prompts),
        generation_log_dir=parsed.generation_log_dir,
        log_count=len(logs),
        logs=logs,
    )


def _purge_generation_logs_for_model(config: TournamentConfig, model_name: str) -> None:
    expected_metadata = generation_log_metadata(config)
    for log_info in list_eval_logs(config.generation_log_dir.as_posix()):
        header = read_eval_log(log_info, header_only=True)
        if header.eval.task != GENERATION_TASK_NAME:
            continue
        if header.eval.model != model_name:
            continue
        if not isinstance(header.eval.metadata, dict):
            continue
        if header.eval.metadata != expected_metadata:
            continue

        log_path = _log_info_path(log_info.name)
        if log_path is None or not log_path.exists():
            continue
        log_path.unlink()


def _log_info_path(value: str) -> Path | None:
    if "://" not in value:
        return Path(value)

    parsed = urlparse(value)
    if parsed.scheme != "file":
        return None
    return Path(parsed.path)
