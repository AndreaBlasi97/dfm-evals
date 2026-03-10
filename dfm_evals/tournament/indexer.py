from pathlib import Path
from typing import Any, Mapping

from inspect_ai.log import (
    EvalLog,
    EvalSample,
    list_eval_logs,
    read_eval_log,
    read_eval_log_samples,
)
from pydantic import BaseModel

from ._provenance import (
    GENERATION_PHASE,
    GENERATION_TASK_NAME,
    TOURNAMENT_PHASE_KEY,
    TOURNAMENT_PROJECT_KEY,
    resolve_tournament_project_id,
)
from .config import TournamentConfig, load_tournament_config
from .store import TournamentStore
from .types import response_id


class ResponseIndexReport(BaseModel):
    """Summary of a response indexing run."""

    logs_seen: int = 0
    logs_processed: int = 0
    logs_skipped_model: int = 0
    logs_skipped_provenance: int = 0
    samples_seen: int = 0
    responses_indexed: int = 0
    responses_inserted: int = 0
    skipped_samples: int = 0
    log_errors: int = 0
    missing_by_model: dict[str, list[str]]

    @property
    def missing_count(self) -> int:
        """Total number of missing model/prompt responses."""
        return sum(len(prompt_ids) for prompt_ids in self.missing_by_model.values())


def index_generation_responses(
    config: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    store: TournamentStore | None = None,
) -> ResponseIndexReport:
    """Index generation logs into tournament response state."""
    parsed = load_tournament_config(config)
    if store is not None:
        return _index_generation_responses(parsed, store)

    with TournamentStore(parsed.state_dir) as opened_store:
        return _index_generation_responses(parsed, opened_store)


def _index_generation_responses(
    config: TournamentConfig,
    store: TournamentStore,
) -> ResponseIndexReport:
    store.initialize_from_config(config)

    expected_project_id = resolve_tournament_project_id(config, store=store)
    expected_models = set(config.contestant_models)
    expected_prompt_ids = [prompt.id for prompt in config.prompts]
    expected_prompt_set = set(expected_prompt_ids)
    report = ResponseIndexReport(missing_by_model={})

    with store.transaction():
        for log_info in list_eval_logs(config.generation_log_dir.as_posix()):
            report.logs_seen += 1
            try:
                header = read_eval_log(log_info, header_only=True)
                if not _matches_generation_provenance(
                    header,
                    expected_project_id=expected_project_id,
                ):
                    report.logs_skipped_provenance += 1
                    continue

                model_name = header.eval.model
                if model_name not in expected_models:
                    report.logs_skipped_model += 1
                    continue
                model_identifier = store.model_identifier(model_name)
                if model_identifier is None:
                    report.logs_skipped_model += 1
                    continue

                report.logs_processed += 1
                for sample in read_eval_log_samples(
                    log_info,
                    all_samples_required=False,
                ):
                    report.samples_seen += 1
                    prompt_id = _resolve_prompt_id(sample, config.prompt_id_field)
                    if prompt_id is None or prompt_id not in expected_prompt_set:
                        report.skipped_samples += 1
                        continue

                    inserted = store.upsert_response(
                        response_id=response_id(
                            model_identifier,
                            prompt_id,
                            source_log=_relative_log_name(
                                config.generation_log_dir, log_info.name
                            ),
                            sample_uuid=sample.uuid,
                            sample_id=str(sample.id),
                            response_text=sample.output.completion,
                        ),
                        model_id=model_identifier,
                        prompt_id=prompt_id,
                        response_text=sample.output.completion,
                        source_log=_relative_log_name(
                            config.generation_log_dir, log_info.name
                        ),
                        source_log_mtime=log_info.mtime,
                        sample_id=str(sample.id),
                        sample_uuid=sample.uuid,
                        commit=False,
                    )
                    report.responses_indexed += 1
                    if inserted:
                        report.responses_inserted += 1
            except Exception:
                report.log_errors += 1

    report.missing_by_model = store.missing_prompt_ids_by_model(
        config.contestant_models, expected_prompt_ids
    )
    return report


def _matches_generation_provenance(
    log: EvalLog,
    *,
    expected_project_id: str,
) -> bool:
    eval_spec = log.eval
    if eval_spec.task != GENERATION_TASK_NAME:
        return False

    metadata = eval_spec.metadata
    if not isinstance(metadata, dict):
        return False

    return (
        metadata.get(TOURNAMENT_PHASE_KEY) == GENERATION_PHASE
        and metadata.get(TOURNAMENT_PROJECT_KEY) == expected_project_id
    )


def _resolve_prompt_id(sample: EvalSample, prompt_id_field: str) -> str | None:
    metadata = sample.metadata if sample.metadata is not None else {}
    prompt_value = metadata.get(prompt_id_field)
    if isinstance(prompt_value, str):
        return prompt_value
    if isinstance(prompt_value, int):
        return str(prompt_value)
    if isinstance(sample.id, str):
        return sample.id
    if isinstance(sample.id, int):
        return str(sample.id)
    return None


def _relative_log_name(base_dir: Path, log_name: str) -> str:
    if "://" in log_name:
        return log_name

    try:
        return Path(log_name).resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return log_name
