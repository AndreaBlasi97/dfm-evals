from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase
from typing import Any

import yaml
from inspect_ai._eval.loader import scorer_from_spec, solver_from_spec
from inspect_ai._eval.task import Task
from inspect_ai._eval.task.epochs import Epochs
from inspect_ai._eval.task.hf import HFFieldSpec
from inspect_ai._eval.task.util import split_spec
from inspect_ai._util.error import PrerequisiteError, pip_dependency_error
from inspect_ai._util.registry import (
    registry_find,
    registry_info,
    registry_lookup,
    registry_unqualified_name,
)
from inspect_ai._util.version import verify_required_version
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.dataset._dataset import DatasetRecord
from inspect_ai.scorer._scorer import Scorer, ScorerSpec
from inspect_ai.solver._solver import Solver, SolverSpec


def task_create_from_hf_extended(task_name: str, **kwargs: Any) -> list[Task]:
    """Load hf/<repo> tasks with dfm-evals extensions (custom scorers + filters)."""
    try:
        from huggingface_hub import errors as hf_errors
        from huggingface_hub import hf_hub_download

        verify_required_version("HuggingFace Tasks", "huggingface_hub", "1.0.0")
    except ImportError:
        raise pip_dependency_error(
            "HuggingFace Dataset Tasks (hf/)", ["huggingface_hub"]
        ) from None

    task_spec, revision = split_spec(task_name.replace("hf/", ""))
    if revision is None:
        revision = kwargs.get("revision", "main")

    repo_id, requested_task_id = _parse_task_spec(task_spec)
    task_configs = _load_task_configs(
        repo_id=repo_id,
        revision=revision,
        hf_hub_download=hf_hub_download,
        entry_not_found_error=hf_errors.EntryNotFoundError,
    )

    tasks: list[Task] = []
    for idx, task_config in enumerate(task_configs):
        task = _build_hf_task(
            task_name=task_name,
            repo_id=repo_id,
            revision=revision,
            task_config=task_config,
            task_index=idx,
            total_task_count=len(task_configs),
            requested_task_id=requested_task_id,
        )
        if task is not None:
            tasks.append(task)

    if len(tasks) == 0:
        raise PrerequisiteError(f"No tasks matching '{task_name}' were found.")

    return tasks


def _load_task_configs(
    *,
    repo_id: str,
    revision: str,
    hf_hub_download: Any,
    entry_not_found_error: type[Exception],
) -> list[dict[str, Any]]:
    try:
        yaml_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="eval.yaml",
                repo_type="dataset",
                revision=revision,
            )
        )
    except entry_not_found_error:
        raise PrerequisiteError(
            f"No 'eval.yaml' file found for Hugging Face Dataset '{repo_id}'"
        ) from None

    global_config = yaml.safe_load(yaml_path.read_text())
    if not isinstance(global_config, dict):
        raise PrerequisiteError("eval.yaml root must be a mapping.")

    task_configs = global_config.get("tasks")
    if not isinstance(task_configs, list) or len(task_configs) == 0:
        raise PrerequisiteError("eval.yaml does not include non-empty 'tasks' list.")

    for idx, task_config in enumerate(task_configs):
        if not isinstance(task_config, dict):
            raise PrerequisiteError(
                f"Task config #{idx} in eval.yaml must be a mapping."
            )

    return task_configs


def _build_hf_task(
    *,
    task_name: str,
    repo_id: str,
    revision: str,
    task_config: dict[str, Any],
    task_index: int,
    total_task_count: int,
    requested_task_id: str | None,
) -> Task | None:
    task_id = task_config.get("id")
    task_label = _task_label(task_id, task_index)

    if total_task_count > 1 and not isinstance(task_id, str):
        raise PrerequisiteError(
            "Task 'id' field is required if there are more than 1 tasks in 'eval.yaml'"
        )
    if requested_task_id is not None and task_id != requested_task_id:
        return None

    field_spec = _build_field_spec(task_config, task_label)
    filters = task_config.get("filters")
    dataset = hf_dataset(
        path=repo_id,
        revision=revision,
        name=str(task_config.get("config", "default")),
        split=str(task_config.get("split", "test")),
        sample_fields=_build_sample_fields(field_spec, filters),
    )

    if bool(task_config.get("shuffle_choices")):
        dataset.shuffle_choices()

    return Task(
        name=f"{task_name}/{task_id}" if total_task_count > 1 else task_name,
        dataset=dataset,
        solver=_build_solvers(task_config),
        scorer=_build_scorers(task_config),
        epochs=_build_epochs(task_config, task_label),
    )


def _task_label(task_id: Any, task_index: int) -> str:
    return str(task_id) if isinstance(task_id, str) and task_id else str(task_index)


def _build_field_spec(task_config: dict[str, Any], task_label: str) -> HFFieldSpec:
    raw_field_spec = task_config.get("field_spec")
    if not isinstance(raw_field_spec, dict):
        raise PrerequisiteError(
            f"Task '{task_label}' missing required mapping field 'field_spec'."
        )

    return HFFieldSpec(**raw_field_spec)


def _build_sample_fields(
    field_spec: HFFieldSpec,
    filters: Any,
) -> Any:
    def record_to_sample_hf(record: DatasetRecord) -> Sample | list[Sample]:
        if filters is not None and not _record_matches_filter(record, filters):
            return []
        return _record_to_sample_hf(record, field_spec)

    return record_to_sample_hf


def _build_epochs(task_config: dict[str, Any], task_label: str) -> Epochs:
    epochs_raw = task_config.get("epochs", 1)
    if not isinstance(epochs_raw, int) or epochs_raw < 1:
        raise PrerequisiteError(
            f"Task '{task_label}' has invalid 'epochs' value: {epochs_raw}"
        )

    epoch_reducer = task_config.get("epoch_reducer")
    if epoch_reducer is not None and not isinstance(epoch_reducer, str):
        raise PrerequisiteError(
            f"Task '{task_label}' has invalid 'epoch_reducer' value."
        )

    return Epochs(epochs_raw, epoch_reducer)


def _build_solvers(task_config: dict[str, Any]) -> list[Solver]:
    return [
        solver_from_spec(SolverSpec(solver=name, args=args, args_passed=args))
        for name, args in _named_spec_entries(task_config, "solvers")
    ]


def _build_scorers(task_config: dict[str, Any]) -> list[Scorer]:
    # Ensure local scorer decorators execute before name resolution.
    import dfm_evals.scorers  # noqa: F401

    return [
        scorer_from_spec(
            ScorerSpec(scorer=_resolve_scorer_name(name)),
            task_path=None,
            **args,
        )
        for name, args in _named_spec_entries(task_config, "scorers")
    ]


def _named_spec_entries(
    task_config: dict[str, Any], field_name: str
) -> list[tuple[str, dict[str, Any]]]:
    raw_entries = task_config.get(field_name)
    if not isinstance(raw_entries, list) or len(raw_entries) == 0:
        raise PrerequisiteError(f"Task config requires non-empty '{field_name}' list.")

    entries: list[tuple[str, dict[str, Any]]] = []
    singular_name = field_name[:-1]
    for idx, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise PrerequisiteError(
                f"Invalid {singular_name} entry #{idx}: expected mapping."
            )

        name = raw_entry.get("name")
        if not isinstance(name, str) or name.strip() == "":
            raise PrerequisiteError(
                f"Invalid {singular_name} entry #{idx}: missing 'name'."
            )

        args = raw_entry.get("args", {})
        if not isinstance(args, dict):
            raise PrerequisiteError(
                f"Invalid {singular_name} entry #{idx}: 'args' must be mapping."
            )

        entries.append((name.strip(), args))

    return entries


def _parse_task_spec(task_spec: str) -> tuple[str, str | None]:
    parts = task_spec.split("/")
    if len(parts) == 2:
        return task_spec, None
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}", parts[2]
    raise ValueError(f"Expected 2 or 3 components in task spec, got {len(parts)}.")


def _sanitize_target(record: DatasetRecord, target: str, is_choices: bool) -> str:
    if target.startswith("literal:"):
        return target.split(":", 1)[1]

    target_value = record[target]
    if isinstance(target_value, int) and is_choices:
        target_value = ascii_uppercase[target_value]
    return str(target_value)


def _sanitize_choices(
    record: DatasetRecord, choices: str | list[str] | None
) -> Any | None:
    if choices is None:
        return None
    if isinstance(choices, list):
        return [record[choice] for choice in choices]
    return record[choices]


_MISSING = object()


def _record_matches_filter(record: DatasetRecord, filter: Any) -> bool:
    if not isinstance(filter, dict):
        raise PrerequisiteError("Task `filters` must be a mapping.")

    if "all" in filter:
        children = filter["all"]
        if not isinstance(children, list) or len(children) == 0:
            raise PrerequisiteError("'all' filter must contain at least one condition.")
        return all(_record_matches_filter(record, child) for child in children)
    if "any" in filter:
        children = filter["any"]
        if not isinstance(children, list) or len(children) == 0:
            raise PrerequisiteError("'any' filter must contain at least one condition.")
        return any(_record_matches_filter(record, child) for child in children)
    if "not" in filter:
        return not _record_matches_filter(record, filter["not"])

    column = filter.get("column")
    op = filter.get("op")
    value = filter.get("value")
    if not isinstance(column, str) or not isinstance(op, str):
        raise PrerequisiteError(
            "Filter condition must include string 'column' and 'op'."
        )

    record_value = record.get(column, _MISSING)
    match op:
        case "exists":
            return record_value is not _MISSING
        case "is_null":
            return record_value is None
        case "not_null":
            return record_value is not _MISSING and record_value is not None
        case "eq":
            return record_value is not _MISSING and record_value == value
        case "ne":
            return record_value is not _MISSING and record_value != value
        case "in" | "not_in":
            if record_value is _MISSING:
                return False
            if not isinstance(value, (list, tuple, set)):
                raise PrerequisiteError(
                    f"Filter op '{op}' requires list/tuple/set value."
                )
            contains = record_value in value
            return contains if op == "in" else not contains
        case "between":
            if record_value is _MISSING:
                return False
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise PrerequisiteError("Filter op 'between' requires [low, high].")
            low, high = value
            try:
                return low <= record_value <= high
            except TypeError as exc:
                raise PrerequisiteError(
                    f"Filter op 'between' cannot compare column '{column}' "
                    f"value {record_value!r} against bounds {value!r}."
                ) from exc
        case "contains" | "not_contains":
            if record_value is _MISSING:
                return False
            contains = _contains_value(record_value, value)
            return contains if op == "contains" else not contains
        case "gt" | "gte" | "lt" | "lte":
            if record_value is _MISSING:
                return False
            try:
                if op == "gt":
                    return record_value > value
                if op == "gte":
                    return record_value >= value
                if op == "lt":
                    return record_value < value
                return record_value <= value
            except TypeError as exc:
                raise PrerequisiteError(
                    f"Filter op '{op}' cannot compare column '{column}' "
                    f"value {record_value!r} against {value!r}."
                ) from exc
        case _:
            raise PrerequisiteError(f"Unsupported filter operation: {op}")


def _contains_value(record_value: Any, value: Any) -> bool:
    if isinstance(record_value, str):
        return isinstance(value, str) and value in record_value
    if isinstance(record_value, (list, tuple, set, dict)):
        return value in record_value
    return False


def _record_to_sample_hf(record: DatasetRecord, field_spec: HFFieldSpec) -> Sample:
    # We support direct literal target and multi-column choices from eval.yaml.
    choices = _sanitize_choices(record, field_spec.choices)
    return Sample(
        input=record[field_spec.input],
        target=_sanitize_target(
            record, field_spec.target, is_choices=choices is not None
        ),
        choices=choices,
        id=record.get(field_spec.id, None),
        metadata=record
        if field_spec.metadata is None
        else {m: record[m] for m in field_spec.metadata},
        sandbox=record.get(field_spec.sandbox),
        files=record.get(field_spec.files),
        setup=record.get(field_spec.setup),
    )


def _resolve_scorer_name(name: str) -> str:
    """Resolve scorer names to the canonical registry key.

    For unqualified names, use them as-is when already resolvable. Otherwise, map
    local dfm_evals scorers by matching their unqualified name.
    """
    if "/" in name:
        return name

    if registry_lookup("scorer", name) is not None:
        return name

    matches = [
        registry_info(scorer_obj).name
        for scorer_obj in registry_find(lambda info: info.type == "scorer")
        if registry_unqualified_name(registry_info(scorer_obj)) == name
        and registry_info(scorer_obj).name.startswith("dfm_evals/")
    ]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise PrerequisiteError(
            f"Ambiguous scorer name '{name}'. Use a fully-qualified scorer name."
        )

    return name


def install_hf_eval_yaml_extensions() -> None:
    """Install the extended hf/eval.yaml loader into inspect_ai.

    inspect_ai binds `task_create_from_hf` into both the hf task module and
    eval loader namespace. Patch both references so CLI task resolution uses
    the extended loader that supports custom scorers and row filters.
    """
    import inspect_ai._eval.loader as inspect_loader
    import inspect_ai._eval.task.hf as inspect_hf_task

    inspect_hf_task.task_create_from_hf = task_create_from_hf_extended
    inspect_loader.task_create_from_hf = task_create_from_hf_extended
