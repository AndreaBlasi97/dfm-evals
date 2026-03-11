from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import TaskState, generate

DEFAULT_DATASET_PATH = Path(__file__).parent / "piqa" / "piqa-dan.json"

PROMPT_TEMPLATE_DA = """Du får et spørgsmål om fysisk hverdagsfornuft og to svarmuligheder.
Vælg den bedste løsning.

Spørgsmål:
{prompt}

Mulighed A:
{solution0}

Mulighed B:
{solution1}

Svar kun med A eller B."""

_RE_FIRST_CHOICE = re.compile(r"^\s*([AaBb])\b")
_RE_ANY_CHOICE = re.compile(r"\b([AaBb])\b")


@task(name="piqa")
def piqa(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
) -> Task:
    path = Path(dataset_path)
    records = _load_records(path)
    _validate_records(records=records, path=path)
    samples = [_record_to_sample(record) for record in records]

    if shuffle:
        random.Random(seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]

    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name="PIQA-da",
            location=str(path),
        ),
        solver=[generate()],
        scorer=piqa_scorer(),
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"PIQA dataset file not found: {path}")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"Expected PIQA dataset JSON list, got: {type(loaded).__name__}")

    records: list[dict[str, Any]] = []
    for idx, raw in enumerate(loaded):
        if not isinstance(raw, dict):
            raise ValueError(f"Record {idx} must be an object, got: {type(raw).__name__}")
        records.append(raw)
    return records


def _record_to_sample(record: dict[str, Any]) -> Sample:
    prompt = _require_text(record, "prompt")
    solution0 = _require_text(record, "solution0")
    solution1 = _require_text(record, "solution1")
    label_raw = record.get("label")

    target_choice = "A" if label_raw == 0 else "B"
    sample_id = str(record.get("id")) if record.get("id") is not None else None

    return Sample(
        id=sample_id,
        input=PROMPT_TEMPLATE_DA.format(
            prompt=prompt,
            solution0=solution0,
            solution1=solution1,
        ),
        target=target_choice,
        metadata={
            "solution0": solution0,
            "solution1": solution1,
        },
    )


def _require_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"PIQA field '{field}' must be a non-empty string.")
    return value.strip()


def _validate_records(records: list[dict[str, Any]], path: Path) -> None:
    malformed: list[str] = []
    for idx, record in enumerate(records):
        issues: list[str] = []
        for field in ("prompt", "solution0", "solution1"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                issues.append(f"{field} must be a non-empty string")

        label_raw = record.get("label")
        if label_raw not in (0, 1):
            issues.append(f"label must be 0 or 1 (got {label_raw!r})")

        if issues:
            malformed.append(
                f"idx={idx}, id={record.get('id')!r}: " + "; ".join(issues)
            )

    if not malformed:
        return

    preview = malformed[:20]
    extra_count = len(malformed) - len(preview)
    extra_suffix = f"\n... (+{extra_count} more malformed rows)" if extra_count > 0 else ""
    message = (
        f"Malformed PIQA rows in {path}:\n"
        + "\n".join(preview)
        + extra_suffix
    )
    raise ValueError(message)


@scorer(metrics=[accuracy()])
def piqa_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion if state.output is not None else ""
        predicted = _extract_choice(
            text=completion,
            solution0=str(state.metadata.get("solution0", "")),
            solution1=str(state.metadata.get("solution1", "")),
        )
        expected = str(target.text).strip().upper()
        is_correct = predicted == expected

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=predicted if predicted is not None else "",
            explanation=f"predicted={predicted!r}, expected={expected!r}",
        )

    return score


def _extract_choice(text: str, solution0: str, solution1: str) -> str | None:
    if not text:
        return None

    match = _RE_FIRST_CHOICE.search(text)
    if match is None:
        match = _RE_ANY_CHOICE.search(text)
    if match is not None:
        token = match.group(1).lower()
        if token == "a":
            return "A"
        if token == "b":
            return "B"

    normalized = _normalize_for_match(text)
    sol0 = _normalize_for_match(solution0)
    sol1 = _normalize_for_match(solution1)

    has_0 = sol0 != "" and sol0 in normalized
    has_1 = sol1 != "" and sol1 in normalized
    if has_0 and not has_1:
        return "A"
    if has_1 and not has_0:
        return "B"

    return None


def _normalize_for_match(text: str) -> str:
    return " ".join(text.lower().split())
