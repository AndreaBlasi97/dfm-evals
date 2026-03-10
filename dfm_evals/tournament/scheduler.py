import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from pydantic import BaseModel

from .config import TournamentConfig
from .rating import TrueSkillEngine
from .store import TournamentStore
from .types import ModelRating, match_id


class ScheduledMatch(BaseModel):
    """One scheduled pairwise match."""

    match_id: str
    model_a: str
    model_b: str
    prompt_id: str
    response_a_id: str
    response_b_id: str
    batch_id: str
    round_index: int
    priority: float
    forced: bool
    pair_matches: int
    prompt_uses: int


class ScheduleBatchResult(BaseModel):
    """Scheduler output for one batch."""

    batch_id: str
    round_index: int
    scheduled: list[ScheduledMatch]
    candidate_pairs: int

    @property
    def exhausted(self) -> bool:
        """Whether no matches were schedulable."""
        return len(self.scheduled) == 0


@dataclass(frozen=True)
class _Candidate:
    acquisition: float
    tie_break: str
    model_a: str
    model_b: str
    prompt_id: str
    response_a_id: str
    response_b_id: str
    forced: bool
    pair_matches: int
    prompt_uses: int


def schedule_match_batch(
    config: TournamentConfig,
    store: TournamentStore,
    *,
    batch_id: str,
    round_index: int,
    seed: int | None = None,
    persist: bool = False,
) -> ScheduleBatchResult:
    """Schedule one information-gain batch under coverage constraints."""
    rng = random.Random(seed if seed is not None else config.seed + round_index)
    conn = store.connection()

    ratings = store.load_active_model_ratings()
    if len(ratings) < 2:
        return ScheduleBatchResult(
            batch_id=batch_id,
            round_index=round_index,
            scheduled=[],
            candidate_pairs=0,
        )

    responses = _load_responses(conn)
    pair_counts, prompt_counts = _load_match_counts(conn)
    prompt_ids = [prompt.id for prompt in config.prompts]
    engine = TrueSkillEngine(
        config.rating_params,
        conservative_k=config.conservative_k,
        elo_scale=config.elo_scale,
    )

    candidates: list[_Candidate] = []
    model_ids = sorted(ratings.keys())
    for model_a, model_b in combinations(model_ids, 2):
        pair_key = _canonical_pair(model_a, model_b)
        pair_match_count = pair_counts.get(pair_key, 0)
        if pair_match_count >= config.max_pair_matches:
            continue

        prompt_choice = _choose_prompt_for_pair(
            model_a=model_a,
            model_b=model_b,
            prompt_ids=prompt_ids,
            responses=responses,
            prompt_counts=prompt_counts,
            max_prompt_uses_per_pair=config.max_prompt_uses_per_pair,
            rng=rng,
        )
        if prompt_choice is None:
            continue

        prompt_id, prompt_uses = prompt_choice
        response_a_id = responses[(model_a, prompt_id)]
        response_b_id = responses[(model_b, prompt_id)]

        forced = pair_match_count < config.min_pair_matches
        acquisition = _information_gain_score(
            engine,
            ratings[model_a],
            ratings[model_b],
        )

        candidates.append(
            _Candidate(
                acquisition=acquisition,
                tie_break=f"{model_a}:{model_b}:{prompt_id}",
                model_a=model_a,
                model_b=model_b,
                prompt_id=prompt_id,
                response_a_id=response_a_id,
                response_b_id=response_b_id,
                forced=forced,
                pair_matches=pair_match_count,
                prompt_uses=prompt_uses,
            )
        )

    candidates.sort(key=_candidate_sort_key)
    selected = candidates[: config.batch_size]

    scheduled = [
        ScheduledMatch(
            match_id=match_id(
                candidate.model_a,
                candidate.model_b,
                candidate.prompt_id,
                round_index,
                batch_id,
            ),
            model_a=candidate.model_a,
            model_b=candidate.model_b,
            prompt_id=candidate.prompt_id,
            response_a_id=candidate.response_a_id,
            response_b_id=candidate.response_b_id,
            batch_id=batch_id,
            round_index=round_index,
            priority=candidate.acquisition,
            forced=candidate.forced,
            pair_matches=candidate.pair_matches,
            prompt_uses=candidate.prompt_uses,
        )
        for candidate in selected
    ]

    if persist and len(scheduled) > 0:
        with store.transaction():
            for scheduled_match in scheduled:
                store.upsert_match(
                    match_id=scheduled_match.match_id,
                    model_a=scheduled_match.model_a,
                    model_b=scheduled_match.model_b,
                    prompt_id=scheduled_match.prompt_id,
                    response_a_id=scheduled_match.response_a_id,
                    response_b_id=scheduled_match.response_b_id,
                    batch_id=scheduled_match.batch_id,
                    round_index=scheduled_match.round_index,
                    status="scheduled",
                    commit=False,
                )

    return ScheduleBatchResult(
        batch_id=batch_id,
        round_index=round_index,
        scheduled=scheduled,
        candidate_pairs=len(candidates),
    )


def _load_responses(conn: Any) -> dict[tuple[str, str], str]:
    rows = conn.execute(
        """
        SELECT model_id, prompt_id, response_id
        FROM responses
        WHERE current = 1
        """
    ).fetchall()
    return {
        (str(row["model_id"]), str(row["prompt_id"])): str(row["response_id"])
        for row in rows
    }


def _load_match_counts(
    conn: Any,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str, str], int]]:
    pair_counts: dict[tuple[str, str], int] = {}
    prompt_counts: dict[tuple[str, str, str], int] = {}

    rows = conn.execute(
        "SELECT model_a, model_b, prompt_id FROM matches"
    ).fetchall()
    for row in rows:
        model_a = str(row["model_a"])
        model_b = str(row["model_b"])
        prompt_id = str(row["prompt_id"])
        pair_key = _canonical_pair(model_a, model_b)
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        prompt_key = (pair_key[0], pair_key[1], prompt_id)
        prompt_counts[prompt_key] = prompt_counts.get(prompt_key, 0) + 1

    return pair_counts, prompt_counts


def _choose_prompt_for_pair(
    *,
    model_a: str,
    model_b: str,
    prompt_ids: list[str],
    responses: dict[tuple[str, str], str],
    prompt_counts: dict[tuple[str, str, str], int],
    max_prompt_uses_per_pair: int,
    rng: random.Random,
) -> tuple[str, int] | None:
    pair_key = _canonical_pair(model_a, model_b)
    candidates: list[tuple[int, str]] = []

    for prompt_id in prompt_ids:
        if (model_a, prompt_id) not in responses or (model_b, prompt_id) not in responses:
            continue
        uses = prompt_counts.get((pair_key[0], pair_key[1], prompt_id), 0)
        if uses < max_prompt_uses_per_pair:
            candidates.append((uses, prompt_id))

    if len(candidates) == 0:
        return None

    min_uses = min(uses for uses, _ in candidates)
    min_prompts = sorted(prompt_id for uses, prompt_id in candidates if uses == min_uses)
    chosen_prompt = rng.choice(min_prompts)
    return chosen_prompt, min_uses


def _canonical_pair(model_a: str, model_b: str) -> tuple[str, str]:
    return (model_a, model_b) if model_a <= model_b else (model_b, model_a)


def _candidate_sort_key(candidate: _Candidate) -> tuple[int, float, int, int, str]:
    return (
        0 if candidate.forced else 1,
        -candidate.acquisition,
        candidate.pair_matches,
        candidate.prompt_uses,
        candidate.tie_break,
    )


def _information_gain_score(
    engine: TrueSkillEngine,
    rating_a: ModelRating,
    rating_b: ModelRating,
) -> float:
    probability = engine.win_probability(rating_a, rating_b)
    return _entropy(probability) * (rating_a.sigma + rating_b.sigma)


def _entropy(probability: float) -> float:
    p = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p)) - ((1.0 - p) * math.log(1.0 - p))
