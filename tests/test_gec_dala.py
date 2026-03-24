import pytest

import dfm_evals.tasks.gec_dala as gec_dala_module
from dfm_evals.tasks.gec_dala import (
    _load_hf_dataset,
    _normalize_split_name,
    record_to_sample,
)


def test_normalize_split_name_accepts_aliases() -> None:
    assert _normalize_split_name("training") == "train"
    assert _normalize_split_name("dev") == "val"
    assert _normalize_split_name("test") == "test"


def test_record_to_sample_maps_corrupted_and_original_fields() -> None:
    sample = record_to_sample(
        record={
            "id": "row-1",
            "corrupted": "Jeg går i skole igår.",
            "original": "Jeg gik i skole i går.",
        },
        prompt_template="Sætning: {{corrupted}}\nSvar:",
        input_field="corrupted",
        target_field="original",
    )

    assert sample.id == "row-1"
    assert sample.input == "Sætning: Jeg går i skole igår.\nSvar:"
    assert sample.target == ["Jeg gik i skole i går."]


def test_record_to_sample_requires_non_empty_target_field() -> None:
    with pytest.raises(
        ValueError,
        match="Record field 'original' must be a non-empty string",
    ):
        record_to_sample(
            record={"corrupted": "Hej", "original": " "},
            prompt_template="{{corrupted}}",
            input_field="corrupted",
            target_field="original",
        )


def test_load_hf_dataset_retries_without_name_on_missing_builder_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_hf_dataset(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        if "name" in kwargs:
            raise ValueError("BuilderConfig 'gec_dala' not found. Available: ['default']")
        return {"ok": True}

    monkeypatch.setattr(gec_dala_module, "hf_dataset", fake_hf_dataset)

    dataset = _load_hf_dataset(
        {
            "path": "giannor/dala_gen_v2",
            "name": "gec_dala",
            "split": "test",
        }
    )

    assert dataset == {"ok": True}
    assert calls == [
        {
            "path": "giannor/dala_gen_v2",
            "name": "gec_dala",
            "split": "test",
        },
        {
            "path": "giannor/dala_gen_v2",
            "split": "test",
        },
    ]
