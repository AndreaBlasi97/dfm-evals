from dfm_evals.tasks.multi_wiki_qa import (
    _extract_answer_texts,
    _is_valid_public_record,
    _normalize_split_name,
    _records_with_unique_ids,
)


def test_extract_answer_texts_strips_and_dedupes() -> None:
    record = {
        "answers": {
            "text": [
                "  Copenhagen  ",
                "Copenhagen",
                "",
                "  Aarhus  ",
                None,
            ]
        }
    }

    assert _extract_answer_texts(record) == ["Copenhagen", "Aarhus"]


def test_is_valid_public_record_requires_cleaned_answer_texts() -> None:
    record = {
        "context": "x" * 40,
        "question": "y" * 20,
        "answers": {"text": [" ", ""]},
    }

    assert _is_valid_public_record(record) is False


def test_records_with_unique_ids_add_suffixes_and_fallback_ids() -> None:
    records = [
        {"id": "shared", "context": "a", "question": "b", "answers": {"text": ["c"]}},
        {"id": "shared", "context": "d", "question": "e", "answers": {"text": ["f"]}},
        {"id": " ", "context": "g", "question": "h", "answers": {"text": ["i"]}},
    ]

    unique_records = _records_with_unique_ids(records, language="da", split="test")

    assert [record["id"] for record in unique_records] == [
        "shared",
        "shared__1",
        "da_test_2",
    ]


def test_normalize_split_name_accepts_aliases() -> None:
    assert _normalize_split_name("training") == "train"
    assert _normalize_split_name("dev") == "val"
