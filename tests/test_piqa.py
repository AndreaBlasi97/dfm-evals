from dfm_evals.tasks.piqa import _extract_choice


def test_extract_choice_maps_letter_a_to_a() -> None:
    assert _extract_choice("A", "Use solution zero.", "Use solution one.") == "A"


def test_extract_choice_maps_letter_b_to_b() -> None:
    assert _extract_choice("B", "Use solution zero.", "Use solution one.") == "B"


def test_extract_choice_rejects_numeric_zero() -> None:
    assert _extract_choice("0", "Use solution zero.", "Use solution one.") is None


def test_extract_choice_rejects_numeric_one() -> None:
    assert _extract_choice("1", "Use solution zero.", "Use solution one.") is None
