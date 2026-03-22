import importlib
import sys
import types

import pytest

from dfm_evals.tasks.ifeval_da import record_to_sample


def _response_language_checker(monkeypatch) -> type[object]:
    ifeval_da_module = importlib.import_module("dfm_evals.tasks.ifeval_da")
    instructions_module = types.ModuleType("instruction_following_eval.instructions")
    instructions_module.Instruction = type("Instruction", (), {})
    registry_module = types.ModuleType(
        "instruction_following_eval.instructions_registry"
    )
    registry_module.INSTRUCTION_DICT = {}
    package_module = types.ModuleType("instruction_following_eval")
    package_module.instructions = instructions_module
    package_module.instructions_registry = registry_module

    monkeypatch.setitem(sys.modules, "instruction_following_eval", package_module)
    monkeypatch.setitem(
        sys.modules,
        "instruction_following_eval.instructions",
        instructions_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "instruction_following_eval.instructions_registry",
        registry_module,
    )

    ifeval_da_module._patch_instruction_registry()
    return registry_module.INSTRUCTION_DICT["language:response_language"]


def test_response_language_checker_rejects_undetectable_text(monkeypatch) -> None:
    checker_cls = _response_language_checker(monkeypatch)

    class FakeLangDetectException(Exception):
        pass

    langdetect_module = types.ModuleType("langdetect")
    langdetect_module.LangDetectException = FakeLangDetectException

    def _raise_detection_error(value: str) -> str:
        del value
        raise FakeLangDetectException("not enough features")

    langdetect_module.detect = _raise_detection_error
    monkeypatch.setitem(sys.modules, "langdetect", langdetect_module)

    checker = checker_cls()
    checker.build_description(language="da")

    assert checker.check_following("...") is False


def test_response_language_checker_accepts_matching_language(monkeypatch) -> None:
    checker_cls = _response_language_checker(monkeypatch)
    langdetect_module = types.ModuleType("langdetect")
    langdetect_module.LangDetectException = RuntimeError
    langdetect_module.detect = lambda value: "da"
    monkeypatch.setitem(sys.modules, "langdetect", langdetect_module)

    checker = checker_cls()
    checker.build_description(language="da")

    assert checker.check_following("Hej verden") is True


def test_record_to_sample_requires_one_kwargs_entry_per_instruction() -> None:
    with pytest.raises(AssertionError):
        record_to_sample(
            {
                "key": "sample-1",
                "prompt": "Hello",
                "instruction_id_list": ["a"],
                "kwargs": [],
            }
        )
