from __future__ import annotations

import threading
from collections.abc import Sequence
from functools import wraps
from typing import Any

try:
    import regex as pattern_re
except ImportError:
    import re as pattern_re

_INSTANCE_LOCK_ATTR = "_dfm_evals_thread_safe_lock"
_PATCH_LOCK = threading.Lock()


def apply_instance_method_rlock_patch(
    target_cls: type,
    method_names: Sequence[str],
) -> None:
    patch_flag = "_dfm_evals_instance_method_rlock_patch"
    if getattr(target_cls, patch_flag, False):
        return

    def get_instance_lock(instance: Any) -> threading.RLock:
        lock = getattr(instance, _INSTANCE_LOCK_ATTR, None)
        if lock is not None:
            return lock

        with _PATCH_LOCK:
            lock = getattr(instance, _INSTANCE_LOCK_ATTR, None)
            if lock is not None:
                return lock

            lock = threading.RLock()
            setattr(instance, _INSTANCE_LOCK_ATTR, lock)
            return lock

    for method_name in method_names:
        original = getattr(target_cls, method_name, None)
        if original is None or not callable(original):
            continue

        @wraps(original)
        def wrapped(
            self: Any,
            *args: Any,
            __original: Any = original,
            **kwargs: Any,
        ) -> Any:
            with get_instance_lock(self):
                return __original(self, *args, **kwargs)

        setattr(target_cls, method_name, wrapped)

    setattr(target_cls, patch_flag, True)


def apply_hermes_tool_parser_thread_safety_patch(
    hermes_parser_cls: type,
    tool_parser_cls: type,
    mistral_tokenizer_cls: type,
) -> None:
    if getattr(hermes_parser_cls, "_dfm_evals_thread_safe_patch", False):
        return

    original_init = hermes_parser_cls.__init__
    cache: dict[int, dict[str, Any]] = {}
    lock = threading.Lock()

    def init_from_cache(self: Any, tokenizer: Any, cached: dict[str, Any]) -> None:
        tool_parser_cls.__init__(self, tokenizer)

        if isinstance(tokenizer, mistral_tokenizer_cls):
            self.model_tokenizer = tokenizer.tokenizer

        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_regex = pattern_re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", pattern_re.DOTALL
        )
        self.scratch_pad_regex = pattern_re.compile(
            r"<scratch_pad>(.*?)</scratch_pad>", pattern_re.DOTALL
        )

        if not getattr(self, "model_tokenizer", None):
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_ids = cached["start_ids"]
        self.tool_call_end_token_ids = cached["end_ids"]
        self.tool_call_start_token_array = cached["start_array"]
        self.tool_call_end_token_array = cached["end_array"]
        self.buffered_delta_text = ""

    def patched_init(self: Any, tokenizer: Any) -> None:
        actual_tokenizer = (
            tokenizer.tokenizer
            if isinstance(tokenizer, mistral_tokenizer_cls)
            else tokenizer
        )
        key = id(actual_tokenizer)
        cached = cache.get(key)

        if cached is not None:
            init_from_cache(self, tokenizer, cached)
            return

        with lock:
            cached = cache.get(key)
            if cached is None:
                original_init(self, tokenizer)
                cache[key] = {
                    "start_ids": list(self.tool_call_start_token_ids),
                    "end_ids": list(self.tool_call_end_token_ids),
                    "start_array": list(self.tool_call_start_token_array),
                    "end_array": list(self.tool_call_end_token_array),
                }
                return

        init_from_cache(self, tokenizer, cached)

    hermes_parser_cls.__init__ = patched_init
    hermes_parser_cls._dfm_evals_thread_safe_patch = True


def apply_fast_tokenizer_thread_safety_patch(fast_tokenizer_cls: type) -> None:
    apply_instance_method_rlock_patch(
        fast_tokenizer_cls,
        method_names=(
            "__call__",
            "encode",
            "encode_plus",
            "batch_encode_plus",
            "_batch_encode_plus",
            "decode",
            "batch_decode",
        ),
    )


def monkey_patch_hermes_tool_parser_thread_safety() -> None:
    from vllm.tokenizers.mistral import MistralTokenizer
    from vllm.tool_parsers.abstract_tool_parser import ToolParser
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    apply_hermes_tool_parser_thread_safety_patch(
        hermes_parser_cls=Hermes2ProToolParser,
        tool_parser_cls=ToolParser,
        mistral_tokenizer_cls=MistralTokenizer,
    )


def monkey_patch_fast_tokenizer_thread_safety() -> None:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    apply_fast_tokenizer_thread_safety_patch(PreTrainedTokenizerFast)


def apply_runtime_thread_safety_patches() -> list[str]:
    monkey_patch_fast_tokenizer_thread_safety()
    monkey_patch_hermes_tool_parser_thread_safety()
    return [
        "transformers-fast-tokenizer",
        "vllm-hermes-tool-parser",
    ]
