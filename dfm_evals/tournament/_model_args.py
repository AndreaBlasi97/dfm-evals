import asyncio
import logging
import os
import re
import shlex
import threading
from functools import lru_cache
from typing import Any, Mapping

import yaml
from inspect_ai.model import Model

VLLM_BASE_URL = "VLLM_BASE_URL"

logger = logging.getLogger(__name__)


def resolve_tournament_model_args(
    model_name: str,
    model_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve shared tournament model args from env and call-site defaults."""
    resolved = resolve_env_model_args()
    if model_args is not None:
        resolved.update(model_args)
    return apply_default_vllm_model_args(model_name, resolved)


def resolve_env_model_args() -> dict[str, Any]:
    """Parse `INSPECT_EVAL_MODEL_ARGS` into kwargs for `inspect_ai.get_model()`."""
    model_args: dict[str, Any] = {}
    env_model_args = os.environ.get("INSPECT_EVAL_MODEL_ARGS")
    if not env_model_args:
        return model_args

    for arg in shlex.split(env_model_args):
        arg = arg.strip()
        if not arg or "=" not in arg:
            continue

        key, value_raw = arg.split("=", maxsplit=1)
        value = yaml.safe_load(value_raw)
        if isinstance(value, str):
            value_parts = value.split(",")
            value = value_parts if len(value_parts) > 1 else value_parts[0]
        model_args[key.replace("-", "_")] = value

    return model_args


def apply_default_vllm_model_args(
    model_name: str,
    model_args: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply default local vLLM args for tournament runs.

    If a model is `vllm/...` and no explicit device/tensor-parallel settings are
    provided, default to all visible CUDA devices so local tournament runs make
    full use of available GPUs.
    """
    resolved = dict(model_args)
    if not model_name.startswith("vllm/"):
        return resolved

    if _has_explicit_vllm_device_args(resolved):
        return resolved

    if _uses_existing_vllm_server(resolved):
        return resolved

    visible_devices = _visible_cuda_devices()
    if len(visible_devices) > 0:
        resolved["device"] = visible_devices

    return resolved


def _has_explicit_vllm_device_args(model_args: Mapping[str, Any]) -> bool:
    return any(
        key in model_args for key in ("device", "devices", "tensor_parallel_size")
    )


def _uses_existing_vllm_server(model_args: Mapping[str, Any]) -> bool:
    return ("base_url" in model_args) or (VLLM_BASE_URL in os.environ)


def _visible_cuda_devices() -> list[int | str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        return _parse_cuda_visible_devices(visible_devices)

    device_count = _cuda_device_count()
    if device_count <= 0:
        return []
    return list(range(device_count))


def _parse_cuda_visible_devices(value: str) -> list[int | str]:
    tokens = [token.strip() for token in value.split(",") if token.strip() != ""]
    if len(tokens) == 0:
        return []
    if len(tokens) == 1 and tokens[0] in ("-1", "none", "None"):
        return []
    if all(re.fullmatch(r"\d+", token) is not None for token in tokens):
        return [int(token) for token in tokens]
    return tokens


@lru_cache(maxsize=1)
def _cuda_device_count() -> int:
    try:
        import torch
    except Exception:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def close_model(model: Model) -> None:
    """Close model resources created for tournament generation or judging."""
    try:
        model.__exit__(None, None, None)
        return
    except RuntimeError as ex:
        if _is_benign_close_error(ex):
            _mark_model_closed(model)
            return
        if "require an async close" not in str(ex):
            logger.warning(f"Error while closing model '{model}': {ex}")
            return
    except Exception as ex:
        if _is_benign_close_error(ex):
            _mark_model_closed(model)
            return
        logger.warning(f"Error while closing model '{model}': {ex}")
        return

    try:
        _run_coroutine(model.__aexit__(None, None, None))
    except Exception as ex:
        if _is_benign_close_error(ex):
            _mark_model_closed(model)
            return
        logger.warning(f"Error while closing model '{model}': {ex}")


def _run_coroutine(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as ex:  # pragma: no cover
            error["value"] = ex

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result.get("value")


def _is_benign_close_error(ex: BaseException) -> bool:
    message = str(ex).strip().lower()
    return message in {
        "event loop is closed",
        "closed event loop",
    }


def _mark_model_closed(model: Model) -> None:
    if hasattr(model, "_closed"):
        try:
            setattr(model, "_closed", True)
        except Exception:
            return
