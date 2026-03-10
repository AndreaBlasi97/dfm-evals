import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def tournament_trace_file(trace_root: Path, phase: str) -> Iterator[None]:
    """Route Inspect trace logs into a writable tournament-local location."""
    existing_trace_file = os.environ.get("INSPECT_TRACE_FILE")
    if existing_trace_file is not None and existing_trace_file.strip() != "":
        yield
        return

    trace_root.mkdir(parents=True, exist_ok=True)
    trace_file = trace_root / f"{phase}-trace-{os.getpid()}.log"
    with _environ_var("INSPECT_TRACE_FILE", trace_file.as_posix()):
        yield


@contextmanager
def _environ_var(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
