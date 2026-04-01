from __future__ import annotations

import sys

from dfm_evals.vllm_patches import apply_runtime_thread_safety_patches


def main() -> None:
    applied = apply_runtime_thread_safety_patches()
    print(
        "dfm_evals runtime patches: " + ", ".join(applied),
        file=sys.stderr,
        flush=True,
    )

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
