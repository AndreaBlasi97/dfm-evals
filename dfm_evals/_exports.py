from __future__ import annotations

from importlib import import_module

REGISTRY_EXPORTS = {
    "danske_talemaader": "dfm_evals.tasks.talemaader.task:danske_talemaader",
    "multi_wiki_qa": "dfm_evals.tasks.multi_wiki_qa:multi_wiki_qa",
    "gec_dala": "dfm_evals.tasks.gec_dala:gec_dala",
    "bfcl": "dfm_evals.tasks.bfcl.bfcl:bfcl",
    "bfcl_da": "dfm_evals.tasks.bfcl.bfcl:bfcl_da",
    "ifeval_da": "dfm_evals.tasks.ifeval_da:ifeval_da",
    "piqa": "dfm_evals.tasks.piqa:piqa",
    "gleu": "dfm_evals.scorers.gleu:gleu",
    "comet": "dfm_evals.scorers.comet:comet",
}

TOP_LEVEL_EXPORTS = (
    "danske_talemaader",
    "multi_wiki_qa",
    "bfcl",
    "bfcl_da",
    "ifeval_da",
    "gleu",
    "comet",
)

TASK_EXPORTS = tuple(
    name
    for name, target in REGISTRY_EXPORTS.items()
    if target.startswith("dfm_evals.tasks.")
)
SCORER_EXPORTS = tuple(
    name
    for name, target in REGISTRY_EXPORTS.items()
    if target.startswith("dfm_evals.scorers.")
)
SCORER_PACKAGE_EXPORTS = {
    **{name: REGISTRY_EXPORTS[name] for name in SCORER_EXPORTS},
    "compute_gleu": "dfm_evals.scorers.gleu:compute_gleu",
    "max_gleu_score": "dfm_evals.scorers.gleu:max_gleu_score",
}
SANDBOX_EXPORTS = {
    "PrimeSandboxEnvironment": "dfm_evals.sandboxes.prime:PrimeSandboxEnvironment",
}
REGISTRY_ONLY_MODULES = (
    "dfm_evals.sandboxes.prime",
    "dfm_evals.tournament.scorer",
)
REGISTRY_MODULES = (
    *REGISTRY_ONLY_MODULES,
    *dict.fromkeys(target.split(":", 1)[0] for target in REGISTRY_EXPORTS.values()),
)


def load_export(
    *,
    name: str,
    exports: dict[str, str],
    namespace: dict[str, object],
    module_name: str,
) -> object:
    target = exports.get(name)
    if target is None:
        raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

    target_module_name, attribute = target.split(":", 1)
    target_module = import_module(target_module_name)
    value = getattr(target_module, attribute)
    namespace[name] = value
    return value
