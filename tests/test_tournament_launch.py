import importlib.util
import json
import os
import shlex
import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


def _install_trueskill_stub() -> None:
    if "trueskill" in sys.modules:
        return

    trueskill = types.ModuleType("trueskill")

    class Rating:
        def __init__(self, mu: float = 25.0, sigma: float = 25.0 / 3.0) -> None:
            self.mu = mu
            self.sigma = sigma

    class TrueSkill:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def cdf(self, value: float) -> float:
            del value
            return 0.5

    def rate_1vs1(
        first: Rating,
        second: Rating,
        drawn: bool = False,
        env: TrueSkill | None = None,
    ) -> tuple[Rating, Rating]:
        del drawn, env
        return first, second

    trueskill.Rating = Rating
    trueskill.TrueSkill = TrueSkill
    trueskill.rate_1vs1 = rate_1vs1
    sys.modules["trueskill"] = trueskill


_install_trueskill_stub()


@lru_cache(maxsize=1)
def _modules() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "lumi" / "tournament_launch.py"
    spec = importlib.util.spec_from_file_location("tournament_launch_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    from dfm_evals.tournament import _run_state as run_state
    from dfm_evals.tournament.config import (
        TournamentConfig,
        TournamentPrompt,
        load_tournament_config,
    )
    from dfm_evals.tournament.store import TournamentStore

    return {
        "TournamentConfig": TournamentConfig,
        "TournamentPrompt": TournamentPrompt,
        "TournamentStore": TournamentStore,
        "load_tournament_config": load_tournament_config,
        "module": module,
        "run_state": run_state,
    }


def _parse_shell_assignments(script: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if line == "":
            continue
        name, raw_value = line.split("=", maxsplit=1)
        if raw_value.startswith("(") and raw_value.endswith(")"):
            inner = raw_value[1:-1].strip()
            parsed[name] = shlex.split(inner) if inner else []
        else:
            tokens = shlex.split(raw_value)
            parsed[name] = tokens[0] if len(tokens) > 0 else ""
    return parsed


def _write_config(tmp_path: Path) -> Path:
    modules = _modules()
    TournamentConfig = modules["TournamentConfig"]
    TournamentPrompt = modules["TournamentPrompt"]

    config = TournamentConfig(
        run_dir=Path("logs/base"),
        project_id="demo-tournament",
        contestant_models=["vllm/org/model-a", "openai/model-b"],
        prompts=[
            TournamentPrompt(id="prompt-1", text="Hello"),
            TournamentPrompt(id="prompt-2", text="World"),
        ],
        judge_model="openai/judge-model",
        judge_prompt_template="Judge:\n{prompt}\nA:{response_a}\nB:{response_b}",
    )

    config_path = tmp_path / "configs" / "tournament.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def _write_launch_map(tmp_path: Path) -> Path:
    contestant_dir = tmp_path / "models" / "contestant-a"
    judge_dir = tmp_path / "models" / "judge-local"
    contestant_dir.mkdir(parents=True, exist_ok=True)
    judge_dir.mkdir(parents=True, exist_ok=True)

    launch_map = {
        "defaults": {
            "tp": 2,
            "dp": 4,
            "ctx": 8192,
            "gpu_mem": 0.88,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        },
        "judge_defaults": {
            "dp": 1,
            "gpu_mem": 0.75,
            "nodes": 1,
        },
        "contestants": {
            "vllm/org/model-a": {
                "mode": "local_vllm",
                "model": "../models/contestant-a",
                "nodes": 2,
                "visible_devices": "0,1",
                "default_chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
            "openai/model-b": {
                "mode": "external_openai",
                "base_url_env": "MODEL_B_BASE_URL",
                "api_key_env": "MODEL_B_API_KEY",
            },
        },
        "judge": {
            "mode": "local_vllm",
            "model": "../models/judge-local",
            "served_model_name": "judge-served",
            "api_key": "judge-secret",
            "enable_auto_tool_choice": False,
        },
    }

    launch_path = tmp_path / "configs" / "tournament.lumi.json"
    launch_path.write_text(json.dumps(launch_map, indent=2) + "\n", encoding="utf-8")
    return launch_path


def _write_launch_definition(tmp_path: Path) -> Path:
    definition_dir = tmp_path / "configs" / "demo"
    definition_dir.mkdir(parents=True, exist_ok=True)
    (definition_dir / "launch-map.yaml").write_text(
        """
defaults:
  tp: 2
contestants:
  vllm/org/model-a:
    mode: local_vllm
    model: ../models/contestant-a
judge:
  mode: external_openai
  base_url_env: OPENAI_BASE_URL
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return definition_dir


def test_emit_spec_shell_resolves_relative_paths_and_defaults(tmp_path: Path) -> None:
    module = _modules()["module"]
    config_path = _write_config(tmp_path)
    launch_path = _write_launch_map(tmp_path)

    launch_map = module.load_launch_map(launch_path)
    assert (
        launch_map.contestants["vllm/org/model-a"].model
        == (tmp_path / "models" / "contestant-a").resolve().as_posix()
    )
    assert (
        launch_map.judge.model
        == (tmp_path / "models" / "judge-local").resolve().as_posix()
    )

    contestant_spec = _parse_shell_assignments(
        module.emit_spec_shell(
            source=config_path,
            launch_map_path=launch_path,
            spec_kind="contestant",
            model_name="vllm/org/model-a",
        )
    )
    assert contestant_spec["SPEC_MODE"] == "local_vllm"
    assert contestant_spec["SPEC_MODEL"] == (tmp_path / "models" / "contestant-a").resolve().as_posix()
    assert contestant_spec["SPEC_SERVED_MODEL_NAME"] == "org/model-a"
    assert contestant_spec["SPEC_TP"] == "2"
    assert contestant_spec["SPEC_DP"] == "4"
    assert contestant_spec["SPEC_NODES"] == "2"
    assert contestant_spec["SPEC_CTX"] == "8192"
    assert contestant_spec["SPEC_GPU_MEM"] == "0.88"
    assert contestant_spec["SPEC_VISIBLE_DEVICES"] == "0,1"
    assert (
        contestant_spec["SPEC_DEFAULT_CHAT_TEMPLATE_KWARGS_JSON"]
        == '{"enable_thinking": false}'
    )
    assert contestant_spec["SPEC_ENABLE_AUTO_TOOL_CHOICE"] == "1"
    assert contestant_spec["SPEC_TOOL_CALL_PARSER"] == "hermes"

    external_spec = _parse_shell_assignments(
        module.emit_spec_shell(
            source=config_path,
            launch_map_path=launch_path,
            spec_kind="contestant",
            model_name="openai/model-b",
        )
    )
    assert external_spec["SPEC_MODE"] == "external_openai"
    assert external_spec["SPEC_SERVED_MODEL_NAME"] == "model-b"
    assert external_spec["SPEC_BASE_URL_ENV"] == "MODEL_B_BASE_URL"
    assert external_spec["SPEC_API_KEY_ENV"] == "MODEL_B_API_KEY"

    judge_spec = _parse_shell_assignments(
        module.emit_spec_shell(
            source=config_path,
            launch_map_path=launch_path,
            spec_kind="judge",
        )
    )
    assert judge_spec["SPEC_MODE"] == "local_vllm"
    assert judge_spec["SPEC_MODEL"] == (tmp_path / "models" / "judge-local").resolve().as_posix()
    assert judge_spec["SPEC_SERVED_MODEL_NAME"] == "judge-served"
    assert judge_spec["SPEC_DP"] == "1"
    assert judge_spec["SPEC_NODES"] == "1"
    assert judge_spec["SPEC_GPU_MEM"] == "0.75"
    assert judge_spec["SPEC_API_KEY"] == "judge-secret"
    assert judge_spec["SPEC_ENABLE_AUTO_TOOL_CHOICE"] == "0"


def test_write_runtime_config_and_emit_target_shell(tmp_path: Path) -> None:
    modules = _modules()
    module = modules["module"]
    config_path = _write_config(tmp_path)
    runtime_path = tmp_path / "runs" / "config" / "runtime.json"
    run_dir = tmp_path / "logs" / "tournament__demo__job-17"

    written_path = module.write_runtime_config(
        source=config_path,
        output=runtime_path,
        run_dir=run_dir,
    )

    assert written_path == runtime_path
    written = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert written["run_dir"] == run_dir.as_posix()
    assert written["project_id"] == "demo-tournament"

    target = _parse_shell_assignments(module.emit_target_shell(source=runtime_path))
    assert target["TARGET_RUN_DIR"] == run_dir.as_posix()
    assert target["TARGET_CONFIG_DIR"] == (run_dir / "config").as_posix()
    assert target["TARGET_INSPECT_DIR"] == (run_dir / "inspect").as_posix()
    assert target["TARGET_GENERATION_LOG_DIR"] == (run_dir / "inspect" / "generation").as_posix()
    assert target["TARGET_JUDGE_LOG_DIR"] == (run_dir / "inspect" / "judge").as_posix()
    assert target["TARGET_STATE_DIR"] == (run_dir / "state").as_posix()
    assert target["TARGET_EXPORTS_DIR"] == (run_dir / "exports").as_posix()
    assert target["TARGET_TRACES_DIR"] == (run_dir / "traces").as_posix()
    assert target["TARGET_SERVICES_DIR"] == (run_dir / "services").as_posix()
    assert target["TARGET_VLLM_LOG_DIR"] == (run_dir / "services" / "vllm").as_posix()
    assert target["TARGET_RUNTIME_CONFIG_PATH"] == (run_dir / "config" / "runtime.json").as_posix()
    assert target["TARGET_RUN_LABEL"] == "tournament__demo__job-17"
    assert target["TARGET_JUDGE_MODEL"] == "openai/judge-model"
    assert target["TARGET_PROJECT_ID"] == "demo-tournament"
    assert target["TARGET_CONTESTANT_MODELS"] == ["vllm/org/model-a", "openai/model-b"]


def test_emit_resource_shell_reports_required_nodes(tmp_path: Path) -> None:
    module = _modules()["module"]
    config_path = _write_config(tmp_path)
    launch_path = _write_launch_map(tmp_path)

    resources = _parse_shell_assignments(
        module.emit_resource_shell(
            source=config_path,
            launch_map_path=launch_path,
            phase="all",
        )
    )

    assert resources["RESOURCE_MAX_NODES"] == "2"
    assert resources["RESOURCE_MAX_LOCAL_WORLD_SIZE"] == "8"


def test_write_runtime_config_applies_model_overrides(tmp_path: Path) -> None:
    modules = _modules()
    module = modules["module"]
    load_tournament_config = modules["load_tournament_config"]
    config_path = _write_config(tmp_path)
    runtime_path = tmp_path / "runs" / "config" / "runtime.json"

    module.write_runtime_config(
        source=config_path,
        output=runtime_path,
        contestant_models=["openai/model-b", "vllm/org/model-c"],
        judge_model="openai/alternate-judge",
    )

    runtime_config = load_tournament_config(runtime_path)
    assert runtime_config.contestant_models == ["openai/model-b", "vllm/org/model-c"]
    assert runtime_config.judge_model == "openai/alternate-judge"


def test_load_launch_map_accepts_definition_dir(tmp_path: Path) -> None:
    module = _modules()["module"]
    definition_dir = _write_launch_definition(tmp_path)
    contestant_dir = tmp_path / "configs" / "models" / "contestant-a"
    contestant_dir.mkdir(parents=True, exist_ok=True)

    launch_map = module.load_launch_map(definition_dir)

    assert (
        launch_map.contestants["vllm/org/model-a"].model
        == contestant_dir.resolve().as_posix()
    )


def test_emit_status_shell_and_main_support_stateful_targets(
    tmp_path: Path,
    capsys,
) -> None:
    modules = _modules()
    module = modules["module"]
    TournamentStore = modules["TournamentStore"]
    load_tournament_config = modules["load_tournament_config"]
    run_state = modules["run_state"]

    config_path = _write_config(tmp_path)
    runtime_path = tmp_path / "runs" / "config" / "runtime.json"
    run_dir = tmp_path / "logs" / "tournament__stateful__job-9"

    module.write_runtime_config(
        source=config_path,
        output=runtime_path,
        run_dir=run_dir,
    )

    runtime_config = load_tournament_config(runtime_path)
    with TournamentStore(runtime_config.state_dir) as store:
        store.initialize_from_config(runtime_config)
        run_state.ensure_defaults(
            store,
            config_json=runtime_config.model_dump_json(),
            run_status="running",
        )

    status = _parse_shell_assignments(module.emit_status_shell(source=runtime_path))
    assert status["STATUS_RUN_STATUS"] == "running"
    assert status["STATUS_CONVERGED"] == "0"
    assert status["STATUS_MISSING_RESPONSES"] == "4"
    assert status["STATUS_MISSING_MODELS_CSV"] == ""
    assert status["STATUS_TOTAL_MATCHES"] == "0"
    assert status["STATUS_RATED_MATCHES"] == "0"
    assert status["STATUS_PENDING_BATCH_ID"] == ""

    exit_code = module.main(
        [
            "emit-target-shell",
            "--source",
            str(runtime_config.state_dir),
            "--stateful",
        ]
    )
    assert exit_code == 0

    emitted = _parse_shell_assignments(capsys.readouterr().out)
    assert emitted["TARGET_RUN_DIR"] == run_dir.as_posix()
    assert emitted["TARGET_RUN_LABEL"] == "tournament__stateful__job-9"


def test_emit_status_shell_can_index_generation_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from dfm_evals.tournament._provenance import (
        GENERATION_PHASE,
        GENERATION_TASK_NAME,
        TOURNAMENT_PHASE_KEY,
        TOURNAMENT_PROJECT_KEY,
        resolve_tournament_project_id,
    )

    modules = _modules()
    module = modules["module"]
    TournamentStore = modules["TournamentStore"]
    load_tournament_config = modules["load_tournament_config"]
    run_state = modules["run_state"]

    config_path = _write_config(tmp_path)
    runtime_path = tmp_path / "runs" / "config" / "runtime.json"
    run_dir = tmp_path / "logs" / "tournament__indexed__job-10"

    module.write_runtime_config(
        source=config_path,
        output=runtime_path,
        run_dir=run_dir,
    )

    runtime_config = load_tournament_config(runtime_path)
    with TournamentStore(runtime_config.state_dir) as store:
        store.initialize_from_config(runtime_config)
        run_state.ensure_defaults(
            store,
            config_json=runtime_config.model_dump_json(),
            run_status="running",
        )

    baseline = _parse_shell_assignments(module.emit_status_shell(source=runtime_path))
    assert baseline["STATUS_MISSING_RESPONSES"] == "4"

    indexer_module = sys.modules["dfm_evals.tournament.indexer"]
    matching_project_id = resolve_tournament_project_id(runtime_config)
    model_a_log = SimpleNamespace(
        name=(runtime_config.generation_log_dir / "model-a.eval").as_posix(),
        mtime=100.0,
    )
    model_b_log = SimpleNamespace(
        name=(runtime_config.generation_log_dir / "model-b.eval").as_posix(),
        mtime=100.0,
    )
    headers = {
        model_a_log.name: SimpleNamespace(
            eval=SimpleNamespace(
                task=GENERATION_TASK_NAME,
                model="vllm/org/model-a",
                metadata={
                    TOURNAMENT_PHASE_KEY: GENERATION_PHASE,
                    TOURNAMENT_PROJECT_KEY: matching_project_id,
                },
            )
        ),
        model_b_log.name: SimpleNamespace(
            eval=SimpleNamespace(
                task=GENERATION_TASK_NAME,
                model="openai/model-b",
                metadata={
                    TOURNAMENT_PHASE_KEY: GENERATION_PHASE,
                    TOURNAMENT_PROJECT_KEY: matching_project_id,
                },
            )
        ),
    }
    samples = {
        model_a_log.name: [
            SimpleNamespace(
                metadata={"prompt_id": "prompt-1"},
                id="prompt-1",
                output=SimpleNamespace(completion="A1"),
                uuid="uuid-a1",
            ),
            SimpleNamespace(
                metadata={"prompt_id": "prompt-2"},
                id="prompt-2",
                output=SimpleNamespace(completion="A2"),
                uuid="uuid-a2",
            ),
        ],
        model_b_log.name: [
            SimpleNamespace(
                metadata={"prompt_id": "prompt-1"},
                id="prompt-1",
                output=SimpleNamespace(completion="B1"),
                uuid="uuid-b1",
            ),
            SimpleNamespace(
                metadata={"prompt_id": "prompt-2"},
                id="prompt-2",
                output=SimpleNamespace(completion="B2"),
                uuid="uuid-b2",
            ),
        ],
    }

    monkeypatch.setattr(
        indexer_module,
        "list_eval_logs",
        lambda path: [model_a_log, model_b_log],
    )
    monkeypatch.setattr(
        indexer_module,
        "read_eval_log",
        lambda log_info, header_only=True: headers[log_info.name],
    )
    monkeypatch.setattr(
        indexer_module,
        "read_eval_log_samples",
        lambda log_info, all_samples_required=False: samples[log_info.name],
    )

    indexed = _parse_shell_assignments(
        module.emit_status_shell(source=runtime_path, index_generation=True)
    )
    assert indexed["STATUS_MISSING_RESPONSES"] == "0"
    assert indexed["STATUS_MISSING_MODELS_CSV"] == ""


def test_emit_status_shell_reports_missing_models_when_indexing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from dfm_evals.tournament._provenance import (
        GENERATION_PHASE,
        GENERATION_TASK_NAME,
        TOURNAMENT_PHASE_KEY,
        TOURNAMENT_PROJECT_KEY,
        resolve_tournament_project_id,
    )

    modules = _modules()
    module = modules["module"]
    TournamentStore = modules["TournamentStore"]
    load_tournament_config = modules["load_tournament_config"]
    run_state = modules["run_state"]

    config_path = _write_config(tmp_path)
    runtime_path = tmp_path / "runs" / "config" / "runtime.json"
    run_dir = tmp_path / "logs" / "tournament__partial__job-11"

    module.write_runtime_config(
        source=config_path,
        output=runtime_path,
        run_dir=run_dir,
    )

    runtime_config = load_tournament_config(runtime_path)
    with TournamentStore(runtime_config.state_dir) as store:
        store.initialize_from_config(runtime_config)
        run_state.ensure_defaults(
            store,
            config_json=runtime_config.model_dump_json(),
            run_status="running",
        )

    indexer_module = sys.modules["dfm_evals.tournament.indexer"]
    matching_project_id = resolve_tournament_project_id(runtime_config)
    model_a_log = SimpleNamespace(
        name=(runtime_config.generation_log_dir / "model-a.eval").as_posix(),
        mtime=100.0,
    )
    headers = {
        model_a_log.name: SimpleNamespace(
            eval=SimpleNamespace(
                task=GENERATION_TASK_NAME,
                model="vllm/org/model-a",
                metadata={
                    TOURNAMENT_PHASE_KEY: GENERATION_PHASE,
                    TOURNAMENT_PROJECT_KEY: matching_project_id,
                },
            )
        ),
    }
    samples = {
        model_a_log.name: [
            SimpleNamespace(
                metadata={"prompt_id": "prompt-1"},
                id="prompt-1",
                output=SimpleNamespace(completion="A1"),
                uuid="uuid-a1",
            ),
            SimpleNamespace(
                metadata={"prompt_id": "prompt-2"},
                id="prompt-2",
                output=SimpleNamespace(completion="A2"),
                uuid="uuid-a2",
            ),
        ],
    }

    monkeypatch.setattr(
        indexer_module,
        "list_eval_logs",
        lambda path: [model_a_log],
    )
    monkeypatch.setattr(
        indexer_module,
        "read_eval_log",
        lambda log_info, header_only=True: headers[log_info.name],
    )
    monkeypatch.setattr(
        indexer_module,
        "read_eval_log_samples",
        lambda log_info, all_samples_required=False: samples[log_info.name],
    )

    indexed = _parse_shell_assignments(
        module.emit_status_shell(source=runtime_path, index_generation=True)
    )
    assert indexed["STATUS_MISSING_RESPONSES"] == "2"
    assert indexed["STATUS_MISSING_MODELS_CSV"] == "openai/model-b"


def test_tournament_submit_dry_run_supports_add_model_phase(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "lumi" / "tournament_submit.sh"
    modules = _modules()
    module = modules["module"]
    TournamentStore = modules["TournamentStore"]
    load_tournament_config = modules["load_tournament_config"]
    run_state = modules["run_state"]

    config_path = _write_config(tmp_path)
    launch_path = _write_launch_map(tmp_path)
    runtime_path = tmp_path / "runs" / "config" / "runtime.json"
    target_run_dir = tmp_path / "logs" / "tournament__demo__job-123"
    module.write_runtime_config(
        source=config_path,
        output=runtime_path,
        run_dir=target_run_dir,
    )
    runtime_config = load_tournament_config(runtime_path)
    with TournamentStore(runtime_config.state_dir) as store:
        store.initialize_from_config(runtime_config)
        run_state.ensure_defaults(
            store,
            config_json=runtime_config.model_dump_json(),
            run_status="running",
        )

    overlay_dir = tmp_path / "overlay"
    overlay_dir.mkdir()
    submit_script = tmp_path / "submit.sbatch"
    submit_script.write_text("#!/bin/bash\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "ENV_FILE": str(tmp_path / "missing.env"),
            "OVERLAY_DIR": str(overlay_dir),
            "SUBMIT_SCRIPT": str(submit_script),
            "SLURM_LOG_DIR": str(tmp_path / "slurm"),
        }
    )

    result = subprocess.run(
        [
            "bash",
            script_path.as_posix(),
            "--phase",
            "add-model",
            "--target",
            target_run_dir.as_posix(),
            "--launch-map",
            launch_path.as_posix(),
            "--model",
            "vllm/org/model-a",
            "--dry-run",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Phase: add-model" in result.stdout
    assert "Add models: vllm/org/model-a" in result.stdout
    assert "Slurm nodes: 2" in result.stdout
    assert "DFM_TOURNAMENT_PHASE=add-model" in result.stdout
    assert "DFM_TOURNAMENT_ADDED_MODELS=vllm/org/model-a" in result.stdout
    assert "sbatch --nodes 2" in result.stdout


def test_tournament_submit_dry_run_supports_max_total_matches_override(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "lumi" / "tournament_submit.sh"
    modules = _modules()
    module = modules["module"]
    TournamentStore = modules["TournamentStore"]
    load_tournament_config = modules["load_tournament_config"]
    run_state = modules["run_state"]

    config_path = _write_config(tmp_path)
    launch_path = _write_launch_map(tmp_path)
    runtime_path = tmp_path / "runs" / "config" / "runtime.json"
    target_run_dir = tmp_path / "logs" / "tournament__demo__job-123"
    module.write_runtime_config(
        source=config_path,
        output=runtime_path,
        run_dir=target_run_dir,
    )
    runtime_config = load_tournament_config(runtime_path)
    with TournamentStore(runtime_config.state_dir) as store:
        store.initialize_from_config(runtime_config)
        run_state.ensure_defaults(
            store,
            config_json=runtime_config.model_dump_json(),
            run_status="running",
        )

    overlay_dir = tmp_path / "overlay"
    overlay_dir.mkdir()
    submit_script = tmp_path / "submit.sbatch"
    submit_script.write_text("#!/bin/bash\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "ENV_FILE": str(tmp_path / "missing.env"),
            "OVERLAY_DIR": str(overlay_dir),
            "SUBMIT_SCRIPT": str(submit_script),
            "SLURM_LOG_DIR": str(tmp_path / "slurm"),
        }
    )

    result = subprocess.run(
        [
            "bash",
            script_path.as_posix(),
            "--phase",
            "resume",
            "--target",
            target_run_dir.as_posix(),
            "--launch-map",
            launch_path.as_posix(),
            "--max-total-matches",
            "240",
            "--dry-run",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Max total matches override: 240" in result.stdout
    assert "DFM_TOURNAMENT_MAX_TOTAL_MATCHES=240" in result.stdout
