from __future__ import annotations

from pathlib import Path

from dfm_evals.tournament._definitions import (
    list_tournament_definitions,
    resolve_tournament_definition,
)
from dfm_evals.tournament._resolve import resolve_tournament_config
from dfm_evals.tournament.config import load_tournament_config


def test_load_tournament_config_expands_prompt_source_from_definition_dir(
    tmp_path: Path,
) -> None:
    definition_dir = tmp_path / "configs" / "tournaments" / "demo"
    definition_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = definition_dir / "creative.jsonl"
    prompts_path.write_text(
        "\n".join(
            [
                '{"id": 1, "title": "First", "cateogry": "Essay", "prompt": "Write one."}',
                '{"id": 2, "title": "Second", "cateogry": "Blog", "prompt": "Write two."}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (definition_dir / "tournament.yaml").write_text(
        """
run_dir: ../../../logs/evals-logs/demo
project_id: demo
contestant_models:
  - vllm/model-a
  - vllm/model-b
prompt_source:
  path: ./creative.jsonl
  format: jsonl
  id_template: creative-writing-da-{id:02d}
  text_field: prompt
  metadata_fields:
    - title
    - cateogry
  metadata_rename:
    cateogry: category
  static_metadata:
    source_file: creative.jsonl
judge_model: openai/judge-model
judge_prompt_template: |
  Judge
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_tournament_config(definition_dir)

    assert config.run_dir == (tmp_path / "logs" / "evals-logs" / "demo").resolve()
    assert [prompt.id for prompt in config.prompts] == [
        "creative-writing-da-01",
        "creative-writing-da-02",
    ]
    assert [prompt.text for prompt in config.prompts] == [
        "Write one.",
        "Write two.",
    ]
    assert config.prompts[0].metadata == {
        "source_file": "creative.jsonl",
        "title": "First",
        "category": "Essay",
    }


def test_resolve_tournament_definition_lists_named_directories(tmp_path: Path) -> None:
    definitions_root = tmp_path / "configs" / "tournaments"
    alpha_dir = definitions_root / "alpha"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    (alpha_dir / "tournament.yaml").write_text("run_dir: ./logs\ncontestant_models: [a, b]\nprompts: [{id: p1, text: hi}]\njudge_model: judge/model\njudge_prompt_template: '{prompt}'\n", encoding="utf-8")
    (alpha_dir / "launch-map.yaml").write_text("contestants: {}\njudge: {mode: external_openai, base_url_env: OPENAI_BASE_URL}\n", encoding="utf-8")

    assert list_tournament_definitions(root=definitions_root) == [alpha_dir.resolve()]
    assert resolve_tournament_definition(
        "alpha",
        kind="config",
        root=definitions_root,
    ) == (alpha_dir / "tournament.yaml").resolve()
    assert resolve_tournament_definition(
        alpha_dir,
        kind="launch_map",
        root=definitions_root,
    ) == (alpha_dir / "launch-map.yaml").resolve()


def test_resolve_tournament_config_accepts_definition_dir(tmp_path: Path) -> None:
    definition_dir = tmp_path / "configs" / "tournaments" / "demo"
    definition_dir.mkdir(parents=True, exist_ok=True)
    (definition_dir / "tournament.yaml").write_text(
        """
run_dir: ../../../logs/evals-logs/demo
contestant_models:
  - model/A
  - model/B
prompts:
  - id: prompt-1
    text: Hello
judge_model: judge/model
judge_prompt_template: |
  Judge
""".strip()
        + "\n",
        encoding="utf-8",
    )

    resolved = resolve_tournament_config(definition_dir)

    assert resolved.contestant_models == ["model/A", "model/B"]
