"""Regression checks for portable configuration and inter-stage handoffs."""
import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import main as cli
from baseline import policy_mapping


def test_stage2_log_records_default_personas_for_branching(tmp_path, monkeypatch):
    import stage2_toc.pipeline as pipeline

    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    (stage1 / "stage1_result.json").write_text("{}")
    (stage1 / "simulation_personas.json").write_text("{}")
    output = tmp_path / "stage2"
    config = {"llm": {"model": "test"}, "paths": {"outputs": {"stage2_toc": str(output)}}}
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "make_run_dir", lambda *a, **k: output)
    monkeypatch.setattr(pipeline, "run_stage2_toc", lambda **kwargs: SimpleNamespace(scenarios=[]))
    result = CliRunner().invoke(cli.app, ["stage2-toc", "test target", "--stage1-result", str(stage1 / "stage1_result.json")])
    assert result.exit_code == 0, result.output
    log = json.loads((output / "run_log.json").read_text())
    assert Path(log["simulation_personas"]).is_file()
    assert Path(log["simulation_personas"]) == stage1 / "simulation_personas.json"


def test_policy_override_routes_every_sampler(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEMOTRON_VALUES_MODULE", "dataset.processed.nemotron_kor")
    (tmp_path / "dataset/usa/example/input_policy").mkdir(parents=True)
    config = cli._apply_policy_override({}, "usa/example")
    persona = config["persona"]
    assert persona["scope_definer"]["values_module"] == "dataset.processed.nemotron_usa"
    assert persona["kg_gen"]["country"] == "usa"
    assert persona["nemotron"]["local_data_dir"].endswith("nemotron_usa/Nemotron-Personas-USA/data")
    assert config["paths"]["outputs"]["stage2_toc"] == "outputs/usa/example/stage2_toc"


def test_provider_model_names_do_not_create_nested_run_paths(tmp_path):
    config = {"llm": {"model": "provider/model"}, "paths": {"outputs": {"stage1": str(tmp_path)}}}
    run = cli.make_run_dir(config, "stage1")
    assert run.parent == tmp_path
    assert "provider_model" in run.name


def test_comparison_registry_uses_explicit_input_paths(tmp_path, monkeypatch):
    policy = tmp_path / "policy.md"
    personas = tmp_path / "simulation_personas.json"
    policy.write_text("Policy input")
    personas.write_text("{}")
    monkeypatch.setattr(policy_mapping, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(policy_mapping, "POLICY_MAPPING", {"custom": {"md_path": "policy.md", "personas_path": "simulation_personas.json"}})
    assert policy_mapping.resolve_md_path("custom") == policy
    assert policy_mapping.resolve_personas_path("custom") == personas
