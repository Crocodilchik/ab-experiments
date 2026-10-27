"""Проверки итоговых материалов и одного настоящего полного запуска с notebook."""

from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

from src.analyze import SCENARIOS, write_analysis
from src.case_materials import METRIC_UNITS, build_metrics, write_case_materials
from src.config import read_config
from src.run_pipeline import run_project
from test_analyze import pack_summary


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cases(tmp_path):
    observed = {
        "null": [("A", 1000, 600, 120, 20), ("B", 1000, 600, 120, 20)],
        "uplift": [("A", 20000, 12000, 2400, 400), ("B", 20000, 13200, 2800, 400)],
        "harmful": [("A", 20000, 12000, 2400, 400), ("B", 20000, 13200, 2800, 800)],
        "broken_assignment": [("A", 600, 360, 72, 20), ("B", 400, 264, 52, 20)],
    }
    analyses = {}
    for name, rows in observed.items():
        source = pack_summary(tmp_path / "prepared" / name, rows, scenario=name)
        analyses[name] = write_analysis(source, tmp_path / "analysis" / name)
    return analyses, read_config(ROOT / "config.json")


def test_summary_keeps_observed_effect_separate_from_configured_effect(cases):
    analyses, config = cases
    before = deepcopy(cases)
    rows = build_metrics(analyses, config)
    assert [row["scenario"] for row in rows] == list(SCENARIOS)
    uplift = rows[1]
    assert uplift["contact_difference"] == pytest.approx(0.02)
    assert uplift["contact_difference_pp"] == pytest.approx(2.0)
    assert uplift["contact_relative_change_percent"] == pytest.approx(100 * 0.02 / 0.12)
    assert rows[-1]["diagnostic_only"] is True
    assert rows[-1]["decision_status"] == "invalid"
    assert set(uplift) == set(METRIC_UNITS)
    assert cases == before


def test_materials_csv_units_conclusions_and_notebook(cases, tmp_path):
    analyses, config = cases
    output = tmp_path / "report"
    write_case_materials(analyses, config, output)
    with (output / "metrics.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert float(rows[1]["contact_difference_pp"]) == pytest.approx(2.0)
    assert rows[-1]["diagnostic_only"] == "True"
    units = json.loads((output / "metrics_schema.json").read_text(encoding="utf-8"))["units"]
    assert units["contact_difference"] == "fraction"
    assert units["contact_difference_pp"] == "percentage_points"
    assert units["contact_relative_change_percent"] == "relative_percent"
    model = json.loads((output / "scenario_design.json").read_text(encoding="utf-8"))
    assert model["scenarios"]["uplift"]["contact_difference"] == pytest.approx(0.012)
    text = (output / "conclusions.md").read_text(encoding="utf-8")
    assert "Задано симулятору и оценено" in text
    assert "+1.2000" in text and "+2.0000" in text
    assert "только как диагностика" in text
    assert "реальных данных" in text
    notebook = nbformat.read(output / "experiment_review.ipynb", as_version=4)
    assert len([cell for cell in notebook.cells if cell.cell_type == "code"]) == 4
    assert all(cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code")
    assert "совместно с AI" in (output / "case_story.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_case_materials(analyses, config, output)


@pytest.mark.parametrize("problem,code", [("missing", "materials_scenarios"), ("incomplete", "materials_incomplete"),
                                        ("decision", "materials_decision"), ("threshold", "materials_protocol")])
def test_unreliable_material_inputs_are_rejected_before_writing(cases, tmp_path, problem, code):
    analyses, config = deepcopy(cases)
    if problem == "missing":
        del analyses["null"]
    elif problem == "incomplete":
        analyses["null"]["complete"] = False
    elif problem == "decision":
        analyses["null"]["decision"]["status"] = "recommend_further_check"
    else:
        # Меняем порог защиты в метаданных, оставляя сам статус неизменным.
        analyses["null"]["parameters"]["srm_alpha"] = 0.0005
    with pytest.raises(ValueError, match=code):
        write_case_materials(analyses, config, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("existing", ["run", "report"])
def test_pipeline_never_overwrites_existing_targets(tmp_path, existing):
    run, report = tmp_path / "run", tmp_path / "report"
    target = run if existing == "run" else report
    target.mkdir()
    (target / "sentinel.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_project(ROOT / "config.json", run, report)
    assert (target / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert not (report if existing == "run" else run).exists()


def test_pipeline_rejects_overlapping_paths_before_creating_them(tmp_path):
    target = tmp_path / "run"
    with pytest.raises(ValueError, match="pipeline_paths"):
        run_project(ROOT / "config.json", target, target / "report")
    assert not target.exists()


def test_pipeline_rejects_invalid_root_config_before_writing(tmp_path):
    config = read_config(ROOT / "config.json")
    config["unexpected"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="pipeline_config"):
        run_project(path, tmp_path / "run", tmp_path / "report")
    assert not (tmp_path / "run").exists()


def test_failed_stage_is_recorded_without_claiming_complete(tmp_path, monkeypatch):
    calls = []
    def fail(*args):
        calls.append(args[-1])
        raise RuntimeError("intentional generator failure")
    monkeypatch.setattr("src.run_pipeline.generate_full", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        run_project(ROOT / "config.json", tmp_path / "run", tmp_path / "report")
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["complete"] is False
    assert manifest["phase"] == "generate:null"
    assert manifest["notebook_executed"] is False
    assert calls == ["null"]
    assert not (tmp_path / "report").exists()


def test_blocked_notebook_socket_fails_promptly_with_incomplete_manifest(tmp_path, monkeypatch):
    config = read_config(ROOT / "config.json")
    config["simulation"]["daily_users_mean"] = 1e-12
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    def blocked(*args, **kwargs):
        raise PermissionError("local socket blocked")
    monkeypatch.setattr("src.run_pipeline.socket.create_connection", blocked)
    with pytest.raises(RuntimeError, match="локальных сокетов"):
        run_project(path, tmp_path / "run", tmp_path / "report")
    manifest = json.loads((tmp_path / "report" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["complete"] is False
    assert manifest["phase"] == "notebook" and manifest["notebook_executed"] is False
    notebook = nbformat.read(tmp_path / "report" / "experiment_review.ipynb", as_version=4)
    assert all(cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code")


def test_cli_runs_all_stages_and_executes_notebook_on_empty_population(tmp_path):
    # Пустая когорта допустима для подготовки; все продуктовые решения должны быть invalid.
    config = read_config(ROOT / "config.json")
    config["simulation"]["daily_users_mean"] = 1e-12
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    run, report = tmp_path / "run", tmp_path / "report"
    args = [sys.executable, "-B", "-m", "src.run_pipeline", "--config", str(path),
            "--run-dir", str(run), "--report-dir", str(report)]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((report / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest == json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["complete"] is True
    assert manifest["notebook_executed"] is True
    assert set(manifest["decisions"].values()) == {"invalid"}
    for key, digest in manifest["artifact_sha256"].items():
        prefix, relative = key.split('/', 1)
        file = (run if prefix == "run" else report) / relative
        assert hashlib.sha256(file.read_bytes()).hexdigest() == digest
    notebook = nbformat.read(report / "experiment_review.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == [1, 2, 3, 4]
    assert all(output.output_type != 'error' for cell in code for output in cell.outputs)
    with (report / "metrics.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert all(row["contact_rate_a"] == "" for row in rows)
    before = (report / "run_manifest.json").read_bytes()
    repeat = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert repeat.returncode != 0
    assert (report / "run_manifest.json").read_bytes() == before
