"""Проверки статистических оценок, полного решения и экспорта результатов."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys

import duckdb
import pytest

from src.analyze import (
    SUMMARY_SCHEMA, analyze_estimates, check_srm, estimate_proportions,
    render_decision, write_analysis, write_estimates,
)
from src.config import read_config


ROOT = Path(__file__).resolve().parents[1]


def pack_summary(path, rows=None, *, scenario="uplift"):
    """Независимые ручные количества, без запуска генератора и simulation на диске."""
    rows = [("A", 1000, 600, 120, 20), ("B", 1000, 660, 132, 20)] if rows is None else rows
    protocol = deepcopy(read_config(ROOT / "config.json")["protocol"])
    quality = {
        "status": "prepared", "experiment_id": protocol["experiment_id"], "scenario": scenario,
        "data_checks_passed": True, "all_windows_mature": True, "summary_ready": True,
        "missing_components": [], "srm_checked": False,
        "source_collection_end": "2026-11-17T00:00:00Z",
        "assigned_users": sum(row[1] for row in rows),
        "user_metric_rows": sum(row[1] for row in rows), "experiment_summary_rows": len(rows),
    }
    path.mkdir(parents=True)
    (path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (path / "quality_report.json").write_text(json.dumps(quality), encoding="utf-8")
    with duckdb.connect() as connection:
        columns = ", ".join(f"{name} {dtype}" for name, dtype in SUMMARY_SCHEMA.items())
        connection.execute(f"CREATE TABLE summary ({columns})").fetchall()
        for variant, n, views, contacts, errors in rows:
            connection.execute("INSERT INTO summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [protocol["experiment_id"], scenario, variant, n, views, contacts, errors,
                 views / n if n else None, contacts / n if n else None,
                 errors / n if n else None, contacts / views if views else None]).fetchall()
        connection.table("summary").write_parquet(str(path / "experiment_summary.parquet"))
    return path


def wilson_bounds(count, n):
    """Формула Wilson из числителя и знаменателя, независимо от statsmodels."""
    z = NormalDist().inv_cdf(0.975)
    p = count / n
    divisor = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / divisor
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / divisor
    return centre - radius, centre + radius


def test_difference_z_and_newcombe_against_independent_formulas():
    result = estimate_proportions(120, 1000, 132, 1000, 0.05)
    d = 0.132 - 0.12
    pooled = 252 / 2000
    z = d / math.sqrt(pooled * (1 - pooled) * (1 / 1000 + 1 / 1000))
    assert result["difference"] == pytest.approx(0.012)
    assert result["difference_pp"] == pytest.approx(1.2)
    assert result["relative_change_percent"] == pytest.approx(10)
    assert result["test"]["z_statistic"] == pytest.approx(z)
    assert result["test"]["p_value"] == pytest.approx(math.erfc(abs(z) / math.sqrt(2)))
    la, ua = wilson_bounds(120, 1000)
    lb, ub = wilson_bounds(132, 1000)
    assert result["interval"]["lower"] == pytest.approx(d - math.sqrt((0.132 - lb) ** 2 + (ua - 0.12) ** 2))
    assert result["interval"]["upper"] == pytest.approx(d + math.sqrt((ub - 0.132) ** 2 + (0.12 - la) ** 2))
    assert result["interval"]["method"] == "newcomb"
    assert result["interval"]["confidence_level"] == 0.95


def test_swapping_groups_changes_sign_and_reverses_interval():
    forward = estimate_proportions(120, 1000, 150, 1000, 0.05)
    reverse = estimate_proportions(150, 1000, 120, 1000, 0.05)
    assert reverse["difference"] == pytest.approx(-forward["difference"])
    assert reverse["interval"]["lower"] == pytest.approx(-forward["interval"]["upper"])
    assert reverse["interval"]["upper"] == pytest.approx(-forward["interval"]["lower"])
    assert reverse["test"]["p_value"] == pytest.approx(forward["test"]["p_value"])


def test_srm_balanced_and_broken_against_one_degree_chi_square():
    assert check_srm(500, 500, 0.5, 0.001)["p_value"] == 1
    broken = check_srm(600, 400, 0.5, 0.001)
    assert broken["expected"] == {"A": 500, "B": 500}
    assert broken["statistic"] == pytest.approx(40)
    assert broken["p_value"] == pytest.approx(math.erfc(math.sqrt(40 / 2)), rel=1e-10, abs=0)
    assert broken["passed"] is False


@pytest.mark.parametrize("n_a,n_b", [(0, 0), (2, 2)])
def test_srm_small_expected_counts_are_not_a_pass(n_a, n_b):
    result = check_srm(n_a, n_b, 0.5, 0.001)
    assert result["applicable"] is False
    assert result["passed"] is False
    assert result["p_value"] is None


@pytest.mark.parametrize("counts,reason", [
    ((0, 0, 10, 100), "empty_group"),
    ((9, 100, 10, 100), "observed_cell_below_10"),
    ((10, 20, 10, 1000), "expected_cell_below_10"),
    ((0, 100, 0, 100), "observed_cell_below_10"),
    ((100, 100, 100, 100), "observed_cell_below_10"),
])
def test_normal_approximation_is_blocked_without_switching_methods(counts, reason):
    result = estimate_proportions(*counts, 0.05)
    assert result["applicability"]["passed"] is False
    assert reason in result["applicability"]["reasons"]
    assert result["test"] is None
    assert result["interval"] is None
    json.dumps(result, allow_nan=False)


def test_ten_in_each_cell_is_allowed_and_guardrail_needs_no_separate_test():
    result = estimate_proportions(10, 20, 10, 20, 0.05, include_test=False)
    assert result["applicability"]["passed"] is True
    assert result["interval"] is not None
    assert result["test"] is None


@pytest.mark.parametrize("bad", [True, 1.0, -1, float("nan")])
def test_counts_reject_noninteger_negative_or_boolean_values(bad):
    with pytest.raises(ValueError, match="counts"):
        estimate_proportions(bad, 100, 20, 100, 0.05)
    with pytest.raises(ValueError, match="counts"):
        check_srm(bad, 100, 0.5, 0.001)


def test_impossible_success_count_is_rejected():
    with pytest.raises(ValueError, match="counts"):
        estimate_proportions(101, 100, 20, 100, 0.05)


def test_analysis_only_uses_prepared_counts_and_protocol(tmp_path):
    source = pack_summary(tmp_path / "source")
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    result = write_estimates(source, tmp_path / "output")
    assert result["complete"] is False
    assert result["missing_components"] == ["product_decision"]
    assert result["decision"] is None
    assert result["inference_eligible"] is True
    assert result["srm_checked"] is True
    assert result["guardrail"]["test"] is None
    assert result["sample_size"]["required_per_arm"] == 11999
    assert result["sample_size"]["below_planned_size"] is True
    assert 0 < result["sample_size"]["power_at_planned_alternative"] < 0.8
    assert json.loads((tmp_path / "output" / "estimates.json").read_text(encoding="utf-8")) == result
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def test_srm_blocks_product_inference_but_keeps_diagnostic_estimates(tmp_path):
    source = pack_summary(tmp_path / "source", [("A", 600, 360, 72, 20), ("B", 400, 264, 52, 20)], scenario="broken_assignment")
    result = analyze_estimates(source)
    assert result["srm"]["expected"] == {"A": 500, "B": 500}
    assert result["inference_eligible"] is False
    assert result["blocked_reasons"] == ["srm_detected"]
    assert result["primary"]["test"] is not None
    assert result["decision"] is None


@pytest.mark.parametrize("rows", [[], [("A", 1000, 600, 120, 20)]])
def test_empty_or_missing_group_is_explicitly_blocked(tmp_path, rows):
    source = pack_summary(tmp_path / "source", rows)
    result = analyze_estimates(source)
    assert result["inference_eligible"] is False
    assert result["primary"]["difference"] is None
    assert result["primary"]["interval"] is None
    assert result["sample_size"]["power_at_planned_alternative"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value,code", [
    ("summary_ready", False, "unprepared_source"),
    ("all_windows_mature", False, "unprepared_source"),
    ("data_checks_passed", False, "unprepared_source"),
    ("source_collection_end", "2026-11-16T00:00:00Z", "immature_analysis"),
    ("assigned_users", 1999, "analysis_participant_count"),
])
@pytest.mark.parametrize("writer", [write_estimates, write_analysis])
def test_bad_preparation_report_blocks_export(tmp_path, field, value, code, writer):
    source = pack_summary(tmp_path / "source")
    path = source / "quality_report.json"
    quality = json.loads(path.read_text(encoding="utf-8"))
    quality[field] = value
    path.write_text(json.dumps(quality), encoding="utf-8")
    with pytest.raises(ValueError, match=code):
        writer(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("rows,code", [
    ([("A", 1000, 600, 120, 20), ("A", 1000, 600, 120, 20)], "analysis_summary_key"),
    ([("C", 1000, 600, 120, 20)], "analysis_summary_key"),
    ([("A", 1000, 100, 120, 20)], "analysis_summary_counts"),
])
def test_bad_summary_keys_or_counts_are_rejected(tmp_path, rows, code):
    source = pack_summary(tmp_path / "source", rows)
    with pytest.raises(ValueError, match=code):
        analyze_estimates(source)


def test_cli_explicit_partial_mode_and_no_overwrite(tmp_path):
    source = pack_summary(tmp_path / "prepared" / "uplift")
    output = tmp_path / "estimates"
    args = [sys.executable, "-B", "-m", "src.analyze", "--input", str(source.parent),
            "--output", str(output), "--scenario", "uplift"]
    env = {**os.environ, "PYTHONUTF8": "1"}
    missing_mode = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert missing_mode.returncode != 0
    success = subprocess.run([*args, "--estimates"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert success.returncode == 0, success.stderr
    before = (output / "uplift" / "estimates.json").read_bytes()
    again = subprocess.run([*args, "--estimates"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert again.returncode != 0
    assert (output / "uplift" / "estimates.json").read_bytes() == before


def test_full_export_keeps_input_and_historical_estimates_unchanged(tmp_path):
    source = pack_summary(tmp_path / "source", [("A", 20000, 12000, 2400, 400), ("B", 20000, 13200, 2800, 400)])
    partial = write_estimates(source, tmp_path / "partial")
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    old_estimates = (tmp_path / "partial" / "estimates.json").read_bytes()
    output = tmp_path / "full"
    full = write_analysis(source, output)
    assert partial["complete"] is False and partial["decision"] is None
    assert full["stage"] == "analysis" and full["complete"] is True
    assert full["missing_components"] == []
    assert full["decision"]["status"] == "recommend_further_check"
    assert full["primary"] == partial["primary"]
    assert full["guardrail"] == partial["guardrail"]
    assert {path.name for path in output.iterdir()} == {"analysis.json", "decision.md"}
    assert json.loads((output / "analysis.json").read_text(encoding="utf-8")) == full
    report = (output / "decision.md").read_text(encoding="utf-8")
    for text in (full["decision"]["reason"], "Newcombe", "Защитная метрика", "SRM", "неокруглённым"):
        assert text in report
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert (tmp_path / "partial" / "estimates.json").read_bytes() == old_estimates
    with pytest.raises(FileExistsError):
        write_analysis(source, output)


def test_full_srm_block_is_complete_and_marked_diagnostic(tmp_path):
    source = pack_summary(tmp_path / "source", [("A", 600, 360, 72, 20), ("B", 400, 264, 52, 20)], scenario="broken_assignment")
    output = tmp_path / "full"
    result = write_analysis(source, output)
    assert result["complete"] is True
    assert result["decision"]["status"] == "invalid"
    assert result["decision"]["guardrail_status"] == "not_assessed"
    assert result["primary"]["test"] is not None
    assert "диагностический" in (output / "decision.md").read_text(encoding="utf-8")


def test_full_report_preserves_evidence_of_primary_decline(tmp_path):
    source = pack_summary(tmp_path / "source", [("A", 20000, 12000, 2400, 400), ("B", 20000, 12000, 2100, 400)])
    output = tmp_path / "full"
    result = write_analysis(source, output)
    assert result["decision"]["status"] == "no_practical_gain"
    assert result["decision"]["primary_decline"] is True
    assert "свидетельство снижения" in (output / "decision.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("rows", [[], [("A", 1000, 600, 120, 20)], [("A", 1, 0, 0, 0), ("B", 1, 0, 0, 0)]])
def test_full_report_handles_absent_or_inapplicable_estimates(tmp_path, rows):
    source = pack_summary(tmp_path / "source", rows)
    output = tmp_path / "full"
    result = write_analysis(source, output)
    assert result["complete"] is True
    assert result["decision"]["status"] == "invalid"
    assert result["primary"]["interval"] is None
    assert "не рассчитан" in (output / "decision.md").read_text(encoding="utf-8")
    json.dumps(result, allow_nan=False)


def test_partial_mode_does_not_call_decision_and_cannot_be_rendered_as_full(tmp_path, monkeypatch):
    source = pack_summary(tmp_path / "source")
    def forbidden(_):
        raise AssertionError("Промежуточный режим не должен принимать решение")
    monkeypatch.setattr("src.analyze.make_decision", forbidden)
    result = write_estimates(source, tmp_path / "partial")
    with pytest.raises(ValueError, match="incomplete_decision"):
        render_decision(result)


def test_full_mode_rejects_scenario_mismatch_before_writing(tmp_path):
    source = pack_summary(tmp_path / "source", scenario="uplift")
    with pytest.raises(ValueError, match="analysis_identity"):
        write_analysis(source, tmp_path / "full", expected_scenario="null")
    assert not (tmp_path / "full").exists()


def test_full_cli_all_scenarios_mode_conflict_and_no_overwrite(tmp_path):
    observed = {
        "null": [("A", 1000, 600, 120, 20), ("B", 1000, 600, 120, 20)],
        "uplift": [("A", 20000, 12000, 2400, 400), ("B", 20000, 13200, 2800, 400)],
        "harmful": [("A", 20000, 12000, 2400, 400), ("B", 20000, 13200, 2800, 800)],
        "broken_assignment": [("A", 600, 360, 72, 20), ("B", 400, 264, 52, 20)],
    }
    for name, rows in observed.items():
        pack_summary(tmp_path / "prepared" / name, rows, scenario=name)
    output = tmp_path / "full"
    args = [sys.executable, "-B", "-m", "src.analyze", "--input", str(tmp_path / "prepared"),
            "--output", str(output), "--scenario", "all", "--full"]
    env = {**os.environ, "PYTHONUTF8": "1"}
    conflict = subprocess.run([*args, "--estimates"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert conflict.returncode != 0 and not output.exists()
    success = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert success.returncode == 0, success.stderr
    expected = {"null": "guardrail_unconfirmed", "uplift": "recommend_further_check",
                "harmful": "guardrail_harm", "broken_assignment": "invalid"}
    for name, status in expected.items():
        result = json.loads((output / name / "analysis.json").read_text(encoding="utf-8"))
        assert result["decision"]["status"] == status
        assert result["complete"] is True
        assert (output / name / "decision.md").is_file()
    before = {str(path): path.read_bytes() for path in output.rglob('*') if path.is_file()}
    again = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert again.returncode != 0
    assert {str(path): path.read_bytes() for path in output.rglob('*') if path.is_file()} == before
