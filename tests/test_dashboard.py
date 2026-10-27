"""Проверки связи источников, знаменателей и автономного HTML-дашборда."""

from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import pytest

from src.analyze import write_analysis
from src.dashboard import build_dashboard_payload, build_figures, render_dashboard, write_dashboard
from src.prepare_data import prepare_scenario
from test_prepare_data import manual, pack


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(manual, tmp_path):
    raw = pack(manual, tmp_path / "raw" / "uplift")
    prepared = tmp_path / "prepared"
    analysis = tmp_path / "analysis"
    prepare_scenario(raw, prepared / "uplift")
    write_analysis(prepared / "uplift", analysis / "uplift")
    return analysis, prepared


def test_dashboard_uses_all_assigned_and_correct_daily_cohorts(inputs):
    payload, sources = build_dashboard_payload(*inputs, ("uplift",))
    model = payload["scenarios"]["uplift"]
    assert payload["synthetic"] is True
    assert model["diagnostic_only"] is True  # Малый ручной пример непригоден для теста.
    assert model["decision"]["status"] == "invalid"
    assert model["figures"]["funnel"]["data"][0]["y"] == [1.0, 0.5, 0.5]
    assert model["figures"]["funnel"]["data"][0]["customdata"] == [2, 1, 1]
    assert model["figures"]["daily-rate"]["data"][0]["x"] == ["2026-11-02", "2026-11-03"]
    assert model["figures"]["daily-rate"]["data"][0]["y"] == [1.0, 0.0]
    assert model["figures"]["daily-count"]["layout"]["yaxis"]["range"][0] == 0
    for feature in ("platform", "is_new_user"):
        assert sum(row["n_users"] for row in model["balance"] if row["feature"] == feature) == 3
    assert len(sources["uplift"]) == 6


def test_effect_figure_uses_pp_asymmetric_interval_and_threshold():
    model = {
        "groups": {"A": dict(n_users=100, n_listing_view=60, n_contact=10),
                   "B": dict(n_users=100, n_listing_view=66, n_contact=11)},
        "daily": [], "parameters": dict(business_contact_effect=0.005, max_error_increase=0.005),
        "primary": dict(difference=0.01, interval=dict(lower=-0.02, upper=0.03)),
        "guardrail": dict(difference=0.001, interval=dict(lower=-0.001, upper=0.004)),
    }
    before = deepcopy(model)
    figures = build_figures(model, dict(primary=4, guardrail=1, daily_count=10, daily_rate=1))
    effect = figures["primary"]
    assert effect["data"][0]["x"] == [1.0]
    assert effect["data"][0]["error_x"]["array"] == pytest.approx([2.0])
    assert effect["data"][0]["error_x"]["arrayminus"] == pytest.approx([3.0])
    assert [shape["x0"] for shape in effect["layout"]["shapes"]] == [0, 0.5]
    assert effect["layout"]["xaxis"]["range"] == [-4, 4]
    assert figures["funnel"]["data"][0]["marker"]["color"] == figures["daily-rate"]["data"][0]["line"]["color"]
    assert model == before


def test_scenario_switch_uses_the_same_axes_even_for_a_missing_group(manual, tmp_path):
    from src.generate_data import write_full
    for scenario in ("uplift", "null"):
        config, tables = deepcopy(manual)
        for table in ("assignments", "events"):
            tables[table]["scenario"] = scenario
        if scenario == "null":
            tables["assignments"]["variant"] = "A"
        raw = tmp_path / "raw" / scenario
        write_full(tables, config, scenario, raw)
        prepare_scenario(raw, tmp_path / "prepared" / scenario)
        write_analysis(tmp_path / "prepared" / scenario, tmp_path / "analysis" / scenario)
    payload, _ = build_dashboard_payload(tmp_path / "analysis", tmp_path / "prepared", ("uplift", "null"))
    models = payload["scenarios"]
    for name, axis in (("primary", "xaxis"), ("guardrail", "xaxis"), ("daily-count", "yaxis"), ("daily-rate", "yaxis")):
        assert models["uplift"]["figures"][name]["layout"][axis]["range"] == models["null"]["figures"][name]["layout"][axis]["range"]
    assert models["null"]["figures"]["funnel"]["data"][1]["y"] == [None, None, None]


class HTMLCheck(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.external_scripts = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "script" and attrs.get("src"):
            self.external_scripts.append(attrs["src"])


def test_export_is_self_contained_hashes_match_and_sources_do_not_change(inputs, tmp_path):
    before = {str(path): path.read_bytes() for root in inputs for path in root.rglob('*') if path.is_file()}
    output = tmp_path / "dashboard"
    manifest = write_dashboard(*inputs, output, ("uplift",))
    html = (output / "dashboard.html").read_text(encoding="utf-8")
    checker = HTMLCheck()
    checker.feed(html)
    assert checker.external_scripts == []
    assert len(checker.ids) == len(set(checker.ids))
    assert {"scenario-select", "funnel", "primary", "guardrail", "daily-count", "daily-rate"} <= set(checker.ids)
    assert "Plotly.react" in html and "addEventListener(\"change\"" in html
    assert "Учебная симуляция" in html and "досрочной остановки" in html
    assert "__DASHBOARD_PAYLOAD__" not in html and "__PLOTLY_BUNDLE__" not in html
    assert manifest["dashboard_sha256"] == hashlib.sha256((output / "dashboard.html").read_bytes()).hexdigest()
    assert set(path.name for path in output.iterdir()) == {"dashboard.html", "dashboard_data.json", "dashboard_manifest.json"}
    assert before == {str(path): path.read_bytes() for root in inputs for path in root.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        write_dashboard(*inputs, output, ("uplift",))


@pytest.mark.parametrize("problem,code", [
    ("incomplete", "dashboard_incomplete"), ("source", "dashboard_source_hash"),
    ("decision", "dashboard_decision"), ("parameters", "dashboard_parameters"),
    ("nonfinite", "dashboard_json"),
])
def test_bad_analysis_is_rejected_before_export(inputs, tmp_path, problem, code):
    path = inputs[0] / "uplift" / "analysis.json"
    analysis = json.loads(path.read_text(encoding="utf-8"))
    if problem == "incomplete":
        analysis["complete"] = False
    elif problem == "source":
        analysis["source_sha256"]["protocol.json"] = "wrong"
    elif problem == "decision":
        analysis["decision"]["status"] = "recommend_further_check"
    elif problem == "parameters":
        analysis["parameters"]["business_contact_effect"] = 0.001
    else:
        analysis["primary"]["difference"] = float("nan")
    path.write_text(json.dumps(analysis), encoding="utf-8")
    with pytest.raises(ValueError, match=code):
        write_dashboard(*inputs, tmp_path / "out", ("uplift",))
    assert not (tmp_path / "out").exists()


def test_wrong_daily_dates_with_same_overall_counts_are_rejected(inputs, tmp_path):
    path = inputs[1] / "uplift" / "daily_summary.parquet"
    with duckdb.connect() as connection:
        connection.read_parquet(str(path)).create_view("source_daily")
        connection.execute("CREATE TABLE altered AS SELECT * FROM source_daily").fetchall()
        connection.execute("UPDATE altered SET assigned_date = assigned_date + INTERVAL '1 day'").fetchall()
        connection.table("altered").write_parquet(str(path))
    with pytest.raises(ValueError, match="dashboard_daily_mart"):
        write_dashboard(*inputs, tmp_path / "out", ("uplift",))
    assert not (tmp_path / "out").exists()


def test_embedded_json_cannot_close_script_element(inputs):
    payload, _ = build_dashboard_payload(*inputs, ("uplift",))
    unsafe = '</script><script>alert("unsafe")</script>&\u2028'
    payload["scenarios"]["uplift"]["decision"]["reason"] = unsafe
    html = render_dashboard(payload)
    start = html.index('<script type="application/json" id="dashboard-data">')
    data = html[start:].split('>', 1)[1].split('</script>', 1)[0]
    assert "<script>" not in data
    assert json.loads(data)["scenarios"]["uplift"]["decision"]["reason"] == unsafe


def test_empty_population_has_no_invented_percentages_or_daily_points(tmp_path):
    from src.config import read_config
    from src.generate_data import generate_full
    config = read_config(ROOT / "config.json")
    config["simulation"]["daily_users_mean"] = 1e-12
    tables = generate_full(config["protocol"], config["simulation"], "uplift")
    raw = pack((config, tables), tmp_path / "raw")
    prepare_scenario(raw, tmp_path / "prepared" / "uplift")
    write_analysis(tmp_path / "prepared" / "uplift", tmp_path / "analysis" / "uplift")
    payload, _ = build_dashboard_payload(tmp_path / "analysis", tmp_path / "prepared", ("uplift",))
    model = payload["scenarios"]["uplift"]
    assert model["groups"]["A"]["n_users"] == 0
    assert model["figures"]["funnel"]["data"][0]["y"] == [None, None, None]
    assert model["figures"]["daily-count"]["data"][0]["x"] == []
    assert model["diagnostic_only"] is True
    json.dumps(payload, allow_nan=False)


def test_cli_build_and_no_overwrite(inputs, tmp_path):
    output = tmp_path / "out"
    args = [sys.executable, "-B", "-m", "src.dashboard", "--analysis", str(inputs[0]),
            "--prepared", str(inputs[1]), "--output", str(output), "--scenario", "uplift"]
    env = {**os.environ, "PYTHONUTF8": "1"}
    first = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert first.returncode == 0, first.stderr
    digest = hashlib.sha256((output / "dashboard.html").read_bytes()).hexdigest()
    again = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert again.returncode != 0
    assert hashlib.sha256((output / "dashboard.html").read_bytes()).hexdigest() == digest
