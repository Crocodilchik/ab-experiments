"""Независимый ручной пример и проверки SQL-подготовки на повреждённых данных."""

import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.config import read_config
from src.generate_data import EVENT_COLUMNS, generate_full, write_full, write_preview
from src.prepare_data import DataQualityError, SQL_DIR, _check_summaries, prepare_scenario


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manual():
    config = read_config(ROOT / "config.json")
    users = pd.DataFrame({
        "user_id": np.array([1, 2, 3], dtype=np.int64),
        "platform": pd.Series(["web", "android", "ios"], dtype="string"),
        "region": pd.Series(["central"] * 3, dtype="string"),
        "is_new_user": [False, True, False],
    })
    assignments = pd.DataFrame({
        "experiment_id": pd.Series(["listing_card_v1"] * 3, dtype="string"),
        "scenario": pd.Series(["uplift"] * 3, dtype="string"),
        "user_id": np.array([1, 2, 3], dtype=np.int64),
        "variant": pd.Series(["A", "A", "B"], dtype="string"),
        "assigned_at": pd.to_datetime([
            "2026-11-02T10:00:00Z", "2026-11-03T11:00:00Z", "2026-11-02T12:00:00Z",
        ], utc=True),
    })
    # Независимый пример: пользователь 2 намеренно не имеет ни одного события.
    events = pd.DataFrame({
        "experiment_id": pd.Series(["listing_card_v1"] * 6, dtype="string"),
        "scenario": pd.Series(["uplift"] * 6, dtype="string"),
        "event_id": np.arange(1, 7, dtype=np.int64),
        "user_id": np.array([1, 1, 1, 1, 3, 3], dtype=np.int64),
        "event_name": pd.Series([
            "search_view", "listing_view", "contact_created", "technical_error", "search_view", "listing_view",
        ], dtype="string"),
        "event_ts": pd.to_datetime([
            "2026-11-02T10:00:00Z", "2026-11-02T10:01:00Z", "2026-11-02T10:02:00Z",
            "2026-11-02T10:03:00Z", "2026-11-02T12:00:00Z", "2026-11-02T12:01:00Z",
        ], utc=True),
        "listing_id": pd.array([None, 501, 501, None, None, 502], dtype="Int64"),
    })[EVENT_COLUMNS]
    return config, {"users": users, "assignments": assignments, "events": events}


def pack(manual, path, *, preview=False):
    config, tables = manual
    writer = write_preview if preview else write_full
    writer(tables, config, "uplift", path)
    return path


def read_flags(output):
    relation = duckdb.read_parquet(str(output / "user_metrics.parquet"))
    return relation.project("user_id, variant, CAST(assigned_date AS VARCHAR), has_listing_view, has_contact, has_error").order("user_id").fetchall()


def read_summary(output, *, daily=False):
    table = "daily_summary" if daily else "experiment_summary"
    day = ", CAST(assigned_date AS VARCHAR)" if daily else ""
    columns = f"variant{day}, n_users, n_listing_view, n_contact, n_error, view_cr, contact_cr, error_rate, contact_given_view_rate"
    order = "variant, assigned_date" if daily else "variant"
    return duckdb.read_parquet(str(output / f"{table}.parquet")).order(order).project(columns).fetchall()


def test_manual_metrics_and_participant_without_events(manual, tmp_path):
    source = pack(manual, tmp_path / "source")
    before = {file.name: file.read_bytes() for file in source.iterdir()}
    report = prepare_scenario(source, tmp_path / "prepared")
    assert read_flags(tmp_path / "prepared") == [
        (1, "A", "2026-11-02", 1, 1, 1),
        (2, "A", "2026-11-03", 0, 0, 0),
        (3, "B", "2026-11-02", 1, 0, 0),
    ]
    assert report["assigned_users"] == report["user_metric_rows"] == 3
    assert report["data_checks_passed"] is True
    assert report["srm_checked"] is False
    assert {file.name: file.read_bytes() for file in source.iterdir()} == before


def test_exact_copy_does_not_change_user_metrics(manual, tmp_path):
    config, tables = manual
    first = pack(manual, tmp_path / "first")
    prepare_scenario(first, tmp_path / "prepared_first")
    tables["events"] = pd.concat([tables["events"], tables["events"].iloc[[2]]], ignore_index=True)
    second = pack((config, tables), tmp_path / "second")
    report = prepare_scenario(second, tmp_path / "prepared_second")
    assert report["exact_duplicates_removed"] == 1
    assert read_flags(tmp_path / "prepared_second") == read_flags(tmp_path / "prepared_first")
    for daily in (False, True):
        assert read_summary(tmp_path / "prepared_second", daily=daily) == read_summary(tmp_path / "prepared_first", daily=daily)


def test_event_key_conflict_blocks_export(manual, tmp_path):
    config, tables = manual
    conflict = tables["events"].iloc[[2]].copy()
    conflict["event_ts"] += pd.Timedelta(seconds=1)
    tables["events"] = pd.concat([tables["events"], conflict], ignore_index=True)
    source = pack((config, tables), tmp_path / "source")
    with pytest.raises(DataQualityError, match="event_key_conflict"):
        prepare_scenario(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_contact_requires_view_of_same_listing(manual, tmp_path):
    config, tables = manual
    tables["events"].loc[2, "listing_id"] = 9999
    source = pack((config, tables), tmp_path / "source")
    with pytest.raises(DataQualityError, match="contact_chain"):
        prepare_scenario(source, tmp_path / "prepared")


def test_event_on_right_boundary_is_excluded_not_conflict(manual, tmp_path):
    config, tables = manual
    tables["events"].loc[2, "event_ts"] = tables["assignments"].loc[0, "assigned_at"] + pd.Timedelta(hours=24)
    source = pack((config, tables), tmp_path / "source")
    report = prepare_scenario(source, tmp_path / "prepared")
    assert report["outside_window_events_excluded"] == 1
    assert read_flags(tmp_path / "prepared")[0] == (1, "A", "2026-11-02", 1, 0, 1)


@pytest.mark.parametrize("problem,code", [
    ("before_assignment", "event_before_assignment"),
    ("unknown_user", "unknown_users"),
    ("unknown_event", "event_names"),
    ("integer_experiment_id", "schema_events"),
])
def test_bad_data_are_rejected(manual, tmp_path, problem, code):
    config, tables = manual
    if problem == "before_assignment":
        tables["events"].loc[0, "event_ts"] -= pd.Timedelta(seconds=1)
    elif problem == "unknown_user":
        tables["assignments"].loc[0, "user_id"] = 999
    elif problem == "unknown_event":
        tables["events"].loc[0, "event_name"] = "unknown"
    else:
        tables["events"]["experiment_id"] = np.ones(len(tables["events"]), dtype=np.int64)
    source = pack((config, tables), tmp_path / "source")
    with pytest.raises(DataQualityError, match=code):
        prepare_scenario(source, tmp_path / "prepared")


def test_preview_is_rejected(manual, tmp_path):
    source = pack(manual, tmp_path / "preview", preview=True)
    with pytest.raises(DataQualityError, match="incomplete_source"):
        prepare_scenario(source, tmp_path / "prepared")


def test_immature_collection_is_rejected(manual, tmp_path):
    source = pack(manual, tmp_path / "source")
    path = source / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["collection_end"] = manifest["enrollment_end"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataQualityError, match="immature_collection"):
        prepare_scenario(source, tmp_path / "prepared")


def test_changed_configuration_snapshot_is_rejected(manual, tmp_path):
    source = pack(manual, tmp_path / "source")
    path = source / "config_snapshot.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["simulation"]["seed"] += 1
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(DataQualityError, match="config_hash"):
        prepare_scenario(source, tmp_path / "prepared")


def test_empty_cohort_is_not_lost_or_fabricated(tmp_path):
    config = read_config(ROOT / "config.json")
    config["simulation"]["daily_users_mean"] = 1e-12
    tables = generate_full(config["protocol"], config["simulation"], "uplift")
    source = pack((config, tables), tmp_path / "source")
    report = prepare_scenario(source, tmp_path / "prepared")
    assert report["assigned_users"] == 0
    assert read_flags(tmp_path / "prepared") == []
    for daily in (False, True):
        assert read_summary(tmp_path / "prepared", daily=daily) == []
    assert report["summary_ready"] is True
    assert report["experiment_summary_rows"] == report["daily_summary_rows"] == 0


def test_scenario_identity_is_checked_before_writing(manual, tmp_path):
    source = pack(manual, tmp_path / "source")
    with pytest.raises(DataQualityError, match="scenario_path"):
        prepare_scenario(source, tmp_path / "prepared", expected_scenario="null")
    assert not (tmp_path / "prepared").exists()


def test_cli_success_failure_and_no_overwrite(manual, tmp_path):
    source = pack(manual, tmp_path / "raw" / "uplift")
    output = tmp_path / "prepared"
    args = [sys.executable, "-B", "-m", "src.prepare_data", "--input", str(source.parent), "--scenario", "uplift", "--output", str(output)]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "uplift" / "quality_report.json").read_text(encoding="utf-8"))
    assert report["summary_ready"] is True
    assert (output / "uplift" / "experiment_summary.parquet").is_file()
    assert (output / "uplift" / "daily_summary.parquet").is_file()
    before = (output / "uplift" / "quality_report.json").read_bytes()
    again = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert again.returncode != 0
    assert (output / "uplift" / "quality_report.json").read_bytes() == before


def test_manual_summaries_export_rates_and_schema(manual, tmp_path):
    source = pack(manual, tmp_path / "source")
    output = tmp_path / "prepared"
    report = prepare_scenario(source, output)
    # Числа рассчитаны вручную: пользователь A без событий остаётся вторым в знаменателе.
    assert read_summary(output) == [
        ("A", 2, 1, 1, 1, 0.5, 0.5, 0.5, 1.0),
        ("B", 1, 1, 0, 0, 1.0, 0.0, 0.0, 0.0),
    ]
    assert read_summary(output, daily=True) == [
        ("A", "2026-11-02", 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0),
        ("A", "2026-11-03", 1, 0, 0, 0, 0.0, 0.0, 0.0, None),
        ("B", "2026-11-02", 1, 1, 0, 0, 1.0, 0.0, 0.0, 0.0),
    ]
    assert {path.name for path in output.iterdir()} == {
        "clean_events.parquet", "user_metrics.parquet", "experiment_summary.parquet",
        "daily_summary.parquet", "protocol.json", "quality_report.json",
    }
    assert json.loads((output / "quality_report.json").read_text(encoding="utf-8")) == report
    assert report["status"] == "prepared"
    assert report["summary_ready"] is True
    assert report["missing_components"] == []
    assert report["srm_checked"] is False
    assert report["experiment_summary_rows"] == 2
    assert report["daily_summary_rows"] == 3
    with duckdb.connect() as connection:
        for table in ("experiment_summary", "daily_summary"):
            connection.read_parquet(str(output / f"{table}.parquet")).create_view(table)
            schema = {row[0]: row[1] for row in connection.execute(f"DESCRIBE {table}").fetchall()}
            for column in ("n_users", "n_listing_view", "n_contact", "n_error"):
                assert schema[column] == "BIGINT"
            for column in ("view_cr", "contact_cr", "error_rate", "contact_given_view_rate"):
                assert schema[column] == "DOUBLE"
            if table == "daily_summary":
                assert schema["assigned_date"] == "DATE"


def test_unequal_days_use_total_counts_and_assignment_date(manual, tmp_path):
    config, tables = manual
    # У A один человек в первый день и двое во второй. Контакт первого — на следующий день.
    extra_user = tables["users"].iloc[[1]].copy()
    extra_user["user_id"] = 4
    tables["users"] = pd.concat([tables["users"], extra_user], ignore_index=True)
    extra_assignment = tables["assignments"].iloc[[1]].copy()
    extra_assignment["user_id"] = 4
    tables["assignments"] = pd.concat([tables["assignments"], extra_assignment], ignore_index=True)
    assigned_at = pd.Timestamp("2026-11-02T23:00:00Z")
    tables["assignments"].loc[0, "assigned_at"] = assigned_at
    for index, seconds in enumerate((0, 60, 7200, 120)):
        tables["events"].loc[index, "event_ts"] = assigned_at + pd.Timedelta(seconds=seconds)
    source = pack((config, tables), tmp_path / "source")
    output = tmp_path / "prepared"
    prepare_scenario(source, output)
    overall = read_summary(output)[0]
    daily = [row for row in read_summary(output, daily=True) if row[0] == "A"]
    assert [(row[1], row[2], row[4]) for row in daily] == [
        ("2026-11-02", 1, 1), ("2026-11-03", 2, 0),
    ]
    for daily_index, overall_index in ((2, 1), (3, 2), (4, 3), (5, 4)):
        assert sum(row[daily_index] for row in daily) == overall[overall_index]
    assert overall[6] == pytest.approx(1 / 3)
    assert sum(row[7] for row in daily) / 2 == pytest.approx(0.5)
    assert overall[6] != pytest.approx(0.5)


def test_observed_group_without_views_has_null_diagnostic(manual, tmp_path):
    config, tables = manual
    tables["users"] = tables["users"].iloc[:2].copy()
    tables["assignments"] = tables["assignments"].iloc[:2].copy()
    tables["events"] = tables["events"].iloc[:0].copy()
    source = pack((config, tables), tmp_path / "source")
    output = tmp_path / "prepared"
    prepare_scenario(source, output)
    # Отсутствующую B не выдумываем. Ноль контактов среди двух участников — определённая доля 0.
    assert read_summary(output) == [("A", 2, 0, 0, 0, 0.0, 0.0, 0.0, None)]
    assert all(row[-1] is None for row in read_summary(output, daily=True))


def test_repeated_logical_events_count_person_once(manual, tmp_path):
    config, tables = manual
    extra = tables["events"].iloc[[2, 3]].copy()
    extra["event_id"] = [7, 8]
    extra["event_ts"] += pd.Timedelta(minutes=5)
    tables["events"] = pd.concat([tables["events"], extra], ignore_index=True)
    source = pack((config, tables), tmp_path / "source")
    output = tmp_path / "prepared"
    report = prepare_scenario(source, output)
    assert report["clean_events"] == 8  # Это новые логические события, а не точные копии.
    assert report["exact_duplicates_removed"] == 0
    assert read_summary(output)[0] == ("A", 2, 1, 1, 1, 0.5, 0.5, 0.5, 1.0)


def test_summary_sql_keeps_experiments_scenarios_and_variants_separate():
    with duckdb.connect() as connection:
        connection.execute("""CREATE TABLE user_metrics AS
            SELECT * FROM (VALUES
                ('e1', 's1', 'A', DATE '2026-11-02', 1, 1, 0),
                ('e1', 's1', 'B', DATE '2026-11-02', 1, 0, 0),
                ('e1', 's2', 'A', DATE '2026-11-02', 0, 0, 1),
                ('e2', 's1', 'A', DATE '2026-11-02', 0, 0, 0)
            ) AS t(experiment_id, scenario, variant, assigned_date,
                   has_listing_view, has_contact, has_error)""").fetchall()
        for filename in ("03_experiment_summary.sql", "04_daily_summary.sql"):
            connection.execute((SQL_DIR / filename).read_text(encoding="utf-8")).fetchall()
        for table in ("experiment_summary", "daily_summary"):
            assert connection.execute(f"""SELECT experiment_id, scenario, variant,
                n_users, n_listing_view, n_contact, n_error FROM {table}
                ORDER BY experiment_id, scenario, variant""").fetchall() == [
                ("e1", "s1", "A", 1, 1, 1, 0),
                ("e1", "s1", "B", 1, 1, 0, 0),
                ("e1", "s2", "A", 1, 0, 0, 1),
                ("e2", "s1", "A", 1, 0, 0, 0),
            ]
        _check_summaries(connection, 4)
        # Число пользователей не меняется, но неверный числитель одного дня должен блокировать экспорт.
        connection.execute("UPDATE daily_summary SET n_contact = 0 WHERE experiment_id = 'e1' AND scenario = 's1' AND variant = 'A'").fetchall()
        with pytest.raises(DataQualityError, match="summary_daily_mismatch"):
            _check_summaries(connection, 4)
