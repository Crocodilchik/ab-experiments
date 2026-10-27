"""Проверки воспроизводимости, окон и крайних вероятностей основы генератора."""

from copy import deepcopy
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.config import read_config
from src.generate_data import generate_preview, write_preview


@pytest.fixture
def config():
    config = read_config(Path(__file__).resolve().parents[1] / "config.json")
    # Компактная когорта ускоряет тесты. Это отдельные тестовые настройки.
    config["simulation"]["daily_users_mean"] = 5
    return config


def generate(config, scenario="uplift"):
    return generate_preview(config["protocol"], config["simulation"], scenario)


def test_reproducible_after_other_scenario_and_no_input_mutation(config):
    before = deepcopy(config)
    first = generate(config)
    generate(config, "harmful")
    second = generate(config)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    assert config == before


def test_scenarios_share_pre_assignment_population(config):
    first, second = generate(config, "null"), generate(config, "broken_assignment")
    pd.testing.assert_frame_equal(first["users"], second["users"])
    pd.testing.assert_frame_equal(
        first["assignments"][["user_id", "assigned_at"]],
        second["assignments"][["user_id", "assigned_at"]],
    )


def test_assignments_preserve_all_users_and_arrival_period(config):
    tables = generate(config)
    users, assignments = tables["users"], tables["assignments"]
    assert len(users) > 0
    assert users["user_id"].is_unique
    assert set(users["user_id"]) == set(assignments["user_id"])
    assert not assignments.duplicated(["experiment_id", "scenario", "user_id"]).any()
    start = pd.Timestamp(config["protocol"]["start_at"])
    assert (assignments["assigned_at"] >= start).all()
    assert (assignments["assigned_at"] < start + pd.Timedelta(days=14)).all()


def test_events_obey_user_windows_and_listing_contract(config):
    tables = generate(config)
    events = tables["events"].merge(tables["assignments"], on=["experiment_id", "scenario", "user_id"], validate="many_to_one")
    assert events["event_id"].is_unique
    assert (events["event_ts"] >= events["assigned_at"]).all()
    assert (events["event_ts"] < events["assigned_at"] + pd.Timedelta(hours=24)).all()
    search = events[events["event_name"] == "search_view"]
    assert len(search) == len(tables["users"])
    assert search["event_ts"].equals(search["assigned_at"])
    assert search["listing_id"].isna().all()
    views = events[events["event_name"] == "listing_view"]
    assert views["listing_id"].notna().all()
    assert (views["event_ts"] > views["assigned_at"]).all()
    # После последнего допустимого просмотра остаётся минимум одна секунда.
    assert (views["event_ts"] < views["assigned_at"] + pd.Timedelta(hours=24) - pd.Timedelta(seconds=1)).all()
    assert "contact_created" not in set(events["event_name"])


def test_extreme_assignment_and_behavior_probabilities(config):
    scenario = config["simulation"]["scenarios"]["uplift"]
    scenario["actual_variant_b_probability"] = 1
    scenario["B"]["view_probability"] = 0
    scenario["B"]["error_probability"] = 1
    tables = generate(config)
    assert set(tables["assignments"]["variant"]) == {"B"}
    assert (tables["events"]["event_name"] == "listing_view").sum() == 0
    assert (tables["events"]["event_name"] == "technical_error").sum() == len(tables["users"])


def test_error_change_does_not_change_views_or_assignments(config):
    scenario = config["simulation"]["scenarios"]["uplift"]
    scenario["A"]["error_probability"] = scenario["B"]["error_probability"] = 0
    first = generate(config)
    scenario["A"]["error_probability"] = scenario["B"]["error_probability"] = 1
    second = generate(config)
    pd.testing.assert_frame_equal(first["assignments"], second["assignments"])
    columns = ["user_id", "event_name", "event_ts", "listing_id"]
    for name in ("search_view", "listing_view"):
        left = first["events"].loc[first["events"]["event_name"] == name, columns].reset_index(drop=True)
        right = second["events"].loc[second["events"]["event_name"] == name, columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize("mean", [5, 1e-12])
def test_parquet_roundtrip_and_explicit_partial_manifest(config, tmp_path, mean):
    config["simulation"]["daily_users_mean"] = mean
    tables = generate(config)
    destination = write_preview(tables, config, "uplift", tmp_path / "preview")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["missing_components"] == ["contact_created", "exact_event_duplicates"]
    for name, original in tables.items():
        relation = duckdb.read_parquet(str(destination / f"{name}.parquet"))
        columns = relation.columns
        # Сравниваем UTC-моменты числом микросекунд. Это проверяет сохранённое
        # время без pandas-конвертера и необязательного pytz для Python datetime.
        time_columns = [column for column in ("assigned_at", "event_ts") if column in columns]
        projection = ", ".join(
            f"epoch_us({column}) AS {column}" if column in time_columns else column
            for column in columns
        )
        restored = relation.project(projection).fetchall()
        assert columns == list(original)
        assert len(restored) == len(original)
        assert [row[columns.index("user_id")] for row in restored] == original["user_id"].tolist()
        for column in time_columns:
            assert "WITH TIME ZONE" in str(relation.types[columns.index(column)])
            assert [row[columns.index(column)] for row in restored] == [value.value // 1000 for value in original[column]]
        if name == "events":
            assert [row[columns.index("listing_id")] is None for row in restored] == original["listing_id"].isna().tolist()
    if mean == 1e-12:
        assert all(table.empty for table in tables.values())
    with pytest.raises(FileExistsError):
        write_preview(tables, config, "uplift", destination)
