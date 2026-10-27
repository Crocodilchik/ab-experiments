"""Контакты, точные копии, полный экспорт и реальный запуск команды."""

from copy import deepcopy
import hashlib
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
from src.generate_data import (
    EVENT_COLUMNS, add_exact_duplicates, generate_contacts, generate_full,
    generate_preview, write_full,
)


ROOT = Path(__file__).resolve().parents[1]
KEY = ["experiment_id", "scenario", "event_id"]


@pytest.fixture
def config():
    config = read_config(ROOT / "config.json")
    config["simulation"]["daily_users_mean"] = 5
    config["simulation"]["duplicate_event_probability"] = 0
    return config


def full(config, name="uplift"):
    return generate_full(config["protocol"], config["simulation"], name)


def canonical(events):
    return events.drop_duplicates().sort_values(KEY).reset_index(drop=True)


def set_behavior(config, view, contact, error=0):
    for variant in ("A", "B"):
        parameters = config["simulation"]["scenarios"]["uplift"][variant]
        parameters.update(view_probability=view, contact_given_view_probability=contact, error_probability=error)


def two_views():
    """Два пользователя с заранее известными назначениями и просмотрами."""
    assigned = pd.to_datetime(["2026-11-02T10:00:00Z"] * 2, utc=True)
    assignments = pd.DataFrame({
        "experiment_id": pd.Series(["listing_card_v1"] * 2, dtype="string"),
        "scenario": pd.Series(["uplift"] * 2, dtype="string"),
        "user_id": np.array([1, 2], dtype=np.int64),
        "variant": pd.Series(["A", "B"], dtype="string"),
        "assigned_at": assigned,
    })
    events = assignments[["experiment_id", "scenario", "user_id"]].copy()
    events["event_id"] = np.array([10, 11], dtype=np.int64)
    events["event_name"] = pd.Series(["listing_view"] * 2, dtype="string")
    events["event_ts"] = assigned + pd.to_timedelta([900, 1200], unit="s")
    events["listing_id"] = pd.array([501, 502], dtype="Int64")
    return events[EVENT_COLUMNS], assignments


@pytest.mark.parametrize("view,contact,expected", [(0, 1, "none"), (1, 0, "none"), (1, 1, "all")])
def test_contact_probability_boundaries(config, view, contact, expected):
    set_behavior(config, view, contact, error=1)
    tables = full(config)
    events = tables["events"]
    contacts = events[events["event_name"] == "contact_created"]
    assert len(contacts) == (len(tables["users"]) if expected == "all" else 0)
    # Наличие технической ошибки не исключает контакт.
    assert (events["event_name"] == "technical_error").sum() == len(tables["users"])


def test_candidates_use_their_variant_and_inputs_are_preserved(config):
    views, assignments = two_views()
    original_views, original_assignments = views.copy(deep=True), assignments.copy(deep=True)
    scenario = config["simulation"]["scenarios"]["uplift"]
    scenario["A"]["contact_given_view_probability"] = 0
    scenario["B"]["contact_given_view_probability"] = 1
    contacts = generate_contacts(views, assignments, config["protocol"], config["simulation"], "uplift")
    assert contacts["user_id"].tolist() == [2]
    assert contacts["listing_id"].tolist() == [502]
    assert contacts["event_id"].min() > views["event_id"].max()
    pd.testing.assert_frame_equal(views, original_views)
    pd.testing.assert_frame_equal(assignments, original_assignments)


def test_repeated_view_produces_one_contact_from_first_listing(config):
    views, assignments = two_views()
    views = views.iloc[[0]].copy()
    repeated = views.copy()
    repeated["event_id"] = 20
    repeated["listing_id"] = pd.array([999], dtype="Int64")
    repeated["event_ts"] += pd.Timedelta(minutes=5)
    views = pd.concat([views, repeated], ignore_index=True)
    set_behavior(config, view=1, contact=1)
    contacts = generate_contacts(views, assignments, config["protocol"], config["simulation"], "uplift")
    assert len(contacts) == 1
    assert contacts["listing_id"].tolist() == [501]


def test_contact_can_use_last_second_but_never_window_end(config):
    views, assignments = two_views()
    views["event_ts"] = assignments["assigned_at"] + pd.Timedelta(hours=24) - pd.Timedelta(seconds=2)
    set_behavior(config, view=1, contact=1)
    contacts = generate_contacts(views, assignments, config["protocol"], config["simulation"], "uplift")
    joined = contacts.merge(assignments, on=["experiment_id", "scenario", "user_id"], validate="one_to_one")
    assert (joined["event_ts"] == joined["assigned_at"] + pd.Timedelta(hours=24) - pd.Timedelta(seconds=1)).all()


def test_contact_generation_reports_no_time_slot(config):
    views, assignments = two_views()
    views["event_ts"] = assignments["assigned_at"] + pd.Timedelta(hours=24) - pd.Timedelta(seconds=1)
    set_behavior(config, view=1, contact=1)
    with pytest.raises(ValueError, match="допустимой секунды"):
        generate_contacts(views, assignments, config["protocol"], config["simulation"], "uplift")


def test_contact_requires_listing_and_assignment(config):
    views, assignments = two_views()
    missing_listing = views.copy()
    missing_listing.loc[0, "listing_id"] = pd.NA
    with pytest.raises(ValueError, match="listing_id"):
        generate_contacts(missing_listing, assignments, config["protocol"], config["simulation"], "uplift")
    with pytest.raises(ValueError, match="назначения"):
        generate_contacts(views, assignments.iloc[0:0], config["protocol"], config["simulation"], "uplift")


@pytest.mark.parametrize("name", ["null", "uplift", "harmful", "broken_assignment"])
def test_full_events_preserve_base_and_obey_contact_chain(config, name):
    for variant in ("A", "B"):
        config["simulation"]["scenarios"][name][variant]["contact_given_view_probability"] = 1
    preview = generate_preview(config["protocol"], config["simulation"], name)
    result = full(config, name)
    events = canonical(result["events"])
    assert not events.duplicated(KEY).any()
    pd.testing.assert_frame_equal(
        canonical(events[events["event_name"] != "contact_created"]), canonical(preview["events"]),
    )
    pd.testing.assert_frame_equal(result["users"], preview["users"])
    pd.testing.assert_frame_equal(result["assignments"], preview["assignments"])
    views = events[events["event_name"] == "listing_view"]
    contacts = events[events["event_name"] == "contact_created"]
    joined = contacts.merge(
        views, on=["experiment_id", "scenario", "user_id", "listing_id"],
        suffixes=("_contact", "_view"), validate="one_to_one",
    ).merge(result["assignments"], on=["experiment_id", "scenario", "user_id"], validate="one_to_one")
    assert len(joined) == len(contacts) == len(views)
    assert (joined["event_ts_contact"] > joined["event_ts_view"]).all()
    assert (joined["event_ts_contact"] < joined["assigned_at"] + pd.Timedelta(hours=24)).all()


@pytest.mark.parametrize("probability,multiplier", [(0, 1), (1, 2)])
def test_duplicate_boundaries_and_exact_original_recovery(config, probability, multiplier):
    logical = full(config)["events"]
    before = logical.copy(deep=True)
    config["simulation"]["duplicate_event_probability"] = probability
    raw = add_exact_duplicates(logical, config["simulation"], "uplift")
    assert len(raw) == len(logical) * multiplier
    assert raw.groupby(KEY).size().eq(multiplier).all()
    pd.testing.assert_frame_equal(canonical(raw), canonical(logical))
    pd.testing.assert_frame_equal(logical, before)
    for frame in (logical, raw):
        contact_users = frame.loc[frame["event_name"] == "contact_created", "user_id"].nunique()
        assert contact_users == logical.loc[logical["event_name"] == "contact_created", "user_id"].nunique()


def test_duplicate_component_refuses_already_duplicated_input(config):
    logical = full(config)["events"]
    raw = pd.concat([logical, logical.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="уникальные логические"):
        add_exact_duplicates(raw, config["simulation"], "uplift")


def test_duplicate_probability_does_not_change_logical_events(config):
    original = deepcopy(config)
    first = full(config)
    full(config, "harmful")
    repeat = full(config)
    assert config == original
    for name in first:
        pd.testing.assert_frame_equal(first[name], repeat[name])
    config["simulation"]["duplicate_event_probability"] = 1
    changed = full(config)
    pd.testing.assert_frame_equal(canonical(first["events"]), canonical(changed["events"]))
    pd.testing.assert_frame_equal(first["assignments"], changed["assignments"])


@pytest.mark.parametrize("mean", [5, 1e-12])
def test_full_parquet_manifest_and_empty_cohort(config, tmp_path, mean):
    config["simulation"]["daily_users_mean"] = mean
    config["simulation"]["duplicate_event_probability"] = 1
    set_behavior(config, view=1, contact=1)
    tables = full(config)
    destination = write_full(tables, config, "uplift", tmp_path / "full")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads((destination / "config_snapshot.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["complete"] is True
    assert manifest["missing_components"] == []
    assert manifest["rows"] == {name: len(frame) for name, frame in tables.items()}
    assert manifest["exact_duplicate_rows"] == manifest["logical_event_count"]
    assert snapshot == config
    assert manifest["config_sha256"] == hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    events = duckdb.read_parquet(str(destination / "events.parquet"))
    names = [row[0] for row in events.project("event_name").fetchall()]
    assert names.count("contact_created") == len(tables["users"]) * 2
    if mean == 1e-12:
        assert all(table.empty for table in tables.values())
    with pytest.raises(FileExistsError):
        write_full(tables, config, "uplift", destination)


def run_cli(arguments):
    return subprocess.run(
        [sys.executable, "-B", "-m", "src.generate_data", *arguments],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def test_full_cli_all_scenarios_and_no_overwrite(config, tmp_path):
    set_behavior(config, view=1, contact=1)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "export"
    arguments = ["--full", "--scenario", "all", "--config", str(path), "--output", str(output)]
    result = run_cli(arguments)
    assert result.returncode == 0, result.stderr
    for name in config["simulation"]["scenarios"]:
        manifest = json.loads((output / name / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["complete"] is True
        assert manifest["scenario"] == name
    manifest_path = output / "uplift" / "manifest.json"
    before = manifest_path.read_bytes()
    repeated = run_cli(arguments)
    assert repeated.returncode != 0
    assert "уже существует" in repeated.stderr
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("arguments", [[], ["--full", "--preview"]])
def test_cli_requires_exactly_one_mode(arguments):
    result = run_cli(arguments)
    assert result.returncode != 0
