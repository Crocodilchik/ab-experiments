"""Проверки публичных правил конфигурации и команды её проверки."""

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from src.config import (
    load_protocol, load_simulation, read_config,
    validate_protocol, validate_simulation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
# Пути описывают внешний контракт, а не внутренние циклы валидатора.
PROBABILITY_PATHS = [
    ("duplicate_event_probability",),
    ("scenarios", "uplift", "actual_variant_b_probability"),
    *[
        ("scenarios", "uplift", variant, field)
        for variant in ("A", "B")
        for field in ("view_probability", "contact_given_view_probability", "error_probability")
    ],
]


@pytest.fixture
def config():
    # Параметризованный пример каждый раз получает чистый исходный словарь.
    return read_config(CONFIG_PATH)


def _set_value(data, path, value):
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value


def _run_cli(path):
    # Проверяем настоящую команду тем же Python, которым запущен pytest.
    return subprocess.run(
        [sys.executable, "-B", "-m", "src.config", "--config", str(path)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def test_scenario_truth_does_not_change_analysis_protocol(config, tmp_path):
    original = load_protocol(CONFIG_PATH)
    config["simulation"]["scenarios"]["uplift"]["B"]["error_probability"] = 0.9
    config["simulation"]["scenarios"]["uplift"]["actual_variant_b_probability"] = 0.4
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    assert load_protocol(changed) == original
    assert original["expected_variant_b_probability"] == 0.5


@pytest.mark.parametrize("invalid", [-0.1, 1.1, True, float("nan")])
def test_invalid_alpha_is_rejected(config, invalid):
    config["protocol"]["alpha"] = invalid
    with pytest.raises(ValueError, match="alpha"):
        validate_protocol(config["protocol"])


def test_typo_is_not_silently_ignored(config):
    config["protocol"]["alhpa"] = config["protocol"].pop("alpha")
    with pytest.raises(ValueError, match="alpha"):
        validate_protocol(config["protocol"])


def test_fractional_duration_is_rejected(config):
    config["protocol"]["enrollment_days"] = 14.5
    with pytest.raises(ValueError, match="enrollment_days"):
        validate_protocol(config["protocol"])


def test_time_without_timezone_is_rejected(config):
    config["protocol"]["start_at"] = "2026-11-02T00:00:00"
    with pytest.raises(ValueError, match="UTC"):
        validate_protocol(config["protocol"])


def test_valid_simulation_accepts_intentionally_broken_assignment(config):
    simulation = config["simulation"]
    simulation["scenarios"]["broken_assignment"]["actual_variant_b_probability"] = 0.4
    validate_simulation(simulation)
    # Валидное описание поломки не изменяет ожидаемое назначение анализа.
    assert config["protocol"]["expected_variant_b_probability"] == 0.5


def test_load_simulation_returns_only_simulation():
    assert load_simulation(CONFIG_PATH) == read_config(CONFIG_PATH)["simulation"]


def test_validators_and_loaders_do_not_mutate_inputs(config):
    before = deepcopy(config)
    file_before = CONFIG_PATH.read_bytes()
    validate_protocol(config["protocol"])
    validate_simulation(config["simulation"])
    load_protocol(CONFIG_PATH)
    load_simulation(CONFIG_PATH)
    assert config == before
    assert CONFIG_PATH.read_bytes() == file_before


@pytest.mark.parametrize("invalid", [-1, 0.5, True, False, None, "42", float("nan")])
def test_invalid_seed_is_rejected(config, invalid):
    config["simulation"]["seed"] = invalid
    with pytest.raises(ValueError, match=r"simulation\.seed"):
        validate_simulation(config["simulation"])


@pytest.mark.parametrize("valid", [0, 42])
def test_nonnegative_integer_seed_is_accepted(config, valid):
    config["simulation"]["seed"] = valid
    validate_simulation(config["simulation"])


@pytest.mark.parametrize(
    "invalid", [0, -5, True, False, None, "3000", float("nan"), float("inf"), -float("inf")]
)
def test_invalid_daily_users_mean_is_rejected(config, invalid):
    config["simulation"]["daily_users_mean"] = invalid
    with pytest.raises(ValueError, match=r"simulation\.daily_users_mean"):
        validate_simulation(config["simulation"])


@pytest.mark.parametrize("valid", [0.5, 3000])
def test_positive_finite_traffic_is_accepted(config, valid):
    config["simulation"]["daily_users_mean"] = valid
    validate_simulation(config["simulation"])


@pytest.mark.parametrize("path", PROBABILITY_PATHS)
@pytest.mark.parametrize(
    "invalid", [-0.1, 1.1, True, False, float("nan"), float("inf"), -float("inf"), "0.2", None]
)
def test_invalid_probability_is_rejected_at_its_real_path(config, path, invalid):
    # Меняем реальное вложенное поле и передаём целый раздел simulation.
    _set_value(config["simulation"], path, invalid)
    expected_path = "simulation." + ".".join(path)
    with pytest.raises(ValueError, match=re.escape(expected_path)):
        validate_simulation(config["simulation"])


@pytest.mark.parametrize("path", PROBABILITY_PATHS)
@pytest.mark.parametrize("valid", [0, 1, 0.0, 1.0])
def test_numeric_probability_boundaries_are_accepted(config, path, valid):
    _set_value(config["simulation"], path, valid)
    validate_simulation(config["simulation"])


@pytest.mark.parametrize("invalid", [-1, 0, 1.5, True, False, "1", None])
def test_invalid_scenario_id_is_rejected(config, invalid):
    config["simulation"]["scenarios"]["null"]["scenario_id"] = invalid
    with pytest.raises(ValueError, match=r"simulation\.scenarios\.null\.scenario_id"):
        validate_simulation(config["simulation"])


def test_duplicate_scenario_id_is_rejected(config):
    scenarios = config["simulation"]["scenarios"]
    scenarios["uplift"]["scenario_id"] = scenarios["null"]["scenario_id"]
    with pytest.raises(ValueError, match=r"simulation\.scenarios\.uplift\.scenario_id"):
        validate_simulation(config["simulation"])


@pytest.mark.parametrize("path", [
    ("seed",),
    ("scenarios", "null"),
    ("scenarios", "uplift", "B"),
    ("scenarios", "uplift", "A", "view_probability"),
])
def test_missing_required_field_is_rejected(config, path):
    parent = config["simulation"]
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    parent_path = ".".join(("simulation", *path[:-1]))
    with pytest.raises(ValueError) as error:
        validate_simulation(config["simulation"])
    assert parent_path in str(error.value)
    assert path[-1] in str(error.value)


def test_unknown_simulation_field_is_rejected(config):
    config["simulation"]["actual_variant_b_probability "] = 0.5
    with pytest.raises(ValueError, match="actual_variant_b_probability "):
        validate_simulation(config["simulation"])


@pytest.mark.parametrize("variant", ["A", "B"])
@pytest.mark.parametrize("invalid", [None, [], 0, "parameters"])
def test_variant_must_be_an_object(config, variant, invalid):
    config["simulation"]["scenarios"]["uplift"][variant] = invalid
    expected_path = f"simulation.scenarios.uplift.{variant}"
    with pytest.raises(ValueError, match=re.escape(expected_path)):
        validate_simulation(config["simulation"])


@pytest.mark.parametrize("invalid", [None, [], "simulation"])
def test_simulation_must_be_an_object(invalid):
    with pytest.raises(ValueError, match="simulation"):
        validate_simulation(invalid)


def test_cli_accepts_valid_file():
    result = _run_cli(CONFIG_PATH)
    assert result.returncode == 0, result.stderr
    assert "Симуляция проверена." in result.stdout
    assert not result.stderr


@pytest.mark.parametrize("path,value", [
    (("daily_users_mean",), 0),
    (("scenarios", "uplift", "actual_variant_b_probability"), True),
    (("scenarios", "uplift", "B", "error_probability"), 1.7),
])
def test_cli_rejects_invalid_file(config, tmp_path, path, value):
    _set_value(config["simulation"], path, value)
    bad_path = tmp_path / "invalid.json"
    bad_path.write_text(json.dumps(config), encoding="utf-8")
    result = _run_cli(bad_path)
    assert result.returncode != 0
    assert "simulation." + ".".join(path) in result.stderr
    assert "Симуляция проверена." not in result.stdout


def test_cli_reports_malformed_json(tmp_path):
    bad_path = tmp_path / "malformed.json"
    bad_path.write_text('{"protocol":', encoding="utf-8")
    result = _run_cli(bad_path)
    assert result.returncode != 0
    assert "Ошибка конфигурации" in result.stderr
    assert "Traceback" not in result.stderr
