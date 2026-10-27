"""Независимые примеры правила решения: пороги, приоритеты и неизменность входа."""

from copy import deepcopy
import math

import pytest

from src.decision import make_decision


def valid_analysis():
    # Числа выбраны вручную для одной определённой положительной ветви.
    return {
        "data_checks_passed": True, "all_windows_mature": True, "inference_eligible": True,
        "srm": {"passed": True, "applicable": True},
        "parameters": {"alpha": 0.05, "business_contact_effect": 0.005, "max_error_increase": 0.005},
        "primary": {"applicability": {"passed": True}, "difference": 0.02,
                    "test": {"p_value": 0.01}, "interval": {"lower": 0.01, "upper": 0.03}},
        "guardrail": {"applicability": {"passed": True}, "interval": {"lower": -0.001, "upper": 0.002}},
    }


def changed(analysis, updates):
    result = deepcopy(analysis)
    for path, value in updates.items():
        target = result
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return result


@pytest.mark.parametrize("updates,status,guardrail,decline", [
    ({}, "recommend_further_check", "passed", False),
    ({"guardrail.interval.upper": 0.005}, "recommend_further_check", "passed", False),
    ({"guardrail.interval.upper": math.nextafter(0.005, math.inf)}, "guardrail_unconfirmed", "unconfirmed", False),
    ({"guardrail.interval.lower": 0.005, "guardrail.interval.upper": 0.01}, "guardrail_unconfirmed", "unconfirmed", False),
    ({"guardrail.interval.lower": math.nextafter(0.005, math.inf), "guardrail.interval.upper": 0.01}, "guardrail_harm", "harm", False),
    ({"primary.interval.lower": 0.005}, "inconclusive", "passed", False),
    ({"primary.interval.lower": math.nextafter(0.005, math.inf)}, "recommend_further_check", "passed", False),
    ({"primary.interval.lower": -0.003, "primary.interval.upper": 0.005,
      "primary.difference": 0.001, "primary.test.p_value": 0.4}, "inconclusive", "passed", False),
    ({"primary.interval.lower": -0.003, "primary.interval.upper": math.nextafter(0.005, -math.inf),
      "primary.difference": 0.001, "primary.test.p_value": 0.4}, "no_practical_gain", "passed", False),
    ({"primary.test.p_value": math.nextafter(0.05, -math.inf)}, "recommend_further_check", "passed", False),
    ({"primary.test.p_value": 0.05}, "inconclusive", "passed", False),
    ({"primary.test.p_value": math.nextafter(0.05, math.inf)}, "inconclusive", "passed", False),
    ({"primary.interval.lower": 0.0}, "inconclusive", "passed", False),
    ({"primary.interval.lower": -0.001}, "inconclusive", "passed", False),
    ({"primary.test.p_value": 0.4}, "inconclusive", "passed", False),
    ({"primary.interval.lower": -0.003, "primary.interval.upper": 0.004,
      "primary.difference": 0.001}, "inconclusive", "passed", False),
    ({"primary.interval.lower": -0.03, "primary.interval.upper": -0.01,
      "primary.difference": -0.02}, "no_practical_gain", "passed", True),
    ({"primary.interval.lower": -0.03, "primary.interval.upper": 0.0,
      "primary.difference": -0.01, "primary.test.p_value": 0.4}, "no_practical_gain", "passed", False),
    ({"primary.interval.lower": -0.02, "primary.interval.upper": 0.02,
      "primary.difference": 0.0, "primary.test.p_value": 1.0}, "inconclusive", "passed", False),
    # Нарушение защиты имеет приоритет даже перед расхождением основной проверки.
    ({"guardrail.interval.lower": 0.006, "guardrail.interval.upper": 0.01,
      "primary.test.p_value": 0.4}, "guardrail_harm", "harm", False),
])
def test_decision_thresholds_and_priorities(updates, status, guardrail, decline):
    analysis = changed(valid_analysis(), updates)
    before = deepcopy(analysis)
    result = make_decision(analysis)
    assert result["status"] == status
    assert result["guardrail_status"] == guardrail
    assert result["primary_decline"] is decline
    assert isinstance(result["reason"], str) and result["reason"].strip()
    assert set(result) == {"status", "guardrail_status", "primary_decline", "reason"}
    assert analysis == before


@pytest.mark.parametrize("flag", [
    "data_checks_passed", "all_windows_mature", "inference_eligible", "srm.applicable", "srm.passed",
    "primary.applicability.passed", "guardrail.applicability.passed",
])
def test_each_validity_gate_blocks_before_accessing_missing_intervals(flag):
    analysis = changed(valid_analysis(), {
        flag: False, "primary.interval": None, "primary.test": None, "guardrail.interval": None,
    })
    result = make_decision(analysis)
    assert result["status"] == "invalid"
    assert result["guardrail_status"] == "not_assessed"
    assert result["primary_decline"] is None


def test_srm_overrides_confirmed_guardrail_harm():
    analysis = changed(valid_analysis(), {
        "srm.passed": False, "guardrail.interval.lower": 0.006, "guardrail.interval.upper": 0.01,
    })
    result = make_decision(analysis)
    assert result["status"] == "invalid"
    assert "SRM" in result["reason"]


@pytest.mark.parametrize("path,value", [
    ("primary.test.p_value", float("nan")),
    ("guardrail.interval.upper", float("inf")),
    ("primary.difference", True),
    ("primary.interval", None),
    ("parameters.alpha", "0.05"),
    ("parameters.alpha", 0),
    ("primary.test.p_value", -0.01),
    ("primary.interval.lower", 0.04),
    ("guardrail.interval.upper", 2.0),
])
def test_missing_nonfinite_or_impossible_numbers_never_produce_a_recommendation(path, value):
    result = make_decision(changed(valid_analysis(), {path: value}))
    assert result["status"] == "invalid"
    assert result["guardrail_status"] == "not_assessed"
    assert result["primary_decline"] is None


@pytest.mark.parametrize("analysis", [{}, None])
def test_missing_analysis_cannot_pass(analysis):
    assert make_decision(analysis)["status"] == "invalid"


def test_decision_does_not_use_scenario_seed_or_simulated_effect():
    expected = make_decision(valid_analysis())
    for name in ("null", "uplift", "harmful", "broken_assignment"):
        analysis = valid_analysis()
        analysis.update(scenario=name, seed=999, simulation={"effect": -0.9})
        assert make_decision(analysis) == expected


def test_changed_protocol_thresholds_are_used_instead_of_hardcoded_values():
    analysis = changed(valid_analysis(), {"parameters.business_contact_effect": 0.015})
    assert make_decision(analysis)["status"] == "inconclusive"
    analysis = changed(valid_analysis(), {"parameters.max_error_increase": 0.001})
    assert make_decision(analysis)["status"] == "guardrail_unconfirmed"
