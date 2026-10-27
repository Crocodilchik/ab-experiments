"""Чтение и проверка протокола и параметров синтетических сценариев.

Здесь проверяются настройки, а не сгенерированные события или SRM.
Валидаторы сообщают об ошибке, но не изменяют и не исправляют входы.
"""

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path


PROTOCOL_FIELDS = {
    "experiment_id", "start_at", "enrollment_days", "outcome_window_hours",
    "expected_variant_b_probability", "baseline_contact_probability",
    "planning_contact_effect", "alpha", "target_power", "business_contact_effect",
    "max_error_increase", "srm_alpha", "ci_method",
}

SIMULATION_FIELDS = {
    "seed", "daily_users_mean", "duplicate_event_probability", "scenarios",
}

SCENARIO_FIELDS = {
    "null", "uplift", "harmful", "broken_assignment",
}

SCENARIO_INSIDES_FIELDS = {
    "scenario_id", "actual_variant_b_probability", "A", "B",
}

SCENARIO_INSIDES_AB_FIELDS = {
    "view_probability", "contact_given_view_probability", "error_probability"
}


def _require_fields(name: str, value: object, expected: set[str]) -> dict:
    """Проверить тип вложенного объекта и полный набор его полей."""
    if not isinstance(value, dict):
        raise ValueError(f"{name}: ожидается объект JSON, получен {type(value).__name__}")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ValueError(
            f"{name}: отсутствуют поля {sorted(missing)}; "
            f"лишние {sorted(extra, key=str)}"
        )
    return value


def read_config(path: str | Path) -> dict:
    """Прочитать JSON и проверить только внешнюю структуру документа."""
    with Path(path).open(encoding="utf-8-sig") as source:
        config = json.load(source)
    _require_fields("config", config, {"protocol", "simulation"})
    for name in ("protocol", "simulation"):
        if not isinstance(config[name], dict):
            raise ValueError(f"{name}: раздел должен быть объектом JSON")
    return config


def _number(name: str, value: object) -> float:
    """Число должно быть конечным; True/False не считаем числами параметров."""
    # В Python bool наследуется от int, но вероятность true была бы ошибкой ввода.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: ожидается число")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name}: число слишком велико") from error
    if not math.isfinite(number):
        raise ValueError(f"{name}: число должно быть конечным")
    return number


def _probability(name: str, value: object) -> float:
    """Числовые 0/1 допустимы; bool, NaN и бесконечности — нет."""
    number = _number(name, value)
    if not 0 <= number <= 1:
        raise ValueError(f"{name}: вероятность должна быть в пределах [0, 1]")
    return number


def _integer(name: str, value: object, minimum: int) -> int:
    """Проверить целый идентификатор/seed без неявного принятия True за 1."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: ожидается целое число не меньше {minimum}, не bool")
    return value


def validate_protocol(protocol: dict) -> None:
    """Отклонить неоднозначные или непригодные для текущего дизайна параметры."""
    _require_fields("protocol", protocol, PROTOCOL_FIELDS)
    experiment_id = protocol["experiment_id"]
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id: нужна непустая строка")

    start_at = protocol["start_at"]
    if not isinstance(start_at, str):
        raise ValueError("start_at: ожидается строка даты и времени UTC")
    try:
        start = datetime.fromisoformat(start_at)
    except ValueError as error:
        raise ValueError("start_at: неверный формат даты и времени") from error
    if start.tzinfo is None or start.utcoffset() != timedelta(0):
        raise ValueError("start_at: нужно явно указать UTC, например 2026-11-02T00:00:00Z")
    if any((start.hour, start.minute, start.second, start.microsecond)):
        raise ValueError("start_at: набор должен начинаться в 00:00 UTC")

    days = protocol["enrollment_days"]
    if type(days) is not int or days < 14 or days % 7:
        raise ValueError("enrollment_days: целое число суток, минимум 14, кратное 7")
    # Эта версия реализует согласованное суточное окно, а не произвольный дизайн.
    if type(protocol["outcome_window_hours"]) is not int or protocol["outcome_window_hours"] != 24:
        raise ValueError("outcome_window_hours: текущий протокол требует 24 часа")

    for name in (
        "expected_variant_b_probability", "baseline_contact_probability",
        "alpha", "target_power", "srm_alpha",
    ):
        if not 0 < _number(name, protocol[name]) < 1:
            raise ValueError(f"{name}: значение должно быть строго между 0 и 1")
    if protocol["expected_variant_b_probability"] != 0.5:
        raise ValueError("expected_variant_b_probability: согласованное назначение — 50/50")

    for name in ("planning_contact_effect", "business_contact_effect", "max_error_increase"):
        if not 0 < _number(name, protocol[name]) < 1:
            raise ValueError(f"{name}: требуется положительная разница в шкале долей")
    if protocol["baseline_contact_probability"] + protocol["planning_contact_effect"] >= 1:
        raise ValueError("baseline + planning_contact_effect: альтернатива должна быть меньше 1")
    if protocol["ci_method"] != "newcomb":
        raise ValueError("ci_method: в этом проекте заранее выбран newcomb")


def load_protocol(path: str | Path) -> dict:
    """Вернуть только протокол: анализу не нужны заданные вероятности сценариев."""
    protocol = read_config(path)["protocol"]
    validate_protocol(protocol)
    return protocol


def validate_simulation(simulation: dict) -> None:
    """Проверить описание симуляции, включая намеренно сломанное назначение."""
    _require_fields("simulation", simulation, SIMULATION_FIELDS)
    _integer("simulation.seed", simulation["seed"], minimum=0)
    traffic = _number("simulation.daily_users_mean", simulation["daily_users_mean"])
    if traffic <= 0:
        raise ValueError("simulation.daily_users_mean: требуется число строго больше нуля")
    _probability("simulation.duplicate_event_probability", simulation["duplicate_event_probability"])

    scenarios = _require_fields("simulation.scenarios", simulation["scenarios"], SCENARIO_FIELDS)
    seen_ids: dict[int, str] = {}
    for name, scenario in scenarios.items():
        path = f"simulation.scenarios.{name}"
        _require_fields(path, scenario, SCENARIO_INSIDES_FIELDS)
        scenario_id = _integer(f"{path}.scenario_id", scenario["scenario_id"], minimum=1)
        if scenario_id in seen_ids:
            raise ValueError(
                f"{path}.scenario_id: идентификатор {scenario_id} уже используется "
                f"в сценарии {seen_ids[scenario_id]}"
            )
        seen_ids[scenario_id] = name

        # Фактическая вероятность может отличаться от ожидаемых 50/50.
        # Согласованность назначения проверит SRM по данным, а не этот валидатор.
        _probability(f"{path}.actual_variant_b_probability", scenario["actual_variant_b_probability"])
        for variant in ("A", "B"):
            variant_path = f"{path}.{variant}"
            parameters = _require_fields(variant_path, scenario[variant], SCENARIO_INSIDES_AB_FIELDS)
            # Одинаковое правило применяется к каждому полю обеих групп ровно один раз.
            for field in sorted(SCENARIO_INSIDES_AB_FIELDS):
                _probability(f"{variant_path}.{field}", parameters[field])


def load_simulation(path: str | Path) -> dict:
    """Вернуть проверенные параметры генератора, без раздела protocol."""
    simulation = read_config(path)["simulation"]
    validate_simulation(simulation)
    return simulation


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Проверить разделы protocol и simulation в конфигурации")
    parser.add_argument("--config", type=Path, default=base_dir / "config.json")
    args = parser.parse_args()
    try:
        # Команда читает один снимок файла и сообщает успех лишь после обеих проверок.
        config = read_config(args.config)
        validate_protocol(config["protocol"])
        validate_simulation(config["simulation"])
    except (OSError, ValueError) as error:
        parser.exit(2, f"Ошибка конфигурации: {error}\n")
    protocol = config["protocol"]
    print(f"Протокол {protocol['experiment_id']} проверен.")
    print(f"Набор: {protocol['enrollment_days']} суток; окно: {protocol['outcome_window_hours']} часа.")
    print("Симуляция проверена.")


# Импорт функций в тесте не должен автоматически запускать командную строку.
if __name__ == "__main__":
    main()
