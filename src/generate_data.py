"""Воспроизводимая симуляция событий: полный набор или неполный предпросмотр.

Генерация не выполняет SQL-очистку, статистические тесты и продуктовый вывод.
Предпросмотр намеренно сохраняется без контактов и точных копий.
"""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import read_config, validate_protocol, validate_simulation


SECONDS_PER_DAY = 24 * 60 * 60
# Числа закреплены по смыслу компонента, а не по порядку вызова функций.
STREAM_IDS = {
    "traffic": 1, "arrival_times": 2, "user_features": 3,
    "assignment": 10, "views": 11, "view_times": 12,
    "errors": 13, "error_times": 14,
    "contacts": 15, "contact_times": 16, "duplicates": 17,
}
EVENT_COLUMNS = [
    "experiment_id", "scenario", "event_id", "user_id",
    "event_name", "event_ts", "listing_id",
]
GENERATOR_VERSION = "stage03-full-v1"


def rng_for(seed: int, component: str, scenario_id: int = 0) -> np.random.Generator:
    """Создать воспроизводимый поток для одного механизма и сценария."""
    # scenario_id = 0 используется только для общей популяции и её прихода.
    # SeedSequence смешивает идентификаторы, сохраняя стабильность их значений.
    sequence = np.random.SeedSequence([STREAM_IDS[component], scenario_id, seed])
    return np.random.Generator(np.random.PCG64(sequence))


def generate_population(protocol: dict, simulation: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Создать общую популяцию и времена прихода; входы уже проверены."""
    seed = simulation["seed"]
    # Число участников случайно; 3000 в сутки — среднее, а не точная квота.
    daily_counts = rng_for(seed, "traffic").poisson(
        simulation["daily_users_mean"], size=protocol["enrollment_days"]
    )
    days = np.repeat(np.arange(protocol["enrollment_days"], dtype=np.int64), daily_counts)
    offsets = rng_for(seed, "arrival_times").integers(0, SECONDS_PER_DAY, size=len(days))
    seconds = days * SECONDS_PER_DAY + offsets
    times = pd.Series(
        pd.Timestamp(protocol["start_at"]) + pd.to_timedelta(seconds, unit="s")
    ).sort_values(ignore_index=True)
    ids = np.arange(1, len(times) + 1, dtype=np.int64)

    features = rng_for(seed, "user_features")
    users = pd.DataFrame({
        "user_id": ids,
        "platform": pd.Series(features.choice(["web", "android", "ios"], len(ids)), dtype="string"),
        "region": pd.Series(features.choice(["central", "northwest", "volga"], len(ids)), dtype="string"),
        "is_new_user": features.random(len(ids)) < 0.3,
    })
    arrivals = pd.DataFrame({"user_id": ids, "assigned_at": times})
    return users, arrivals


def generate_assignments(
    arrivals: pd.DataFrame, protocol: dict, simulation: dict, scenario_name: str,
) -> pd.DataFrame:
    """Один вариант на пользователя; назначения не зависят от его исходов."""
    scenario = simulation["scenarios"][scenario_name]
    rng = rng_for(simulation["seed"], "assignment", scenario["scenario_id"])
    variants = np.where(rng.random(len(arrivals)) < scenario["actual_variant_b_probability"], "B", "A")
    assignments = arrivals.copy()
    assignments["experiment_id"] = pd.Series([protocol["experiment_id"]] * len(arrivals), dtype="string")
    assignments["scenario"] = pd.Series([scenario_name] * len(arrivals), dtype="string")
    assignments["variant"] = pd.Series(variants, dtype="string")
    return assignments[["experiment_id", "scenario", "user_id", "variant", "assigned_at"]]


def _event_rows(
    participants: pd.DataFrame, event_name: str, offsets: np.ndarray,
    listing_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Собрать события выбранных пользователей с задержками в секундах."""
    participants = participants.reset_index(drop=True)
    events = participants[["experiment_id", "scenario", "user_id"]].copy()
    events["event_name"] = pd.Series([event_name] * len(participants), dtype="string")
    events["event_ts"] = participants["assigned_at"] + pd.to_timedelta(offsets, unit="s")
    # Nullable Int64 сохраняет смысл идентификатора при отсутствующем listing_id.
    values = [None] * len(participants) if listing_ids is None else listing_ids
    events["listing_id"] = pd.array(values, dtype="Int64")
    return events


def generate_base_events(
    assignments: pd.DataFrame, protocol: dict, simulation: dict, scenario_name: str,
) -> pd.DataFrame:
    """Показ выдачи, максимум один просмотр и одна ошибка на участника."""
    scenario = simulation["scenarios"][scenario_name]
    seed, scenario_id = simulation["seed"], scenario["scenario_id"]
    window_seconds = protocol["outcome_window_hours"] * 60 * 60
    is_b = assignments["variant"].to_numpy() == "B"
    view_probability = np.where(is_b, scenario["B"]["view_probability"], scenario["A"]["view_probability"])
    error_probability = np.where(is_b, scenario["B"]["error_probability"], scenario["A"]["error_probability"])
    viewers = assignments.loc[rng_for(seed, "views", scenario_id).random(len(assignments)) < view_probability]
    errors = assignments.loc[rng_for(seed, "errors", scenario_id).random(len(assignments)) < error_probability]

    # Последняя секунда окна зарезервирована: после просмотра должно оставаться
    # место хотя бы для одного контакта, строго позже просмотра и раньше конца.
    view_offsets = rng_for(seed, "view_times", scenario_id).integers(1, window_seconds - 1, size=len(viewers))
    error_offsets = rng_for(seed, "error_times", scenario_id).integers(0, window_seconds, size=len(errors))
    listing_ids = 1000 + viewers["user_id"].to_numpy() % 500
    events = pd.concat([
        _event_rows(assignments, "search_view", np.zeros(len(assignments), dtype=np.int64)),
        _event_rows(viewers, "listing_view", view_offsets, listing_ids),
        _event_rows(errors, "technical_error", error_offsets),
    ], ignore_index=True)
    # Уникальность — на сценарий эксперимента; сортировка не меняет сам event_id.
    events["event_id"] = np.arange(1, len(events) + 1, dtype=np.int64)
    return events[EVENT_COLUMNS].sort_values(["event_ts", "event_id"], ignore_index=True)


def generate_contacts(
    events: pd.DataFrame, assignments: pd.DataFrame, protocol: dict,
    simulation: dict, scenario_name: str,
) -> pd.DataFrame:
    """Вернуть только новые контакты по логическим событиям до добавления копий.

Один пользователь получает один розыгрыш контакта. Если просмотров несколько,
используется первый; объявление контакта берётся именно из этого просмотра.
"""
    scenario = simulation["scenarios"][scenario_name]
    seed, scenario_id = simulation["seed"], scenario["scenario_id"]
    window_seconds = protocol["outcome_window_hours"] * 60 * 60
    keys = ["experiment_id", "scenario", "user_id"]
    views = events.loc[events["event_name"] == "listing_view"].copy()
    # Схема и типы сохраняются даже при отсутствии просмотревших.
    if views.empty:
        return events[EVENT_COLUMNS].iloc[0:0].copy()
    views = views.sort_values(["event_ts", "event_id"]).drop_duplicates(keys)
    candidates = views.merge(
        assignments[keys + ["variant", "assigned_at"]],
        on=keys, how="left", validate="many_to_one",
    )
    if candidates["assigned_at"].isna().any() or not candidates["variant"].isin(["A", "B"]).all():
        raise ValueError("generate_contacts: у просмотра нет корректного назначения A/B")
    if candidates["listing_id"].isna().any():
        raise ValueError("generate_contacts: у просмотра отсутствует listing_id")
    view_delays = (candidates["event_ts"] - candidates["assigned_at"]).dt.total_seconds()
    if ((view_delays < 0) | (view_delays >= window_seconds)).any():
        raise ValueError("generate_contacts: просмотр вне индивидуального окна")

    # Маска имеет длину кандидатов, а не всех назначенных пользователей.
    is_b = candidates["variant"].to_numpy() == "B"
    probability = np.where(
        is_b, scenario["B"]["contact_given_view_probability"],
        scenario["A"]["contact_given_view_probability"],
    )
    mask = rng_for(seed, "contacts", scenario_id).random(len(candidates)) < probability
    contacts = candidates.loc[mask].reset_index(drop=True)
    if contacts.empty:
        return events[EVENT_COLUMNS].iloc[0:0].copy()

    # _event_rows ожидает задержку от назначения. Нижняя целая секунда строго
    # позже просмотра, верхняя граница окна исключена методом integers.
    delays = (contacts["event_ts"] - contacts["assigned_at"]).dt.total_seconds().to_numpy()
    lower_bounds = np.floor(delays).astype(np.int64) + 1
    if (lower_bounds >= window_seconds).any():
        raise ValueError("generate_contacts: после просмотра не осталось допустимой секунды")
    offsets = rng_for(seed, "contact_times", scenario_id).integers(lower_bounds, window_seconds)
    result = _event_rows(
        contacts, "contact_created", offsets,
        contacts["listing_id"].to_numpy(dtype=np.int64),
    )
    # ID исходных событий не меняются; новые продолжают их последовательность.
    first_id = int(events["event_id"].max()) + 1
    result["event_id"] = np.arange(first_id, first_id + len(result), dtype=np.int64)
    return result[EVENT_COLUMNS].sort_values(["event_ts", "event_id"], ignore_index=True)


def add_exact_duplicates(
    logical_events: pd.DataFrame, simulation: dict, scenario_name: str,
) -> pd.DataFrame:
    """Каждому исходному событию независимо добавить не больше одной копии."""
    key = ["experiment_id", "scenario", "event_id"]
    if logical_events.duplicated(key).any():
        raise ValueError("add_exact_duplicates: вход должен содержать уникальные логические события")
    if logical_events.empty:
        return logical_events.copy()
    scenario_id = simulation["scenarios"][scenario_name]["scenario_id"]
    rng = rng_for(simulation["seed"], "duplicates", scenario_id)
    mask = rng.random(len(logical_events)) < simulation["duplicate_event_probability"]
    copies = logical_events.loc[mask].copy()
    # event_id и все остальные поля копий сохраняются. Индекс pandas — не ID.
    return pd.concat([logical_events, copies], ignore_index=True).sort_values(
        ["event_ts", "event_id"], ignore_index=True,
    )

def generate_preview(protocol: dict, simulation: dict, scenario_name: str) -> dict[str, pd.DataFrame]:
    """Собрать неполный набор без изменения настроек и глобального RNG."""
    validate_protocol(protocol)
    validate_simulation(simulation)
    if scenario_name not in simulation["scenarios"]:
        raise ValueError(f"Неизвестный сценарий: {scenario_name}")
    users, arrivals = generate_population(protocol, simulation)
    assignments = generate_assignments(arrivals, protocol, simulation, scenario_name)
    events = generate_base_events(assignments, protocol, simulation, scenario_name)
    return {"users": users, "assignments": assignments, "events": events}


def generate_full(protocol: dict, simulation: dict, scenario_name: str) -> dict[str, pd.DataFrame]:
    """Сохранить базовые исходы, добавить контакты и только затем точные копии."""
    tables = generate_preview(protocol, simulation, scenario_name)
    base_events = tables["events"]
    contacts = generate_contacts(base_events, tables["assignments"], protocol, simulation, scenario_name)
    logical_events = pd.concat([base_events, contacts], ignore_index=True).sort_values(
        ["event_ts", "event_id"], ignore_index=True,
    )
    tables["events"] = add_exact_duplicates(logical_events, simulation, scenario_name)
    return tables


def _write_dataset(
    tables: dict[str, pd.DataFrame], config: dict, scenario_name: str, output_dir: str | Path,
    *, complete: bool,
) -> Path:
    """Общая запись файлов; режим задают отдельные функции экспорта."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with duckdb.connect() as connection:
        for name in ("users", "assignments", "events"):
            connection.from_df(tables[name]).write_parquet(str(output_dir / f"{name}.parquet"))
    protocol = config["protocol"]
    start = pd.Timestamp(protocol["start_at"])
    enrollment_end = start + pd.Timedelta(days=protocol["enrollment_days"])
    canonical_config = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    events = tables["events"]
    logical_count = len(events.drop_duplicates())
    manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rng_algorithm": "PCG64",
        "status": "complete" if complete else "partial_preview",
        "complete": complete,
        "missing_components": [] if complete else ["contact_created", "exact_event_duplicates"],
        "experiment_id": protocol["experiment_id"],
        "scenario": scenario_name,
        "seed": config["simulation"]["seed"],
        "config_sha256": hashlib.sha256(canonical_config).hexdigest(),
        "start_at": start.isoformat(),
        "enrollment_end": enrollment_end.isoformat(),
        "collection_end": (enrollment_end + pd.Timedelta(hours=protocol["outcome_window_hours"])).isoformat(),
        "rows": {name: len(table) for name, table in tables.items()},
        "logical_event_count": logical_count,
        "exact_duplicate_rows": len(events) - logical_count,
        "raw_event_counts": {str(name): int(count) for name, count in events["event_name"].value_counts().items()},
        "variant_counts": {str(name): int(count) for name, count in tables["assignments"]["variant"].value_counts().items()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "duckdb": duckdb.__version__},
    }
    # Снимок сохраняет сами параметры, а хеш позволяет проверить их совпадение.
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir


def write_preview(
    tables: dict[str, pd.DataFrame], config: dict, scenario_name: str, output_dir: str | Path,
) -> Path:
    """Экспорт предпросмотра, который всегда явно помечен неполным."""
    return _write_dataset(tables, config, scenario_name, output_dir, complete=False)


def write_full(
    tables: dict[str, pd.DataFrame], config: dict, scenario_name: str, output_dir: str | Path,
) -> Path:
    """Экспорт результата generate_full; отсутствие контактов при p=0 допустимо."""
    return _write_dataset(tables, config, scenario_name, output_dir, complete=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Синтетические события: полный запуск или предпросмотр")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--scenario", choices=("null", "uplift", "harmful", "broken_assignment", "all"), default="all")
    parser.add_argument("--output", type=Path, help="Новый каталог экспорта; сценарии сохраняются внутри него")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preview", action="store_true", help="Неполный предпросмотр без контактов и копий")
    modes.add_argument("--full", action="store_true", help="Полная генерация с контактами и точными копиями")
    args = parser.parse_args()
    output = args.output if args.output is not None else root / "data" / ("raw" if args.full else "preview")
    try:
        config = read_config(args.config)
        validate_protocol(config["protocol"])
        validate_simulation(config["simulation"])
        scenarios = tuple(config["simulation"]["scenarios"]) if args.scenario == "all" else (args.scenario,)
        # Проверяем все цели до записи первого сценария, чтобы не затирать файлы.
        for name in scenarios:
            if (output / name).exists():
                raise ValueError(f"Каталог уже существует: {output / name}. Выбери новый --output")
        for name in scenarios:
            generator = generate_full if args.full else generate_preview
            writer = write_full if args.full else write_preview
            tables = generator(config["protocol"], config["simulation"], name)
            destination = writer(tables, config, name, output / name)
            print(f"{name}: участников {len(tables['users'])}, событий {len(tables['events'])}; {destination}")
    except (OSError, ValueError, duckdb.Error) as error:
        parser.exit(2, f"Ошибка генерации: {error}\n")
    if args.full:
        print("Полные синтетические данные созданы. Статистический анализ ещё не выполнен.")
    else:
        print("Неполный предпросмотр: контакты и точные копии пропущены. Итоговый анализ недопустим.")


if __name__ == "__main__":
    main()
