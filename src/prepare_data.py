"""Проверить сырой набор, удалить копии, собрать витрину и агрегаты.

Этот этап не проверяет SRM и не делает статистический продуктовый вывод.
"""

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import duckdb

from src.config import read_config, validate_protocol, validate_simulation


SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
SCHEMAS = {
    "users": {"user_id": "BIGINT", "platform": "VARCHAR", "region": "VARCHAR", "is_new_user": "BOOLEAN"},
    "assignments": {
        "experiment_id": "VARCHAR", "scenario": "VARCHAR", "user_id": "BIGINT",
        "variant": "VARCHAR", "assigned_at": "TIMESTAMP WITH TIME ZONE",
    },
    "raw_events": {
        "experiment_id": "VARCHAR", "scenario": "VARCHAR", "event_id": "BIGINT",
        "user_id": "BIGINT", "event_name": "VARCHAR",
        "event_ts": "TIMESTAMP WITH TIME ZONE", "listing_id": "BIGINT",
    },
}


class DataQualityError(ValueError):
    """Подготовку нельзя продолжать с этим набором данных."""


def _scalar(connection, query: str):
    # Полностью читаем результат: соединение DuckDB готово к следующему запросу.
    return connection.execute(query).fetchall()[0][0]


def _reject_rows(connection, code: str, query: str) -> None:
    count = _scalar(connection, f"SELECT count(*) FROM ({query}) AS problems")
    if count:
        raise DataQualityError(f"[{code}] Проверка не пройдена, проблемных строк/ключей: {count}")


def _utc(value: str, field: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise DataQualityError(f"[manifest_dates] Неверное время {field}") from error
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise DataQualityError(f"[manifest_dates] {field} должен быть явно задан в UTC")
    return moment


def _read_source(source: Path) -> tuple[dict, dict]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("complete") is not True or manifest.get("status") != "complete" or manifest.get("missing_components") != []:
        raise DataQualityError("[incomplete_source] Нужен полный экспорт; предпросмотр не подходит")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise DataQualityError("[schema_version] Поддерживается формат экспорта версии 1")
    config = read_config(source / "config_snapshot.json")
    validate_protocol(config["protocol"])
    validate_simulation(config["simulation"])
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("config_sha256"):
        raise DataQualityError("[config_hash] Снимок параметров не совпадает с хешем manifest")
    protocol = config["protocol"]
    if manifest.get("experiment_id") != protocol["experiment_id"] or not isinstance(manifest.get("scenario"), str) or manifest["scenario"] not in config["simulation"]["scenarios"]:
        raise DataQualityError("[source_identity] Несогласованные эксперимент или сценарий")
    rows = manifest.get("rows")
    if not isinstance(rows, dict) or set(rows) != {"users", "assignments", "events"} or any(type(value) is not int or value < 0 for value in rows.values()):
        raise DataQualityError("[manifest_rows] Ожидаются неотрицательные количества трёх таблиц")
    start = _utc(protocol["start_at"], "protocol.start_at")
    enrollment_end = start + timedelta(days=protocol["enrollment_days"])
    if _utc(manifest.get("start_at"), "start_at") != start or _utc(manifest.get("enrollment_end"), "enrollment_end") != enrollment_end:
        raise DataQualityError("[manifest_dates] Период набора не совпадает с протоколом")
    # Не используем максимум времени событий как доказательство полного сбора.
    # Для симуляции граница наблюдения объявлена генератором в manifest.
    if _utc(manifest.get("collection_end"), "collection_end") < enrollment_end + timedelta(hours=protocol["outcome_window_hours"]):
        raise DataQualityError("[immature_collection] Плановый период наблюдения ещё не завершён")
    return manifest, protocol


def _load_tables(connection, source: Path, manifest: dict, protocol: dict) -> None:
    for table, filename in (("users", "users"), ("assignments", "assignments"), ("raw_events", "events")):
        connection.read_parquet(str(source / f"{filename}.parquet")).create_view(table)
        description = connection.execute(f"DESCRIBE {table}").fetchall()
        actual = {row[0]: row[1] for row in description}
        if actual != SCHEMAS[table]:
            raise DataQualityError(f"[schema_{filename}] Неверные поля или типы: {actual}")
        if _scalar(connection, f"SELECT count(*) FROM {table}") != manifest.get("rows", {}).get(filename):
            raise DataQualityError(f"[row_count_{filename}] Число строк не совпадает с manifest")
    connection.execute(
        """CREATE TEMP TABLE run_settings AS SELECT
        ?::VARCHAR AS experiment_id, ?::VARCHAR AS scenario,
        ?::TIMESTAMPTZ AS start_at, ?::TIMESTAMPTZ AS enrollment_end,
        ?::TIMESTAMPTZ AS collection_end, ?::BIGINT AS window_hours""",
        [protocol["experiment_id"], manifest["scenario"], manifest["start_at"],
         manifest["enrollment_end"], manifest["collection_end"], protocol["outcome_window_hours"]],
    ).fetchall()


def _check_raw_data(connection) -> None:
    for table, schema in SCHEMAS.items():
        required = [name for name in schema if name != "listing_id"]
        condition = " OR ".join(f"{name} IS NULL" for name in required)
        _reject_rows(connection, f"nulls_{table}", f"SELECT * FROM {table} WHERE {condition}")
    checks = {
        "users_key": "SELECT user_id FROM users GROUP BY user_id HAVING count(*) > 1",
        "users_values": "SELECT * FROM users WHERE user_id <= 0 OR platform NOT IN ('web', 'android', 'ios') OR trim(region) = ''",
        "assignment_key": "SELECT experiment_id, scenario, user_id FROM assignments GROUP BY experiment_id, scenario, user_id HAVING count(*) > 1",
        "assignment_values": "SELECT * FROM assignments WHERE user_id <= 0 OR variant NOT IN ('A', 'B')",
        "assignment_identity": "SELECT a.* FROM assignments a CROSS JOIN run_settings r WHERE a.experiment_id <> r.experiment_id OR a.scenario <> r.scenario",
        "event_identity": "SELECT e.* FROM raw_events e CROSS JOIN run_settings r WHERE e.experiment_id <> r.experiment_id OR e.scenario <> r.scenario",
        "assignment_period": "SELECT a.* FROM assignments a CROSS JOIN run_settings r WHERE a.assigned_at < r.start_at OR a.assigned_at >= r.enrollment_end",
        "unknown_users": "SELECT a.* FROM assignments a LEFT JOIN users u USING (user_id) WHERE u.user_id IS NULL",
        "event_names": "SELECT * FROM raw_events WHERE event_name NOT IN ('search_view', 'listing_view', 'contact_created', 'technical_error')",
        "event_ids": "SELECT * FROM raw_events WHERE event_id <= 0 OR user_id <= 0 OR listing_id <= 0",
        "listing_required": "SELECT * FROM raw_events WHERE event_name IN ('listing_view', 'contact_created') AND listing_id IS NULL",
        "search_listing": "SELECT * FROM raw_events WHERE event_name = 'search_view' AND listing_id IS NOT NULL",
        "event_assignment": "SELECT e.* FROM raw_events e LEFT JOIN assignments a USING (experiment_id, scenario, user_id) WHERE a.user_id IS NULL",
        "event_before_assignment": "SELECT e.* FROM raw_events e JOIN assignments a USING (experiment_id, scenario, user_id) WHERE e.event_ts < a.assigned_at",
        "immature_windows": "SELECT a.* FROM assignments a CROSS JOIN run_settings r WHERE a.assigned_at + r.window_hours * INTERVAL '1 hour' > r.collection_end",
        # Сначала различаем полные строки, затем проверяем повтор ключа.
        "event_key_conflict": "SELECT experiment_id, scenario, event_id FROM (SELECT DISTINCT * FROM raw_events) GROUP BY experiment_id, scenario, event_id HAVING count(*) > 1",
    }
    for code, query in checks.items():
        _reject_rows(connection, code, query)


def _check_contact_chain(connection) -> None:
    # Проверяем контакты в индивидуальном окне. Поздние сырые события в метрику
    # не попадут и не превращаются из-за одного времени в повреждённые записи.
    _reject_rows(connection, "contact_chain", """
        SELECT c.* FROM clean_events c
        JOIN assignments a USING (experiment_id, scenario, user_id)
        CROSS JOIN run_settings r
        WHERE c.event_name = 'contact_created'
          AND c.event_ts >= a.assigned_at
          AND c.event_ts < a.assigned_at + r.window_hours * INTERVAL '1 hour'
          AND NOT EXISTS (
              SELECT 1 FROM clean_events v
              WHERE v.experiment_id = c.experiment_id AND v.scenario = c.scenario
                AND v.user_id = c.user_id AND v.listing_id = c.listing_id
                AND v.event_name = 'listing_view'
                AND v.event_ts >= a.assigned_at AND v.event_ts < c.event_ts
          )
    """)


def _check_summaries(connection, n_assigned: int) -> None:
    """Проверить сохранение участников и согласованность двух уровней итогов."""
    for table in ("experiment_summary", "daily_summary"):
        if _scalar(connection, f"SELECT COALESCE(SUM(n_users), 0) FROM {table}") != n_assigned:
            raise DataQualityError(f"[summary_participant_loss] {table} не сохранила всех назначенных")
        _reject_rows(connection, "summary_counts", f"""SELECT * FROM {table}
            WHERE n_users <= 0 OR n_listing_view < 0 OR n_contact < 0 OR n_error < 0
               OR n_listing_view > n_users OR n_contact > n_listing_view OR n_error > n_users""")
    # Складываем количества по дням. Доли здесь не усредняются.
    # EXCEPT ALL в обе стороны обнаруживает лишнюю, пропавшую или изменённую группу.
    _reject_rows(connection, "summary_daily_mismatch", """
        WITH daily_totals AS (
            SELECT experiment_id, scenario, variant,
                   SUM(n_users) AS n_users, SUM(n_listing_view) AS n_listing_view,
                   SUM(n_contact) AS n_contact, SUM(n_error) AS n_error
            FROM daily_summary GROUP BY experiment_id, scenario, variant
        ), overall AS (
            SELECT experiment_id, scenario, variant, n_users, n_listing_view, n_contact, n_error
            FROM experiment_summary
        )
        (SELECT * FROM daily_totals EXCEPT ALL SELECT * FROM overall)
        UNION ALL
        (SELECT * FROM overall EXCEPT ALL SELECT * FROM daily_totals)
    """)


def prepare_scenario(source: str | Path, output: str | Path, *, expected_scenario: str | None = None) -> dict:
    """Проверить один экспорт и записать полную подготовку данных этапа 4."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Каталог уже существует: {output}")
    manifest, protocol = _read_source(source)
    if expected_scenario is not None and manifest["scenario"] != expected_scenario:
        raise DataQualityError("[scenario_path] Имя каталога не совпадает со сценарием")
    with duckdb.connect() as connection:
        _load_tables(connection, source, manifest, protocol)
        _check_raw_data(connection)
        connection.execute((SQL_DIR / "01_clean_events.sql").read_text(encoding="utf-8")).fetchall()
        _check_contact_chain(connection)
        connection.execute((SQL_DIR / "02_user_metrics.sql").read_text(encoding="utf-8")).fetchall()
        n_assigned = _scalar(connection, "SELECT count(*) FROM assignments")
        if _scalar(connection, "SELECT count(*) FROM user_metrics") != n_assigned:
            raise DataQualityError("[participant_loss] Витрина не сохранила всех назначенных")
        _reject_rows(connection, "mart_key", "SELECT experiment_id, scenario, user_id FROM user_metrics GROUP BY experiment_id, scenario, user_id HAVING count(*) > 1")
        _reject_rows(connection, "funnel", "SELECT * FROM user_metrics WHERE has_contact > has_listing_view")
        for filename in ("03_experiment_summary.sql", "04_daily_summary.sql"):
            connection.execute((SQL_DIR / filename).read_text(encoding="utf-8")).fetchall()
        _check_summaries(connection, n_assigned)
        n_raw = _scalar(connection, "SELECT count(*) FROM raw_events")
        n_clean = _scalar(connection, "SELECT count(*) FROM clean_events")
        outside = _scalar(connection, """SELECT count(*) FROM clean_events e
            JOIN assignments a USING (experiment_id, scenario, user_id)
            CROSS JOIN run_settings r
            WHERE e.event_ts >= a.assigned_at + r.window_hours * INTERVAL '1 hour'""")
        report = {
            "status": "prepared",
            "experiment_id": protocol["experiment_id"], "scenario": manifest["scenario"],
            "source": str(source.resolve()), "config_sha256": manifest["config_sha256"],
            "source_collection_end": manifest["collection_end"],
            "raw_events": n_raw, "clean_events": n_clean,
            "exact_duplicates_removed": n_raw - n_clean,
            "assigned_users": n_assigned, "user_metric_rows": n_assigned,
            "outside_window_events_excluded": outside,
            "data_checks_passed": True, "all_windows_mature": True,
            "experiment_summary_rows": _scalar(connection, "SELECT count(*) FROM experiment_summary"),
            "daily_summary_rows": _scalar(connection, "SELECT count(*) FROM daily_summary"),
            "srm_checked": False, "summary_ready": True,
            "missing_components": [],
        }
        # Не создаём папку результата при провале проверок; сырые файлы не меняем.
        output.mkdir(parents=True, exist_ok=False)
        for table in ("clean_events", "user_metrics", "experiment_summary", "daily_summary"):
            connection.table(table).write_parquet(str(output / f"{table}.parquet"))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Проверки качества, SQL-витрина и агрегаты по группам и дням")
    parser.add_argument("--input", type=Path, required=True, help="Каталог, содержащий папки сценариев")
    parser.add_argument("--output", type=Path, default=root / "data" / "processed")
    parser.add_argument("--scenario", choices=("null", "uplift", "harmful", "broken_assignment", "all"), default="all")
    args = parser.parse_args()
    names = ("null", "uplift", "harmful", "broken_assignment") if args.scenario == "all" else (args.scenario,)
    try:
        for name in names:
            if (args.output / name).exists():
                raise FileExistsError(f"Каталог уже существует: {args.output / name}")
        for name in names:
            report = prepare_scenario(args.input / name, args.output / name, expected_scenario=name)
            print(f"{name}: участников {report['assigned_users']}, удалено копий {report['exact_duplicates_removed']}")
    except (OSError, ValueError, KeyError, duckdb.Error) as error:
        parser.exit(2, f"Ошибка подготовки: {error}\n")
    print("Витрины и агрегаты готовы. SRM и статистический вывод ещё не выполнены.")


if __name__ == "__main__":
    main()
