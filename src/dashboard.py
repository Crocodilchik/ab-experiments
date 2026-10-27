"""Этап 6: автономный HTML с наблюдаемыми данными и готовым решением этапа 5."""

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import duckdb
import plotly
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from src.analyze import SCENARIOS, SUMMARY_SCHEMA, load_analysis_input
from src.decision import make_decision
from src.prepare_data import _check_summaries, _reject_rows


TEMPLATE = Path(__file__).with_name("dashboard_template.html")
COLORS = {"A": "#2563eb", "B": "#0f766e"}
DECISION_LABELS = {
    "invalid": "Продуктовый вывод заблокирован",
    "guardrail_harm": "Подтверждён недопустимый рост ошибок",
    "guardrail_unconfirmed": "Безопасность не подтверждена",
    "recommend_further_check": "Есть основания для дальнейшей проверки",
    "no_practical_gain": "Практически важный выигрыш не подтверждён",
    "inconclusive": "Результат неопределённый",
}
USER_FIELDS = {
    "experiment_id": "VARCHAR", "scenario": "VARCHAR", "user_id": "BIGINT",
    "variant": "VARCHAR", "assigned_at": "TIMESTAMP WITH TIME ZONE", "assigned_date": "DATE",
    "platform": "VARCHAR", "is_new_user": "BOOLEAN",
    "has_listing_view": "INTEGER", "has_contact": "INTEGER", "has_error": "INTEGER",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_constant(value):
    raise ValueError(f"[dashboard_json] Недопустимое значение JSON: {value}")


def _observed_diagnostics(prepared: Path, protocol: dict, groups: dict, scenario: str) -> tuple[list, list]:
    """Сверить дневные итоги с пользователями; сегменты описывают только состав групп."""
    with duckdb.connect() as connection:
        for table in ("experiment_summary", "daily_summary", "user_metrics"):
            connection.read_parquet(str(prepared / f"{table}.parquet")).create_view(table)
            actual = {row[0]: row[1] for row in connection.execute(f"DESCRIBE {table}").fetchall()}
            expected = {**SUMMARY_SCHEMA, "assigned_date": "DATE"} if table == "daily_summary" else SUMMARY_SCHEMA
            if table == "user_metrics":
                if any(actual.get(name) != dtype for name, dtype in USER_FIELDS.items()):
                    raise ValueError("[dashboard_schema] Неверные поля пользовательской витрины")
            elif actual != expected:
                raise ValueError(f"[dashboard_schema] Неверная схема {table}")
        for table, columns in (("daily_summary", ("experiment_id", "scenario", "variant", "assigned_date", "n_users", "n_listing_view", "n_contact", "n_error")),
                               ("user_metrics", USER_FIELDS)):
            condition = " OR ".join(f"{column} IS NULL" for column in columns)
            _reject_rows(connection, "dashboard_nulls", f"SELECT * FROM {table} WHERE {condition}")
        _reject_rows(connection, "dashboard_user_key", "SELECT experiment_id, scenario, user_id FROM user_metrics GROUP BY ALL HAVING count(*) > 1")
        _reject_rows(connection, "dashboard_flags", """SELECT * FROM user_metrics
            WHERE user_id <= 0 OR variant NOT IN ('A', 'B') OR platform NOT IN ('web', 'android', 'ios')
               OR has_listing_view NOT IN (0, 1) OR has_contact NOT IN (0, 1) OR has_error NOT IN (0, 1)
               OR has_contact > has_listing_view
               OR assigned_date <> CAST(assigned_at AT TIME ZONE 'UTC' AS DATE)""")
        for table in ("daily_summary", "user_metrics"):
            invalid = connection.execute(f"SELECT count(*) FROM {table} WHERE experiment_id <> ? OR scenario <> ?",
                                         [protocol["experiment_id"], scenario]).fetchall()[0][0]
            if invalid:
                raise ValueError("[dashboard_identity] В диагностике смешаны эксперименты или сценарии")
        start = datetime.fromisoformat(protocol["start_at"])
        end = start + timedelta(days=protocol["enrollment_days"])
        if connection.execute("SELECT count(*) FROM user_metrics WHERE assigned_at < ? OR assigned_at >= ?",
                              [start, end]).fetchall()[0][0]:
            raise ValueError("[dashboard_dates] Назначения вне периода набора")
        _reject_rows(connection, "dashboard_daily_key", "SELECT experiment_id, scenario, variant, assigned_date FROM daily_summary GROUP BY ALL HAVING count(*) > 1")
        # Проверка не усредняет дневные доли: сравниваются количества пользователей.
        _check_summaries(connection, sum(group["n_users"] for group in groups.values()))
        _reject_rows(connection, "dashboard_daily_mart", """
            WITH observed AS (
                SELECT experiment_id, scenario, variant, assigned_date, COUNT(*) AS n_users,
                    SUM(has_listing_view) AS n_listing_view, SUM(has_contact) AS n_contact, SUM(has_error) AS n_error
                FROM user_metrics GROUP BY experiment_id, scenario, variant, assigned_date
            ), daily AS (
                SELECT experiment_id, scenario, variant, assigned_date, n_users, n_listing_view, n_contact, n_error
                FROM daily_summary
            )
            (SELECT * FROM observed EXCEPT ALL SELECT * FROM daily)
            UNION ALL
            (SELECT * FROM daily EXCEPT ALL SELECT * FROM observed)
        """)
        daily = [dict(zip(("assigned_date", "variant", "n_users", "n_listing_view", "n_contact", "n_error"), row))
                 for row in connection.execute("""SELECT CAST(assigned_date AS VARCHAR), variant,
                     n_users, n_listing_view, n_contact, n_error FROM daily_summary ORDER BY assigned_date, variant""").fetchall()]
        balance = [dict(zip(("feature", "level", "variant", "n_users"), row))
                   for row in connection.execute("""
                       SELECT 'platform', platform, variant, count(*) FROM user_metrics GROUP BY platform, variant
                       UNION ALL
                       SELECT 'is_new_user', CASE WHEN is_new_user THEN 'Новый' ELSE 'Возвращающийся' END,
                           variant, count(*) FROM user_metrics GROUP BY is_new_user, variant
                       ORDER BY 1, 2, 3
                   """).fetchall()]
    return daily, balance


def load_dashboard_scenario(analysis_dir: Path, prepared: Path, scenario: str) -> tuple[dict, dict]:
    """Связать полный анализ с тем же набором; не пересчитывать z-тест и интервалы."""
    analysis_dir, prepared = Path(analysis_dir), Path(prepared)
    analysis_path = analysis_dir / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(analysis, dict) or type(analysis.get("schema_version")) is not int or analysis["schema_version"] != 1 or analysis.get("stage") != "analysis" or analysis.get("complete") is not True or analysis.get("missing_components") != []:
        raise ValueError("[dashboard_incomplete] Нужен полный analysis.json этапа 5")
    protocol, quality, groups = load_analysis_input(prepared, expected_scenario=scenario)
    if analysis.get("scenario") != scenario or analysis.get("experiment_id") != protocol["experiment_id"] or analysis.get("groups") != groups:
        raise ValueError("[dashboard_identity] Анализ не соответствует наблюдаемым группам")
    expected_hashes = {name: _hash(prepared / name) for name in ("protocol.json", "quality_report.json", "experiment_summary.parquet")}
    if analysis.get("source_sha256") != expected_hashes:
        raise ValueError("[dashboard_source_hash] Подготовка отличается от входа статистического анализа")
    parameter_names = ("alpha", "srm_alpha", "expected_variant_b_probability", "ci_method", "business_contact_effect", "max_error_increase")
    if analysis.get("parameters") != {name: protocol[name] for name in parameter_names}:
        raise ValueError("[dashboard_parameters] Пороги анализа не совпадают с протоколом")
    if analysis.get("decision") != make_decision(analysis):
        raise ValueError("[dashboard_decision] Сохранённое решение не соответствует правилу")
    for name, count_field in (("primary", "n_contact"), ("guardrail", "n_error")):
        metric = analysis[name]
        for variant, group in groups.items():
            expected_rate = group[count_field] / group["n_users"] if group["n_users"] else None
            if metric[variant] != {"n_users": group["n_users"], "n_success": group[count_field], "rate": expected_rate}:
                raise ValueError("[dashboard_metric] Оценка использует другие числители или знаменатели")
    daily, balance = _observed_diagnostics(prepared, protocol, groups, scenario)
    start = datetime.fromisoformat(protocol["start_at"])
    model = {
        "scenario": scenario, "experiment_id": analysis["experiment_id"],
        "decision": analysis["decision"], "decision_label": DECISION_LABELS[analysis["decision"]["status"]],
        "diagnostic_only": analysis["decision"]["status"] == "invalid",
        "groups": groups, "primary": analysis["primary"], "guardrail": analysis["guardrail"],
        "srm": analysis["srm"], "parameters": analysis["parameters"], "sample_size": analysis["sample_size"],
        "passport": {"start_at": protocol["start_at"],
                     "enrollment_end": (start + timedelta(days=protocol["enrollment_days"])).isoformat(),
                     "collection_end": quality["source_collection_end"],
                     "outcome_window_hours": protocol["outcome_window_hours"],
                     "all_windows_mature": analysis["all_windows_mature"],
                     "duplicates_removed": quality.get("exact_duplicates_removed")},
        "daily": daily, "balance": balance,
    }
    files = {"analysis.json": analysis_path, **{name: prepared / name for name in (
        "protocol.json", "quality_report.json", "experiment_summary.parquet", "daily_summary.parquet", "user_metrics.parquet",
    )}}
    return model, {name: _hash(path) for name, path in files.items()}


def _layout(figure: go.Figure, *, height=330) -> go.Figure:
    figure.update_layout(template="plotly_white", height=height, autosize=True,
                         margin=dict(l=60, r=25, t=25, b=65), separators=", ",
                         font=dict(family="Segoe UI, Arial, sans-serif", size=13, color="#26344a"),
                         legend=dict(orientation="h", y=1.12, x=0),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


def build_figures(model: dict, ranges: dict) -> dict:
    """Все графики используют одну модель сценария и постоянные шкалы отчёта."""
    funnel = go.Figure()
    counts = go.Figure()
    conversion = go.Figure()
    for variant, group in model["groups"].items():
        n = group["n_users"]
        values = [n, group["n_listing_view"], group["n_contact"]]
        funnel.add_bar(name=variant, marker_color=COLORS[variant],
                       x=["Назначены", "Просмотрели", "Создали контакт"],
                       y=[value / n if n else None for value in values], customdata=values,
                       hovertemplate="%{x}<br>Пользователей: %{customdata:,d}<br>От всех назначенных: %{y:.2%}<extra>%{fullData.name}</extra>")
        rows = [row for row in model["daily"] if row["variant"] == variant]
        dates = [row["assigned_date"] for row in rows]
        counts.add_scatter(name=variant, mode="lines+markers", x=dates, y=[row["n_users"] for row in rows],
                           line_color=COLORS[variant], hovertemplate="Назначены: %{y:,d}<extra>%{fullData.name}</extra>")
        conversion.add_scatter(name=variant, mode="lines+markers", x=dates,
                               y=[row["n_contact"] / row["n_users"] for row in rows],
                               customdata=[[row["n_contact"], row["n_users"]] for row in rows], line_color=COLORS[variant],
                               hovertemplate="Конверсия: %{y:.2%}<br>Контакты / назначены: %{customdata[0]} / %{customdata[1]}<extra>%{fullData.name}</extra>")
    _layout(funnel).update_layout(barmode="group", yaxis=dict(title="От всех назначенных", tickformat=".0%", range=[0, 1.05]))
    for figure, axis in ((counts, dict(title="Назначенные пользователи", range=[0, ranges["daily_count"]])),
                         (conversion, dict(title="Контакт / все назначенные", tickformat=".0%", range=[0, ranges["daily_rate"]]))):
        _layout(figure).update_layout(hovermode="x unified", xaxis=dict(title="UTC-день назначения", type="date", tickformat="%d.%m"), yaxis=axis)
        if not model["daily"]:
            figure.add_annotation(text="Нет назначенных участников", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figures = {"funnel": funnel, "daily-count": counts, "daily-rate": conversion}
    for name, threshold in (("primary", "business_contact_effect"), ("guardrail", "max_error_increase")):
        metric = model[name]
        figure = go.Figure()
        interval, difference = metric["interval"], metric["difference"]
        if difference is not None:
            error = dict(type="data", symmetric=False, array=[100 * (interval["upper"] - difference)],
                         arrayminus=[100 * (difference - interval["lower"])], thickness=3, width=8) if interval else None
            figure.add_scatter(x=[100 * difference], y=[0], mode="markers", marker=dict(size=15, color="#334155"),
                               error_x=error, name="B − A", hovertemplate="B − A: %{x:.4f} п.п.<extra></extra>")
        if interval is None:
            figure.add_annotation(text="Интервал не рассчитан: метод неприменим", x=0.5, y=0.86, xref="paper", yref="paper", showarrow=False)
        figure.add_vline(x=0, line_color="#64748b", line_width=1)
        figure.add_vline(x=100 * model["parameters"][threshold], line_color="#b45309", line_dash="dash", line_width=2)
        _layout(figure, height=210).update_layout(showlegend=False, margin=dict(l=20, r=25, t=20, b=60),
            xaxis=dict(title="Разница B − A, процентные пункты", range=[-ranges[name], ranges[name]]),
            yaxis=dict(visible=False, range=[-1, 1]))
        figures[name] = figure
    # Plotly JSON сериализуем штатным способом, не преобразуя массивы через str().
    return {name: json.loads(figure.to_json()) for name, figure in figures.items()}


def build_dashboard_payload(analysis_root: Path, prepared_root: Path, scenarios=SCENARIOS) -> tuple[dict, dict]:
    models, sources = {}, {}
    for scenario in scenarios:
        if scenario not in SCENARIOS or scenario in models:
            raise ValueError("[dashboard_scenario] Неизвестный или повторный сценарий")
        models[scenario], sources[scenario] = load_dashboard_scenario(Path(analysis_root) / scenario, Path(prepared_root) / scenario, scenario)
    if not models:
        raise ValueError("[dashboard_empty] Не выбран ни один сценарий")
    if len({model["experiment_id"] for model in models.values()}) != 1:
        raise ValueError("[dashboard_experiment] Нельзя смешивать разные эксперименты")
    ranges = {name: 115 * max([0.005] + [abs(value) for model in models.values() for value in (
        model[name]["difference"], *([model[name]["interval"][key] for key in ("lower", "upper")] if model[name]["interval"] else []),
        model["parameters"]["business_contact_effect" if name == "primary" else "max_error_increase"],
    ) if value is not None]) for name in ("primary", "guardrail")}
    daily_rows = [row for model in models.values() for row in model["daily"]]
    ranges["daily_count"] = 1.15 * max([1] + [row["n_users"] for row in daily_rows])
    ranges["daily_rate"] = min(1, max([0.2] + [1.15 * row["n_contact"] / row["n_users"] for row in daily_rows]))
    for model in models.values():
        model["figures"] = build_figures(model, ranges)
    return {"schema_version": 1, "synthetic": True, "scenario_order": list(models), "scenarios": models}, sources


def render_dashboard(payload: dict) -> str:
    # JSON внутри script не должен допускать закрывающий </script> из строк данных.
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = TEMPLATE.read_text(encoding="utf-8")
    return template.replace("__DASHBOARD_PAYLOAD__", serialized, 1).replace("__PLOTLY_BUNDLE__", get_plotlyjs(), 1)


def write_dashboard(analysis_root: Path, prepared_root: Path, output: Path, scenarios=SCENARIOS) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Каталог уже существует: {output}")
    payload, sources = build_dashboard_payload(analysis_root, prepared_root, scenarios)
    html = render_dashboard(payload)
    manifest = {"schema_version": 1, "stage": "dashboard", "complete": True, "synthetic": True,
                "scenarios": payload["scenario_order"], "plotly_version": plotly.__version__,
                "source_sha256": sources, "dashboard_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest()}
    # Все проверки и построение завершаются до создания выходной папки.
    output.mkdir(parents=True, exist_ok=False)
    (output / "dashboard.html").write_text(html, encoding="utf-8", newline="\n")
    (output / "dashboard_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "dashboard_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Автономный HTML-дашборд синтетического A/B-эксперимента")
    parser.add_argument("--analysis", type=Path, required=True, help="Корень полного анализа этапа 5")
    parser.add_argument("--prepared", type=Path, required=True, help="Корень соответствующей подготовки этапа 4")
    parser.add_argument("--output", type=Path, default=root / "learning" / "stage_06_output")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    args = parser.parse_args()
    try:
        manifest = write_dashboard(args.analysis, args.prepared, args.output, SCENARIOS if args.scenario == "all" else (args.scenario,))
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as error:
        parser.exit(2, f"Ошибка дашборда: {error}\n")
    print(f"Готово: {args.output / 'dashboard.html'}; сценариев: {len(manifest['scenarios'])}; без CDN и Python-сервера.")


if __name__ == "__main__":
    main()
