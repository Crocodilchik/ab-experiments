"""Этап 5: статистические оценки и полное решение по протоколу.

Вход — protocol.json, quality_report.json и experiment_summary.parquet этапа 4.
Параметры simulation и заданный генератору эффект анализу не нужны.
"""

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path

import duckdb
from scipy.stats import chisquare
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_effectsize, proportions_ztest

from src.config import validate_protocol
from src.decision import make_decision


SUMMARY_SCHEMA = {
    "experiment_id": "VARCHAR", "scenario": "VARCHAR", "variant": "VARCHAR",
    "n_users": "BIGINT", "n_listing_view": "BIGINT", "n_contact": "BIGINT", "n_error": "BIGINT",
    "view_cr": "DOUBLE", "contact_cr": "DOUBLE", "error_rate": "DOUBLE",
    "contact_given_view_rate": "DOUBLE",
}
SCENARIOS = ("null", "uplift", "harmful", "broken_assignment")


def _count(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"[counts] {name}: нужно неотрицательное целое без bool")


def _open_probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"[probability] {name}: нужно конечное число строго между 0 и 1")


def check_srm(n_a: int, n_b: int, expected_b_probability: float, alpha: float) -> dict:
    """Проверить назначение по заранее ожидаемым вероятностям, а не наблюдаемым."""
    _count("n_a", n_a)
    _count("n_b", n_b)
    _open_probability("expected_b_probability", expected_b_probability)
    _open_probability("srm_alpha", alpha)
    total = n_a + n_b
    expected = [total * (1 - expected_b_probability), total * expected_b_probability]
    result = {
        "method": "chi_square_goodness_of_fit", "alpha": alpha,
        "observed": {"A": n_a, "B": n_b},
        "expected": {"A": expected[0], "B": expected[1]},
        "statistic": None, "p_value": None, "passed": False, "applicable": False,
        "reason": None,
    }
    # Для малой ожидаемой ячейки не выдаём успешный результат приближения.
    if min(expected) < 5:
        result["reason"] = "srm_expected_count_below_5"
        return result
    statistic, p_value = chisquare([n_a, n_b], f_exp=expected)
    result.update(statistic=float(statistic), p_value=float(p_value), applicable=True,
                  passed=bool(p_value >= alpha), reason=None if p_value >= alpha else "srm_detected")
    return result


def estimate_proportions(count_a: int, n_a: int, count_b: int, n_b: int,
                         alpha: float, *, include_test: bool = True) -> dict:
    """Оценить B − A и проверить заранее выбранное нормальное приближение.

    count — число пользователей с исходом, n — все назначенные пользователи.
    При малом числе исходов доли сохраняются, но тест и интервал не подбираются заново.
    """
    for name, value in (("count_a", count_a), ("n_a", n_a), ("count_b", count_b), ("n_b", n_b)):
        _count(name, value)
    if count_a > n_a or count_b > n_b:
        raise ValueError("[counts] Число пользователей с исходом не может превышать размер группы")
    _open_probability("alpha", alpha)
    rate_a = count_a / n_a if n_a else None
    rate_b = count_b / n_b if n_b else None
    difference = rate_b - rate_a if rate_a is not None and rate_b is not None else None
    observed = [[count_a, n_a - count_a], [count_b, n_b - count_b]]
    reasons = []
    expected = None
    if not n_a or not n_b:
        reasons.append("empty_group")
    else:
        pooled = (count_a + count_b) / (n_a + n_b)
        expected = [[n_a * pooled, n_a * (1 - pooled)],
                    [n_b * pooled, n_b * (1 - pooled)]]
        if min(value for row in observed for value in row) < 10:
            reasons.append("observed_cell_below_10")
        if min(value for row in expected for value in row) < 10:
            reasons.append("expected_cell_below_10")
    result = {
        "A": {"n_users": n_a, "n_success": count_a, "rate": rate_a},
        "B": {"n_users": n_b, "n_success": count_b, "rate": rate_b},
        "difference": difference,
        "difference_pp": 100 * difference if difference is not None else None,
        "relative_change_percent": 100 * difference / rate_a if rate_a and difference is not None else None,
        "applicability": {"passed": not reasons, "reasons": reasons,
                          "observed_cells": observed, "expected_cells": expected},
        "interval": None, "test": None,
    }
    if reasons:
        return result
    # Первый аргумент библиотеки — B: и знак эффекта, и интервал относятся к B − A.
    lower, upper = confint_proportions_2indep(
        count_b, n_b, count_a, n_a, method="newcomb", compare="diff", alpha=alpha,
    )
    result["interval"] = {"method": "newcomb", "confidence_level": 1 - alpha,
                          "lower": float(lower), "upper": float(upper)}
    if include_test:
        z_statistic, p_value = proportions_ztest(
            [count_b, count_a], [n_b, n_a], value=0, alternative="two-sided", prop_var=False,
        )
        result["test"] = {"method": "two_proportion_z", "alternative": "two-sided",
                          "z_statistic": float(z_statistic), "p_value": float(p_value)}
    return result


def load_analysis_input(source: str | Path, *, expected_scenario: str | None = None) -> tuple[dict, dict, dict]:
    """Прочитать проверенные количества; готовые дробные колонки не являются входом теста."""
    source = Path(source)
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    quality = json.loads((source / "quality_report.json").read_text(encoding="utf-8"))
    if not isinstance(quality, dict) or quality.get("status") != "prepared" or any(
        quality.get(field) is not True for field in ("data_checks_passed", "all_windows_mature", "summary_ready")
    ) or quality.get("missing_components") != []:
        raise ValueError("[unprepared_source] Нужен завершённый результат этапа 4")
    scenario = quality.get("scenario")
    if quality.get("experiment_id") != protocol["experiment_id"] or scenario not in SCENARIOS or (expected_scenario is not None and scenario != expected_scenario):
        raise ValueError("[analysis_identity] Несогласованные эксперимент или сценарий")
    collection_end = datetime.fromisoformat(quality["source_collection_end"])
    planned_end = datetime.fromisoformat(protocol["start_at"]) + timedelta(
        days=protocol["enrollment_days"], hours=protocol["outcome_window_hours"],
    )
    if collection_end.tzinfo is None or collection_end.utcoffset() != timedelta(0) or collection_end < planned_end:
        raise ValueError("[immature_analysis] Все окна должны созреть до анализа")
    for field in ("assigned_users", "user_metric_rows", "experiment_summary_rows"):
        _count(field, quality.get(field))
    with duckdb.connect() as connection:
        connection.read_parquet(str(source / "experiment_summary.parquet")).create_view("summary")
        schema = {row[0]: row[1] for row in connection.execute("DESCRIBE summary").fetchall()}
        if schema != SUMMARY_SCHEMA:
            raise ValueError("[analysis_schema] Неверная схема итоговой таблицы")
        rows = connection.execute("SELECT experiment_id, scenario, variant, n_users, n_listing_view, n_contact, n_error FROM summary").fetchall()
    groups = {variant: {"n_users": 0, "n_listing_view": 0, "n_contact": 0, "n_error": 0} for variant in ("A", "B")}
    seen = set()
    for experiment_id, row_scenario, variant, n_users, n_views, n_contact, n_error in rows:
        if experiment_id != protocol["experiment_id"] or row_scenario != scenario or variant not in groups or variant in seen:
            raise ValueError("[analysis_summary_key] Неверный или повторный ключ группы")
        for name, value in (("n_users", n_users), ("n_listing_view", n_views), ("n_contact", n_contact), ("n_error", n_error)):
            _count(name, value)
        if not n_users or not 0 <= n_contact <= n_views <= n_users or n_error > n_users:
            raise ValueError("[analysis_summary_counts] Количества противоречат пользовательской воронке")
        groups[variant] = dict(n_users=n_users, n_listing_view=n_views, n_contact=n_contact, n_error=n_error)
        seen.add(variant)
    total = sum(group["n_users"] for group in groups.values())
    if total != quality["assigned_users"] or total != quality["user_metric_rows"] or len(rows) != quality["experiment_summary_rows"]:
        raise ValueError("[analysis_participant_count] Количества не совпадают с отчётом подготовки")
    return protocol, quality, groups


def analyze_estimates(source: str | Path, *, expected_scenario: str | None = None) -> dict:
    """Собрать только расчёты: промежуточный режим не выдаёт продуктового решения."""
    source = Path(source)
    protocol, quality, groups = load_analysis_input(source, expected_scenario=expected_scenario)
    a, b = groups["A"], groups["B"]
    srm = check_srm(a["n_users"], b["n_users"], protocol["expected_variant_b_probability"], protocol["srm_alpha"])
    primary = estimate_proportions(a["n_contact"], a["n_users"], b["n_contact"], b["n_users"], protocol["alpha"])
    # Для защиты нужен двусторонний интервал; отдельный p-value ошибок не заменяет его.
    guardrail = estimate_proportions(a["n_error"], a["n_users"], b["n_error"], b["n_users"], protocol["alpha"], include_test=False)
    blocked_reasons = []
    if not srm["passed"]:
        blocked_reasons.append(srm["reason"])
    for name, metric in (("primary", primary), ("guardrail", guardrail)):
        blocked_reasons.extend(f"{name}:{reason}" for reason in metric["applicability"]["reasons"])
    # Диагностика набора использует плановую альтернативу, а не наблюдаемый эффект.
    planned_effect = abs(float(proportion_effectsize(
        protocol["baseline_contact_probability"] + protocol["planning_contact_effect"],
        protocol["baseline_contact_probability"],
    )))
    power_model = NormalIndPower()
    required_per_arm = math.ceil(power_model.solve_power(
        effect_size=planned_effect, alpha=protocol["alpha"], power=protocol["target_power"],
        ratio=1, alternative="two-sided",
    ))
    planned_power = float(power_model.power(
        effect_size=planned_effect, nobs1=a["n_users"], alpha=protocol["alpha"],
        ratio=b["n_users"] / a["n_users"], alternative="two-sided",
    )) if a["n_users"] and b["n_users"] else None
    return {
        "schema_version": 1, "stage": "estimates", "complete": False,
        "missing_components": ["product_decision"], "decision": None,
        "experiment_id": protocol["experiment_id"], "scenario": quality["scenario"],
        "source": str(source.resolve()),
        "source_sha256": {filename: hashlib.sha256((source / filename).read_bytes()).hexdigest()
                          for filename in ("protocol.json", "quality_report.json", "experiment_summary.parquet")},
        "data_checks_passed": True, "all_windows_mature": True, "srm_checked": True,
        "inference_eligible": not blocked_reasons, "blocked_reasons": blocked_reasons,
        "parameters": {name: protocol[name] for name in (
            "alpha", "srm_alpha", "expected_variant_b_probability", "ci_method",
            "business_contact_effect", "max_error_increase",
        )},
        "sample_size": {
            "required_per_arm": required_per_arm,
            "below_planned_size": a["n_users"] < required_per_arm or b["n_users"] < required_per_arm,
            "power_at_planned_alternative": planned_power,
            "planning_effect": protocol["planning_contact_effect"],
        },
        "groups": groups, "srm": srm, "primary": primary, "guardrail": guardrail,
    }


def write_estimates(source: str | Path, output: str | Path, *, expected_scenario: str | None = None) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Каталог уже существует: {output}")
    result = analyze_estimates(source, expected_scenario=expected_scenario)
    # Запрещаем NaN/Infinity: неопределённые значения должны быть JSON null.
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    (output / "estimates.json").write_text(serialized, encoding="utf-8")
    return result


def analyze_full(source: str | Path, *, expected_scenario: str | None = None) -> dict:
    """Завершить расчёт решения, даже если результат отрицательный или недостоверный."""
    result = analyze_estimates(source, expected_scenario=expected_scenario)
    decision = make_decision(result)
    # result создан здесь заново; исходные таблицы и исторические estimates не меняются.
    result.update(stage="analysis", complete=True, missing_components=[], decision=decision)
    return result


def _display(value, *, multiplier: float = 1, suffix: str = "") -> str:
    return "не определено" if value is None else f"{value * multiplier:.4f}{suffix}"


def _display_interval(metric: dict) -> str:
    interval = metric["interval"]
    if interval is None:
        return "не рассчитан: метод неприменим"
    confidence = 100 * interval["confidence_level"]
    return (f"{confidence:g}%: [{100 * interval['lower']:.4f}; "
            f"{100 * interval['upper']:.4f}] п.п.")


def render_decision(result: dict) -> str:
    """Представить уже принятое решение человеку, без повторной классификации."""
    if result.get("complete") is not True or not isinstance(result.get("decision"), dict):
        raise ValueError("[incomplete_decision] Для отчёта нужен полный результат с решением")
    decision = result["decision"]
    labels = {
        "invalid": "Анализ недостоверен / не завершён",
        "guardrail_harm": "Недопустимое ухудшение ошибок подтверждено",
        "guardrail_unconfirmed": "Безопасность по критерию не подтверждена",
        "recommend_further_check": "Есть основания для дальнейшей проверки изменения",
        "no_practical_gain": "Практически важный выигрыш не подтверждён",
        "inconclusive": "Результат неопределённый",
    }
    guardrail_labels = {"not_assessed": "не оценивается", "passed": "пройдена",
                       "harm": "подтверждён недопустимый рост ошибок", "unconfirmed": "безопасность не подтверждена"}
    primary, guardrail, srm = result["primary"], result["guardrail"], result["srm"]
    parameters = result["parameters"]
    lines = [
        f"# Решение: {result['scenario']}", "",
        "Учебный эксперимент на синтетических данных.", "",
        f"Эксперимент: {result['experiment_id']}.", "",
        f"**Итог: {labels[decision['status']]}**", "", decision["reason"], "",
    ]
    if decision["status"] == "invalid":
        lines.extend(["**Оценки ниже имеют только диагностический смысл и не служат основанием для запуска изменения.**", ""])
    lines.extend([
        "## Наблюдаемые данные", "",
        "| Вариант | Назначены | С контактом | С ошибкой |",
        "|---|---:|---:|---:|",
    ])
    for variant in ("A", "B"):
        group = result["groups"][variant]
        lines.append(f"| {variant} | {group['n_users']} | {group['n_contact']} | {group['n_error']} |")
    lines.extend([
        "", "## Основная метрика", "",
        f"Конверсия A: {_display(primary['A']['rate'], multiplier=100, suffix='%')}; "
        f"B: {_display(primary['B']['rate'], multiplier=100, suffix='%')}.", "",
        f"Разница B − A: {_display(primary['difference'], multiplier=100, suffix=' п.п.')}", "",
        f"Интервал Newcombe: {_display_interval(primary)}", "",
        f"Двусторонний p-value: {format(primary['test']['p_value'], '.6g') if primary['test'] else 'не рассчитывался'}; "
        f"alpha = {parameters['alpha']:g}.", "",
        f"Бизнес-порог: {_display(parameters['business_contact_effect'], multiplier=100, suffix=' п.п.')}", "",
        "## Защитная метрика", "",
        f"Доля с ошибкой A: {_display(guardrail['A']['rate'], multiplier=100, suffix='%')}; "
        f"B: {_display(guardrail['B']['rate'], multiplier=100, suffix='%')}.", "",
        f"Разница ошибок B − A: {_display(guardrail['difference'], multiplier=100, suffix=' п.п.')}", "",
        f"Интервал Newcombe: {_display_interval(guardrail)}", "",
        f"Допустимый рост ошибок: {_display(parameters['max_error_increase'], multiplier=100, suffix=' п.п.')}", "",
        f"Защита: {guardrail_labels[decision['guardrail_status']]}.", "",
        "## Достоверность и набор", "",
        f"SRM: ожидаемые A/B = {srm['expected']['A']:g}/{srm['expected']['B']:g}; "
        f"порог p < {parameters['srm_alpha']:g}.", "",
        f"p-value SRM: {format(srm['p_value'], '.6g') if srm['p_value'] is not None else 'не рассчитывался'}. "
        f"Проверка {'пройдена' if srm['passed'] else 'не пройдена'}.", "",
        f"Плановый размер: {result['sample_size']['required_per_arm']} на вариант; "
        f"недобор относительно плана: {'да' if result['sample_size']['below_planned_size'] else 'нет'}.", "",
    ])
    if srm["p_value"] == 0:
        lines.extend(["Нулевой p-value SRM — результат машинной точности для крайне малого числа, а не математическое утверждение о нулевой вероятности.", ""])
    if decision["primary_decline"]:
        lines.extend(["Есть свидетельство снижения основной метрики: верхняя граница интервала ниже нуля.", ""])
    lines.extend([
        "Решение рассчитано по неокруглённым значениям. Числа в этом документе округлены для чтения.", "",
        "complete=true означает завершённый расчёт решения, включая недостоверный или отрицательный итог.", "",
    ])
    return "\n".join(lines)


def write_analysis(source: str | Path, output: str | Path, *, expected_scenario: str | None = None) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Каталог уже существует: {output}")
    result = analyze_full(source, expected_scenario=expected_scenario)
    # Готовим оба представления до создания каталога: ошибки расчёта не оставят экспорт.
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    markdown = render_decision(result)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(serialized, encoding="utf-8")
    (output / "decision.md").write_text(markdown, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Статистические оценки и решение по протоколу этапа 5")
    parser.add_argument("--input", type=Path, required=True, help="Каталог подготовленных сценариев этапа 4")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "analysis")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimates", action="store_true", help="Только оценки, без продуктового решения")
    mode.add_argument("--full", action="store_true", help="Полный анализ с решением и читаемым отчётом")
    args = parser.parse_args()
    names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    try:
        for name in names:
            if (args.output / name).exists():
                raise FileExistsError(f"Каталог уже существует: {args.output / name}")
        for name in names:
            writer = write_analysis if args.full else write_estimates
            result = writer(args.input / name, args.output / name, expected_scenario=name)
            status = result["decision"]["status"] if args.full else "промежуточные оценки, без решения"
            print(f"{name}: SRM пройден={result['srm']['passed']}, итог={status}")
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as error:
        parser.exit(2, f"Ошибка расчёта: {error}\n")
    if args.full:
        print("Полный расчёт сохранён: analysis.json и decision.md, complete=true.")
    else:
        print("Расчёты сохранены как complete=false. Продуктовой рекомендации в этом режиме нет.")


if __name__ == "__main__":
    main()
