"""Продуктовое правило из protocol.md, раздел 6, без повторного расчёта статистики."""

import math


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _finite(value) -> bool:
    # NaN, бесконечность и bool не должны незаметно участвовать в сравнении порогов.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _result(status: str, guardrail_status: str, primary_decline: bool | None, reason: str) -> dict:
    return dict(status=status, guardrail_status=guardrail_status,
                primary_decline=primary_decline, reason=reason)


def make_decision(analysis: dict) -> dict:
    """Выбрать ровно один итог, сохранив входной словарь неизменным.

    Порядок важен: достоверность → защита → согласованность теста/интервала → эффект.
    Имя сценария, seed и параметры генератора в принятии решения не участвуют.
    """
    analysis = _mapping(analysis)
    srm = _mapping(analysis.get("srm"))
    primary = _mapping(analysis.get("primary"))
    guardrail = _mapping(analysis.get("guardrail"))
    checks = (
        (analysis.get("data_checks_passed") is True, "проверки качества данных не пройдены"),
        (analysis.get("all_windows_mature") is True, "полнота индивидуальных окон не подтверждена"),
        (srm.get("applicable") is True, "критерий SRM неприменим"),
        (srm.get("passed") is True, "проверка распределения назначений SRM не пройдена"),
        (_mapping(primary.get("applicability")).get("passed") is True, "метод основной метрики неприменим"),
        (_mapping(guardrail.get("applicability")).get("passed") is True, "метод защитной метрики неприменим"),
    )
    failures = [reason for passed, reason in checks if not passed]
    if analysis.get("inference_eligible") is not True and not failures:
        failures.append("общая пригодность анализа не подтверждена")
    if failures:
        # При заблокированном анализе интервалы могут быть None: их ещё не читаем.
        return _result("invalid", "not_assessed", None, "Анализ недостоверен или не завершён: " + "; ".join(failures) + ".")

    parameters = _mapping(analysis.get("parameters"))
    interval = _mapping(primary.get("interval"))
    error_interval = _mapping(guardrail.get("interval"))
    lower, upper = interval.get("lower"), interval.get("upper")
    error_lower, error_upper = error_interval.get("lower"), error_interval.get("upper")
    p_value = _mapping(primary.get("test")).get("p_value")
    difference = primary.get("difference")
    alpha = parameters.get("alpha")
    business = parameters.get("business_contact_effect")
    max_error = parameters.get("max_error_increase")
    numbers = (lower, upper, error_lower, error_upper, p_value, difference, alpha, business, max_error)
    if not all(_finite(value) for value in numbers):
        return _result("invalid", "not_assessed", None, "Отсутствуют конечные численные оценки или пороги; решение невозможно.")
    if not (-1 <= lower <= upper <= 1 and -1 <= error_lower <= error_upper <= 1
            and -1 <= difference <= 1 and 0 <= p_value <= 1
            and 0 < alpha < 1 and 0 < business < 1 and 0 < max_error < 1):
        return _result("invalid", "not_assessed", None, "Численные оценки или пороги имеют недопустимые границы.")

    primary_decline = upper < 0
    if error_upper <= max_error:
        guardrail_status = "passed"
    elif error_lower > max_error:
        return _result("guardrail_harm", "harm", primary_decline,
                       "Подтверждено недопустимое ухудшение ошибок: нижняя граница выше допустимого роста. Изменение не рекомендуется.")
    else:
        return _result("guardrail_unconfirmed", "unconfirmed", primary_decline,
                       "Безопасность по защитному критерию не подтверждена: верхняя граница выше допустимого роста, но нижняя его не превышает. Изменение не рекомендуется.")

    # Двусторонний тест сравниваем с исключением нуля в любом направлении.
    significant = p_value < alpha
    excludes_zero = lower > 0 or upper < 0
    if significant != excludes_zero:
        return _result("inconclusive", guardrail_status, primary_decline,
                       "Тест и интервал расходятся по значимости относительно нуля; результат неопределённый.")
    if significant and difference > 0 and lower > business:
        return _result("recommend_further_check", guardrail_status, primary_decline,
                       "Защита пройдена, положительный эффект значим, нижняя граница выше бизнес-порога. Есть основания для следующей проверки или ограниченного запуска в симулируемом продукте.")
    if upper < business:
        reason = "Верхняя граница основной разницы ниже бизнес-порога: практически важный выигрыш не подтверждён."
        if primary_decline:
            reason += " Интервал целиком ниже нуля: есть свидетельство снижения основной метрики."
        return _result("no_practical_gain", guardrail_status, primary_decline, reason)
    return _result("inconclusive", guardrail_status, primary_decline,
                   "Данных недостаточно для уверенной положительной рекомендации по заранее заданным условиям.")
