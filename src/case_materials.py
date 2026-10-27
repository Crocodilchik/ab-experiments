"""Итоговые материалы: наблюдаемые результаты отдельно от параметров симулятора."""

import csv
import json
from pathlib import Path

import nbformat

from src.analyze import SCENARIOS
from src.config import validate_protocol, validate_simulation
from src.decision import make_decision


METRIC_UNITS = {
    "experiment_id": "identifier", "scenario": "identifier", "synthetic": "boolean",
    "n_users_a": "users", "n_users_b": "users", "n_contact_a": "users", "n_contact_b": "users",
    "contact_rate_a": "fraction", "contact_rate_b": "fraction", "contact_difference": "fraction",
    "contact_difference_pp": "percentage_points", "contact_relative_change_percent": "relative_percent",
    "contact_p_value": "probability", "contact_ci_lower": "fraction", "contact_ci_upper": "fraction",
    "n_error_a": "users", "n_error_b": "users", "error_rate_a": "fraction", "error_rate_b": "fraction",
    "error_difference": "fraction", "error_ci_lower": "fraction", "error_ci_upper": "fraction",
    "srm_p_value": "probability", "srm_passed": "boolean", "inference_eligible": "boolean",
    "diagnostic_only": "boolean", "decision_status": "category", "guardrail_status": "category",
}


def _display(value, scale=1, signed=False) -> str:
    if value is None:
        return "не определено"
    return format(scale * value, "+.4f" if signed else ".4f")


def _interval(metric) -> str:
    interval = metric["interval"]
    return "не рассчитан" if interval is None else f"[{_display(interval['lower'], 100, True)}; {_display(interval['upper'], 100, True)}]"


def build_metrics(analyses: dict, config: dict) -> list[dict]:
    """Сводная таблица использует только оценки, без подстановки заданного эффекта."""
    validate_protocol(config["protocol"])
    validate_simulation(config["simulation"])
    if set(analyses) != set(SCENARIOS):
        raise ValueError("[materials_scenarios] Нужны четыре отдельных анализа")
    # Не допускаем NaN/Infinity в машинных материалах.
    json.dumps(analyses, allow_nan=False)
    rows = []
    for scenario in SCENARIOS:
        result = analyses[scenario]
        if (result.get("stage") != "analysis" or result.get("complete") is not True
                or result.get("missing_components") != [] or result.get("scenario") != scenario
                or result.get("experiment_id") != config["protocol"]["experiment_id"]):
            raise ValueError("[materials_incomplete] Нужны согласованные полные результаты")
        if result.get("decision") != make_decision(result):
            raise ValueError("[materials_decision] Решение расходится с протоколом")
        parameters = ("alpha", "srm_alpha", "expected_variant_b_probability", "ci_method", "business_contact_effect", "max_error_increase")
        if result["parameters"] != {name: config["protocol"][name] for name in parameters}:
            raise ValueError("[materials_protocol] Пороги не совпадают со снимком протокола")
        primary, error, groups = result["primary"], result["guardrail"], result["groups"]
        rows.append({
            "experiment_id": result["experiment_id"], "scenario": scenario, "synthetic": True,
            "n_users_a": groups["A"]["n_users"], "n_users_b": groups["B"]["n_users"],
            "n_contact_a": groups["A"]["n_contact"], "n_contact_b": groups["B"]["n_contact"],
            "contact_rate_a": primary["A"]["rate"], "contact_rate_b": primary["B"]["rate"],
            "contact_difference": primary["difference"], "contact_difference_pp": primary["difference_pp"],
            "contact_relative_change_percent": primary["relative_change_percent"],
            "contact_p_value": primary["test"]["p_value"] if primary["test"] else None,
            "contact_ci_lower": primary["interval"]["lower"] if primary["interval"] else None,
            "contact_ci_upper": primary["interval"]["upper"] if primary["interval"] else None,
            "n_error_a": groups["A"]["n_error"], "n_error_b": groups["B"]["n_error"],
            "error_rate_a": error["A"]["rate"], "error_rate_b": error["B"]["rate"],
            "error_difference": error["difference"],
            "error_ci_lower": error["interval"]["lower"] if error["interval"] else None,
            "error_ci_upper": error["interval"]["upper"] if error["interval"] else None,
            "srm_p_value": result["srm"]["p_value"], "srm_passed": result["srm"]["passed"],
            "inference_eligible": result["inference_eligible"],
            "diagnostic_only": result["decision"]["status"] == "invalid",
            "decision_status": result["decision"]["status"], "guardrail_status": result["decision"]["guardrail_status"],
        })
    return rows


def build_design(config: dict) -> dict:
    """Описание заданных вероятностей для обучения; не вход статистического анализа."""
    scenarios = {}
    for name in SCENARIOS:
        scenario = config["simulation"]["scenarios"][name]
        rates = {variant: scenario[variant]["view_probability"] * scenario[variant]["contact_given_view_probability"]
                 for variant in ("A", "B")}
        scenarios[name] = {
            "contact_probability_a": rates["A"], "contact_probability_b": rates["B"],
            "contact_difference": rates["B"] - rates["A"],
            "error_difference": scenario["B"]["error_probability"] - scenario["A"]["error_probability"],
            "actual_variant_b_probability": scenario["actual_variant_b_probability"],
        }
    return {"synthetic": True, "purpose": "Описание модели, не вход теста или правила решения",
            "seed": config["simulation"]["seed"], "protocol": config["protocol"], "scenarios": scenarios}


def render_conclusions(analyses: dict, config: dict) -> str:
    protocol = config["protocol"]
    design = build_design(config)
    lines = [
        "# Итоги учебного A/B-эксперимента", "",
        "Синтетические данные маркетплейса. Это демонстрация аналитического метода; реальных данных или результатов Авито здесь нет.", "",
        "## Задача и дизайн", "",
        "Проверить, меняет ли новая карточка вероятность контакта с продавцом без недопустимого роста технических ошибок.", "",
        f"Эксперимент: {protocol['experiment_id']}. Seed симулятора: {design['seed']}. "
        f"Начало виртуального набора: {protocol['start_at']}; набор {protocol['enrollment_days']} суток, "
        f"после него ещё {protocol['outcome_window_hours']} часа сбора событий.", "",
        "Единица назначения и анализа — пользователь. Основной знаменатель — все назначенные в вариант; окно [назначение, назначение + 24 часа). "
        "Повторные контакты не создают новых независимых наблюдений. Сценарии анализируются раздельно на общей синтетической когорте; их нельзя складывать в одну независимую выборку.", "",
        f"План: baseline {_display(protocol['baseline_contact_probability'], 100)}%, "
        f"MDE {_display(protocol['planning_contact_effect'], 100)} п.п., "
        f"мощность {_display(protocol['target_power'], 100)}% против нулевого эффекта. "
        f"Бизнес-порог {_display(protocol['business_contact_effect'], 100)} п.п. — отдельное условие рекомендации.", "",
        f"Двусторонний z-тест, alpha={protocol['alpha']}; интервалы Newcombe с уровнем {_display(1-protocol['alpha'], 100)}%. "
        f"SRM: ожидается 50/50, порог {protocol['srm_alpha']}. "
        f"Допустимый рост ошибок {_display(protocol['max_error_increase'], 100)} п.п. проверяется верхней границей двустороннего интервала.", "",
        "## Наблюдаемые результаты", "",
        "| Сценарий | N A / B | Конверсия A / B, % | B − A, п.п. | Интервал контактов, п.п. | Интервал ошибок, п.п. | Решение |",
        "|---|---|---|---:|---|---|---|",
    ]
    for name in SCENARIOS:
        result = analyses[name]
        p, g = result["primary"], result["groups"]
        lines.append(f"| {name} | {g['A']['n_users']} / {g['B']['n_users']} | "
                     f"{_display(p['A']['rate'], 100)} / {_display(p['B']['rate'], 100)} | "
                     f"{_display(p['difference'], 100, True)} | {_interval(p)} | "
                     f"{_interval(result['guardrail'])} | {result['decision']['status']} |")
    lines.extend(["", "Решение использует исходную точность. Таблица округлена для чтения; полный CSV и analysis.json сохраняют численные оценки. "
                  "При invalid числа приведены только как диагностика, даже если контактный p-value мал.", ""])
    for name in SCENARIOS:
        result = analyses[name]
        p = result["primary"]["test"]
        lines.extend([f"### {name}", "", result["decision"]["reason"], "",
                      f"Контактный p-value: {format(p['p_value'], '.8g') if p else 'не рассчитывался'}; "
                      f"SRM пройден: {'да' if result['srm']['passed'] else 'нет'}. "
                      f"Пригодность для продуктового вывода: {'да' if result['inference_eligible'] else 'нет'}.", ""])
    lines.extend([
        "## Задано симулятору и оценено по данным — разные величины", "",
        "Следующая таблица описывает известные параметры учебной модели. Эти значения не использовались для построения оценки, p-value или выбора решения.", "",
        "| Сценарий | Задана разница контактов, п.п. | Оценена разница, п.п. | Задана разница ошибок, п.п. |",
        "|---|---:|---:|---:|",
    ])
    for name in SCENARIOS:
        lines.append(f"| {name} | {_display(design['scenarios'][name]['contact_difference'], 100, True)} | "
                     f"{_display(analyses[name]['primary']['difference'], 100, True)} | "
                     f"{_display(design['scenarios'][name]['error_difference'], 100, True)} |")
    lines.extend([
        "", "Случайные наблюдения не обязаны равняться заданным вероятностям. Одинаковые настройки null не гарантируют нулевую наблюдаемую разницу. "
        "В broken_assignment ожидаемое правило анализа остаётся 50/50; известная поломка генератора не исправляет достоверность результата.", "",
        "## Ограничения", "",
        "- Вероятности поведения заданы моделью. Переносимость результатов на реальный продукт не проверена.",
        "- Пользователи считаются независимыми; рыночное взаимодействие покупателей и продавцов не моделируется.",
        "- Защита покрывает технические ошибки. Качество контактов, выручка, жалобы и долгосрочные эффекты не измеряются.",
        "- Baseline, MDE, бизнес-порог и допустимый рост ошибок учебные и не выведены из экономики продукта.",
        "- Непрохождение значимости не доказывает равенство вариантов. Пройденная защита относится только к выбранному критерию.",
        "- Это один заранее выбранный seed, без перебора удачных исходов, досрочной остановки или продления после результатов.",
        "- Плановая мощность не равна вероятности выполнить совместно все условия положительной рекомендации.", "",
        "## Как читать машинные файлы", "",
        "metrics.csv: UTF-8, разделитель — запятая, десятичный знак — точка. Пустое поле означает неопределённое значение, а не ноль. "
        "fraction — доля 0–1; percentage_points — абсолютная разница в п.п.; relative_percent — относительное изменение в процентах. "
        "Полный словарь единиц — metrics_schema.json. Нулевой машинный p-value SRM может быть следствием предела точности вычисления.", "",
        "Воспроизведение подтверждается run_manifest.json. Дашборд находится в dashboard/dashboard.html; "
        "notebook experiment_review.ipynb читает соседние CSV и описание модели.", "",
    ])
    return "\n".join(lines)


def render_story(analyses: dict, config: dict) -> str:
    uplift = analyses["uplift"]
    group = uplift["groups"]
    p = config["protocol"]
    return f"""# Рассказ о проекте на пять минут

## 0:00–0:45. Задача

Это учебный кейс на синтетических данных маркетплейса. Я рассматриваю изменение карточки объявления и оцениваю вероятность контакта с продавцом. Положительной конверсии недостаточно: изменение должно пройти защиту по техническим ошибкам. Это не эксперимент Авито и не доказательство коммерческого эффекта.

## 0:45–1:35. Дизайн

Единица назначения и анализа — пользователь, ожидаемое распределение 50/50. Основная метрика — хотя бы один контакт среди всех назначенных, включая людей без событий. Набор фиксирован заранее: {p['enrollment_days']} суток; окно каждого участника — {p['outcome_window_hours']} часа. Плановый MDE {_display(p['planning_contact_effect'],100)} п.п. используется для выборки, бизнес-порог {_display(p['business_contact_effect'],100)} п.п. — для решения. После результатов срок и seed не меняются.

## 1:35–2:25. Данные

Генератор воспроизводимо создаёт четыре сценария. SQL сначала различает точные копии и конфликты ключа, затем строит одну строку на участника. LEFT JOIN сохраняет людей без событий; временные условия находятся в ON. Контакт требует более раннего просмотра того же объявления. В итогах суммируются пользовательские флаги, а не строки действий.

## 2:25–3:20. Оценка

До вывода проверяются качество, зрелость окон, SRM и применимость нормального приближения. Контакты оцениваются двусторонним z-тестом и интервалом Newcombe для B − A. Защита сравнивает верхнюю границу интервала ошибок с допустимым ухудшением. Интервалы и тест могут расходиться у границы: правило заранее предусматривает неопределённый результат.

## 3:20–4:15. Четыре результата

Для иллюстрации в uplift назначены {group['A']['n_users']} в A и {group['B']['n_users']} в B. Наблюдаемая разница контактов {_display(uplift['primary']['difference'],100,True)} п.п.; интервал {_interval(uplift['primary'])} п.п. Итог этого запуска — {uplift['decision']['status']}. Вся таблица включает также null ({analyses['null']['decision']['status']}), harmful ({analyses['harmful']['decision']['status']}) и broken_assignment ({analyses['broken_assignment']['decision']['status']}). Я показываю все сценарии и не выбираю удачный p-value. При SRM оценки остаются диагностическими.

## 4:15–5:00. Ограничения и воспроизводимость

Истинные вероятности известны только потому, что это симуляция; оценка строится по наблюдениям. Не моделируются взаимодействия покупателей и продавцов, выручка и долгосрочные эффекты. Один локальный запуск создаёт данные, SQL-витрины, анализ, автономный HTML и итоговые материалы. Manifest фиксирует конфигурацию, код, версии и хеши артефактов. Проект разрабатывался как обучение совместно с AI; история реализации и объяснения сохранены в learning.
"""


def build_notebook():
    """Notebook читает соседние артефакты, не запускает новый эксперимент."""
    nbf = nbformat.v4
    return nbf.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, cells=[
        nbf.new_markdown_cell("# Разбор учебного A/B-эксперимента\n\nСинтетические данные. Notebook использует готовые результаты, не меняет seed и не пересчитывает продуктовое правило. Запускается из папки итоговых материалов; CSV и JSON должны лежать рядом."),
        nbf.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\n\nfolder = Path.cwd()\nmetrics = pd.read_csv(folder / 'metrics.csv')\ndesign = json.loads((folder / 'scenario_design.json').read_text(encoding='utf-8'))\nprint('Синтетические данные; seed:', design['seed'])\nprint('Сценариев:', len(metrics))"),
        nbf.new_markdown_cell("## Назначенные и контактная конверсия\n\nЗнаменатели включают участников без событий. contact_rate — доля, contact_difference_pp — процентные пункты. NaN в таблице означает неопределённое значение, а не ноль."),
        nbf.new_code_cell("metrics[['scenario', 'n_users_a', 'n_users_b', 'n_contact_a', 'n_contact_b', 'contact_rate_a', 'contact_rate_b', 'contact_difference_pp', 'contact_relative_change_percent']]"),
        nbf.new_markdown_cell("## Неопределённость и решение\n\nПри diagnostic_only=True значения нельзя использовать для продуктовой рекомендации. Точный интервал хранится в шкале долей; маленький p-value сам по себе не заменяет защиту и бизнес-порог."),
        nbf.new_code_cell("metrics[['scenario', 'contact_p_value', 'contact_ci_lower', 'contact_ci_upper', 'error_ci_lower', 'error_ci_upper', 'srm_p_value', 'srm_passed', 'diagnostic_only', 'decision_status']]"),
        nbf.new_markdown_cell("## Параметры модели и наблюдения\n\nЭто сравнение для объяснения симуляции. Статистический анализ и выбор решения не используют заданный генератору эффект."),
        nbf.new_code_cell("model = pd.DataFrame.from_dict(design['scenarios'], orient='index').rename_axis('scenario').reset_index()\ncomparison = metrics[['scenario', 'contact_difference_pp', 'decision_status']].merge(model[['scenario', 'contact_difference']], on='scenario', validate='one_to_one')\ncomparison = comparison.rename(columns={'contact_difference_pp': 'observed_difference_pp', 'contact_difference': 'configured_difference'})\ncomparison['configured_difference_pp'] = 100 * comparison['configured_difference']\ncomparison[['scenario', 'configured_difference_pp', 'observed_difference_pp', 'decision_status']]"),
        nbf.new_markdown_cell("## Вывод\n\nРазбор каждого решения и ограничения находятся в conclusions.md. Интерактивные воронка, интервалы и дневные графики — в dashboard/dashboard.html. Сценарии не объединяются в одну выборку. Реальная выручка, качество лидов и переносимость на настоящий маркетплейс не оценивались."),
    ])


def write_case_materials(analyses: dict, config: dict, output: Path) -> None:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Каталог уже существует: {output}")
    rows = build_metrics(analyses, config)
    conclusions, story = render_conclusions(analyses, config), render_story(analyses, config)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(METRIC_UNITS))
        writer.writeheader()
        writer.writerows(rows)
    for name, content in (("conclusions.md", conclusions), ("case_story.md", story)):
        (output / name).write_text(content, encoding="utf-8")
    (output / "metrics_schema.json").write_text(json.dumps({"schema_version": 1, "synthetic": True, "undefined": "empty CSV field", "units": METRIC_UNITS}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "scenario_design.json").write_text(json.dumps(build_design(config), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    nbformat.write(build_notebook(), output / "experiment_review.ipynb")
