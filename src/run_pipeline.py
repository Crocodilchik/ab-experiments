"""Один воспроизводимый запуск всех этапов в новые каталоги; без перезаписи."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys

from nbclient import NotebookClient
import nbformat

from src.analyze import SCENARIOS, write_analysis
from src.case_materials import write_case_materials
from src.config import validate_protocol, validate_simulation
from src.dashboard import write_dashboard
from src.generate_data import generate_full, write_full
from src.prepare_data import prepare_scenario


ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_notebook(path: Path, runtime: Path) -> None:
    """Запустить тем же Python и сохранить реальные выводы ячеек, включая ошибку."""
    # Ядро Jupyter общается с клиентом через локальные сокеты. Проверяем доступ
    # заранее: ограниченная среда должна дать ясную ошибку, а не зависший запуск.
    try:
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with socket.create_connection(server.getsockname(), timeout=3):
                pass
    except OSError as error:
        raise RuntimeError("Jupyter требует доступных локальных сокетов 127.0.0.1; текущая среда блокирует выполнение ядра") from error
    runtime = runtime.resolve()
    folders = {"JUPYTER_RUNTIME_DIR": runtime / "jupyter", "IPYTHONDIR": runtime / "ipython", "TEMP": runtime / "tmp", "TMP": runtime / "tmp"}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **{name: str(folder) for name, folder in folders.items()},
           "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
           "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=90, startup_timeout=30, allow_errors=False,
                            kernel_name="python3", resources={"metadata": {"path": str(path.parent.resolve())}})
    client.create_kernel_manager().ip = "127.0.0.1"
    try:
        if sys.platform == "win32":
            # ZeroMQ требует add_reader, которого нет у Windows ProactorEventLoop.
            # Локальный Runner не меняет глобальную политику циклов событий.
            with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
                runner.run(client.async_execute(env=env))
        else:
            client.execute(env=env)
    finally:
        nbformat.write(notebook, path)


def run_project(config_path: Path, run_dir: Path, report_dir: Path) -> dict:
    config_path, run_dir, report_dir = Path(config_path).resolve(), Path(run_dir).resolve(), Path(report_dir).resolve()
    if run_dir.is_relative_to(report_dir) or report_dir.is_relative_to(run_dir):
        raise ValueError("[pipeline_paths] Каталоги данных и отчёта должны быть разными и не вложенными")
    for path in (run_dir, report_dir):
        if path.exists():
            raise FileExistsError(f"Каталог уже существует: {path}")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if not isinstance(config, dict) or set(config) != {"protocol", "simulation"}:
        raise ValueError("[pipeline_config] Ожидаются ровно разделы protocol и simulation")
    validate_protocol(config["protocol"])
    validate_simulation(config["simulation"])
    files = [*sorted((ROOT / "src").glob("*.py")), *sorted((ROOT / "src").glob("*.html")),
             *sorted((ROOT / "sql").glob("*.sql")), ROOT / "requirements.txt", ROOT / "protocol.md", ROOT / "data_contracts.md"]
    manifest = {
        "schema_version": 1, "status": "running", "complete": False, "synthetic": True,
        "started_at": _now(), "finished_at": None, "phase": "configuration", "scenarios": list(SCENARIOS),
        "seed": config["simulation"]["seed"], "config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "run_directory": str(run_dir), "report_directory": str(report_dir),
        "source_sha256": {str(path.relative_to(ROOT)).replace('\\', '/'): _hash(path) for path in files},
        "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scipy", "statsmodels", "duckdb", "plotly", "nbformat", "nbclient", "ipykernel",
        )}},
        "notebook_executed": False, "decisions": {}, "error": None,
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config_snapshot.json").write_bytes(config_bytes)

    def save_manifest():
        serialized = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        (run_dir / "run_manifest.json").write_text(serialized, encoding="utf-8")
        if report_dir.is_dir():
            (report_dir / "run_manifest.json").write_text(serialized, encoding="utf-8")

    save_manifest()
    try:
        analyses = {}
        for scenario in SCENARIOS:
            manifest["phase"] = f"generate:{scenario}"
            save_manifest()
            tables = generate_full(config["protocol"], config["simulation"], scenario)
            write_full(tables, config, scenario, run_dir / "raw" / scenario)
            manifest["phase"] = f"prepare:{scenario}"
            save_manifest()
            prepare_scenario(run_dir / "raw" / scenario, run_dir / "prepared" / scenario, expected_scenario=scenario)
            manifest["phase"] = f"analyze:{scenario}"
            save_manifest()
            analyses[scenario] = write_analysis(run_dir / "prepared" / scenario, run_dir / "analysis" / scenario, expected_scenario=scenario)
            manifest["decisions"][scenario] = analyses[scenario]["decision"]["status"]
        manifest["phase"] = "case_materials"
        save_manifest()
        write_case_materials(analyses, config, report_dir)
        manifest["phase"] = "dashboard"
        save_manifest()
        write_dashboard(run_dir / "analysis", run_dir / "prepared", report_dir / "dashboard")
        manifest["phase"] = "notebook"
        save_manifest()
        execute_notebook(report_dir / "experiment_review.ipynb", run_dir / ".notebook_runtime")
        manifest["notebook_executed"] = True
        # Временные файлы Jupyter и сам manifest не входят в рекурсивные хеши.
        manifest["artifact_sha256"] = {
            **{f"run/{path.relative_to(run_dir).as_posix()}": _hash(path) for path in sorted(run_dir.rglob('*'))
               if path.is_file() and '.notebook_runtime' not in path.relative_to(run_dir).parts and path.name != 'run_manifest.json'},
            **{f"report/{path.relative_to(report_dir).as_posix()}": _hash(path) for path in sorted(report_dir.rglob('*'))
               if path.is_file() and path.name != 'run_manifest.json'},
        }
        manifest.update(status="complete", complete=True, phase="complete", finished_at=_now())
        save_manifest()
        return manifest
    except Exception as error:
        manifest.update(status="failed", complete=False, finished_at=_now(), error=f"{type(error).__name__}: {error}")
        save_manifest()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный воспроизводимый учебный A/B-кейс")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--run-dir", type=Path, required=True, help="Новый каталог данных всех этапов")
    parser.add_argument("--report-dir", type=Path, required=True, help="Новый каталог материалов, рекомендуется внутри learning")
    args = parser.parse_args()
    try:
        result = run_project(args.config, args.run_dir, args.report_dir)
    except Exception as error:
        parser.exit(2, f"Ошибка полного запуска: {error}\n")
    print(f"Полный запуск завершён; notebook выполнен: {result['notebook_executed']}")
    print(f"Материалы: {args.report_dir}; дашборд: {args.report_dir / 'dashboard' / 'dashboard.html'}")
    for name, status in result["decisions"].items():
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
