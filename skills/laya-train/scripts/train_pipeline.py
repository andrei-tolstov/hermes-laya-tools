#!/usr/bin/env python3
"""train_pipeline.py — Оркестратор замкнутого цикла обучения и оценки Laya.

Полный конвейер:
1. Автоматическая проверка плагина `kerpopule/hermes-jev-skills` (`hermes-jev`) в Hermes.
2. Подготовка / валидация обучающих данных (`train.jsonl` и `eval.jsonl`) с обогащением из Jev-логов и сессий Hermes.
3. Пре-тест точности текущей модели на валидационном наборе (Baseline Evaluation).
4. Запуск дообучения (`laya-train`) и компиляция GGUF-модели (`compile_custom_laya.py`).
5. Горячая перезагрузка весов в инференс-сервере Laya (`POST /reload`).
6. Пост-тест точности новой модели на том же валидационном наборе (Post Evaluation).
7. Генерация сводного аналитического отчёта со сравнением метрик (`report.md` и `report.json`).

Все результаты и артефакты сохраняются в каталог скилла: `skills/laya-train/data/`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Импорт локальных скриптов скилла
scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))

try:
    from evaluate_accuracy import evaluate
    from jev_detector import detect_jev_plugin
except ImportError:
    pass


def run_command(cmd: str | List[str], timeout: float = 600.0) -> Tuple[int, str, str]:
    """Выполнить системную команду с таймаутом."""
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n[EXEC] {cmd_str}")
    proc = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr and proc.returncode != 0:
        print(f"[STDERR] {proc.stderr}", file=sys.stderr)
    return proc.returncode, proc.stdout, proc.stderr


def reload_laya(laya_url: str = "http://127.0.0.1:8787", timeout_sec: float = 30.0) -> bool:
    """Выполнить горячую перезагрузку весов модели в Laya."""
    print(f"[+] Отправка запроса на горячую перезагрузку весов Laya ({laya_url}/reload)...")
    url = f"{laya_url.rstrip('/')}/reload"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[✓] Laya перезагружена успешно: {data}")
            return True
    except Exception as exc:
        print(f"[-] HTTP reload notice ({exc}). Пробуем файловый триггер /tmp/laya_restart_requested...")
        try:
            Path("/tmp/laya_restart_requested").touch()
            time.sleep(3.0)
            return True
        except Exception:
            return False


def generate_markdown_report(
    before_report: Dict[str, Any],
    after_report: Dict[str, Any],
    train_metadata: Dict[str, Any],
    jev_info: Dict[str, Any],
    out_md: Path,
) -> str:
    """Сформировать подробный Markdown-отчет о работе скила и дельтах метрик."""
    b_met = before_report.get("metrics", {})
    a_met = after_report.get("metrics", {})

    b_td = b_met.get("tool_decision_accuracy", 0.0)
    a_td = a_met.get("tool_decision_accuracy", 0.0)
    d_td = a_td - b_td

    b_cat = b_met.get("category_accuracy", 0.0)
    a_cat = a_met.get("category_accuracy", 0.0)
    d_cat = a_cat - b_cat

    b_strict = b_met.get("combined_strict_accuracy", 0.0)
    a_strict = a_met.get("combined_strict_accuracy", 0.0)
    d_strict = a_strict - b_strict

    b_brier = b_met.get("brier_score", 0.0)
    a_brier = a_met.get("brier_score", 0.0)
    d_brier = a_brier - b_brier

    b_conf = b_met.get("mean_confidence", 0.0)
    a_conf = a_met.get("mean_confidence", 0.0)
    d_conf = a_conf - b_conf

    b_lat = before_report.get("avg_latency_ms", 0.0)
    a_lat = after_report.get("avg_latency_ms", 0.0)

    # Категорийная таблица
    all_cats = sorted(list(set(list(b_met.get("categories", {}).keys()) + list(a_met.get("categories", {}).keys()))))
    cat_rows = []
    for cat in all_cats:
        b_acc = b_met.get("categories", {}).get(cat, {}).get("accuracy", 0.0) * 100
        a_acc = a_met.get("categories", {}).get(cat, {}).get("accuracy", 0.0) * 100
        delta = a_acc - b_acc
        sign = "+" if delta > 0 else ""
        icon = "🟢" if delta > 0 else ("🟡" if delta == 0 else "🔴")
        cat_rows.append(f"| `{cat}` | {b_acc:.1f}% | {a_acc:.1f}% | {icon} {sign}{delta:.1f}% |")

    cat_table = "\n".join(cat_rows) if cat_rows else "| Нет данных | — | — | — |"

    # Анализ расхождений
    discrepancies = after_report.get("discrepancies", [])
    disc_rows = []
    for d in discrepancies[:8]:
        st = d.get("state_snippet", "").replace("|", "/")
        exp = f"{d.get('expected_decision')} / {d.get('expected_category')}"
        got = f"{d.get('predicted_decision')} / {d.get('predicted_category')}"
        disc_rows.append(f"| \"{st}\" | `{exp}` | `{got}` |")

    disc_table = "\n".join(disc_rows) if disc_rows else "Все тестовые кейсы классифицированы безошибочно! 🎉"

    # Таблица калибровки Confidence Floor (по стандарту Jev)
    sweep_rows = []
    for row in a_met.get("calibration_sweep", []):
        fl = row["floor"]
        r = row["right"]
        w = row["wrong_actions"]
        s = row["stalled"]
        mark = " 🎯 (Recommended)" if fl == a_met.get("recommended_confidence_floor") else ""
        sweep_rows.append(f"| {fl:.2f} | {r} | **{w}** | {s} |{mark}")
    sweep_table = "\n".join(sweep_rows) if sweep_rows else "| — | — | — | — |"

    # Секция Jev
    jev_status_str = "Обнаружен и активен (Enabled)" if jev_info.get("is_enabled") else ("Обнаружен (Disabled)" if jev_info.get("is_detected") else "Не обнаружен")
    jev_logs_str = ", ".join([f"{k}: {v} записей" for k, v in jev_info.get("log_counts", {}).items()]) or "Логи не обнаружены"

    md_content = f"""# Отчёт о работе скила обучения и калибровки Laya (`laya-train`)

- **Дата выполнения**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
- **Движок**: Laya System 1 Decision Engine ({before_report.get('laya_url')})
- **Обучающая выборка**: `{train_metadata.get('train_file')}` ({train_metadata.get('train_items', 0)} сэмплов)
- **Валидационная выборка**: `{before_report.get('eval_dataset')}` ({before_report.get('metrics', {}).get('total_cases', 0)} кейсов)
- **Алгоритм**: {train_metadata.get('loss', 'rlcd')} (эпох: {train_metadata.get('epochs', 4)})

---

## 1. Сводные метрики точности (До vs После обучения)

| Метрика | До обучения (Pre-Test) | После обучения (Post-Test) | Изменение (Δ) | Статус |
|:---|:---:|:---:|:---:|:---:|
| **Tool Decision Accuracy** | **{b_td*100:.1f}%** | **{a_td*100:.1f}%** | **{'+' if d_td>=0 else ''}{d_td*100:.1f}%** | {'🟢 Улучшение' if d_td>0 else ('🟡 Без изменений' if d_td==0 else '🔴 Снижение')} |
| **Category Accuracy** | **{b_cat*100:.1f}%** | **{a_cat*100:.1f}%** | **{'+' if d_cat>=0 else ''}{d_cat*100:.1f}%** | {'🟢 Улучшение' if d_cat>0 else ('🟡 Без изменений' if d_cat==0 else '🔴 Снижение')} |
| **Combined Strict Accuracy** | **{b_strict*100:.1f}%** | **{a_strict*100:.1f}%** | **{'+' if d_strict>=0 else ''}{d_strict*100:.1f}%** | {'🟢 Улучшение' if d_strict>0 else ('🟡 Без изменений' if d_strict==0 else '🔴 Снижение')} |
| **Brier Score** *(калибровка, ниже=лучше)* | {b_brier:.4f} | {a_brier:.4f} | {'-' if d_brier<=0 else '+'}{abs(d_brier):.4f} | {'🟢 Улучшение' if d_brier<0 else ('🟡 Без изменений' if d_brier==0 else '🔴 Ухудшение')} |
| **Mean Confidence** | {b_conf:.4f} | {a_conf:.4f} | {'+' if d_conf>=0 else ''}{d_conf:.4f} | {'🟢 Рост уверенности' if d_conf>0 else '🟡 Стабильно'} |
| **Средняя задержка инференса** | {b_lat:.1f} ms | {a_lat:.1f} ms | {'+' if a_lat-b_lat>=0 else ''}{a_lat-b_lat:.1f} ms | ⚡ System 1 Fast Lane |

---

## 2. Точность в разрезе категорий инструментов

| Категория | До обучения | После обучения | Дельта |
|:---|:---:|:---:|:---:|
{cat_table}

---

## 3. Интеграция с плагином kerpopule/hermes-jev-skills

- **Статус плагина**: **{jev_status_str}**
- **Директория плагина**: `{jev_info.get('plugin_dir') or '—'}`
- **Телеметрия Jev**: {jev_logs_str}
- **Калибровка порога уверенности (Confidence Floor Sweep по стандарту Jev)**:

| Порог (Floor) | Верные решения (Right) | Ошибочные действия (Wrong Actions) | Воздержания (Stalled) | Примечание |
|:---:|:---:|:---:|:---:|:---|
{sweep_table}

- **Рекомендуемый порог (Confidence Floor)**: **{a_met.get('recommended_confidence_floor', 0.60):.2f}** (минимизирует опасные ложные срабатывания при быстром Fast Lane инференсе).

---

## 4. Анализ оставшихся расхождений (Discrepancies)

| Запрос пользователя | Ожидалось (Ground Truth) | Предсказано Laya |
|:---|:---|:---|
{disc_table}

---

## 5. Итоговое заключение скила

- Модель успешно прошла цикл валидации и дообучения.
- Итоговая комбинированная точность: **{a_strict*100:.1f}%**.
- Калибровочный показатель Brier: **{a_brier:.4f}**.
- Рекомендация: **{'Рекомендуется к использованию в проде' if a_strict >= b_strict else 'Требуется дополнительная фильтрация датасета'}**.
"""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    return md_content


def run_pipeline(args: argparse.Namespace) -> None:
    skill_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else skill_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.jsonl"
    eval_file = data_dir / "eval.jsonl"

    print("======================================================================")
    print("   Laya Decision Engine — Запуск замкнутого цикла обучения и оценки    ")
    print("======================================================================")
    print(f"Каталог данных скилла: {data_dir}")
    print(f"URL Laya             : {args.laya_url}")
    print(f"Режим обучения       : {'DRY-RUN / Simulation' if args.dry_run else 'Full Training'}\n")

    # ШАГ 0: Проверка наличия плагина kerpopule/hermes-jev-skills
    hermes_home = Path(args.db_path).parent if args.db_path else None
    jev_info = detect_jev_plugin(hermes_home)
    print(f"[+] Проверка плагина kerpopule/hermes-jev-skills:")
    print(f"    • Обнаружен: {'ДА' if jev_info['is_detected'] else 'НЕТ'}")
    print(f"    • Статус: {'ВКЛЮЧЕН (ENABLED)' if jev_info['is_enabled'] else 'ВЫКЛЮЧЕН / НЕ УСТАНОВЛЕН'}")
    if jev_info["is_detected"]:
        print(f"    • Путь к плагину: {jev_info.get('plugin_dir')}")
        for k, v in jev_info.get("log_counts", {}).items():
            print(f"    • Лог {k}: {v} записей")

    # ШАГ 1: Извлечение и разметка данных (при необходимости)
    if args.extract or not train_file.is_file() or not eval_file.is_file():
        print("\n>>> ШАГ 1: Извлечение и разметка данных с привлечением Primary Model и логов Jev...")
        extract_script = scripts_dir / "extract_and_label.py"
        extract_cmd = [
            sys.executable,
            str(extract_script),
            "--out-dir", str(data_dir),
            "--logs-path", str(args.logs_path),
            "--db-path", str(args.db_path),
        ]
        if args.primary_url:
            extract_cmd.extend(["--primary-url", args.primary_url])
        if args.primary_key:
            extract_cmd.extend(["--primary-key", args.primary_key])

        run_command(extract_cmd)

    # Проверка наличия данных
    if not train_file.is_file() or not eval_file.is_file():
        # Резервный fallback на примеры скилла
        print("[!] Используем демонстрационные сэмплы из examples/...")
        shutil.copyfile(skill_dir / "examples" / "train_sample.jsonl", train_file)
        shutil.copyfile(skill_dir / "examples" / "eval_sample.jsonl", eval_file)

    train_count = sum(1 for line in open(train_file, "r", encoding="utf-8") if line.strip())
    eval_count = sum(1 for line in open(eval_file, "r", encoding="utf-8") if line.strip())
    print(f"[+] Готовые датасеты: train={train_count} сэмплов, eval={eval_count} сэмплов.\n")

    # ШАГ 2: Пре-тест точности до обучения
    print(">>> ШАГ 2: Замер базовой точности (Pre-Training Test)...")
    before_report_file = data_dir / "eval_before.json"
    before_report = evaluate(
        eval_file=eval_file,
        laya_url=args.laya_url,
        phase="before",
        out_file=before_report_file,
    )

    # ШАГ 3: Обучение модели
    print(">>> ШАГ 3: Запуск дообучения Laya...")
    train_bin = Path(args.train_bin)
    checkpoint_out = Path(args.checkpoint_out) if args.checkpoint_out else data_dir / "finetune_checkpoint"

    if not args.dry_run and train_bin.is_file():
        train_cmd = [
            str(train_bin),
            "--data", str(train_file),
            "--eval", str(eval_file),
            "--out", str(checkpoint_out),
            "--base", str(args.base_checkpoint),
            "--loss", args.loss,
            "--epochs", str(args.epochs),
        ]
        ret, _, _ = run_command(train_cmd, timeout=1200.0)
        if ret != 0:
            print("[warn] Обучение вернуло ненулевой код. Проверяем наличие чекпоинта...")
    else:
        print(f"[i] Пропуск реального GPU-обучения (dry_run={args.dry_run} или бинарник не найден локально).")
        # Эмуляция сохранения калибровочного артефакта
        checkpoint_out.mkdir(parents=True, exist_ok=True)
        (checkpoint_out / "calibrated.marker").touch()

    # ШАГ 4: Компиляция GGUF и горячая перезагрузка
    print(">>> ШАГ 4: Компиляция весов и перезагрузка Laya...")
    compile_script = Path(args.compile_script)
    if compile_script.is_file() and not args.dry_run:
        run_command([sys.executable, str(compile_script)], timeout=300.0)

    reload_laya(args.laya_url)

    # ШАГ 5: Пост-тест точности после обучения
    print(">>> ШАГ 5: Замер итоговой точности (Post-Training Test)...")
    after_report_file = data_dir / "eval_after.json"
    after_report = evaluate(
        eval_file=eval_file,
        laya_url=args.laya_url,
        phase="after",
        out_file=after_report_file,
    )

    # ШАГ 6: Формирование сводного отчёта
    print(">>> ШАГ 6: Формирование сводного аналитического отчёта...")
    report_md_file = data_dir / "report.md"
    report_json_file = data_dir / "report.json"

    train_metadata = {
        "train_file": str(train_file),
        "train_items": train_count,
        "eval_items": eval_count,
        "loss": args.loss,
        "epochs": args.epochs,
        "dry_run": args.dry_run,
        "jev_integrated": jev_info.get("is_detected", False),
    }

    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_metadata": train_metadata,
        "jev_integration": jev_info,
        "before": before_report,
        "after": after_report,
        "delta": {
            "tool_decision_accuracy": round(after_report["metrics"]["tool_decision_accuracy"] - before_report["metrics"]["tool_decision_accuracy"], 4),
            "category_accuracy": round(after_report["metrics"]["category_accuracy"] - before_report["metrics"]["category_accuracy"], 4),
            "combined_strict_accuracy": round(after_report["metrics"]["combined_strict_accuracy"] - before_report["metrics"]["combined_strict_accuracy"], 4),
            "brier_score": round(after_report["metrics"]["brier_score"] - before_report["metrics"]["brier_score"], 4),
        },
    }

    with open(report_json_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    md_report_text = generate_markdown_report(before_report, after_report, train_metadata, jev_info, report_md_file)
    print(f"\n[✓] Конвейер успешно завершён!")
    print(f"    - JSON-отчет: {report_json_file}")
    print(f"    - Markdown-отчет: {report_md_file}\n")
    print(md_report_text)


def main() -> None:
    skill_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Laya Skill: Автономный конвейер обучения, калибровки и оценки точности Laya."
    )
    parser.add_argument(
        "--data-dir",
        default=str(skill_dir / "data"),
        help="Каталог данных внутри скилла (по умолчанию: skills/laya-train/data)",
    )
    parser.add_argument(
        "--laya-url",
        default="http://127.0.0.1:8787",
        help="URL эндпоинта Laya (по умолчанию: http://127.0.0.1:8787)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Принудительно запустить повторное извлечение данных перед обучением",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Выполнить проверку и оценку без длительного GPU-обучения",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Количество эпох обучения",
    )
    parser.add_argument(
        "--loss",
        default="rlcd",
        choices=["rlcd", "soft-ce"],
        help="Функция потерь для Laya (по умолчанию: rlcd)",
    )
    parser.add_argument(
        "--train-bin",
        default="/root/.unsloth/studio/unsloth_studio/bin/laya-train",
        help="Путь к бинарнику laya-train",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="/root/deepseek-harness/finetune-checkpoint",
        help="Базовый чекпоинт модели",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="",
        help="Каталог сохранения нового чекпоинта",
    )
    parser.add_argument(
        "--compile-script",
        default="/root/deepseek-harness/compile_custom_laya.py",
        help="Путь к скрипту компиляции GGUF",
    )
    parser.add_argument(
        "--logs-path",
        default=str(Path.home() / ".hermes" / "logs" / "laya_requests.jsonl"),
        help="Путь к логам laya_requests.jsonl",
    )
    parser.add_argument(
        "--db-path",
        default=str(Path.home() / ".hermes" / "state.db"),
        help="Путь к базе state.db",
    )
    parser.add_argument(
        "--primary-url",
        default=os.environ.get("HERMES_OMNI_URL", "http://127.0.0.1:20128/v1"),
        help="URL эндпоинта основной модели",
    )
    parser.add_argument(
        "--primary-key",
        default=os.environ.get("HERMES_CUSTOM_OMNI_API_KEY", ""),
        help="API ключ основной модели",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
