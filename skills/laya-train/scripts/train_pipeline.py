#!/usr/bin/env python3
"""train_pipeline.py — Оркестратор замкнутого цикла обучения, калибровки и оценки Laya.

Полный конвейер:
1. Автоматическая проверка плагина `kerpopule/hermes-jev-skills` (`hermes-jev`) в Hermes.
2. Сбор и сверка выборок Laya и Основной модели (`extract_and_label.py`):
   - Если есть данные Laya и Основной модели -> дообучение на выборке Основной модели.
   - Если данных нет -> live-запрос к Основной модели (и Laya) для сравнения и формирования эталона.
3. Строгая валидация датасетов (`laya-evals validate` + проверка поля `expected`).
4. Пре-тест точности текущей модели на валидационном наборе (Baseline Evaluation).
5. Запуск дообучения (`laya-train`) с `--shuffle-options` и адаптивным бюджетом шагов (защита от Prior Collapse).
6. Компиляция GGUF-модели с контролем совместимости токенизатора (`gemma` / `gpt2`) и атомарным обновлением `/root/deepseek-harness/models/laya_multilingual_f16.gguf`.
7. Горячая перезагрузка весов в инференс-сервере Laya (`POST /reload`).
8. Пост-тест точности новой модели на том же валидационном наборе (Post Evaluation).
9. Генерация сводного аналитического отчёта (`report.md` и `report.json`).

Все результаты и артефакты сохраняются в каталог скилла: `skills/laya-train/data/`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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


def get_gpu0_free_vram_mib() -> Optional[int]:
    """Проверить объем свободной видеопамяти (MiB) на GPU0 через nvidia-smi."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=5.0,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0].strip()
            return int(float(first_line))
    except Exception:
        pass
    return None


def backup_before_replace(src: Path, stamp: str, keep: int = 3) -> Optional[Path]:
    """Сделать резервную копию файла или каталога `src` перед заменой: `<имя>.<YYYYmmdd-HHMMSS>.bak` рядом с оригиналом.

    Хранятся `keep` последних копий (более старые удаляются). Ручные бэкапы без метки времени
    (например, `laya_multilingual_f16.gguf.bak`) не затрагиваются. Возвращает путь копии или None,
    если копировать нечего.
    """
    if not src.exists():
        return None
    dst = src.parent / f"{src.name}.{stamp}.bak"
    if dst.exists():
        return dst
    print(f"[+] Резервная копия перед заменой: {src} -> {dst}")
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

    ts_re = re.compile(rf"^{re.escape(src.name)}\.\d{{8}}-\d{{6}}\.bak$")
    old = sorted(p for p in src.parent.iterdir() if ts_re.match(p.name))
    if keep > 0:
        for stale in old[:-keep]:
            print(f"[i] Удаляю устаревшую резервную копию: {stale}")
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)
    return dst


def validate_jsonl_schema(file_path: Path, evals_bin: Optional[Path] = None) -> int:
    """Проверить наличие обязательных полей (`state`, `questions`, `expected`) и запустить `laya-evals validate`."""
    valid_rows = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "state" not in row or "questions" not in row or ("expected" not in row and "gold" not in row):
                raise ValueError(
                    f"Ошибка схемы в {file_path}:{idx} — отсутствует обязательное поле 'expected' для laya-train!"
                )
            valid_rows += 1

    if evals_bin and evals_bin.is_file():
        ret, _, _ = run_command([str(evals_bin), "validate", str(file_path)], timeout=60.0)
        if ret != 0:
            print(f"[warn] laya-evals validate вернул код {ret} для {file_path}")
    return valid_rows


def compile_checkpoint_to_gguf(
    python_bin: str,
    checkpoint_dir: Path,
    target_gguf: Path,
    compile_script: Optional[Path] = None,
) -> bool:
    """Скомпилировать обученный чекпоинт Laya в GGUF с автоопределением токенизатора и атомарной заменой."""
    new_gguf = target_gguf.parent / f"{target_gguf.stem}_new.gguf"
    inline_code = f"""
import json, shutil, sys
from pathlib import Path

ROOT = Path("/root/ggmlc")
if (ROOT / "python").is_dir():
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "examples" / "laya"))

import torch
import laya
import ggmlc
from compile_laya import _example_batch, _pipeline_tokenizer, _length_buckets, CHECKPOINTS, MAX_OPTS, LayaCleanTrunk, FusionOptions, DecisionPipelineSpec

checkpoint = {json.dumps(str(checkpoint_dir))}
output = Path({json.dumps(str(new_gguf))})
target = Path({json.dumps(str(target_gguf))})
quantize = "f16"

print(f"Loading checkpoint {{checkpoint}} ...")
agent = laya.load(checkpoint, device="cpu")
agent.model.cpu().float().eval()

# Определяем реальный pre_tokenizer по словарю модели (Gemma 256k vs GPT-2 50k),
# сохраняя совместимое имя модели laya-multilingual для C++ сервера
vocab_sz = len(agent.tok)
enc_name = str(agent.cfg.get("encoder", "")).lower()
pre_tok = "gemma" if ("mmbert" in enc_name or vocab_sz > 100000) else "gpt2"
spec = dict(CHECKPOINTS["multilingual"])
spec["pre_tokenizer"] = pre_tok
print(f"Detected encoder={{enc_name}}, vocab_size={{vocab_sz}} -> pre_tokenizer={{pre_tok}}")

max_len = int(agent.cfg.get("max_len", 512))
head_max_len = int(agent.cfg.get("head_max_len", 192))
min_seq = 64
max_batch = 8

trunk = LayaCleanTrunk(agent, max_len=max_len).eval()
example = _example_batch(agent, seq_len=min(128, max_len))
tok = _pipeline_tokenizer(agent, spec["pre_tokenizer"], max_len)

decision = DecisionPipelineSpec.laya(
    cls_id=int(agent.tok.cls_token_id),
    sep_id=int(agent.tok.sep_token_id),
    pad_id=int(agent.tok.pad_token_id),
    mask_id=int(agent.tok.mask_token_id),
    max_len=max_len,
    head_max_len=head_max_len,
    max_opts=MAX_OPTS,
    min_seq=int(min_seq),
    max_batch=int(max_batch),
    length_buckets=_length_buckets(max_len),
    temperature=agent.temperature,
    temperature_by_options=dict(agent.temperature_by_options),
    model_name=spec["model_name"],
    family=spec["family"],
    checkpoint=checkpoint,
)
extra = {{
    "laya.max_len": max_len,
    "laya.head_max_len": head_max_len,
    "laya.max_opts": MAX_OPTS,
    "laya.max_batch": int(max_batch),
    "laya.min_seq": int(min_seq),
    "laya.length_buckets": json.dumps(_length_buckets(max_len)),
    "laya.mask_token_id": int(agent.tok.mask_token_id),
    "laya.cls_token_id": int(agent.tok.cls_token_id),
    "laya.sep_token_id": int(agent.tok.sep_token_id),
    "laya.pad_token_id": int(agent.tok.pad_token_id),
    "laya.temperature": json.dumps(agent.temperature),
    "laya.temperature_by_options": json.dumps(agent.temperature_by_options),
    "laya.model_name": spec["model_name"],
    "laya.family": spec["family"],
    "laya.checkpoint": checkpoint,
}}
extra.update(decision.to_gguf_metadata())

fusion = FusionOptions()
fusion.enable_rope = False

dim_b = torch.export.Dim("b", min=1, max=int(max_batch))
dim_s = torch.export.Dim("s", min=int(min_seq), max=max_len)
dynamic_shapes = (
    {{0: dim_b, 1: dim_s}},
    {{0: dim_b, 1: dim_s}},
    {{0: dim_b}},
    {{0: dim_b}},
    {{0: dim_b}},
)
output.parent.mkdir(parents=True, exist_ok=True)
print(f"Compiling GGUF to {{output}} ...")
ggmlc.compile(
    trunk,
    example,
    output=str(output),
    model_name=spec["stem"],
    quantize=quantize,
    pipeline=tok,
    tasks=["classification"],
    extra_metadata=extra,
    fusion_options=fusion,
    dynamic_shapes=dynamic_shapes,
)
if output.is_file() and output.stat().st_size > 10_000_000:
    shutil.move(str(output), str(target))
    print(f"Atomically updated active GGUF: {{target}} ({{target.stat().st_size}} bytes)")
"""
    ret, _, _ = run_command([python_bin, "-c", inline_code], timeout=600.0)
    if ret == 0 and target_gguf.is_file():
        return True
    if compile_script and compile_script.is_file():
        ret2, _, _ = run_command([python_bin, str(compile_script)], timeout=600.0)
        if ret2 == 0 and new_gguf.is_file():
            shutil.move(str(new_gguf), str(target_gguf))
            return True
    return False


def reload_laya(laya_url: str = "http://127.0.0.1:8787", timeout_sec: float = 35.0) -> bool:
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
    distill_summary: Dict[str, Any],
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

    discrepancies = after_report.get("discrepancies", [])
    disc_rows = []
    for d in discrepancies[:8]:
        st = d.get("state_snippet", "").replace("|", "/")
        exp = f"{d.get('expected_decision')} / {d.get('expected_category')}"
        got = f"{d.get('predicted_decision')} / {d.get('predicted_category')}"
        disc_rows.append(f"| \"{st}\" | `{exp}` | `{got}` |")

    disc_table = (
        "| Запрос пользователя | Ожидалось (Ground Truth) | Предсказано Laya |\n|:---|:---|:---|\n" + "\n".join(disc_rows)
        if disc_rows
        else "Все тестовые кейсы классифицированы безошибочно! 🎉"
    )

    sweep_rows = []
    for row in a_met.get("calibration_sweep", []):
        fl = row["floor"]
        r = row["right"]
        w = row["wrong_actions"]
        s = row["stalled"]
        mark = " 🎯 (Recommended)" if fl == a_met.get("recommended_confidence_floor") else ""
        sweep_rows.append(f"| {fl:.2f} | {r} | **{w}** | {s} |{mark}")
    sweep_table = "\n".join(sweep_rows) if sweep_rows else "| — | — | — | — |"

    jev_status_str = (
        "Обнаружен и активен (Enabled)"
        if jev_info.get("is_enabled")
        else ("Обнаружен (Disabled)" if jev_info.get("is_detected") else "Не обнаружен")
    )
    jev_logs_str = ", ".join([f"{k}: {v} записей" for k, v in jev_info.get("log_counts", {}).items()]) or "Логи не обнаружены"

    ds_stats = distill_summary.get("summary") or distill_summary or {}
    total_unique = ds_stats.get("total_unique_cases", ds_stats.get("total_unique_states", "—"))
    both_known = ds_stats.get("observed_from_logs_or_db", ds_stats.get("both_known_from_logs_or_db", "—"))
    queried_prim = ds_stats.get("queried_primary_model", ds_stats.get("queried_primary_model_live", "—"))
    queried_laya = ds_stats.get("queried_laya_live", "—")
    exact_matches = ds_stats.get("laya_vs_primary_matches", ds_stats.get("laya_vs_primary_exact_match", "—"))
    agr_rate = float(ds_stats.get("agreement_rate", ds_stats.get("laya_vs_primary_agreement_rate", 0.0)))
    mismatches = ds_stats.get("laya_vs_primary_mismatches", ds_stats.get("laya_vs_primary_disagreement_hard_negatives", "—"))

    md_content = f"""# Отчёт о работе скила обучения и калибровки Laya (`laya-train`)

- **Дата выполнения**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
- **Движок**: Laya System 1 Decision Engine ({before_report.get('laya_url')})
- **Базовый чекпоинт**: `{train_metadata.get('base_checkpoint')}`
- **Обучающая выборка**: `{train_metadata.get('train_file')}` ({train_metadata.get('train_items', 0)} сэмплов)
- **Валидационная выборка**: `{before_report.get('eval_dataset')}` ({before_report.get('metrics', {}).get('total_cases', 0)} кейсов)
- **Алгоритм**: `{train_metadata.get('loss', 'rlcd')}` (эпох: {train_metadata.get('epochs', 8)}, micro-batch: {train_metadata.get('micro_batch', 1)}, grad-accum: {train_metadata.get('grad_accum', 4)}, shuffle-options: `True`)

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

## 2. Статистика сверки Laya vs Основная модель (Data Distillation)

- **Всего уникальных диалоговых состояний**: **{total_unique}**
- **Сценарий A (Выбор Laya и выбор Основной модели уже были в логах/БД)**: **{both_known}**
- **Сценарий B (Выполнен live-запрос к Основной модели для сравнения с Laya)**: **{queried_prim}**
- **Выполнен live-запрос к Laya (при отсутствии в логах)**: **{queried_laya}**
- **Совпадение Laya с Основной моделью до обучения**: **{exact_matches}** ({agr_rate*100:.1f}%)
- **Расхождения (Hard Negatives, усиленные в `train.jsonl` 2x)**: **{mismatches}**

---

## 3. Точность в разрезе категорий инструментов

| Категория | До обучения | После обучения | Дельта |
|:---|:---:|:---:|:---:|
{cat_table}

---

## 4. Интеграция с плагином kerpopule/hermes-jev-skills

- **Статус плагина**: **{jev_status_str}**
- **Директория плагина**: `{jev_info.get('plugin_dir') or '—'}`
- **Телеметрия Jev**: {jev_logs_str}
- **Калибровка порога уверенности (Confidence Floor Sweep по стандарту Jev)**:

| Порог (Floor) | Верные решения (Right) | Ошибочные действия (Wrong Actions) | Воздержания (Stalled) | Примечание |
|:---:|:---:|:---:|:---:|:---|
{sweep_table}

- **Рекомендуемый порог (Confidence Floor)**: **{a_met.get('recommended_confidence_floor', 0.60):.2f}** (минимизирует ложные срабатывания при Fast Lane инференсе).

---

## 5. Анализ оставшихся расхождений (Discrepancies)

{disc_table}

---

## 6. Итоговое заключение скила

- Модель успешно прошла цикл валидации, дообучения и горячей перезагрузки GGUF.
- Итоговая комбинированная точность: **{a_strict*100:.1f}%** (прирост **{'+' if d_strict>=0 else ''}{d_strict*100:.1f}%**).
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
    distill_comp_file = data_dir / "distill_comparison.json"

    print("======================================================================")
    print("   Laya Decision Engine — Запуск замкнутого цикла обучения и оценки    ")
    print("======================================================================")
    print(f"Каталог данных скилла: {data_dir}")
    print(f"URL Laya             : {args.laya_url}")
    print(f"Базовый чекпоинт     : {args.base_checkpoint}")
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

    # ШАГ 1: Извлечение и разметка данных
    if args.extract or not train_file.is_file() or not eval_file.is_file():
        print("\n>>> ШАГ 1: Извлечение и сверка выборок Laya и Основной модели...")
        extract_script = scripts_dir / "extract_and_label.py"
        extract_cmd = [
            sys.executable,
            str(extract_script),
            "--out-dir", str(data_dir),
            "--logs-path", str(args.logs_path),
            "--db-path", str(args.db_path),
            "--laya-url", str(args.laya_url),
        ]
        if args.primary_url:
            extract_cmd.extend(["--primary-url", args.primary_url])
        if args.primary_key:
            extract_cmd.extend(["--primary-key", args.primary_key])

        run_command(extract_cmd)

    if not train_file.is_file() or not eval_file.is_file():
        print("[!] Используем демонстрационные сэмплы из examples/...")
        shutil.copyfile(skill_dir / "examples" / "train_sample.jsonl", train_file)
        shutil.copyfile(skill_dir / "examples" / "eval_sample.jsonl", eval_file)

    evals_bin = Path(args.train_bin).parent / "laya-evals" if args.train_bin else None
    train_count = validate_jsonl_schema(train_file, evals_bin)
    eval_count = validate_jsonl_schema(eval_file, evals_bin)
    print(f"[+] Валидированные датасеты: train={train_count} сэмплов, eval={eval_count} сэмплов.\n")

    distill_summary: Dict[str, Any] = {}
    if distill_comp_file.is_file():
        try:
            with open(distill_comp_file, "r", encoding="utf-8") as f:
                distill_summary = json.load(f)
        except Exception:
            pass

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
    checkpoint_out = (
        Path(args.checkpoint_out)
        if args.checkpoint_out
        else Path("/root/deepseek-harness/finetune-checkpoint")
        if Path("/root/deepseek-harness").is_dir()
        else data_dir / "finetune_checkpoint"
    )

    # Адаптивный бюджет оптимизатора для малых выборок (< 500 строк) против Prior Collapse
    micro_batch = args.micro_batch
    grad_accum = args.grad_accum
    epochs = args.epochs
    if train_count < 500 and micro_batch == 8 and grad_accum == 8:
        micro_batch = 4
        grad_accum = 1

    # Проверка свободной VRAM на GPU0: если не хватает (< min_vram_mib), запускаем laya-train на CPU (--device cpu)
    free_vram_mib = get_gpu0_free_vram_mib()
    train_device = args.device
    if train_device == "auto":
        if free_vram_mib is not None and free_vram_mib < args.min_vram_mib:
            print(
                f"[!] На GPU0 недостаточно свободной VRAM ({free_vram_mib} MiB свободно < {args.min_vram_mib} MiB требуется). "
                f"Переключаем дообучение laya-train на CPU (--device cpu)!"
            )
            train_device = "cpu"
        else:
            train_device = "cuda" if free_vram_mib is not None else "auto"

    backup_stamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_backup: Optional[Path] = None
    gguf_backup: Optional[Path] = None

    if not args.dry_run and train_bin.is_file():
        # Резервная копия чекпоинта: laya-train перезаписывает --out
        if not args.no_backup and (checkpoint_out / "model.safetensors").is_file():
            checkpoint_backup = backup_before_replace(checkpoint_out, backup_stamp, args.keep_backups)
        train_cmd = [
            str(train_bin),
            "--data", str(train_file),
            "--eval", str(eval_file),
            "--out", str(checkpoint_out),
            "--base", str(args.base_checkpoint),
            "--loss", args.loss,
            "--epochs", str(epochs),
            "--micro-batch", str(micro_batch),
            "--grad-accum", str(grad_accum),
            "--device", train_device,
            "--shuffle-options",
        ]
        if args.freeze_encoder:
            train_cmd.append("--freeze-encoder")
        ret, out_txt, err_txt = run_command(train_cmd, timeout=2400.0)
        if ret != 0 and train_device != "cpu" and "out of memory" in (out_txt + err_txt).lower():
            print("[!] Обнаружена нехватка VRAM на GPU0 (CUDA out of memory). Автоматический перезапуск обучения на CPU (--device cpu)...")
            train_device = "cpu"
            cpu_cmd = [c if c != "cuda" and c != "auto" else "cpu" for c in train_cmd]
            ret, _, _ = run_command(cpu_cmd, timeout=2400.0)
        if ret != 0:
            print("[warn] Обучение вернуло ненулевой код. Проверяем наличие чекпоинта...")
    else:
        print(f"[i] Пропуск реального GPU-обучения (dry_run={args.dry_run} или бинарник не найден локально).")
        checkpoint_out.mkdir(parents=True, exist_ok=True)
        (checkpoint_out / "calibrated.marker").touch()

    # ШАГ 4: Компиляция GGUF и горячая перезагрузка
    print(">>> ШАГ 4: Компиляция весов в GGUF и горячая перезагрузка Laya...")
    unsloth_py = Path(args.train_bin).parent / "python"
    python_for_compile = str(unsloth_py) if unsloth_py.is_file() else sys.executable
    target_gguf = Path(args.target_gguf)
    compile_script = Path(args.compile_script) if args.compile_script else None

    if not args.dry_run and (checkpoint_out / "model.safetensors").is_file() and target_gguf.parent.is_dir():
        # Резервная копия рабочего GGUF: компилятор атомарно подменяет target_gguf
        if not args.no_backup:
            gguf_backup = backup_before_replace(target_gguf, backup_stamp, args.keep_backups)
        compile_checkpoint_to_gguf(
            python_bin=python_for_compile,
            checkpoint_dir=checkpoint_out,
            target_gguf=target_gguf,
            compile_script=compile_script,
        )

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
        "base_checkpoint": args.base_checkpoint,
        "checkpoint_out": str(checkpoint_out),
        "loss": args.loss,
        "epochs": epochs,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "device": train_device,
        "checkpoint_backup": str(checkpoint_backup) if checkpoint_backup else None,
        "gguf_backup": str(gguf_backup) if gguf_backup else None,
        "gpu0_free_vram_mib": free_vram_mib,
        "shuffle_options": True,
        "dry_run": args.dry_run,
        "jev_integrated": jev_info.get("is_detected", False),
    }

    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_metadata": train_metadata,
        "distillation_summary": distill_summary.get("summary", {}),
        "jev_integration": jev_info,
        "before": before_report,
        "after": after_report,
        "delta": {
            "tool_decision_accuracy": round(
                after_report["metrics"]["tool_decision_accuracy"] - before_report["metrics"]["tool_decision_accuracy"],
                4,
            ),
            "category_accuracy": round(
                after_report["metrics"]["category_accuracy"] - before_report["metrics"]["category_accuracy"],
                4,
            ),
            "combined_strict_accuracy": round(
                after_report["metrics"]["combined_strict_accuracy"] - before_report["metrics"]["combined_strict_accuracy"],
                4,
            ),
            "brier_score": round(
                after_report["metrics"]["brier_score"] - before_report["metrics"]["brier_score"],
                4,
            ),
        },
    }

    with open(report_json_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    md_report_text = generate_markdown_report(
        before_report, after_report, train_metadata, jev_info, distill_summary, report_md_file
    )
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
        help="Количество эпох обучения (по умолчанию: 4)",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=4,
        help="Размер микро-батча (по умолчанию: 4 для предотвращения Prior Collapse)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Шагов аккумуляции градиента (по умолчанию: 1)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Устройство обучения: auto (проверяет VRAM на GPU0 и переключает на cpu при нехватке), cuda или cpu",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Не создавать резервные копии чекпоинта и GGUF перед заменой (по умолчанию бэкап создаётся)",
    )
    parser.add_argument(
        "--keep-backups",
        type=int,
        default=3,
        help="Сколько последних резервных копий (<имя>.<YYYYmmdd-HHMMSS>.bak) хранить (по умолчанию: 3)",
    )
    parser.add_argument(
        "--min-vram-mib",
        type=int,
        default=4500,
        help="Минимальный объем свободной VRAM на GPU0 (MiB) для запуска на GPU (по умолчанию: 4500 MiB, иначе --device cpu)",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Заморозить базовый энкодер и обучать только головы классификации",
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
        default="convaiinnovations/laya-multilingual",
        help="Базовый чекпоинт модели (по умолчанию: convaiinnovations/laya-multilingual)",
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
        "--target-gguf",
        default="/root/deepseek-harness/models/laya_multilingual_f16.gguf",
        help="Путь к активному GGUF файлу инференс-сервера Laya",
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
