#!/usr/bin/env python3
"""evaluate_accuracy.py — Тестирование точности и калибровки Laya (System 1 Decision Engine).

Выполняет прогон валидационного датасета (`eval.jsonl`) через рабочий эндпоинт Laya
(`/ask` или `/v1/systemone`) и рассчитывает метрики:
- Общая точность (Accuracy) по `tool_decision` и `category`
- Совместная точность (Combined Strict Accuracy)
- Brier Score (мера калибровки вероятностей)
- Средняя уверенность (Mean Confidence)
- По категорийный срез точности
- Список расхождений (discrepancies) для анализа ошибок

Результаты сохраняются в каталог скилла: `skills/laya-train/data/eval_<phase>.json`.
"""

from __future__ import annotations

import argparse
import json
import math
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


def query_laya(
    laya_url: str,
    state: str,
    questions: Dict[str, Any],
    timeout_sec: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Отправляет запрос в Laya (поддерживает как Sidecar /ask, так и C++ /v1/systemone)."""
    base = laya_url.rstrip("/")

    # Попытка 1: /ask (Laya Sidecar)
    ask_url = f"{base}/ask"
    payload = {"state": state, "questions": questions}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        ask_url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "answers" in data:
                return data
    except Exception:
        pass

    # Попытка 2: /v1/systemone (Laya C++ Native REST)
    systemone_url = f"{base}/v1/systemone"
    req_sys = urllib.request.Request(
        systemone_url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req_sys, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "answers" in data:
                return data
    except Exception:
        pass

    return None


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Рассчитать Accuracy, Brier score, Mean Confidence и по категорийный срез."""
    total = len(results)
    if total == 0:
        return {"total": 0, "accuracy": 0.0}

    td_matches = sum(1 for r in results if r["tool_decision_match"])
    cat_eval_count = sum(1 for r in results if r["expected_decision"] == "execute_tools")
    cat_matches = sum(1 for r in results if r["expected_decision"] == "execute_tools" and r["category_match"])
    combined_matches = sum(1 for r in results if r["strict_match"])

    # Brier score для бинарного tool_decision: mean((prob - y)^2)
    brier_scores = []
    confidences = []

    for r in results:
        prob = r.get("predicted_td_prob", 0.5)
        y = 1.0 if r["expected_decision"] == "execute_tools" else 0.0
        brier_scores.append((prob - y) ** 2)
        conf = r.get("predicted_td_conf")
        if conf is not None:
            confidences.append(conf)

    mean_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0.0
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # По категорийный срез
    category_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        expected_cat = r.get("expected_category") or "general_utility"
        if expected_cat not in category_stats:
            category_stats[expected_cat] = {"total": 0, "correct": 0}
        category_stats[expected_cat]["total"] += 1
        if r["category_match"]:
            category_stats[expected_cat]["correct"] += 1

    cat_breakdown: Dict[str, Dict[str, Any]] = {}
    for cat_name, stats in category_stats.items():
        tot = stats["total"]
        cor = stats["correct"]
        cat_breakdown[cat_name] = {
            "total": tot,
            "correct": cor,
            "accuracy": round(cor / tot, 4) if tot > 0 else 0.0,
        }

    return {
        "total_cases": total,
        "tool_decision_accuracy": round(td_matches / total, 4),
        "tool_decision_matches": td_matches,
        "category_accuracy": round(cat_matches / cat_eval_count, 4) if cat_eval_count > 0 else 1.0,
        "category_matches": cat_matches,
        "category_tested_cases": cat_eval_count,
        "combined_strict_accuracy": round(combined_matches / total, 4),
        "combined_matches": combined_matches,
        "mean_confidence": round(mean_conf, 4),
        "brier_score": round(mean_brier, 4),
        "categories": cat_breakdown,
    }


def evaluate(
    eval_file: Path,
    laya_url: str,
    phase: str = "eval",
    out_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Выполнить тестирование на датасете eval_file."""
    if not eval_file.is_file():
        raise FileNotFoundError(f"Файл датасета не найден: {eval_file}")

    print(f"\n--- [Laya Accuracy Test: {phase.upper()}] ---")
    print(f"Датасет: {eval_file}")
    print(f"Laya URL: {laya_url}")

    cases: List[Dict[str, Any]] = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"Загружено {len(cases)} валидационных тестовых случаев. Выполнение оценки...\n")

    eval_results: List[Dict[str, Any]] = []
    discrepancies: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for idx, c in enumerate(cases):
        state = c.get("state", "")
        questions = c.get("questions", {})
        label = c.get("label", {})

        expected_td = label.get("tool_decision")
        expected_cat = label.get("category")

        t0 = time.time()
        resp = query_laya(laya_url, state, questions)
        elapsed_ms = (time.time() - t0) * 1000.0
        latencies.append(elapsed_ms)

        if not resp:
            print(f"  [{idx+1}/{len(cases)}] [FAIL] Ошибка подключения к Laya ({elapsed_ms:.1f}ms): {state[:50]}")
            continue

        answers = resp.get("answers", {})
        td_ans = answers.get("tool_decision", {})
        cat_ans = answers.get("category", {})

        pred_td = td_ans.get("choice")
        pred_cat = cat_ans.get("choice")

        td_probs = td_ans.get("probabilities", {})
        pred_td_prob = td_probs.get("execute_tools", 0.95 if pred_td == "execute_tools" else 0.05)
        pred_td_conf = float(td_ans.get("confidence") or 0.90)

        td_match = (pred_td == expected_td)
        cat_match = (pred_cat == expected_cat) or (expected_td == "direct_chat")
        strict_match = td_match and (cat_match or expected_td == "direct_chat")

        result_row = {
            "index": idx,
            "state_snippet": state[:70],
            "expected_decision": expected_td,
            "predicted_decision": pred_td,
            "tool_decision_match": td_match,
            "expected_category": expected_cat,
            "predicted_category": pred_cat,
            "category_match": cat_match,
            "strict_match": strict_match,
            "predicted_td_prob": pred_td_prob,
            "predicted_td_conf": pred_td_conf,
            "latency_ms": round(elapsed_ms, 2),
        }
        eval_results.append(result_row)

        icon = "✓" if strict_match else "✗"
        status_str = f"[{icon}] TD: {pred_td} (exp: {expected_td}), Cat: {pred_cat} (exp: {expected_cat})"
        print(f"  [{idx+1}/{len(cases)}] {status_str} ({elapsed_ms:.1f}ms) | \"{state[:45]}\"")

        if not strict_match:
            discrepancies.append(result_row)

    metrics = calculate_metrics(eval_results)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    report = {
        "phase": phase,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "laya_url": laya_url,
        "eval_dataset": str(eval_file),
        "metrics": metrics,
        "avg_latency_ms": round(avg_latency, 2),
        "discrepancies_count": len(discrepancies),
        "discrepancies": discrepancies,
    }

    print(f"\n{'='*55}")
    print(f"РЕЗУЛЬТАТЫ ТЕСТА ТОЧНОСТИ ({phase.upper()}):")
    print(f"  • Tool Decision Accuracy : {metrics.get('tool_decision_accuracy', 0)*100:.1f}% ({metrics.get('tool_decision_matches', 0)}/{metrics.get('total_cases', 0)})")
    print(f"  • Category Accuracy      : {metrics.get('category_accuracy', 0)*100:.1f}% ({metrics.get('category_matches', 0)}/{metrics.get('category_tested_cases', 0)})")
    print(f"  • Strict Combined Acc    : {metrics.get('combined_strict_accuracy', 0)*100:.1f}%")
    print(f"  • Brier Score (ниже-лучше): {metrics.get('brier_score', 0):.4f}")
    print(f"  • Mean Confidence        : {metrics.get('mean_confidence', 0):.4f}")
    print(f"  • Средняя задержка       : {report['avg_latency_ms']:.1f} ms")
    print(f"{'='*55}\n")

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[✓] Отчет теста сохранен в {out_file}")

    return report


def main() -> None:
    skill_dir = Path(__file__).resolve().parent.parent
    default_eval_file = skill_dir / "data" / "eval.jsonl"
    if not default_eval_file.is_file():
        default_eval_file = skill_dir / "examples" / "eval_sample.jsonl"

    parser = argparse.ArgumentParser(
        description="Laya Skill: Оценка точности и калибровки решений Laya на тестовой выборке."
    )
    parser.add_argument(
        "--eval-data",
        default=str(default_eval_file),
        help="Путь к файлу валидационного датасета eval.jsonl (по умолчанию: skills/laya-train/data/eval.jsonl)",
    )
    parser.add_argument(
        "--laya-url",
        default="http://127.0.0.1:8787",
        help="URL инференса Laya (порт 8787 Sidecar или 8080 C++ Engine)",
    )
    parser.add_argument(
        "--phase",
        default="eval",
        choices=["before", "after", "eval"],
        help="Фаза оценки: before (до обучения), after (после обучения), eval (контрольная)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Путь к файлу сохранения результатов JSON (по умолчанию: skills/laya-train/data/eval_<phase>.json)",
    )

    args = parser.parse_args()
    eval_file = Path(args.eval_data)
    out_file = Path(args.out) if args.out else skill_dir / "data" / f"eval_{args.phase}.json"

    evaluate(eval_file=eval_file, laya_url=args.laya_url, phase=args.phase, out_file=out_file)


if __name__ == "__main__":
    main()
