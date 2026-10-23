#!/usr/bin/env python3
"""extract_and_label.py — Сбор и экспертная разметка данных для обучения Laya.

Извлекает реальные диалоговые траектории из:
1. Аудит-логов плагина Laya (`laya_requests.jsonl`);
2. Базы данных сессий Hermes (`state.db`);
3. Логов решений и теневых прогонов плагина `kerpopule/hermes-jev-skills` (`jev-decisions.jsonl`, `jev-shadow.jsonl`).

Если в сессиях/логах отсутствует явный вызов инструмента или выбор неоднозначен,
скрипт автоматически запрашивает эталонное решение (Ground Truth) у основной модели Hermes
(Primary LLM Teacher) через OpenAI-совместимый эндпоинт.

Выходные файлы сохраняются в каталог скилла: `skills/laya-train/data/`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
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

# Импорт детектора Jev
try:
    from .jev_detector import detect_jev_plugin
except ImportError:
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from jev_detector import detect_jev_plugin

# Стандартные категории Laya
CATEGORY_CRITERIA: Dict[str, str] = {
    "marketplace_and_commerce": "Online marketplaces, shopping, e-commerce, product search, item cards, reviews, orders, sellers, and prices (Ozon, Wildberries, Yandex Market, Avito, AliExpress, Taobao, Lamoda, Detmir, Drom, Citilink, DNS)",
    "desktop_and_computer_use": "Desktop GUI interaction, native OS control, computer use, window management, screen inspection, and workspace preview",
    "filesystem": "File and directory operations (reading, writing, editing, listing, searching files, directory trees, and patching)",
    "terminal_execution": "Executing shell, bash, powershell commands, terminal scripts, command-line processes, and code execution",
    "web_and_search": "Internet web search, web page extraction, downloading pages, HTTP APIs, and online retrieval",
    "browser_automation": "Browser UI automation, web page navigation, clicking, typing, web snapshots, and DOM inspection",
    "system_and_tasks": "Background process management, scheduled cron jobs, secure vault storage, session search, and task tracking",
    "meta_and_skills": "Skills management, subagent delegation, memory storage, task coordination, and decision models",
    "multimedia_and_vision": "Image/video analysis, speech synthesis (TTS), audio, visual recognition, and generative media",
    "code_and_git": "Source code inspection, git version control, repository commits, AST and code analysis",
    "devops_containers": "Docker containers, system administration, services, infrastructure, and Kubernetes",
    "external_messaging": "Sending notifications or messages to external services (Telegram, Discord, Slack, Feishu) or interactive confirmation dialogs",
    "general_utility": "Calculation, formatting, time, data transformations, and general utilities",
}

# Маппинг популярных инструментов в категории
TOOL_TO_CATEGORY: Dict[str, str] = {
    "read_file": "filesystem",
    "write_file": "filesystem",
    "edit_file": "filesystem",
    "view_file": "filesystem",
    "list_dir": "filesystem",
    "search_files": "filesystem",
    "grep_search": "filesystem",
    "patch": "filesystem",
    "patch_file": "filesystem",
    "terminal": "terminal_execution",
    "execute_code": "terminal_execution",
    "bash": "terminal_execution",
    "exec": "terminal_execution",
    "web_search": "web_and_search",
    "web_extract": "web_and_search",
    "fetch_web_page": "web_and_search",
    "browser_exec": "browser_automation",
    "browser_navigate": "browser_automation",
    "browser_click": "browser_automation",
    "computer_use": "desktop_and_computer_use",
    "desktop_preview": "desktop_and_computer_use",
    "focus_pane": "desktop_and_computer_use",
    "cronjob_manage": "system_and_tasks",
    "browser_vault": "system_and_tasks",
    "process_manage": "system_and_tasks",
    "todo_list": "system_and_tasks",
    "delegate_task": "meta_and_skills",
    "memory": "meta_and_skills",
    "clarify": "meta_and_skills",
    "skill_manage": "meta_and_skills",
    "skill_view": "meta_and_skills",
    "skills_list": "meta_and_skills",
    "vision_analyze": "multimedia_and_vision",
    "video_analyze": "multimedia_and_vision",
    "text_to_speech": "multimedia_and_vision",
    "git_status": "code_and_git",
    "git_commit": "code_and_git",
    "git_diff": "code_and_git",
    "git_log": "code_and_git",
}


def infer_category_from_tool_name(tool_name: str) -> str:
    """Определить категорию по имени инструмента (включая MCP marketplace)."""
    t = tool_name.lower()
    if t in TOOL_TO_CATEGORY:
        return TOOL_TO_CATEGORY[t]

    if "marketplace" in t or any(m in t for m in ("ozon", "wb", "wildberries", "yandex", "avito", "aliexpress", "drom", "citilink", "dns", "cian", "mpstats")):
        return "marketplace_and_commerce"
    if "computer" in t or "desktop" in t or "mouse" in t or "keyboard" in t or "window" in t:
        return "desktop_and_computer_use"
    if "browser" in t:
        return "browser_automation"
    if "git" in t:
        return "code_and_git"
    if "docker" in t or "container" in t:
        return "devops_containers"
    if any(k in t for k in ("read", "write", "file", "dir", "grep", "search")):
        return "filesystem"
    if any(k in t for k in ("term", "bash", "sh", "exec", "cmd")):
        return "terminal_execution"
    if any(k in t for k in ("web", "url", "http", "fetch")):
        return "web_and_search"
    if any(k in t for k in ("telegram", "discord", "slack", "notify", "msg")):
        return "external_messaging"
    if any(k in t for k in ("skill", "agent", "memory", "task")):
        return "meta_and_skills"
    if any(k in t for k in ("image", "vision", "video", "voice", "audio", "tts")):
        return "multimedia_and_vision"

    return "general_utility"


def query_primary_model(
    state_text: str,
    base_url: str = "http://127.0.0.1:20128/v1",
    api_key: str = "",
    model: str = "andrei-router",
    timeout: int = 25,
) -> Optional[Dict[str, str]]:
    """Запрос к основной модели Hermes (Teacher Judge) для разметки эталона при неопределенности."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    prompt_categories = "\n".join([f"- {k}: {v}" for k, v in CATEGORY_CRITERIA.items()])

    system_prompt = (
        "You are an expert AI decision judge for the Hermes Agent System 1 Router (Laya Engine).\n"
        "Your task is to analyze the user request and determine ground truth decisions:\n"
        "1. tool_decision: 'execute_tools' (if shell, file, web, coding, computer or tools are needed) "
        "OR 'direct_chat' (if pure conversational answer, math, reasoning, or text explanation).\n"
        "2. category: pick precisely one primary tool category from the valid list below:\n"
        f"{prompt_categories}\n\n"
        "Respond ONLY with a valid JSON object strictly matching this schema:\n"
        '{"tool_decision": "execute_tools" | "direct_chat", "category": "<valid_category_name>", "reasoning": "<short_explanation>"}'
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Request: {state_text}"},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "Bearer none",
            "User-Agent": "Laya-Teacher-Judge/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            td = parsed.get("tool_decision")
            cat = parsed.get("category")
            if td in ("execute_tools", "direct_chat") and cat in CATEGORY_CRITERIA:
                return {"tool_decision": td, "category": cat}
    except Exception:
        pass

    return None


def extract_from_logs(log_path: Path) -> List[Dict[str, Any]]:
    """Извлечь записи из аудит-лога laya_requests.jsonl."""
    records: List[Dict[str, Any]] = []
    if not log_path.is_file():
        return records

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    raw_state = (
                        entry.get("state")
                        or entry.get("state_snippet")
                        or entry.get("request", {}).get("state")
                    )
                    if not raw_state:
                        continue

                    state = raw_state.strip()
                    if state.startswith("USER:"):
                        state = state[5:].strip()

                    if len(state) < 2:
                        continue

                    # Решения
                    decision = entry.get("decision")
                    winning_cat = entry.get("winning_category")
                    raw_answers = entry.get("raw_answers", {})
                    if not decision and "tool_decision" in raw_answers:
                        decision = raw_answers.get("tool_decision", {}).get("choice")
                    if not winning_cat and "category" in raw_answers:
                        winning_cat = raw_answers.get("category", {}).get("choice")

                    actual_exec = entry.get("actual_executed_tools", [])

                    tool_decision: Optional[str] = None
                    category: Optional[str] = None

                    if actual_exec:
                        tool_decision = "execute_tools"
                        category = infer_category_from_tool_name(actual_exec[0])
                    elif decision in ("execute_tools", "direct_chat"):
                        tool_decision = decision
                        category = winning_cat or "general_utility"

                    records.append({
                        "state": state,
                        "tool_decision": tool_decision,
                        "category": category,
                        "source": "laya_requests_log",
                        "actual_executed_tools": actual_exec,
                    })
                except Exception:
                    continue
    except Exception as exc:
        print(f"  [warn] Ошибка чтения {log_path}: {exc}")

    return records


def extract_from_sqlite(db_path: Path) -> List[Dict[str, Any]]:
    """Извлечь сессии и фактические вызовы инструментов из SQLite Hermes state.db."""
    records: List[Dict[str, Any]] = []
    if not db_path.is_file():
        return records

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages';")
        if not cur.fetchone():
            conn.close()
            return records

        # В схеме Hermes колонка времени называется `timestamp` (REAL)
        query = """
            SELECT m.session_id, m.role, m.content, m.tool_calls, m.tool_name, m.timestamp
            FROM messages m
            WHERE m.role = 'user'
            ORDER BY m.timestamp DESC
            LIMIT 400;
        """
        user_messages = cur.execute(query).fetchall()

        for u_row in user_messages:
            user_prompt = (u_row["content"] or "").strip()
            if not user_prompt or len(user_prompt) < 3:
                continue

            sess_id = u_row["session_id"]
            ts = u_row["timestamp"]

            # Ищем последующие действия агента в рамках этой сессии
            action_query = """
                SELECT role, content, tool_calls, tool_name
                FROM messages
                WHERE session_id = ? AND timestamp >= ? AND role IN ('assistant', 'tool')
                ORDER BY timestamp ASC
                LIMIT 5;
            """
            agent_actions = cur.execute(action_query, (sess_id, ts)).fetchall()

            executed_tools: List[str] = []
            for act in agent_actions:
                tc = act["tool_calls"]
                tn = act["tool_name"]
                if tc:
                    try:
                        tc_parsed = json.loads(tc) if isinstance(tc, str) else tc
                        if isinstance(tc_parsed, list):
                            for call in tc_parsed:
                                fn = call.get("function", {}).get("name") or call.get("name")
                                if fn and fn not in executed_tools:
                                    executed_tools.append(fn)
                    except Exception:
                        pass
                if tn and tn not in executed_tools:
                    executed_tools.append(tn)

            if executed_tools:
                cat = infer_category_from_tool_name(executed_tools[0])
                records.append({
                    "state": user_prompt,
                    "tool_decision": "execute_tools",
                    "category": cat,
                    "executed_tools": executed_tools,
                    "source": "hermes_state_tool_exec",
                    "timestamp": ts,
                    "session_id": sess_id,
                })
            else:
                records.append({
                    "state": user_prompt,
                    "tool_decision": None,
                    "category": None,
                    "source": "hermes_state_no_tool",
                    "timestamp": ts,
                    "session_id": sess_id,
                })

        conn.close()
    except Exception as exc:
        print(f"  [warn] Ошибка чтения базы SQLite {db_path}: {exc}")

    return records


def extract_from_jev_logs(
    jev_log_path: Path,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Извлечь данные решений плагина kerpopule/hermes-jev-skills и сопоставить с сессиями Hermes."""
    records: List[Dict[str, Any]] = []
    if not jev_log_path.is_file():
        return records

    user_timeline: List[Tuple[float, str, str]] = []
    if db_path and db_path.is_file():
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT timestamp, session_id, content FROM messages WHERE role = 'user' AND content IS NOT NULL ORDER BY timestamp ASC;"
            ).fetchall()
            for ts_raw, sid, content in rows:
                if not content:
                    continue
                try:
                    ts_val = float(ts_raw)
                    if ts_val > 0:
                        user_timeline.append((ts_val, sid, content.strip()))
                except Exception:
                    pass
            conn.close()
        except Exception:
            pass

    try:
        with open(jev_log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = float(entry.get("ts", 0.0))
                    kind = entry.get("kind", "")
                    mode = entry.get("mode", "")
                    reason = entry.get("reason", "")
                    model = entry.get("model", "")
                    tier = entry.get("tier")

                    matched_prompt: Optional[str] = None
                    if user_timeline and ts > 0:
                        # Ищем сообщение пользователя в окне +-25 сек от вызова Jev
                        candidates = [
                            (abs(ts - u_ts), prompt)
                            for u_ts, sid, prompt in user_timeline
                            if abs(ts - u_ts) <= 25.0
                        ]
                        if candidates:
                            candidates.sort(key=lambda x: x[0])
                            matched_prompt = candidates[0][1]

                    if not matched_prompt:
                        continue

                    tool_decision: Optional[str] = None
                    category: Optional[str] = None

                    r_low = reason.lower() if reason else ""
                    if any(w in r_low for w in ("code", "terminal", "command", "bash", "shell")):
                        tool_decision = "execute_tools"
                        category = "terminal_execution"
                    elif any(w in r_low for w in ("file", "read", "write", "patch", "dir")):
                        tool_decision = "execute_tools"
                        category = "filesystem"
                    elif any(w in r_low for w in ("search", "web", "tavily", "extract")):
                        tool_decision = "execute_tools"
                        category = "web_and_search"
                    elif any(w in r_low for w in ("marketplace", "ozon", "wb", "avito")):
                        tool_decision = "execute_tools"
                        category = "marketplace_and_commerce"
                    elif tier == "cheap" or "simple conversational" in r_low:
                        tool_decision = "direct_chat"
                        category = "general_utility"

                    records.append({
                        "state": matched_prompt,
                        "tool_decision": tool_decision,
                        "category": category,
                        "source": f"hermes_jev_{kind}",
                        "jev_metadata": {
                            "kind": kind,
                            "mode": mode,
                            "tier": tier,
                            "reason": reason,
                            "model": model,
                        },
                    })
                except Exception:
                    continue
    except Exception as exc:
        print(f"  [warn] Ошибка чтения логов Jev {jev_log_path}: {exc}")

    return records


def format_laya_training_item(state: str, tool_decision: str, category: str, source: str = "hermes") -> Dict[str, Any]:
    """Сформировать обучающий образец в стандартном формате вопросов Laya."""
    return {
        "state": state,
        "questions": {
            "tool_decision": {
                "type": "choice",
                "instructions": (
                    "Determine whether external tools or terminal execution are required "
                    "to fulfill the user request, or if direct conversational response is appropriate."
                ),
                "criteria": {
                    "direct_chat": "Conversational reply, reasoning, code explanation without tool execution",
                    "execute_tools": "Executing shell, reading or writing files, web queries, or interacting with tools",
                },
            },
            "category": {
                "type": "choice",
                "instructions": "Select the primary category of tool required for this task.",
                "criteria": CATEGORY_CRITERIA,
            },
        },
        "label": {
            "tool_decision": tool_decision,
            "category": category,
        },
        "metadata": {
            "source": source,
        },
    }


def main() -> None:
    skill_dir = Path(__file__).resolve().parent.parent
    default_data_dir = skill_dir / "data"

    parser = argparse.ArgumentParser(
        description="Laya Skill: Извлечение и разметка обучающих данных из сессий Hermes, логов плагина и kerpopule/hermes-jev-skills."
    )
    parser.add_argument(
        "--logs-path",
        default=str(Path.home() / ".hermes" / "logs" / "laya_requests.jsonl"),
        help="Путь к файлу логов laya_requests.jsonl",
    )
    parser.add_argument(
        "--db-path",
        default=str(Path.home() / ".hermes" / "state.db"),
        help="Путь к файлу базы данных Hermes state.db",
    )
    parser.add_argument(
        "--check-jev",
        action="store_true",
        default=True,
        help="Проверять наличие плагина kerpopule/hermes-jev-skills и извлекать его логи (по умолчанию: True)",
    )
    parser.add_argument(
        "--jev-logs-path",
        default=str(Path.home() / ".hermes" / "logs" / "jev-decisions.jsonl"),
        help="Путь к логам решений Jev (jev-decisions.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(default_data_dir),
        help="Каталог для сохранения train.jsonl и eval.jsonl (по умолчанию: skills/laya-train/data)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.20,
        help="Доля выборки для валидационного набора eval.jsonl (по умолчанию: 0.20)",
    )
    parser.add_argument(
        "--primary-url",
        default=os.environ.get("HERMES_OMNI_URL", "http://127.0.0.1:20128/v1"),
        help="URL эндпоинта основной модели для разметки (Teacher Judge)",
    )
    parser.add_argument(
        "--primary-key",
        default=os.environ.get("HERMES_CUSTOM_OMNI_API_KEY", ""),
        help="API ключ основной модели",
    )
    parser.add_argument(
        "--primary-model",
        default=os.environ.get("HERMES_MODEL", "andrei-router"),
        help="Имя основной модели для разметки",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=400,
        help="Максимальное количество извлекаемых сэмплов",
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [Laya Train Skill] Сбор и разметка данных ===")
    print(f"Логи Laya: {args.logs_path}")
    print(f"База Hermes: {args.db_path}")
    print(f"Каталог сохранения: {out_dir}\n")

    items_pool: List[Dict[str, Any]] = []
    seen_prompts = set()

    # 1. Проверка плагина kerpopule/hermes-jev-skills
    jev_info = detect_jev_plugin(Path(args.db_path).parent if args.db_path else None)
    print(f"[*] Проверка плагина kerpopule/hermes-jev-skills:")
    print(f"    - Обнаружен: {'ДА' if jev_info['is_detected'] else 'НЕТ'}")
    print(f"    - Статус: {'ВКЛЮЧЕН (ENABLED)' if jev_info['is_enabled'] else 'ВЫКЛЮЧЕН / НЕ УСТАНОВЛЕН'}")

    # 2. Если плагин установлен, извлекаем логи Jev
    if jev_info["is_detected"] and args.check_jev:
        jev_log_file = Path(args.jev_logs_path)
        if not jev_log_file.is_file() and jev_info.get("logs", {}).get("decisions"):
            jev_log_file = Path(jev_info["logs"]["decisions"])

        if jev_log_file.is_file():
            jev_items = extract_from_jev_logs(jev_log_file, Path(args.db_path))
            print(f"[+] Из логов Jev ({jev_log_file.name}) извлечено {len(jev_items)} сопоставленных записей.")
            for item in jev_items:
                st = item["state"]
                if st not in seen_prompts:
                    seen_prompts.add(st)
                    items_pool.append(item)

    # 3. Извлечение из логов плагина Laya
    log_file = Path(args.logs_path)
    if log_file.is_file():
        log_items = extract_from_logs(log_file)
        print(f"[+] Из логов laya_requests.jsonl извлечено {len(log_items)} записей.")
        for item in log_items:
            st = item["state"]
            if st not in seen_prompts:
                seen_prompts.add(st)
                items_pool.append(item)
    else:
        print(f"[-] Файл логов {log_file} не найден.")

    # 4. Извлечение из базы сессий Hermes
    db_file = Path(args.db_path)
    if db_file.is_file():
        db_items = extract_from_sqlite(db_file)
        print(f"[+] Из базы сессий state.db извлечено {len(db_items)} записей.")
        for item in db_items:
            st = item["state"]
            if st not in seen_prompts:
                seen_prompts.add(st)
                items_pool.append(item)

    print(f"\n[+] Всего уникальных обращений для анализа: {len(items_pool)}")

    # 5. Экспертная разметка с обращением к основной модели (Teacher Judge)
    labeled_dataset: List[Dict[str, Any]] = []
    queried_primary_count = 0

    for idx, item in enumerate(items_pool[: args.max_samples]):
        state = item["state"]
        tool_decision = item.get("tool_decision")
        category = item.get("category")
        source = item.get("source", "hermes_session")

        # Если эталон отсутствует или неоднозначен — запрашиваем основную модель (Teacher LLM)
        if not tool_decision or not category:
            teacher_res = query_primary_model(
                state_text=state,
                base_url=args.primary_url,
                api_key=args.primary_key,
                model=args.primary_model,
            )
            if teacher_res:
                tool_decision = teacher_res["tool_decision"]
                category = teacher_res["category"]
                queried_primary_count += 1
                source += "+teacher_llm_judge"
            else:
                tool_decision = "direct_chat"
                category = "general_utility"
                source += "+fallback_default"

        labeled_item = format_laya_training_item(state, tool_decision, category, source=source)
        labeled_dataset.append(labeled_item)

    print(f"[+] Разметка завершена: {len(labeled_dataset)} образцов (запросов к основной модели: {queried_primary_count}).")

    if not labeled_dataset:
        print("[-] Датасет пуст. Использование демонстрационных сэмплов из examples/...")
        sample_file = skill_dir / "examples" / "train_sample.jsonl"
        if sample_file.is_file():
            with open(sample_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        labeled_dataset.append(json.loads(line))

    # 6. Разделение на train и eval
    eval_count = max(2, int(len(labeled_dataset) * args.eval_ratio))
    train_data = labeled_dataset[:-eval_count] if len(labeled_dataset) > eval_count else labeled_dataset
    eval_data = labeled_dataset[-eval_count:] if len(labeled_dataset) > eval_count else labeled_dataset[:2]

    train_file = out_dir / "train.jsonl"
    eval_file = out_dir / "eval.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(eval_file, "w", encoding="utf-8") as f:
        for r in eval_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[✓] Готово! Данные сохранены в каталог навыка {out_dir}:")
    print(f"    - train.jsonl: {len(train_data)} записей -> {train_file}")
    print(f"    - eval.jsonl:  {len(eval_data)} записей -> {eval_file}")


if __name__ == "__main__":
    main()
