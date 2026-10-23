#!/usr/bin/env python3
"""extract_and_label.py — Сбор и экспертная разметка данных для обучения Laya.

Извлекает реальные диалоговые траектории из сессий Hermes (`state.db`) и
аудит-логов плагина (`laya_requests.jsonl`).

Если в сессии отсутствует явный вызов инструмента или выбор неоднозначен,
скрипт автоматически запрашивает эталонное суждение у основной модели Hermes
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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    timeout_sec: float = 20.0,
) -> Optional[Dict[str, str]]:
    """Запрос к основной модели Hermes (Teacher LLM) для получения эталонной разметки."""
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    categories_list = "\n".join(f"- {k}: {v}" for k, v in CATEGORY_CRITERIA.items())

    system_prompt = (
        "You are an expert AI Router and Decision Supervisor for an autonomous agent.\n"
        "Your task is to analyze user requests and determine the optimal routing decisions:\n"
        "1. 'tool_decision': must be either 'execute_tools' (if the user request requires running shell commands, "
        "editing/reading files, web searches, computer control, marketplace lookups, or executing external tools) "
        "or 'direct_chat' (if this is a purely conversational greeting, text discussion, explanation, or general knowledge without tools).\n"
        "2. 'category': if tools are required, pick the single best category from the available list below. If direct_chat, category should be null or 'general_utility'.\n\n"
        f"Available categories:\n{categories_list}\n\n"
        "Respond ONLY with a valid JSON object with keys 'tool_decision', 'category', and 'reasoning'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User request:\n{state_text}"},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 200,
    }

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            # Clean markdown codeblocks if model wraps output
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)
            decision = parsed.get("tool_decision", "direct_chat")
            category = parsed.get("category")
            if decision not in ("execute_tools", "direct_chat"):
                decision = "direct_chat"
            if category and category not in CATEGORY_CRITERIA:
                category = "general_utility"
            return {
                "tool_decision": decision,
                "category": category or "general_utility",
                "reasoning": str(parsed.get("reasoning", "")),
                "source": f"primary_model:{model}",
            }
    except Exception as exc:
        print(f"  [warn] Primary model query failed ({exc}), falling back to local heuristic.")
        return None


def extract_from_logs(log_path: Path) -> List[Dict[str, Any]]:
    """Извлечь примеры из логов laya_requests.jsonl."""
    if not log_path.is_file():
        return []

    records: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            # Игнорировать телеметрию исполнения и ошибки
            if item.get("kind") == "tool_telemetry" or item.get("error"):
                continue

            state = str(item.get("state_snippet") or "").strip()
            if not state or len(state) < 3:
                continue

            # Очистка преамбулы USER:
            if state.startswith("USER:"):
                state = state[5:].strip()

            decision = item.get("decision")
            category = item.get("winning_category") or item.get("category")

            # Если решение зафиксировано с высокой вероятностью или было подтверждено исполнением
            if decision in ("direct_chat", "execute_tools"):
                records.append({
                    "state": state,
                    "tool_decision": decision,
                    "category": category or ("general_utility" if decision == "execute_tools" else "filesystem"),
                    "source": "laya_requests_log",
                    "probability": item.get("probability", 1.0),
                })

    return records


def extract_from_sqlite(db_path: Path) -> List[Dict[str, Any]]:
    """Извлечь диалоги и вызовы инструментов из Hermes state.db."""
    if not db_path.is_file():
        return []

    records: List[Dict[str, Any]] = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Выбираем пары User -> Assistant с вызовами инструментов
        query = """
            SELECT m1.content AS user_prompt,
                   m2.content AS assistant_response,
                   m2.tool_calls,
                   m2.tool_name
            FROM messages m1
            JOIN messages m2 ON m1.session_id = m2.session_id AND m2.id > m1.id
            WHERE m1.role = 'user'
              AND m2.role = 'assistant'
              AND length(m1.content) > 3
            ORDER BY m1.id ASC
            LIMIT 500
        """
        cur.execute(query)
        rows = cur.fetchall()

        seen_states = set()
        for row in rows:
            user_prompt = str(row["user_prompt"] or "").strip()
            # Пропускаем системные промпты и дубликаты
            if not user_prompt or user_prompt in seen_states or user_prompt.startswith("/"):
                continue
            seen_states.add(user_prompt)

            tool_calls = row["tool_calls"]
            tool_name = row["tool_name"]

            executed_tools = []
            if tool_calls:
                try:
                    tc_parsed = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
                    if isinstance(tc_parsed, list):
                        for tc in tc_parsed:
                            fn_name = tc.get("function", {}).get("name") or tc.get("name")
                            if fn_name:
                                executed_tools.append(fn_name)
                except Exception:
                    pass

            if tool_name and tool_name not in executed_tools:
                executed_tools.append(tool_name)

            if executed_tools:
                first_tool = executed_tools[0]
                cat = infer_category_from_tool_name(first_tool)
                records.append({
                    "state": user_prompt,
                    "tool_decision": "execute_tools",
                    "category": cat,
                    "executed_tools": executed_tools,
                    "source": "hermes_state_tool_exec",
                })
            else:
                # Нет вызова инструментов: кандидат на direct_chat или требует оценки основной модели
                records.append({
                    "state": user_prompt,
                    "tool_decision": None,  # Требует валидации через Primary Model
                    "category": None,
                    "source": "hermes_state_no_tool",
                })

        conn.close()
    except Exception as exc:
        print(f"  [warn] SQLite extraction warning: {exc}")

    return records


def format_laya_training_item(state: str, tool_decision: str, category: str) -> Dict[str, Any]:
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
    }


def main() -> None:
    skill_dir = Path(__file__).resolve().parent.parent
    default_data_dir = skill_dir / "data"

    parser = argparse.ArgumentParser(
        description="Laya Skill: Извлечение и разметка обучающих данных из сессий Hermes и логов плагина."
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
        "--out-dir",
        default=str(default_data_dir),
        help="Каталог для сохранения сформированных файлов train.jsonl и eval.jsonl (по умолчанию: skills/laya-train/data)",
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
        default=300,
        help="Максимальное количество извлекаемых сэмплов",
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [Laya Train Skill] Сбор и разметка данных ===")
    print(f"Логи: {args.logs_path}")
    print(f"База данных: {args.db_path}")
    print(f"Целевой каталог: {out_dir}\n")

    items_pool: List[Dict[str, Any]] = []
    seen_prompts = set()

    # 1. Извлечение из логов плагина
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
        print(f"[-] Файл логов {log_file} не найден локально (будут использованы сессии или тестовые образцы).")

    # 2. Извлечение из базы сессий Hermes
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

    # 3. Разметка с привлечением основной модели (Teacher Judge)
    labeled_dataset: List[Dict[str, Any]] = []
    queried_primary_count = 0

    for idx, item in enumerate(items_pool[: args.max_samples]):
        state = item["state"]
        tool_decision = item.get("tool_decision")
        category = item.get("category")

        # Если эталон отсутствует или не полон — запрашиваем основную модель
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
            else:
                # Локальный эвристический fallback
                tool_decision = "direct_chat"
                category = "general_utility"

        labeled_item = format_laya_training_item(state, tool_decision, category)
        labeled_dataset.append(labeled_item)

    print(f"[+] Разметка завершена: {len(labeled_dataset)} образцов (запросов к основной модели: {queried_primary_count}).")

    if not labeled_dataset:
        print("[-] Датасет пуст. Создание демонстрационных эталонных сэмплов для калибровки...")
        sample_file = skill_dir / "examples" / "train_sample.jsonl"
        if sample_file.is_file():
            with open(sample_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        labeled_dataset.append(json.loads(line))

    # 4. Разделение на train и eval
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
