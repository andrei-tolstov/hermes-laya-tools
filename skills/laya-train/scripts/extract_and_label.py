#!/usr/bin/env python3
"""extract_and_label.py — Сбор, сравнение выбора Laya с основной моделью и разметка данных для дообучения Laya.

Извлекает реальные диалоговые траектории из:
1. Аудит-логов плагина Laya (`laya_requests.jsonl`) — включая связку предложений Laya с телеметрией вызовов (`tool_telemetry`);
2. Базы данных сессий Hermes (`state.db`) — фактические вызовы инструментов основной моделью;
3. Логов решений плагина `kerpopule/hermes-jev-skills` (`jev-decisions.jsonl`, `jev-shadow.jsonl`).

Логика формирования обучающей выборки (Laya vs Primary Model):
- Если ЕСТЬ информация, что предлагает Laya, и что использует основная модель
  (из `tool_telemetry` в `laya_requests.jsonl` или фактических `tool_calls` в `state.db`),
  то Laya дообучается непосредственно на выборке основной модели (сравнение фиксируется в `metadata`,
  а расхождения маркируются как `hard_negative`).
- Если информации НЕТ (нет данных о том, что использовала основная модель, или нет предложения Laya),
  то выполняется прямой запрос к основной модели Hermes (`query_primary_model`) и/или к Laya (`query_laya_choice`)
  для сравнения с выбором Laya и формирования эталона `expected`.

Выходные файлы сохраняются в каталог скилла: `skills/laya-train/data/`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Стандартные категории Laya (синхронизированы с categorizer.py и router.py)
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

# Эталонная схема вопросов (совпадает с router.py в hermes-laya-tools)
STANDARD_QUESTIONS: Dict[str, Any] = {
    "tool_decision": {
        "type": "choice",
        "instructions": "Determine whether this request requires tool/command execution or is a pure conversational response.",
        "criteria": {
            "execute_tools": "Requires executing tools, running terminal commands, reading or editing files, searching web, or external actions",
            "direct_chat": "Pure conversational chat, greeting, text explanation, or answering general knowledge without any tools",
        },
    },
    "category": {
        "type": "choice",
        "instructions": "If tools are required, which tool category best fits this request?",
        "criteria": CATEGORY_CRITERIA,
    },
}

# Маппинг инструментов в категории
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
    "read_terminal": "terminal_execution",
    "close_terminal": "terminal_execution",
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
    "desktop_project": "desktop_and_computer_use",
    "drive_preview": "desktop_and_computer_use",
    "annotate_preview": "desktop_and_computer_use",
    "focus_pane": "desktop_and_computer_use",
    "read_window_below": "desktop_and_computer_use",
    "set_theme": "desktop_and_computer_use",
    "show_tip": "desktop_and_computer_use",
    "gui_tour": "desktop_and_computer_use",
    "setup_beat": "desktop_and_computer_use",
    "cronjob_manage": "system_and_tasks",
    "browser_vault": "system_and_tasks",
    "process_manage": "system_and_tasks",
    "session_search": "system_and_tasks",
    "todo_list": "system_and_tasks",
    "manage_connections": "system_and_tasks",
    "manage_catalog": "meta_and_skills",
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

_WORD_SPLIT = re.compile(r"[^\w]+", re.UNICODE)
_ACK_WORDS: Set[str] = {
    "ок", "окей", "да", "ага", "угу", "хорошо", "понял", "понятно", "ясно",
    "спасибо", "спс", "благодарю", "привет", "здравствуйте", "хай", "добрый",
    "день", "вечер", "утро", "отлично", "ладно", "супер", "принято", "точно",
    "согласен", "верно", "кто", "ты", "как", "дела",
    "ok", "okay", "yes", "yeah", "yep", "sure", "thanks", "thx", "thank",
    "you", "got", "it", "understood", "hello", "hi", "hey", "good",
    "morning", "evening", "great", "fine", "acknowledged", "agreed",
}
_ACTION_WORDS: Set[str] = {
    "запусти", "выполни", "сделай", "удали", "создай", "открой", "покажи",
    "найди", "проверь", "напиши", "скачай", "исправь", "перезапусти", "клонируй",
    "настрой", "установи", "прочитай", "сохрани", "сравни", "продолжай",
    "run", "exec", "execute", "start", "stop", "delete", "remove", "create",
    "open", "show", "find", "search", "check", "write", "download", "fix",
    "git", "bash", "curl", "python", "cat", "grep", "ls", "cd", "mkdir",
}
_NOISE_EXACT: Set[str] = {
    "jev", "exit", "test", "test jev 2", "quit", "/jev", "/laya",
    "продолжай", "продолжай\n\nпродолжай", "выполняй", "да выполняй",
    "пароль не нужен, выполняй план",
}


def load_hermes_env(hermes_home: Optional[Path] = None) -> Dict[str, str]:
    """Загрузить переменные окружения из ~/.hermes/.env и локального .env."""
    home = hermes_home or (Path.home() / ".hermes")
    candidates = [
        home / ".env",
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    loaded: Dict[str, str] = {}
    for env_file in candidates:
        if env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and v:
                        loaded[k] = v
                        if k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass
    return loaded


def looks_trivial_greeting(text: str) -> bool:
    """Проверить, является ли запрос коротким приветствием или подтверждением без вызова тулов."""
    if not text:
        return False
    words = [w.lower() for w in _WORD_SPLIT.split(text.strip()) if w]
    if not words or len(words) > 6:
        return False
    if any(w in _ACTION_WORDS for w in words):
        return False
    return all(w in _ACK_WORDS for w in words)


def clean_user_state(raw_text: Optional[str]) -> Optional[str]:
    """Очистить пользовательский запрос от системных префиксов, инъекций и дампов вложений."""
    if not raw_text or not isinstance(raw_text, str):
        return None

    text = raw_text.strip()

    # Отбрасываем строки внутренней телеметрии тулов
    if text.startswith("executed=") and "hit=" in text:
        return None

    # Убираем ролевые префиксы
    for prefix in ("USER:", "SYSTEM:", "ASSISTANT:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Убираем блок --- Attached Context --- (огромные дампы страниц)
    if "--- Attached Context ---" in text:
        text = text.split("--- Attached Context ---", 1)[0].strip()

    # Убираем системные уведомления [System: ...]
    text = re.sub(r"\[System:[^\]]*\]", "", text).strip()

    # Отбрасываем чисто системные инъекции фоновых процессов и автозагрузки скилов
    if text.startswith("[IMPORTANT: Background process") or text.startswith("[IMPORTANT: The user has invoked"):
        return None

    # Очищаем преамбулы [/plan ...] и служебные инструкции режима планирования
    if text.startswith("[/plan") or "For this turn, you are in PLAN MODE" in text or "Interaction style:" in text:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        kept = [
            p for p in parts
            if not p.startswith(("[/plan", "-", "*", "For this turn, you are in PLAN MODE", "Interaction style:"))
        ]
        text = kept[-1] if kept else ""

    text = text.strip()
    if len(text) < 2:
        return None
    if text.lower() in _NOISE_EXACT:
        return None
    if text.startswith(("For this turn, you are in PLAN MODE", "Interaction style:")):
        return None

    return text[:1000]


def infer_category_from_tool_name(tool_name: str) -> str:
    """Определить категорию по имени инструмента (включая MCP marketplace)."""
    t = tool_name.lower().strip()
    if t in TOOL_TO_CATEGORY:
        return TOOL_TO_CATEGORY[t]

    if "marketplace" in t or any(
        m in t for m in ("ozon", "wb", "wildberries", "yandex", "avito", "aliexpress", "drom", "citilink", "dns", "cian", "mpstats")
    ):
        return "marketplace_and_commerce"
    if any(k in t for k in ("computer", "desktop", "mouse", "keyboard", "window", "pane", "preview")):
        return "desktop_and_computer_use"
    if "browser" in t:
        return "browser_automation"
    if "git" in t:
        return "code_and_git"
    if "docker" in t or "container" in t:
        return "devops_containers"
    if any(k in t for k in ("read", "write", "file", "dir", "grep", "search_files", "patch")):
        return "filesystem"
    if any(k in t for k in ("term", "bash", "sh", "exec", "cmd")):
        return "terminal_execution"
    if any(k in t for k in ("web", "url", "http", "fetch")):
        return "web_and_search"
    if any(k in t for k in ("telegram", "discord", "slack", "notify", "msg")):
        return "external_messaging"
    if any(k in t for k in ("cron", "process", "todo", "session", "vault", "connection")):
        return "system_and_tasks"
    if any(k in t for k in ("skill", "agent", "memory", "delegate", "clarify", "jev_")):
        return "meta_and_skills"
    if any(k in t for k in ("image", "vision", "video", "voice", "audio", "tts")):
        return "multimedia_and_vision"

    return "general_utility"


def infer_primary_category(executed_tools: List[str]) -> str:
    """Выбрать основную категорию по списку фактически вызванных основной моделью инструментов."""
    if not executed_tools:
        return "general_utility"
    # Фильтруем вспомогательные мета-инструменты поиска тулов, если после них был вызван целевой инструмент
    bridge_names = {"tool_search", "tool_describe", "tool_call", "request_toolset", "skills_list", "skill_view"}
    target_tools = [t for t in executed_tools if t not in bridge_names]
    active_list = target_tools if target_tools else executed_tools
    cats = [infer_category_from_tool_name(t) for t in active_list]
    # Приоритет первой целевой категории или наиболее частой
    counts = Counter(cats)
    return counts.most_common(1)[0][0]


def query_laya_choice(
    laya_url: str,
    state_text: str,
    questions: Optional[Dict[str, Any]] = None,
    timeout: float = 8.0,
) -> Optional[Dict[str, Any]]:
    """Запросить выбор Laya для заданного state (через /ask или /v1/systemone)."""
    base = laya_url.rstrip("/")
    q_payload = questions or STANDARD_QUESTIONS
    body = json.dumps({"state": state_text, "questions": q_payload}, ensure_ascii=False).encode("utf-8")

    for path in ("/ask", "/v1/systemone"):
        req = urllib.request.Request(
            f"{base}{path}",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                answers = data.get("answers", {})
                if answers:
                    td_ans = answers.get("tool_decision", {})
                    cat_ans = answers.get("category", {})
                    td_probs = td_ans.get("probabilities", {})
                    cat_probs = cat_ans.get("probabilities", {})
                    td_choice = td_ans.get("choice")
                    cat_choice = cat_ans.get("choice")
                    td_conf = max(td_probs.values()) if td_probs else float(td_ans.get("answer_confidence") or td_ans.get("confidence") or 0.5)
                    cat_conf = max(cat_probs.values()) if cat_probs else float(cat_ans.get("answer_confidence") or cat_ans.get("confidence") or 0.5)
                    if td_choice in ("execute_tools", "direct_chat"):
                        return {
                            "tool_decision": td_choice,
                            "category": cat_choice or "general_utility",
                            "confidence": round(float(td_conf), 4),
                            "category_confidence": round(float(cat_conf), 4),
                            "source": "laya_live_query",
                        }
        except Exception:
            continue
    return None


def _parse_json_from_llm(content: str) -> Optional[Dict[str, Any]]:
    """Извлечь JSON-объект из текстового ответа LLM (включая markdown-блоки и блоки <think>)."""
    if not content:
        return None
    text = content.strip()
    # Удаляем возможный блок размышлений <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            text = match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None


def query_primary_model(
    state_text: str,
    laya_choice: Optional[Dict[str, Any]] = None,
    base_url: str = "http://127.0.0.1:20128/v1",
    api_key: str = "",
    model: str = "andrei-router",
    reasoning_effort: str = "xhigh",
    timeout: int = 90,
) -> Optional[Dict[str, Any]]:
    """Запрос к основной модели Hermes (Primary Model Teacher) с enable_thinking=True и reasoning_effort='xhigh'."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    prompt_categories = "\n".join([f"- {k}: {v}" for k, v in CATEGORY_CRITERIA.items()])

    laya_context = ""
    if laya_choice and laya_choice.get("tool_decision"):
        laya_context = (
            f"\nLaya System 1 proposed: tool_decision='{laya_choice.get('tool_decision')}', "
            f"category='{laya_choice.get('category')}' (prob={laya_choice.get('confidence', 0.0):.2f}).\n"
            "Verify whether Laya's proposal is right or wrong, and provide the authoritative ground truth."
        )

    system_prompt = (
        "You are the Primary Model Teacher for the Hermes Agent System 1 Router (Laya Engine).\n"
        "Analyze the user request and determine the authoritative ground truth routing decision:\n"
        "1. tool_decision:\n"
        "   - 'execute_tools': if fulfilling the request requires any tools, terminal/shell commands, "
        "reading/writing files, searching marketplaces (Ozon, WB, Avito, etc.), web search, browser, desktop GUI, skills, or system actions.\n"
        "   - 'direct_chat': ONLY if it is a pure greeting, simple acknowledgement, creative text generation, or general explanation requiring zero external tools.\n"
        "2. category: pick EXACTLY ONE category key from this list:\n"
        f"{prompt_categories}\n"
        "(If tool_decision is 'direct_chat', set category to 'general_utility'.)\n\n"
        "Respond ONLY with a valid JSON object matching this exact schema:\n"
        '{"tool_decision": "execute_tools" | "direct_chat", "category": "<valid_category_key>", "reasoning": "<short_reason>"}'
    )

    user_content = f"User Request: {state_text}{laya_context}"

    # Включаем глубокое размышление (enable_thinking=True) и максимальный уровень мышления (reasoning_effort="xhigh").
    # Не передаем response_format={"type": "json_object"}, т.к. провайдер omni/coder390 возвращает 502.
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "reasoning_effort": reasoning_effort,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    req_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}" if api_key else "Bearer none",
            "User-Agent": "Laya-Primary-Teacher/2.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            parsed = _parse_json_from_llm(content)
            if parsed:
                td = parsed.get("tool_decision")
                cat = parsed.get("category")
                # Нормализация близких имен категорий, если модель сократила название
                if cat not in CATEGORY_CRITERIA and isinstance(cat, str):
                    cat = infer_category_from_tool_name(cat)
                if td in ("execute_tools", "direct_chat") and cat in CATEGORY_CRITERIA:
                    return {
                        "tool_decision": td,
                        "category": cat if td == "execute_tools" else "general_utility",
                        "reasoning": str(parsed.get("reasoning", ""))[:200],
                        "source": "primary_model_queried",
                    }
    except Exception as exc:
        print(f"  [warn] Ошибка запроса к основной модели ({url}): {exc}")

    return None


def extract_from_logs(log_path: Path) -> List[Dict[str, Any]]:
    """Извлечь предложения Laya и связать их с последующими записями tool_telemetry из laya_requests.jsonl."""
    if not log_path.is_file():
        return []

    routing_entries: List[Dict[str, Any]] = []
    by_turn: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_session_last: Dict[str, Dict[str, Any]] = {}

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                raw_answers = entry.get("raw_answers") or {}
                decision = str(entry.get("decision") or "")
                session_id = str(entry.get("session_id") or "")
                turn_id = str(entry.get("turn_id") or "")

                # 1. Запись телеметрии фактического вызова инструмента основной моделью (_on_post_tool_call)
                if raw_answers.get("kind") == "tool_telemetry" or decision.startswith("tool_exec:"):
                    tool_name = raw_answers.get("tool_name")
                    if not tool_name:
                        avail = entry.get("available_tools") or []
                        if avail:
                            tool_name = avail[0]
                    hit = raw_answers.get("hit")
                    if hit is None:
                        hit = decision == "tool_exec:hit"

                    target_req = by_turn.get((session_id, turn_id)) or by_session_last.get(session_id)
                    if target_req is not None and tool_name:
                        target_req["telemetry_tools"].append(tool_name)
                        target_req["telemetry_hits"].append(bool(hit))
                    continue

                # 2. Запись маршрутизации Laya (_route_hierarchical / _route_flat)
                raw_state = (
                    entry.get("state")
                    or entry.get("state_snippet")
                    or (entry.get("request") or {}).get("state")
                )
                state = clean_user_state(raw_state)
                if not state:
                    continue

                td_ans = raw_answers.get("tool_decision") or {}
                cat_ans = raw_answers.get("category") or {}

                laya_td = td_ans.get("choice") or (decision if decision in ("execute_tools", "direct_chat") else None)
                laya_cat = cat_ans.get("choice") or entry.get("winning_category")

                td_probs = td_ans.get("probabilities") or {}
                cat_probs = cat_ans.get("probabilities") or {}
                laya_conf = (
                    max(td_probs.values())
                    if td_probs
                    else float(entry.get("probability") or td_ans.get("confidence") or 0.5)
                )
                laya_cat_conf = (
                    max(cat_probs.values())
                    if cat_probs
                    else float(entry.get("probability") or cat_ans.get("confidence") or 0.5)
                )

                req_record: Dict[str, Any] = {
                    "state": state,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "timestamp": entry.get("timestamp"),
                    "laya_choice": {
                        "tool_decision": laya_td,
                        "category": laya_cat or "general_utility",
                        "confidence": round(float(laya_conf), 4),
                        "category_confidence": round(float(laya_cat_conf), 4),
                        "selected_tools": entry.get("selected_tools") or [],
                        "source": "laya_requests_log",
                    } if laya_td in ("execute_tools", "direct_chat") else None,
                    "telemetry_tools": list(entry.get("actual_executed_tools") or []),
                    "telemetry_hits": [],
                }
                routing_entries.append(req_record)
                if session_id and turn_id:
                    by_turn[(session_id, turn_id)] = req_record
                if session_id:
                    by_session_last[session_id] = req_record

    except Exception as exc:
        print(f"  [warn] Ошибка чтения {log_path}: {exc}")

    records: List[Dict[str, Any]] = []
    for req in routing_entries:
        telemetry_tools = req["telemetry_tools"]
        telemetry_hits = req["telemetry_hits"]
        primary_choice: Optional[Dict[str, Any]] = None

        # Если есть телеметрия вызовов основной модели — это прямое наблюдение выбора основной модели
        if telemetry_tools:
            primary_choice = {
                "tool_decision": "execute_tools",
                "category": infer_primary_category(telemetry_tools),
                "executed_tools": telemetry_tools,
                "telemetry_hit": all(telemetry_hits) if telemetry_hits else True,
                "source": "primary_observed_telemetry",
            }

        records.append({
            "state": req["state"],
            "session_id": req["session_id"],
            "turn_id": req["turn_id"],
            "laya_choice": req["laya_choice"],
            "primary_choice": primary_choice,
            "source": "laya_requests_log",
        })

    return records


def extract_from_sqlite(db_path: Path) -> List[Dict[str, Any]]:
    """Извлечь сессии и фактические вызовы инструментов основной модели из SQLite Hermes state.db."""
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

        rows = cur.execute(
            """
            SELECT id, session_id, role, content, tool_calls, tool_name, timestamp
            FROM messages
            ORDER BY session_id, timestamp ASC, id ASC;
            """
        ).fetchall()
        conn.close()

        sessions: Dict[str, List[sqlite3.Row]] = {}
        for r in rows:
            sessions.setdefault(str(r["session_id"]), []).append(r)

        for sess_id, msgs in sessions.items():
            for idx, m in enumerate(msgs):
                if m["role"] != "user":
                    continue
                state = clean_user_state(m["content"])
                if not state:
                    continue

                executed_tools: List[str] = []
                has_assistant_reply = False

                for next_m in msgs[idx + 1 :]:
                    if next_m["role"] == "user":
                        break
                    if next_m["role"] == "assistant":
                        if (next_m["content"] or "").strip():
                            has_assistant_reply = True
                        tc = next_m["tool_calls"]
                        if tc:
                            try:
                                tc_parsed = json.loads(tc) if isinstance(tc, str) else tc
                                if isinstance(tc_parsed, list):
                                    for call in tc_parsed:
                                        fn = (call.get("function") or {}).get("name") or call.get("name")
                                        if fn and fn not in executed_tools:
                                            executed_tools.append(fn)
                            except Exception:
                                pass
                    elif next_m["role"] == "tool":
                        tn = next_m["tool_name"]
                        if tn and tn not in executed_tools:
                            executed_tools.append(tn)

                primary_choice: Optional[Dict[str, Any]] = None
                if executed_tools:
                    primary_choice = {
                        "tool_decision": "execute_tools",
                        "category": infer_primary_category(executed_tools),
                        "executed_tools": executed_tools,
                        "source": "primary_observed_state_db",
                    }
                elif has_assistant_reply and looks_trivial_greeting(state):
                    primary_choice = {
                        "tool_decision": "direct_chat",
                        "category": "general_utility",
                        "executed_tools": [],
                        "source": "primary_observed_trivial_chat",
                    }

                records.append({
                    "state": state,
                    "session_id": sess_id,
                    "timestamp": m["timestamp"],
                    "laya_choice": None,
                    "primary_choice": primary_choice,
                    "source": "hermes_state_db",
                })

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
                cleaned = clean_user_state(content)
                if not cleaned:
                    continue
                try:
                    ts_val = float(ts_raw)
                    if ts_val > 0:
                        user_timeline.append((ts_val, str(sid), cleaned))
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
                    matched_sid: str = ""
                    if user_timeline and ts > 0:
                        candidates = [
                            (abs(ts - u_ts), sid, prompt)
                            for u_ts, sid, prompt in user_timeline
                            if abs(ts - u_ts) <= 25.0
                        ]
                        if candidates:
                            candidates.sort(key=lambda x: x[0])
                            matched_sid = candidates[0][1]
                            matched_prompt = candidates[0][2]

                    if not matched_prompt:
                        continue

                    records.append({
                        "state": matched_prompt,
                        "session_id": matched_sid,
                        "laya_choice": None,
                        "primary_choice": None,
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


def format_laya_training_item(
    state: str,
    primary_choice: Dict[str, Any],
    laya_choice: Optional[Dict[str, Any]] = None,
    match: Optional[bool] = None,
    is_hard_negative: bool = False,
    jev_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Сформировать обучающий образец в строгом формате Laya (`expected` + `label`)."""
    td = primary_choice["tool_decision"]
    cat = primary_choice.get("category") or "general_utility"

    # Для laya-train и laya-evals validate:
    # Если direct_chat, обучаем голову tool_decision (не засоряя голову category искусственным general_utility).
    # Если execute_tools, обучаем обе головы: tool_decision и category.
    expected: Dict[str, Any] = {"tool_decision": td}
    if td == "execute_tools":
        expected["category"] = cat

    label: Dict[str, Any] = {
        "tool_decision": td,
        "category": cat,
    }

    tags = [f"decision:{td}", f"cat:{cat}", f"source:{primary_choice.get('source', 'primary')}"]
    if is_hard_negative:
        tags.append("priority:hard_negative")

    metadata: Dict[str, Any] = {
        "source": primary_choice.get("source", "primary"),
        "primary_model": primary_choice,
        "laya_proposal": laya_choice,
        "match": match,
        "hard_negative": is_hard_negative,
    }
    if jev_metadata:
        metadata["jev"] = jev_metadata

    return {
        "state": state,
        "questions": STANDARD_QUESTIONS,
        "expected": expected,
        "label": label,
        "tags": tags,
        "metadata": metadata,
    }


def main() -> None:
    skill_dir = Path(__file__).resolve().parent.parent
    default_data_dir = skill_dir / "data"

    # Загружаем переменные из ~/.hermes/.env до инициализации аргументов CLI
    load_hermes_env()

    parser = argparse.ArgumentParser(
        description="Laya Skill: Сбор, сравнение выбора Laya с основной моделью и разметка датасета."
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
        help="Проверять наличие плагина kerpopule/hermes-jev-skills и извлекать его логи",
    )
    parser.add_argument(
        "--jev-logs-path",
        default=str(Path.home() / ".hermes" / "logs" / "jev-decisions.jsonl"),
        help="Путь к логам решений Jev (jev-decisions.jsonl)",
    )
    parser.add_argument(
        "--laya-url",
        default=os.environ.get("LAYA_SIDECAR_URL", "http://127.0.0.1:8787"),
        help="URL эндпоинта Laya для получения выбора Laya, если его нет в логах",
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
        "--oversample-hard-negatives",
        type=int,
        default=1,
        help="Кратность дублирования примеров, где Laya ошиблась (по умолчанию: 1, без дублирования для чистоты split_calibration в laya-train)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=400,
        help="Максимальное количество уникальных сэмплов",
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Повторно подгружаем .env из домашней папки Hermes, если передан кастомный --db-path
    if args.db_path:
        load_hermes_env(Path(args.db_path).parent)
        if not args.primary_key:
            args.primary_key = os.environ.get("HERMES_CUSTOM_OMNI_API_KEY", "")

    print("=== [Laya Train Skill] Сбор и сравнение решений: Laya vs Основная модель ===")
    print(f"Логи Laya        : {args.logs_path}")
    print(f"База Hermes      : {args.db_path}")
    print(f"Эндпоинт Laya    : {args.laya_url}")
    print(f"Основная модель  : {args.primary_model} ({args.primary_url}, ключ: {'ДА' if args.primary_key else 'НЕТ'})")
    print(f"Каталог данных   : {out_dir}\n")

    # Пул уникальных запросов, объединяющий информацию от Laya, основной модели и Jev
    merged_by_state: Dict[str, Dict[str, Any]] = {}

    def merge_record(rec: Dict[str, Any]) -> None:
        st = rec["state"]
        if st not in merged_by_state:
            merged_by_state[st] = {
                "state": st,
                "session_id": rec.get("session_id") or "",
                "laya_choice": rec.get("laya_choice"),
                "primary_choice": rec.get("primary_choice"),
                "jev_metadata": rec.get("jev_metadata"),
                "sources": [rec.get("source", "unknown")],
            }
        else:
            existing = merged_by_state[st]
            src = rec.get("source", "unknown")
            if src not in existing["sources"]:
                existing["sources"].append(src)
            if rec.get("laya_choice") and not existing.get("laya_choice"):
                existing["laya_choice"] = rec["laya_choice"]
            if rec.get("primary_choice"):
                # Приоритет у фактического вызова инструмента (execute_tools) над отсутствием тулов
                curr_p = existing.get("primary_choice")
                new_p = rec["primary_choice"]
                if not curr_p or (
                    new_p.get("tool_decision") == "execute_tools"
                    and curr_p.get("tool_decision") != "execute_tools"
                ):
                    existing["primary_choice"] = new_p
            if rec.get("jev_metadata") and not existing.get("jev_metadata"):
                existing["jev_metadata"] = rec["jev_metadata"]

    # 1. Проверка плагина kerpopule/hermes-jev-skills
    jev_info = detect_jev_plugin(Path(args.db_path).parent if args.db_path else None)
    print("[*] Проверка плагина kerpopule/hermes-jev-skills:")
    print(f"    - Обнаружен: {'ДА' if jev_info['is_detected'] else 'НЕТ'}")
    print(f"    - Статус: {'ВКЛЮЧЕН (ENABLED)' if jev_info['is_enabled'] else 'ВЫКЛЮЧЕН / НЕ УСТАНОВЛЕН'}")

    # 2. Извлечение из логов плагина Laya (предложения Laya + связка с tool_telemetry)
    log_file = Path(args.logs_path)
    if log_file.is_file():
        log_items = extract_from_logs(log_file)
        telemetry_linked = sum(1 for x in log_items if x.get("primary_choice") is not None)
        print(
            f"[+] Из laya_requests.jsonl извлечено {len(log_items)} запросов Laya "
            f"(с прямой телеметрией вызовов основной модели: {telemetry_linked})."
        )
        for item in log_items:
            merge_record(item)
    else:
        print(f"[-] Файл логов {log_file} не найден.")

    # 3. Извлечение из базы сессий Hermes (фактические вызовы тулов основной моделью)
    db_file = Path(args.db_path)
    if db_file.is_file():
        db_items = extract_from_sqlite(db_file)
        db_with_tools = sum(
            1 for x in db_items if (x.get("primary_choice") or {}).get("tool_decision") == "execute_tools"
        )
        print(
            f"[+] Из базы сессий state.db извлечено {len(db_items)} чистых реплик "
            f"(с вызовами инструментов основной моделью: {db_with_tools})."
        )
        for item in db_items:
            merge_record(item)

    # 4. Если плагин Jev установлен, обогащаем данными Jev
    if jev_info["is_detected"] and args.check_jev:
        jev_log_file = Path(args.jev_logs_path)
        if not jev_log_file.is_file() and jev_info.get("logs", {}).get("decisions"):
            jev_log_file = Path(jev_info["logs"]["decisions"])
        if jev_log_file.is_file():
            jev_items = extract_from_jev_logs(jev_log_file, Path(args.db_path))
            print(f"[+] Из логов Jev ({jev_log_file.name}) сопоставлено {len(jev_items)} записей.")
            for item in jev_items:
                merge_record(item)

    pool = list(merged_by_state.values())[: args.max_samples]
    print(f"\n[+] Всего уникальных пользовательских запросов после очистки и слияния: {len(pool)}")

    # 5. Развилка логики получения данных для дообучения:
    # - Если ЕСТЬ информация, что предлагает Laya и что использует основная модель -> обучаем на выборке основной модели.
    # - Если информации НЕТ -> делаем запрос к основной модели (и/или Laya) для сравнения с выбором Laya.
    labeled_dataset: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []

    observed_both_count = 0
    queried_primary_count = 0
    queried_laya_count = 0
    matches_count = 0
    mismatches_count = 0

    for item in pool:
        state = item["state"]
        laya_choice = item.get("laya_choice")
        primary_choice = item.get("primary_choice")

        # Шаг 5.1: Если нет информации, что предлагает Laya для этого запроса — опрашиваем Laya
        if not laya_choice and args.laya_url:
            laya_choice = query_laya_choice(args.laya_url, state)
            if laya_choice:
                queried_laya_count += 1

        # Шаг 5.2: Проверяем наличие информации о том, что использует основная модель
        if primary_choice is not None:
            # СЛУЧАЙ 1: Есть информация, что предлагает Laya и что использует основная модель
            observed_both_count += 1
        else:
            # СЛУЧАЙ 2: Информации о выборе основной модели нет -> делаем запрос к основной модели для сравнения с Laya
            primary_choice = query_primary_model(
                state_text=state,
                laya_choice=laya_choice,
                base_url=args.primary_url,
                api_key=args.primary_key,
                model=args.primary_model,
            )
            if primary_choice:
                queried_primary_count += 1
            else:
                # Безопасный эвристический резерв только при недоступности API основной модели
                if looks_trivial_greeting(state):
                    primary_choice = {
                        "tool_decision": "direct_chat",
                        "category": "general_utility",
                        "source": "fallback_heuristic",
                    }
                else:
                    primary_choice = {
                        "tool_decision": (laya_choice or {}).get("tool_decision") or "direct_chat",
                        "category": (laya_choice or {}).get("category") or "general_utility",
                        "source": "fallback_default",
                    }

        # Шаг 5.3: Сравнение выбора Laya с выбором основной модели
        match: Optional[bool] = None
        if laya_choice and primary_choice:
            td_match = laya_choice.get("tool_decision") == primary_choice.get("tool_decision")
            if primary_choice.get("tool_decision") == "execute_tools":
                cat_match = laya_choice.get("category") == primary_choice.get("category")
            else:
                cat_match = True
            telemetry_hit = primary_choice.get("telemetry_hit", True)
            match = bool(td_match and cat_match and telemetry_hit)
            if match:
                matches_count += 1
            else:
                mismatches_count += 1

        is_hard_negative = (match is False) or (
            laya_choice is not None and float(laya_choice.get("confidence", 1.0)) < 0.58
        )

        formatted = format_laya_training_item(
            state=state,
            primary_choice=primary_choice,
            laya_choice=laya_choice,
            match=match,
            is_hard_negative=is_hard_negative,
            jev_metadata=item.get("jev_metadata"),
        )
        labeled_dataset.append(formatted)

        comparisons.append({
            "state": state[:100],
            "laya_decision": (laya_choice or {}).get("tool_decision"),
            "laya_category": (laya_choice or {}).get("category"),
            "laya_confidence": (laya_choice or {}).get("confidence"),
            "primary_decision": primary_choice.get("tool_decision"),
            "primary_category": primary_choice.get("category"),
            "primary_source": primary_choice.get("source"),
            "executed_tools": primary_choice.get("executed_tools", []),
            "match": match,
            "hard_negative": is_hard_negative,
        })

    print(f"\n[+] Сравнение и разметка завершены ({len(labeled_dataset)} уникальных кейсов):")
    print(f"    • Известен выбор основной модели из логов/сессий : {observed_both_count}")
    print(f"    • Запрошено у основной модели (Teacher LLM)      : {queried_primary_count}")
    print(f"    • Запрошено у Laya (для полного сравнения)       : {queried_laya_count}")
    print(f"    • Совпадений Laya с основной моделью (Match)     : {matches_count}")
    print(f"    • Расхождений Laya с основной моделью (Mismatch) : {mismatches_count}")

    if not labeled_dataset:
        print("[-] Датасет пуст. Использование демонстрационных сэмплов из examples/...")
        sample_file = skill_dir / "examples" / "train_sample.jsonl"
        if sample_file.is_file():
            with open(sample_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if "expected" not in row and "label" in row:
                            row["expected"] = dict(row["label"])
                        labeled_dataset.append(row)

    # 6. Стратифицированное разделение на train и eval (без пересечения state!)
    exec_items = [r for r in labeled_dataset if r["expected"].get("tool_decision") == "execute_tools"]
    chat_items = [r for r in labeled_dataset if r["expected"].get("tool_decision") == "direct_chat"]

    eval_exec_n = max(1, int(round(len(exec_items) * args.eval_ratio))) if len(exec_items) > 2 else 0
    eval_chat_n = max(1, int(round(len(chat_items) * args.eval_ratio))) if len(chat_items) > 2 else 0

    eval_data = exec_items[-eval_exec_n:] + chat_items[-eval_chat_n:] if (eval_exec_n or eval_chat_n) else labeled_dataset[-2:]
    eval_states = {r["state"] for r in eval_data}
    base_train_data = [r for r in labeled_dataset if r["state"] not in eval_states]
    if not base_train_data:
        base_train_data = list(labeled_dataset)

    # 7. Hard Negative Oversampling (только в обучающей выборке, чтобы не нарушать held-out eval!)
    train_data: List[Dict[str, Any]] = []
    oversampled_extra = 0
    for r in base_train_data:
        train_data.append(r)
        if args.oversample_hard_negatives > 1 and (r.get("metadata") or {}).get("hard_negative"):
            for _ in range(args.oversample_hard_negatives - 1):
                train_data.append(r)
                oversampled_extra += 1

    train_file = out_dir / "train.jsonl"
    eval_file = out_dir / "eval.jsonl"
    comparison_file = out_dir / "distill_comparison.json"

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(eval_file, "w", encoding="utf-8") as f:
        for r in eval_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    comparison_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_unique_cases": len(labeled_dataset),
        "observed_from_logs_or_db": observed_both_count,
        "queried_primary_model": queried_primary_count,
        "queried_laya_live": queried_laya_count,
        "laya_vs_primary_matches": matches_count,
        "laya_vs_primary_mismatches": mismatches_count,
        "agreement_rate": round(matches_count / max(1, matches_count + mismatches_count), 4),
        "train_base_cases": len(base_train_data),
        "train_oversampled_hard_negatives": oversampled_extra,
        "train_total_rows": len(train_data),
        "eval_total_rows": len(eval_data),
        "comparisons": comparisons,
    }
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Готово! Данные сохранены в каталог навыка {out_dir}:")
    print(f"    - train.jsonl            : {len(train_data)} записей (базовых: {len(base_train_data)}, hard_negative oversample: +{oversampled_extra}) -> {train_file}")
    print(f"    - eval.jsonl             : {len(eval_data)} независимых held-out записей -> {eval_file}")
    print(f"    - distill_comparison.json: сводка сравнения Laya vs Primary Model -> {comparison_file}")


if __name__ == "__main__":
    main()
