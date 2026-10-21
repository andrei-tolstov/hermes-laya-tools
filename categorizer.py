"""ToolCategorizer — Daily tool classification and category caching for Laya.

Categorizes all available tools (built-in + MCP) into semantic groups once a day (24h TTL)
and persists the grouping to disk so it survives restarts.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern

try:
    from .client import LayaClient
except ImportError:
    from client import LayaClient

logger = logging.getLogger("hermes.plugin.laya.categorizer")


DEFAULT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "filesystem": {
        "name": "filesystem",
        "description": "File and directory operations (reading, writing, editing, listing, searching files)",
        "patterns": [r"fs", r"file", r"directory", r"read_", r"write_", r"edit_", r"str_replace", r"patch"],
    },
    "terminal_execution": {
        "name": "terminal_execution",
        "description": "Executing shell, bash, powershell commands, terminal scripts, and command-line processes",
        "patterns": [r"bash", r"\bsh\b", r"pwsh", r"terminal", r"command", r"shell", r"exec"],
    },
    "web_and_search": {
        "name": "web_and_search",
        "description": "Internet web search, downloading web pages, HTTP APIs, and online fetching",
        "patterns": [r"web_search", r"search", r"fetch", r"curl", r"http", r"url", r"scrape"],
    },
    "browser_automation": {
        "name": "browser_automation",
        "description": "Browser UI automation, web page navigation, clicking, typing, and desktop inspection",
        "patterns": [r"browser", r"puppeteer", r"playwright", r"stagehand", r"chrome", r"page", r"screenshot"],
    },
    "code_and_git": {
        "name": "code_and_git",
        "description": "Source code inspection, git version control, repository commits, and code analysis",
        "patterns": [r"git", r"repo", r"branch", r"commit", r"diff", r"lint", r"ast", r"code"],
    },
    "devops_containers": {
        "name": "devops_containers",
        "description": "Docker containers, system administration, processes, services, and infrastructure",
        "patterns": [r"docker", r"container", r"process", r"service", r"system", r"k8s"],
    },
    "external_messaging": {
        "name": "external_messaging",
        "description": "Sending notifications or messages to external services (Telegram, Slack, webhooks) or interactive confirmation dialogs",
        "patterns": [r"telegram", r"slack", r"notify", r"alert", r"webhook", r"ask_user", r"approval", r"question"],
    },
    "general_utility": {
        "name": "general_utility",
        "description": "Calculation, formatting, time, data transformations, and general utilities",
        "patterns": [r"calc", r"math", r"time", r"date", r"json", r"format"],
    },
}


def extract_tool_name(tool: Any) -> str:
    """Extract tool name from function schema dict or object."""
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
        if tool.get("name"):
            return str(tool["name"])
    return getattr(tool, "name", str(tool))


def extract_tool_desc(tool: Any) -> str:
    """Extract tool description from function schema dict or object."""
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("description"):
            return str(fn["description"])
        if tool.get("description"):
            return str(tool["description"])
    return getattr(tool, "description", "")


class ToolCategorizer:
    """Categorizes tools into semantic categories with 24h caching."""

    def __init__(
        self,
        laya_client: LayaClient,
        cache_path: Path | str,
        interval_ms: int = 86_400_000,
        categories_def: Optional[Dict[str, Dict[str, Any]]] = None,
        custom_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.laya = laya_client
        self.cache_path = Path(cache_path).expanduser()
        self.interval_sec = max(60.0, interval_ms / 1000.0)
        self.logger = custom_logger or logger

        # Compiled patterns
        self.categories_def = categories_def or DEFAULT_CATEGORIES
        self._compiled_patterns: Dict[str, List[Pattern[str]]] = {}
        for cat_id, cat_info in self.categories_def.items():
            raw_pats = cat_info.get("patterns", [])
            self._compiled_patterns[cat_id] = [
                re.compile(p, re.IGNORECASE) if isinstance(p, str) else p for p in raw_pats
            ]

        self._in_memory_categories: Optional[Dict[str, Dict[str, Any]]] = None
        self._last_categorized_at: float = 0.0
        self._tools_signature: str = ""
        self._known_tools: List[Any] = []

        self._load_cache()

    def get_category_descriptions(self) -> Dict[str, str]:
        """Category descriptions for Laya questions."""
        return {cat_id: info.get("description", "") for cat_id, info in self.categories_def.items()}

    def get_categories(
        self, tools: List[Any], force_refresh: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Get tools partitioned by semantic category.

        Uses cached partitioning if within interval and tools signature matches.
        """
        if not tools:
            if self._known_tools:
                tools = self._known_tools
            else:
                return {
                    cat_id: {"description": info["description"], "tools": []}
                    for cat_id, info in self.categories_def.items()
                }
        else:
            self._known_tools = tools

        current_sig = self._compute_signature(tools)
        now = time.time()
        is_expired = (now - self._last_categorized_at) >= self.interval_sec

        if (
            not force_refresh
            and not is_expired
            and self._tools_signature == current_sig
            and self._in_memory_categories is not None
        ):
            self.logger.debug("Using cached tool categories (TTL valid).")
            return self._in_memory_categories

        reason = (
            "force-refresh"
            if force_refresh
            else ("daily TTL refresh" if is_expired else "toolset signature changed")
        )
        self.logger.info("Categorizing %d tools (reason: %s)...", len(tools), reason)

        categorized = self._categorize_all(tools)
        self._in_memory_categories = categorized
        self._last_categorized_at = now
        self._tools_signature = current_sig

        self._save_cache()
        return categorized

    def recategorize(self, tools: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Force immediate re-categorization of tools and return a structured summary report."""
        target_tools = tools or self._known_tools
        if not target_tools:
            try:
                from agent import model_tools  # type: ignore[import-not-found]
                defs = model_tools.get_tool_definitions()
                if isinstance(defs, list):
                    target_tools = defs
            except Exception:
                pass

        if not target_tools:
            return {
                "status": "warning",
                "message": "Нет доступных инструментов для разбивки",
                "categories_count": 0,
                "total_tools": 0,
                "counts": {},
            }

        start_time = time.time()
        categories = self.get_categories(target_tools, force_refresh=True)
        elapsed_ms = round((time.time() - start_time) * 1000.0, 1)

        counts = {
            cat_id: len(data.get("tools", []))
            for cat_id, data in categories.items()
            if data.get("tools")
        }
        total_categorized = sum(counts.values())

        cat_breakdown = ", ".join(
            f"{cat}: {cnt}" for cat, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        )
        summary = (
            f"Разбито {total_categorized} инструментов по {len(counts)} категориям "
            f"за {elapsed_ms} мс ({cat_breakdown})"
        )

        self.logger.info("Recategorization completed: %s", summary)
        return {
            "status": "ok",
            "message": summary,
            "total_tools": total_categorized,
            "categories_count": len(counts),
            "counts": counts,
            "latency_ms": elapsed_ms,
        }


    def _categorize_all(self, tools: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Categorize all tools using pattern matching + Laya fallback."""
        groups: Dict[str, Dict[str, Any]] = {
            cat_id: {"description": info["description"], "tools": []}
            for cat_id, info in self.categories_def.items()
        }

        unclassified: List[Any] = []

        for tool in tools:
            matched_cat = self._match_pattern(tool)
            if matched_cat and matched_cat in groups:
                groups[matched_cat]["tools"].append(tool)
            else:
                unclassified.append(tool)

        if unclassified:
            self._classify_with_laya(unclassified, groups)

        return groups

    def _match_pattern(self, tool: Any) -> Optional[str]:
        name = extract_tool_name(tool)
        desc = extract_tool_desc(tool)
        target = f"{name} {desc}"

        for cat_id, pats in self._compiled_patterns.items():
            for pat in pats:
                if pat.search(target):
                    return cat_id
        return None

    def _classify_with_laya(
        self, unclassified_tools: List[Any], groups: Dict[str, Dict[str, Any]]
    ) -> None:
        """Batch classify unknown tools with Laya System 1."""
        criteria = self.get_category_descriptions()
        batch_size = 8

        for i in range(0, len(unclassified_tools), batch_size):
            chunk = unclassified_tools[i : i + batch_size]
            questions: Dict[str, Any] = {}

            for idx, tool in enumerate(chunk):
                name = extract_tool_name(tool)
                desc = (extract_tool_desc(tool) or "no description")[:120].replace("\n", " ")
                questions[f"tool_{idx}"] = {
                    "type": "choice",
                    "instructions": f"Categorize the tool '{name}' ({desc}) into the best category:",
                    "criteria": criteria,
                }

            try:
                resp = self.laya.ask("Classify the following tools into categories", questions)
                answers = resp.get("answers", {})
                for idx, tool in enumerate(chunk):
                    choice = answers.get(f"tool_{idx}", {}).get("choice")
                    target_cat = choice if choice in groups else "general_utility"
                    groups[target_cat]["tools"].append(tool)
            except Exception as exc:
                self.logger.warning("Laya classification fallback failed: %s", exc)
                for tool in chunk:
                    groups["general_utility"]["tools"].append(tool)

    def _compute_signature(self, tools: List[Any]) -> str:
        names = [extract_tool_name(t) for t in tools]
        names.sort()
        return "|".join(names)

    def _load_cache(self) -> None:
        if not self.cache_path.is_file():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return

            self._last_categorized_at = float(data.get("lastCategorizedAt", 0.0))
            self._tools_signature = str(data.get("toolsSignature", ""))
            raw_cats = data.get("categories", {})

            loaded_groups: Dict[str, Dict[str, Any]] = {}
            for cat_id, group in raw_cats.items():
                loaded_groups[cat_id] = {
                    "description": group.get("description", self.categories_def.get(cat_id, {}).get("description", "")),
                    "tools": group.get("tools", []),
                }

            if loaded_groups:
                self._in_memory_categories = loaded_groups
                self.logger.debug("Loaded tool categories cache from %s", self.cache_path)
        except Exception as exc:
            self.logger.warning("Failed to load categories cache from %s: %s", self.cache_path, exc)

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "lastCategorizedAt": self._last_categorized_at,
                "toolsSignature": self._tools_signature,
                "categories": self._in_memory_categories or {},
            }
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.logger.debug("Saved categories cache to %s", self.cache_path)
        except Exception as exc:
            self.logger.warning("Failed to save categories cache to %s: %s", self.cache_path, exc)
