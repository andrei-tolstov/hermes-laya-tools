"""Audit and operational logger for Laya System 1 Decision Engine.

Provides structured JSONL logging for every request/response, latency,
decisions made, and shadow mode status.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.plugin.laya")


class LayaAuditLogger:
    """Thread-safe persistent JSONL audit logger for Laya interactions."""

    def __init__(self, log_path: Path | str, min_level: str = "INFO") -> None:
        self.log_path = Path(log_path).expanduser()
        self._lock = threading.Lock()
        self._ensure_dir()

        # Set logger level
        numeric_level = getattr(logging, min_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

    def _ensure_dir(self) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                self.log_path.touch()
        except Exception as exc:
            logger.warning("Could not create log file %s: %s", self.log_path, exc)

    def log_interaction(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        mode: str = "active",
        decision: str = "execute_tools",
        winning_category: Optional[str] = None,
        probability: float = 0.0,
        selected_tools: Optional[List[str]] = None,
        all_available_tools: Optional[List[str]] = None,
        state_snippet: str = "",
        questions: Optional[Dict[str, Any]] = None,
        raw_answers: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an audit entry for a Laya evaluation turn."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        is_shadow = mode.lower() == "shadow"
        selected = selected_tools or []
        available = all_available_tools or []

        entry = {
            "timestamp": now,
            "session_id": session_id,
            "turn_id": turn_id,
            "mode": mode,
            "shadow": is_shadow,
            "decision": decision,
            "winning_category": winning_category,
            "probability": round(probability, 4) if probability else 0.0,
            "selected_tools": selected,
            "selected_count": len(selected),
            "available_tools_count": len(available),
            "available_tools": available,
            "state_snippet": (state_snippet or "")[:300],
            "questions": questions or {},
            "raw_answers": raw_answers or {},
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
            "error": error,
        }

        # Write to JSONL file
        self._write_file(entry)

        # Emit human-readable operational log
        self._emit_op_log(entry)

        return entry

    def _write_file(self, entry: Dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        try:
            with self._lock:
                self._ensure_dir()
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as exc:
            logger.warning("Failed to write Laya audit log to %s: %s", self.log_path, exc)

    def _emit_op_log(self, entry: Dict[str, Any]) -> None:
        mode = entry["mode"]
        decision = entry["decision"]
        prob = entry["probability"] * 100.0
        tools = entry["selected_tools"]
        cat = entry["winning_category"]
        lat = f"{entry['latency_ms']}ms" if entry["latency_ms"] is not None else "n/a"

        if entry.get("error"):
            logger.warning(
                "⚠️ [Laya %s] Evaluation failed (%s): %s. Fallback applied.",
                mode.capitalize(),
                lat,
                entry["error"],
            )
            return

        if entry["shadow"]:
            # Shadow Mode Log
            if decision == "direct_chat":
                logger.info(
                    "👻 [Laya Shadow] (Simulated direct_chat %.1f%%, %s). Primary model surface UNCHANGED (all %d tools kept).",
                    prob,
                    lat,
                    entry["available_tools_count"],
                )
            else:
                logger.info(
                    "👻 [Laya Shadow] (Simulated category '%s' %.1f%%, %s). Would select %d tools: %s. Primary model surface UNCHANGED (all %d tools kept).",
                    cat,
                    prob,
                    lat,
                    len(tools),
                    ", ".join(tools[:5]) + ("..." if len(tools) > 5 else ""),
                    entry["available_tools_count"],
                )
        else:
            # Active Mode Log
            if decision == "direct_chat":
                logger.info(
                    "💬 [Laya System 1] Direct chat classified (%.1f%%, %s). Tools stripped (tools=[]).",
                    prob,
                    lat,
                )
            else:
                logger.info(
                    "⚡ [Laya System 1] Category '%s' (%.1f%%, %s). Selected %d tools: %s",
                    cat,
                    prob,
                    lat,
                    len(tools),
                    ", ".join(tools[:6]) + ("..." if len(tools) > 6 else ""),
                )
