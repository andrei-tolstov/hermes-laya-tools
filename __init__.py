"""Hermes Laya Tools Plugin.

Smart System 1 Tool Routing & Top-10 Selection for Hermes Agent,
powered by Laya Decision Engine.
Includes Shadow Mode and comprehensive JSONL request audit logging.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Dict, Optional

try:
    from .categorizer import ToolCategorizer, extract_tool_name
    from .client import LayaClient
    from .config import LayaConfig
    from .logger import LayaAuditLogger
    from .router import ToolRouter
except ImportError:
    from categorizer import ToolCategorizer, extract_tool_name
    from client import LayaClient
    from config import LayaConfig
    from logger import LayaAuditLogger
    from router import ToolRouter

logger = logging.getLogger("hermes.plugin.laya")

# Global singleton plugin components
_config: Optional[LayaConfig] = None
_client: Optional[LayaClient] = None
_categorizer: Optional[ToolCategorizer] = None
_audit_logger: Optional[LayaAuditLogger] = None
_router: Optional[ToolRouter] = None

# Track original toolsets per session to support cleanly restoring or inspecting tools
_saved_agent_tools: Dict[str, list[Any]] = {}
_session_lock = threading.Lock()


def _get_agent_from_runtime(session_id: str | None = None) -> Any:
    """Find the live agent object from stack or active worker threads."""
    frame = sys._getframe()
    try:
        while frame is not None:
            cand = frame.f_locals.get("agent")
            if cand is not None and hasattr(cand, "tools"):
                if not session_id or str(getattr(cand, "session_id", "")) == str(session_id):
                    return cand
            frame = frame.f_back
    finally:
        del frame

    if session_id:
        try:
            frames = sys._current_frames()
            for tid, t_frame in frames.items():
                curr = t_frame
                while curr is not None:
                    cand = curr.f_locals.get("agent")
                    if cand is not None and hasattr(cand, "tools"):
                        if str(getattr(cand, "session_id", "")) == str(session_id):
                            return cand
                    curr = curr.f_back
        except Exception:
            pass
    return None


def register(ctx: Any) -> None:
    """Hermes Agent plugin registration entrypoint."""
    global _config, _client, _categorizer, _audit_logger, _router

    _config = LayaConfig.from_env()
    _client = LayaClient(
        sidecar_url=_config.sidecar_url,
        timeout_ms=_config.timeout_ms,
    )
    _audit_logger = LayaAuditLogger(
        log_path=_config.log_path,
        min_level=_config.log_level,
    )
    _categorizer = ToolCategorizer(
        laya_client=_client,
        cache_path=_config.cache_path,
        interval_ms=_config.categorization_interval_ms,
    )
    _router = ToolRouter(
        laya_client=_client,
        config=_config,
        categorizer=_categorizer,
        audit_logger=_audit_logger,
    )

    # Register hooks
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)

    # Register /laya slash command
    try:
        ctx.register_command(
            "laya",
            handler=_laya_command,
            description="Manage Laya Decision Engine (status, shadow, active, reload)",
            args_hint="status|shadow|active|reload",
        )
    except Exception as exc:
        logger.debug("Failed to register /laya command: %s", exc)

    logger.info(
        "hermes-laya-tools plugin loaded (sidecar: %s, mode: %s, hierarchical: %s, max_tools: %d)",
        _config.sidecar_url,
        _config.mode,
        _config.hierarchical,
        _config.max_tools,
    )


def _on_pre_llm_call(session_id: str = "", turn_id: str = "", **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Hook invoked before LLM request assembly.

    In active mode: narrows agent.tools to Laya Top-10 / strips if direct chat.
    In shadow mode: runs full Laya evaluation, records audit log, but leaves agent.tools intact.
    """
    global _router, _config, _saved_agent_tools

    if not _router or not _config or not _config.enabled:
        return None

    agent = kwargs.get("agent") or _get_agent_from_runtime(session_id)
    if agent is None or not hasattr(agent, "tools"):
        return None

    all_tools = getattr(agent, "tools", None)
    if not isinstance(all_tools, list) or not all_tools:
        return None

    with _session_lock:
        if session_id and session_id not in _saved_agent_tools:
            _saved_agent_tools[session_id] = list(all_tools)

    effective_tools, decision = _router.route_tools(
        tools=all_tools,
        context=kwargs,
        session_id=session_id,
        turn_id=turn_id,
    )

    if _config.is_shadow:
        # SHADOW MODE: Strictly DO NOT alter agent.tools and DO NOT inject context!
        return None

    # ACTIVE MODE: Apply filtered toolset to agent
    agent.tools = effective_tools

    # Optional chat notification
    if _config.notify_in_chat:
        notice = _format_notice(decision)
        if notice:
            return {"context": notice}

    return None


def _format_notice(decision: Dict[str, Any]) -> str:
    """Format a concise System 1 notice."""
    dec = decision.get("decision")
    prob = decision.get("probability", 0.0) * 100.0
    lat = decision.get("latency_ms")
    lat_str = f"{lat:.0f}ms" if lat is not None else ""

    if dec == "direct_chat":
        return f"💬 [Laya System 1] Direct conversational request ({prob:.1f}%). Tools stripped."
    if dec == "execute_tools":
        cat = decision.get("category", "general")
        tools = decision.get("selected_tools", [])
        tools_str = ", ".join(tools[:5]) + ("..." if len(tools) > 5 else "")
        return f"⚡ [Laya System 1] Category: '{cat}' ({prob:.1f}%). Selected {len(tools)} tools: {tools_str} ({lat_str})"
    return ""


def _on_session_end(session_id: str = "", **_kwargs: Any) -> None:
    """Clean up session data when session terminates."""
    with _session_lock:
        _saved_agent_tools.pop(session_id, None)


def _laya_command(raw_args: str = "") -> str:
    """Handle /laya CLI slash commands."""
    global _config, _client, _router

    if not _config or not _client or not _router:
        return "Laya plugin not initialized."

    cmd = (raw_args or "status").strip().lower()

    if cmd == "status":
        health = _client.health()
        status_str = "online (healthy)" if health else "unreachable"
        last_dec = _router.last_decision or {}
        return (
            f"Laya Engine: {status_str} ({_config.sidecar_url})\n"
            f"Mode: {_config.mode.upper()} {'(SHADOW: primary output untouched)' if _config.is_shadow else ''}\n"
            f"Enabled: {_config.enabled}\n"
            f"Max Tools: {_config.max_tools}\n"
            f"Hierarchical: {_config.hierarchical}\n"
            f"Log File: {_config.log_path}\n"
            f"Last Decision: {last_dec.get('decision', 'none')} "
            f"(cat: {last_dec.get('category')}, tools: {len(last_dec.get('selected_tools', []))})"
        )

    if cmd == "shadow":
        _config.mode = "shadow"
        _config.notify_in_chat = False
        return "Laya mode switched to SHADOW (evaluation and logging active, tools unmodified)."

    if cmd == "active":
        _config.mode = "active"
        _config.notify_in_chat = True
        return "Laya mode switched to ACTIVE (hierarchical filtering applied to model tools)."

    if cmd == "reload":
        try:
            res = _client.reload_model()
            return f"Model reloaded: {res}"
        except Exception as e:
            return f"Model reload error: {e}"

    return "Usage: /laya [status | shadow | active | reload]"
