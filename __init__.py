"""Hermes Laya Tools Plugin.

Smart System 1 Tool Routing & Top-10 Selection for Hermes Agent,
powered by Laya Decision Engine.
Includes Shadow Mode, Hermes WebUI Settings integration, and comprehensive JSONL request audit logging.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Dict, Optional, Set

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
_CTX: Any = None
_config: Optional[LayaConfig] = None
_client: Optional[LayaClient] = None
_categorizer: Optional[ToolCategorizer] = None
_audit_logger: Optional[LayaAuditLogger] = None
_router: Optional[ToolRouter] = None

# Track original toolsets per session to support cleanly restoring or inspecting tools
_saved_agent_tools: Dict[str, list[Any]] = {}
_audited_turns: Set[str] = set()
_session_lock = threading.Lock()


def _get_tools_fallback() -> list[Any]:
    """Retrieve default tool definitions from Hermes model_tools if agent is unreachable."""
    try:
        from agent import model_tools
        tools = model_tools.get_tool_definitions()
        if isinstance(tools, list) and tools:
            return tools
    except Exception:
        pass
    return []


def _get_agent_from_runtime(session_id: str | None = None) -> Any:
    """Find the live agent object from stack or active worker threads."""
    frame = sys._getframe()
    try:
        while frame is not None:
            for name in ("agent", "self", "a"):
                cand = frame.f_locals.get(name)
                if cand is not None and hasattr(cand, "tools"):
                    cand_sess = str(getattr(cand, "session_id", ""))
                    if not session_id or cand_sess == str(session_id) or not cand_sess:
                        return cand
            frame = frame.f_back
    finally:
        del frame

    try:
        frames = sys._current_frames()
        for tid, t_frame in frames.items():
            curr = t_frame
            while curr is not None:
                for name in ("agent", "self", "a"):
                    cand = curr.f_locals.get(name)
                    if cand is not None and hasattr(cand, "tools"):
                        cand_sess = str(getattr(cand, "session_id", ""))
                        if not session_id or cand_sess == str(session_id) or not cand_sess:
                            return cand
                curr = curr.f_back
    except Exception:
        pass
    return None


def register(ctx: Any) -> None:
    """Hermes Agent plugin registration entrypoint."""
    global _config, _client, _categorizer, _audit_logger, _router, _CTX

    _CTX = ctx
    _config = LayaConfig.from_ctx_or_env(ctx=ctx)
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
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)

    # Register /laya slash command
    try:
        ctx.register_command(
            "laya",
            handler=_laya_command,
            description="Manage Laya Decision Engine (status, shadow, active, config, reload, recategorize)",
            args_hint="status|shadow|active|config|reload|recategorize",
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
    global _router, _config, _saved_agent_tools, _audited_turns, _CTX

    if not _router or not _config or not _config.enabled:
        return None

    # Sync any updated settings from Hermes UI if context is available
    if _CTX is not None:
        try:
            _config.reload(ctx=_CTX)
        except Exception:
            pass

    # Handle immediate recategorization trigger from settings UI
    if _config.recategorize_now and _categorizer is not None:
        try:
            _config.recategorize_now = False
            recat_res = _categorizer.recategorize(kwargs.get("tools") or _get_tools_fallback())
            if _config.notify_in_chat:
                return {"context": f"[Laya] ⚡ {recat_res.get('message', 'Recategorized')}"}
        except Exception as e:
            logger.warning("Failed to run recategorize_now: %s", e)

    agent = kwargs.get("agent") or _get_agent_from_runtime(session_id)
    all_tools = getattr(agent, "tools", None) if agent is not None else None
    if not isinstance(all_tools, list) or not all_tools:
        all_tools = kwargs.get("tools") or _get_tools_fallback()

    if not isinstance(all_tools, list) or not all_tools:
        return None

    turn_key = f"{session_id}:{turn_id}"
    with _session_lock:
        _audited_turns.add(turn_key)
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

    # ACTIVE MODE: Apply filtered toolset to agent if agent instance was found
    if agent is not None and hasattr(agent, "tools"):
        agent.tools = effective_tools

    # Optional chat notification
    if _config.notify_in_chat:
        notice = _format_notice(decision)
        if notice:
            return {"context": notice}

    return None


def _on_pre_api_request(**kwargs: Any) -> None:
    """Observer hook fired right before an LLM provider call.

    Guarantees every request is evaluated and audited in Laya, even if
    pre_llm_call was bypassed or agent was detached.
    """
    global _router, _config, _audited_turns, _CTX

    if not _router or not _config or not _config.enabled:
        return

    # Sync any updated settings from Hermes UI if context is available
    if _CTX is not None:
        try:
            _config.reload(ctx=_CTX)
        except Exception:
            pass

    session_id = str(kwargs.get("session_id") or "")
    turn_id = str(kwargs.get("turn_id") or "")
    turn_key = f"{session_id}:{turn_id}"

    with _session_lock:
        if turn_key in _audited_turns:
            return
        if len(_audited_turns) > 1000:
            _audited_turns.clear()
        _audited_turns.add(turn_key)

    req = kwargs.get("request") or {}
    body = req.get("body") if isinstance(req, dict) else {}
    tools = body.get("tools") or []

    if not isinstance(tools, list) or not tools:
        tools = _get_tools_fallback()

    if not tools:
        return

    _router.route_tools(
        tools=tools,
        context=kwargs,
        session_id=session_id,
        turn_id=turn_id,
    )


def _on_post_tool_call(
    tool_name: str = "",
    args: Any = None,
    session_id: str = "",
    status: Any = None,
    **extra: Any,
) -> None:
    """Record tool execution telemetry and evaluate router accuracy (hit/miss)."""
    global _router, _audit_logger, _config
    if not _router or not _audit_logger or not tool_name:
        return

    last_dec = _router.last_decision or {}
    selected = last_dec.get("selected_tools")
    if not isinstance(selected, list):
        return

    is_hit = tool_name in selected
    decision_kind = last_dec.get("decision", "unknown")
    turn_id = str(extra.get("turn_id") or "")
    mode = _config.mode if _config else "unknown"

    telemetry_payload = {
        "kind": "tool_telemetry",
        "tool_name": tool_name,
        "hit": is_hit,
        "decision": decision_kind,
        "selected_tools": selected,
        "status": status,
    }
    _audit_logger.log_interaction(
        session_id=session_id,
        turn_id=turn_id,
        mode=mode,
        decision=f"tool_exec:{'hit' if is_hit else 'miss'}",
        selected_tools=selected,
        all_available_tools=[tool_name],
        state_snippet=f"executed={tool_name} hit={is_hit}",
        raw_answers=telemetry_payload,
    )


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
    global _config, _client, _router, _categorizer, _CTX

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
            f"Kill Switch Active: {_config.is_killed}\n"
            f"Max Tools: {_config.max_tools}\n"
            f"Hierarchical: {_config.hierarchical}\n"
            f"Safety Threshold: {_config.direct_chat_threshold:.2f} (min conf: {_config.min_confidence:.2f})\n"
            f"Trivial Bypass: {_config.trivial_bypass}\n"
            f"Cache Repeat Asks: {_config.cache_repeat_asks} (size: {_config.cache_size})\n"
            f"Always Include: {sorted(list(_config.always_include))}\n"
            f"Log File: {_config.log_path}\n"
            f"Last Decision: {last_dec.get('decision', 'none')} "
            f"(cat: {last_dec.get('category')}, tools: {len(last_dec.get('selected_tools', []))})"
        )

    if cmd == "config":
        return (
            f"Current Laya Configuration:\n"
            f"  enabled: {_config.enabled}\n"
            f"  is_killed: {_config.is_killed}\n"
            f"  mode: {_config.mode}\n"
            f"  hierarchical: {_config.hierarchical}\n"
            f"  max_tools: {_config.max_tools}\n"
            f"  sidecar_url: {_config.sidecar_url}\n"
            f"  timeout_ms: {_config.timeout_ms}\n"
            f"  fallback_mode: {_config.fallback_mode}\n"
            f"  notify_in_chat: {_config.notify_in_chat}\n"
            f"  direct_chat_threshold: {_config.direct_chat_threshold}\n"
            f"  min_confidence: {_config.min_confidence}\n"
            f"  trivial_bypass: {_config.trivial_bypass}\n"
            f"  cache_repeat_asks: {_config.cache_repeat_asks}\n"
            f"  cache_size: {_config.cache_size}\n"
            f"  privacy_mode: {_config.privacy_mode}\n"
            f"  always_include: {sorted(list(_config.always_include))}\n"
            f"  log_level: {_config.log_level}\n"
            f"  log_path: {_config.log_path}\n"
            f"  cache_path: {_config.cache_path}\n"
        )

    if cmd == "shadow":
        _config.mode = "shadow"
        _config.notify_in_chat = False
        return "Laya mode switched to SHADOW (evaluation and logging active, tools unmodified)."

    if cmd == "active":
        _config.mode = "active"
        _config.notify_in_chat = True
        return "Laya mode switched to ACTIVE (hierarchical filtering applied to model tools)."

    if cmd in ("recategorize", "categorize"):
        if _categorizer is None:
            return "Categorizer not initialized."
        res = _categorizer.recategorize()
        return f"⚡ Laya Recategorize: {res.get('message', 'done')}"

    if cmd == "reload":
        if _CTX is not None:
            _config.reload(ctx=_CTX)
        else:
            _config.reload()
        try:
            res = _client.reload_model()
            return f"Configuration reloaded. Model reloaded: {res}"
        except Exception as e:
            return f"Configuration reloaded. Model reload notice: {e}"

    return "Usage: /laya [status | shadow | active | config | reload | recategorize]"

