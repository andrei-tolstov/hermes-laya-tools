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
    from .config import ENRICH_TOP3_MODE, ENRICH_TOP_N, LayaConfig, clear_yaml_setting
    from .logger import LayaAuditLogger
    from .router import ToolRouter
    from .tool_collector import discovery_tool_names, get_bridge_tools
except ImportError:
    from categorizer import ToolCategorizer, extract_tool_name
    from client import LayaClient
    from config import ENRICH_TOP3_MODE, ENRICH_TOP_N, LayaConfig, clear_yaml_setting
    from logger import LayaAuditLogger
    from router import ToolRouter
    from tool_collector import discovery_tool_names, get_bridge_tools

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
_bridge_warned: Set[str] = set()
_session_lock = threading.Lock()


def _get_tools_fallback() -> list[Any]:
    """Retrieve full uncollapsed tool definitions across hot + catalog/MCP tools."""
    try:
        from .tool_collector import collect_all_tools
    except ImportError:
        from tool_collector import collect_all_tools

    try:
        tools = collect_all_tools()
        if isinstance(tools, list) and tools:
            return tools
    except Exception as exc:
        logger.debug("Failed to collect all tools in fallback: %s", exc)

    try:
        from agent import model_tools
        tools = model_tools.get_tool_definitions(skip_tool_search_assembly=True, quiet_mode=True)
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

    # Register CLI subcommand 'hermes laya'
    if hasattr(ctx, "register_cli_command"):
        try:
            ctx.register_cli_command(
                "laya",
                help="Manage Laya Decision Engine (status, shadow, active, config, reload, recategorize)",
                setup_fn=_setup_laya_cli,
                handler_fn=_handle_laya_cli,
            )
        except Exception as exc:
            logger.debug("Failed to register CLI command hermes laya: %s", exc)

    logger.info(
        "hermes-laya-tools plugin loaded (sidecar: %s, mode: %s, hierarchical: %s, max_tools: %d)",
        _config.sidecar_url,
        _config.mode,
        _config.hierarchical,
        _config.max_tools,
    )


def _setup_laya_cli(parser: Any) -> None:
    """Setup CLI parser for `hermes laya`."""
    parser.add_argument(
        "subcommand",
        nargs="?",
        default="status",
        choices=["status", "recategorize", "categorize", "shadow", "active", "top3", "config", "reload"],
        help="Laya action: status | recategorize | shadow | active | top3 | config | reload",
    )


def _handle_laya_cli(args: Any) -> int:
    """Execute `hermes laya <subcommand>`."""
    cmd = getattr(args, "subcommand", "status")
    output = _laya_command(cmd)
    print(output)
    return 0


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
            clear_yaml_setting("recategorize_now")
            recat_res = _categorizer.recategorize(kwargs.get("tools") or _get_tools_fallback())
            summary_msg = recat_res.get("summary") or recat_res.get("message", "done")
            logger.info("Recategorize trigger executed: %s", summary_msg)
            return {"context": f"⚡ [Laya] {summary_msg}"}
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
        # Enrich mode adds tools to agent.tools; always enrich the ORIGINAL standard list,
        # otherwise tools added on earlier turns would pile up as if they were standard.
        if _config.is_enrich_top3 and session_id:
            all_tools = list(_saved_agent_tools.get(session_id) or all_tools)
        original_tools = _saved_agent_tools.get(session_id) if session_id else None

    effective_tools, decision = _router.route_tools(
        tools=all_tools,
        context=kwargs,
        session_id=session_id,
        turn_id=turn_id,
        agent=agent,
        original_tools=original_tools,
    )

    if _config.is_shadow:
        # SHADOW MODE: Strictly DO NOT alter agent.tools and DO NOT inject context!
        return None

    # ACTIVE MODE: Apply filtered toolset to agent if agent instance was found
    if agent is not None and hasattr(agent, "tools"):
        _apply_effective_tools(agent, effective_tools)

    # Optional chat notification
    if _config.notify_in_chat:
        notices = [n for n in (_format_bridge_warning(session_id), _format_notice(decision)) if n]
        if notices:
            return {"context": "\n".join(notices)}

    return None


def _apply_effective_tools(agent: Any, effective_tools: list[Any]) -> None:
    """Expose ``effective_tools`` to the model AND to Hermes' tool-call validation.

    Hermes validates every tool call against ``agent.valid_tool_names`` (built once at agent init),
    not against ``agent.tools``; a tool shown to the model but absent there is rejected with
    "Tool '...' does not exist". The set is only ever widened, so tools Laya merely left out of
    this turn's list stay callable too.
    """
    agent.tools = effective_tools
    if not hasattr(agent, "valid_tool_names"):
        return
    names = {n for n in (extract_tool_name(t) for t in effective_tools) if n}
    current = getattr(agent, "valid_tool_names", None)
    merged = set(current) if isinstance(current, (set, frozenset, list, tuple)) else set()
    try:
        agent.valid_tool_names = merged | names
    except Exception as exc:
        logger.debug("Could not sync agent.valid_tool_names: %s", exc)


def _format_bridge_warning(session_id: str) -> str:
    """One-time chat warning when Hermes' discovery tools could not be found."""
    status = _router.bridge_status if _router else None
    if not status or not status.get("hermes_available") or status.get("ok"):
        return ""
    with _session_lock:
        if session_id in _bridge_warned:
            return ""
        if len(_bridge_warned) > 1000:
            _bridge_warned.clear()
        _bridge_warned.add(session_id)
    problems = []
    if status.get("unresolved"):
        problems.append(f"нет стандартных тулзов Hermes: {', '.join(status['unresolved'])}")
    problems.extend(status.get("api_issues") or [])
    return "⚠️ [Laya] Прогрессивное раскрытие тулзов Hermes изменилось — " + "; ".join(problems)


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
        agent=_get_agent_from_runtime(session_id),
    )


def _should_unlock_full_toolset(tool_name: str, session_id: str) -> bool:
    """Did the model just reach beyond the tools Laya exposed (discovery tool or hidden tool)?"""
    if not _router or not _config or not _config.enabled or _config.is_killed or _config.is_shadow:
        return False
    if _router.is_unlocked(session_id):
        return False
    if tool_name in discovery_tool_names():
        return True
    # Hermes unwraps tool_call(calls=[{name,...}]), so a hidden tool reached through the bridge
    # arrives here under its own name although it was not in the list sent to the model.
    exposed = (_router.last_decision or {}).get("exposed_tools")
    return isinstance(exposed, list) and tool_name not in exposed


def _unlock_full_toolset(session_id: str) -> None:
    """Give the model the full tool list from the very next LLM call (this turn and later ones).

    ``pre_llm_call`` fires once per user turn, so mid-turn the only way to widen the list the
    next iteration sends is to rewrite ``agent.tools`` / ``agent.valid_tool_names`` here.
    """
    if _router is None:
        return
    _router.unlock_session(session_id)
    agent = _get_agent_from_runtime(session_id)
    if agent is None or not isinstance(getattr(agent, "tools", None), list):
        return
    bridge_tools = get_bridge_tools(
        agent.tools,
        fallback_sources=[_saved_agent_tools.get(session_id), _router._bridge_by_session.get(session_id or "")],
        rebuild=True,
    )
    full_tools = _router.build_full_access_tools(list(agent.tools), agent, bridge_tools)
    _apply_effective_tools(agent, full_tools)
    names = [extract_tool_name(t) for t in full_tools]
    _router.last_decision = {
        "decision": "bridge_unlocked",
        "category": None,
        "probability": 1.0,
        "selected_tools": names,
        "exposed_tools": names,
        "shadow": False,
        "fallback_reason": "bridge_tool_called",
        "error": None,
    }
    logger.info("[Laya] Discovery tool used - full tool list (%d tools) exposed to the model", len(names))


def _on_post_tool_call(
    tool_name: str = "",
    args: Any = None,
    session_id: str = "",
    status: Any = None,
    **extra: Any,
) -> None:
    """Record tool telemetry, and open the full tool list once the model uses discovery tools."""
    unlock = bool(tool_name) and _should_unlock_full_toolset(tool_name, session_id)
    try:
        _log_tool_telemetry(tool_name, session_id, status, extra)
    finally:
        if unlock:
            try:
                _unlock_full_toolset(session_id)
            except Exception as exc:
                logger.warning("Failed to expose the full tool list after %s: %s", tool_name, exc)


def _log_tool_telemetry(tool_name: str, session_id: str, status: Any, extra: Dict[str, Any]) -> None:
    """Record tool execution telemetry and evaluate router accuracy (hit/miss)."""
    global _router, _audit_logger, _config
    if not _router or not _audit_logger or not tool_name:
        return

    last_dec = _router.last_decision or {}
    selected = last_dec.get("selected_tools")
    if not isinstance(selected, list):
        return

    # Discovery tools are always exposed, so using them is never a routing miss.
    is_hit = tool_name in selected or tool_name in discovery_tool_names()
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
    if dec == "enrich_top3":
        cat = decision.get("category") or "general"
        picked = decision.get("selected_tools", [])
        added = decision.get("added_tools", [])
        if decision.get("trivial_bypass") or not picked:
            return ""
        added_str = ", ".join(added) if added else "все уже в стандартном списке"
        return (
            f"➕ [Laya System 1] Топ-3 ({cat}, {prob:.1f}%): {', '.join(picked)}. "
            f"Добавлено: {added_str} ({lat_str})"
        )
    return ""


def _on_session_end(session_id: str = "", **_kwargs: Any) -> None:
    """Clean up session data when session terminates."""
    with _session_lock:
        _saved_agent_tools.pop(session_id, None)
        _bridge_warned.discard(session_id)
    if _router is not None:
        _router.forget_session(session_id)


def _format_bridge_status() -> str:
    """Status line for /laya status: state of Hermes' tool_search/tool_describe/tool_call."""
    status = _router.bridge_status if _router else None
    if not status:
        return "not checked yet"
    if not status.get("hermes_available"):
        return "n/a (not running inside Hermes)"
    if status.get("ok"):
        return "OK (" + ", ".join(sorted(extract_tool_name(t) for t in status["bridge_tools"])) + ")"
    problems = []
    if status.get("unresolved"):
        problems.append("MISSING " + ", ".join(status["unresolved"]))
    problems.extend(status.get("api_issues") or [])
    return "ERROR: " + "; ".join(problems)


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
        dist = _categorizer.last_distribution_summary if _categorizer else "none"
        if _categorizer and (dist == "none" or dist.startswith("Всего тулзов: 0")):
            try:
                recat_res = _categorizer.recategorize()
                dist = recat_res.get("summary") or dist
            except Exception:
                pass
        return (
            f"Laya Engine: {status_str} ({_config.sidecar_url})\n"
            f"Mode: {_config.mode.upper()} {'(SHADOW: primary output untouched)' if _config.is_shadow else ''}"
            f"{'(Добавление топ 3: стандартный список + топ-3 от Laya)' if _config.is_enrich_top3 else ''}\n"
            f"Tool Distribution: {dist}\n"
            f"Enabled: {_config.enabled}\n"
            f"Kill Switch Active: {_config.is_killed}\n"
            f"Max Tools: {_config.max_tools}\n"
            f"Hierarchical: {_config.hierarchical}\n"
            f"Safety Threshold: {_config.direct_chat_threshold:.2f} (min conf: {_config.min_confidence:.2f})\n"
            f"Trivial Bypass: {_config.trivial_bypass}\n"
            f"Hermes bridge tools: {_format_bridge_status()}\n"
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

    if cmd in ("top3", "enrich", "enrich_top3"):
        _config.mode = ENRICH_TOP3_MODE
        _config.notify_in_chat = True
        return (
            f"Laya mode switched to «Добавление топ 3» (standard tool list kept, "
            f"Laya top {ENRICH_TOP_N} tools missing from it are added)."
        )

    if cmd in ("recategorize", "categorize"):
        if _categorizer is None:
            return "Categorizer not initialized."
        res = _categorizer.recategorize()
        summary = res.get("summary") or res.get("message", "done")
        return f"⚡ Laya: {summary}"

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

    return "Usage: /laya [status | shadow | active | top3 | config | reload | recategorize]"

