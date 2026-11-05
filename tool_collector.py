"""Tool Collector for Hermes Laya Tools.

Extracts the full catalog of available tools (hot + deferred/catalog/MCP tools),
ensuring Laya evaluates and routes across all 80+ tools instead of only the collapsed
hot subset.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("hermes.plugin.laya.tool_collector")

# Hermes bridge tools used for progressive disclosure
BRIDGE_TOOL_NAMES: frozenset[str] = frozenset({"tool_search", "tool_describe", "tool_call"})

# Connector/catalog management tools: using them surfaces new tools, so they also unlock the full list
DISCOVERY_TOOL_NAMES: frozenset[str] = frozenset({"manage_catalog", "manage_connections"})


def extract_tool_name(tool: Any) -> str:
    """Extract tool name from function schema dict or object."""
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
        if tool.get("name"):
            return str(tool["name"])
    return getattr(tool, "name", str(tool))


def hermes_bridge_names() -> Optional[frozenset]:
    """Bridge tool names as declared by the running Hermes (``None`` outside Hermes)."""
    try:
        from tools.tool_search_catalog import BRIDGE_TOOL_NAMES as hermes_names  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return frozenset(str(n) for n in hermes_names)
    except Exception:
        return None


def bridge_tool_names() -> frozenset:
    """Bridge tool names to protect: our known set plus anything Hermes declares today."""
    declared = hermes_bridge_names()
    return BRIDGE_TOOL_NAMES | declared if declared else BRIDGE_TOOL_NAMES


def discovery_tool_names() -> frozenset:
    """Every tool through which the model discovers or reaches hidden tools/connectors."""
    return bridge_tool_names() | DISCOVERY_TOOL_NAMES


def is_bridge_tool(tool: Any) -> bool:
    """Check if the given tool is a progressive disclosure bridge tool."""
    name = extract_tool_name(tool)
    return name in bridge_tool_names()


def _build_hermes_bridge_schemas(deferred_count: int = 1) -> List[Any]:
    """Ask Hermes itself for the canonical bridge schemas (``tools.tool_search``)."""
    try:
        from tools.tool_search import bridge_tool_schemas  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        built = bridge_tool_schemas(max(int(deferred_count), 1))
        return list(built) if isinstance(built, (list, tuple)) else []
    except Exception as exc:
        logger.debug("Hermes bridge_tool_schemas() failed: %s", exc)
        return []


def _bridge_sort_key(name: str) -> Tuple[int, str]:
    order = ("tool_search", "tool_describe", "tool_call")
    return (order.index(name) if name in order else len(order), name)


def get_bridge_tools(
    tools: Optional[List[Any]],
    fallback_sources: Optional[List[Optional[List[Any]]]] = None,
    rebuild: bool = False,
    deferred_count: int = 1,
) -> List[Any]:
    """Return bridge tool definitions (``tool_search``/``tool_describe``/``tool_call``).

    Looked up in order: ``tools``, each of ``fallback_sources`` (e.g. the session's original
    tool list or the bridge schemas remembered for the session) and — only if ``rebuild`` — the
    schemas Hermes itself builds. This keeps bridge tools available after ``agent.tools`` was
    rewritten by an earlier turn.
    """
    names = bridge_tool_names()
    found: Dict[str, Any] = {}
    for source in [tools, *(fallback_sources or [])]:
        if not isinstance(source, list):
            continue
        for t in source:
            name = extract_tool_name(t)
            if name in names and name not in found:
                found[name] = t

    if rebuild and any(n not in found for n in names):
        for t in _build_hermes_bridge_schemas(deferred_count):
            name = extract_tool_name(t)
            if name in names and name not in found:
                found[name] = t

    return [found[n] for n in sorted(found, key=_bridge_sort_key)]


def running_inside_hermes() -> bool:
    """True when the Hermes runtime (not just a test harness) hosts this plugin."""
    try:
        import importlib.util

        return importlib.util.find_spec("hermes_cli") is not None
    except Exception:
        return False


def verify_bridge_tools(
    tools: Optional[List[Any]],
    fallback_sources: Optional[List[Optional[List[Any]]]] = None,
    deferred_count: int = 1,
) -> Dict[str, Any]:
    """Check that the standard Hermes discovery tools are obtainable and report drift.

    Result keys:
      ``bridge_tools``       resolved schemas (tools -> fallbacks -> Hermes builder)
      ``unresolved``         names that could not be obtained anywhere
      ``rebuilt``            names obtained only by asking Hermes to rebuild them
      ``hermes_available``   running inside Hermes, so the bridge tools are expected to exist
      ``api_issues``         human-readable Hermes API drift (renamed/removed bridge tools)
      ``ok``                 nothing unresolved and no API drift
    """
    names = bridge_tool_names()
    without_rebuild = {extract_tool_name(t) for t in get_bridge_tools(tools, fallback_sources)}
    resolved = get_bridge_tools(tools, fallback_sources, rebuild=True, deferred_count=deferred_count)
    resolved_names = {extract_tool_name(t) for t in resolved}

    api_issues: List[str] = []
    hermes_available = running_inside_hermes()
    if hermes_available:
        declared = hermes_bridge_names()
        if declared is None:
            api_issues.append("tools.tool_search_catalog.BRIDGE_TOOL_NAMES is not importable")
        elif declared != BRIDGE_TOOL_NAMES:
            api_issues.append(
                f"Hermes bridge tools {sorted(declared)} differ from the plugin's known "
                f"{sorted(BRIDGE_TOOL_NAMES)}"
            )
        built = {extract_tool_name(t) for t in _build_hermes_bridge_schemas(deferred_count)}
        if not built:
            api_issues.append("tools.tool_search.bridge_tool_schemas() is unavailable")
        elif names - built:
            api_issues.append(f"Hermes bridge_tool_schemas() no longer builds: {sorted(names - built)}")

    unresolved = sorted(names - resolved_names)
    return {
        "bridge_tools": resolved,
        "unresolved": unresolved,
        "rebuilt": sorted(resolved_names - without_rebuild),
        "hermes_available": hermes_available,
        "api_issues": api_issues,
        "ok": not unresolved and not api_issues,
    }


def _tool_calls_of(message: Any) -> List[str]:
    """Names of tools called in (or answered by) one conversation message."""
    out: List[str] = []
    if isinstance(message, dict):
        calls = message.get("tool_calls")
        role = message.get("role")
        name = message.get("name") if role == "tool" else None
    else:
        calls = getattr(message, "tool_calls", None)
        role = getattr(message, "role", None)
        name = getattr(message, "name", None) if role == "tool" else None
    if name:
        out.append(str(name))
    for call in calls or []:
        fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
        fn_name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        if fn_name:
            out.append(str(fn_name))
    return out


def messages_used_bridge_tools(messages: Any) -> bool:
    """True if the conversation already contains a call to a discovery/bridge tool."""
    if not isinstance(messages, (list, tuple)):
        return False
    names = discovery_tool_names()
    return any(n in names for m in messages for n in _tool_calls_of(m))


def _ensure_mcp_servers_registered() -> None:
    """Best-effort check to register configured MCP servers if not already registered."""
    try:
        from pathlib import Path
        import hermes_yaml as yaml  # type: ignore[import-not-found]
        from tools.mcp_tool_discovery import register_mcp_servers  # type: ignore[import-not-found]

        cfg_path = Path("~/.hermes/config.yaml").expanduser()
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            mcp_servers = data.get("mcp_servers")
            if isinstance(mcp_servers, dict) and mcp_servers:
                register_mcp_servers(mcp_servers)
    except Exception as exc:
        logger.debug("MCP registration check non-fatal notice: %s", exc)


def collect_all_tools(
    agent: Any = None,
    extra_tools: Optional[List[Any]] = None,
    include_bridge_tools: bool = False,
) -> List[Any]:
    """Collect the complete, uncollapsed catalog of all tools available to Hermes.

    Combines:
    1. model_tools.get_tool_definitions(skip_tool_search_assembly=True)
       (includes all core, plugin, and registered MCP tools without deferral collapse)
    2. tools.registry.registry definitions for any dynamically declared tools
    3. agent.tools (runtime-injected tools like memory or turn-scoped tools)
    4. extra_tools explicitly passed by caller

    Filters out bridge tools (tool_search, tool_describe, tool_call) by default
    so that Laya routes over genuine capabilities.
    """
    tools_by_name: Dict[str, Any] = {}

    # 1. Attempt to register configured MCP servers if in Hermes environment
    _ensure_mcp_servers_registered()

    # 2. Extract uncollapsed definitions from model_tools
    enabled_toolsets = getattr(agent, "enabled_toolsets", None) if agent is not None else None
    disabled_toolsets = getattr(agent, "disabled_toolsets", None) if agent is not None else None
    quiet_mode = getattr(agent, "quiet_mode", True) if agent is not None else True

    try:
        from agent import model_tools  # type: ignore[import-not-found]
    except ImportError:
        try:
            import model_tools  # type: ignore[import-not-found]
        except ImportError:
            model_tools = None

    if model_tools is not None:
        try:
            uncollapsed = model_tools.get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=quiet_mode,
                skip_tool_search_assembly=True,
            )
            if isinstance(uncollapsed, list):
                for t in uncollapsed:
                    name = extract_tool_name(t)
                    if name:
                        tools_by_name[name] = t
        except Exception as exc:
            logger.debug("Failed to get uncollapsed definitions from model_tools: %s", exc)

    # 3. Check central registry if available
    try:
        from tools.registry import registry  # type: ignore[import-not-found]
        snapshot = registry._snapshot_entries()
        missing_names = {entry.name for entry in snapshot if entry.name not in tools_by_name}
        if missing_names:
            defs = registry.get_definitions(missing_names, quiet=True)
            for t in defs:
                name = extract_tool_name(t)
                if name:
                    tools_by_name.setdefault(name, t)
    except Exception as exc:
        logger.debug("Central registry lookup notice: %s", exc)

    # 4. Include runtime agent.tools
    if agent is not None and hasattr(agent, "tools") and isinstance(agent.tools, list):
        for t in agent.tools:
            name = extract_tool_name(t)
            if name:
                tools_by_name.setdefault(name, t)

    # 5. Include caller's extra tools
    if extra_tools and isinstance(extra_tools, list):
        for t in extra_tools:
            name = extract_tool_name(t)
            if name:
                tools_by_name.setdefault(name, t)

    # 6. Filter or keep bridge tools
    if not include_bridge_tools:
        bridge_names = bridge_tool_names()
        result = [t for name, t in tools_by_name.items() if name not in bridge_names]
    else:
        result = list(tools_by_name.values())

    return result
