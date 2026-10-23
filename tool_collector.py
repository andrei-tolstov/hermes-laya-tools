"""Tool Collector for Hermes Laya Tools.

Extracts the full catalog of available tools (hot + deferred/catalog/MCP tools),
ensuring Laya evaluates and routes across all 80+ tools instead of only the collapsed
hot subset.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("hermes.plugin.laya.tool_collector")

# Hermes bridge tools used for progressive disclosure
BRIDGE_TOOL_NAMES: frozenset[str] = frozenset({"tool_search", "tool_describe", "tool_call"})


def extract_tool_name(tool: Any) -> str:
    """Extract tool name from function schema dict or object."""
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
        if tool.get("name"):
            return str(tool["name"])
    return getattr(tool, "name", str(tool))


def is_bridge_tool(tool: Any) -> bool:
    """Check if the given tool is a progressive disclosure bridge tool."""
    name = extract_tool_name(tool)
    return name in BRIDGE_TOOL_NAMES


def get_bridge_tools(tools: List[Any]) -> List[Any]:
    """Extract bridge tool definitions from a tool list."""
    if not isinstance(tools, list):
        return []
    return [t for t in tools if is_bridge_tool(t)]


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
        result = [t for name, t in tools_by_name.items() if name not in BRIDGE_TOOL_NAMES]
    else:
        result = list(tools_by_name.values())

    return result
