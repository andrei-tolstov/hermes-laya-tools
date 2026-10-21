"""Tests for ToolRouter including Shadow Mode and Level 1/2 routing."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from categorizer import ToolCategorizer, extract_tool_name
from config import LayaConfig
from logger import LayaAuditLogger
from router import ToolRouter


def _sample_tools():
    return [
        {"function": {"name": "fs_read", "description": "Read file"}},
        {"function": {"name": "fs_write", "description": "Write file"}},
        {"function": {"name": "fs_list", "description": "List dir"}},
        {"function": {"name": "bash", "description": "Run bash"}},
        {"function": {"name": "web_search", "description": "Search web"}},
    ]


def test_router_direct_chat_active_mode(tmp_path):
    mock_laya = MagicMock()
    # Level 1 returns direct_chat
    mock_laya.ask.return_value = {
        "ok": True,
        "latency_ms": 30.0,
        "answers": {
            "tool_decision": {"choice": "direct_chat", "probabilities": {"direct_chat": 0.99}},
            "category": {"choice": "none"},
        },
    }

    cfg = LayaConfig(mode="active", log_path=tmp_path / "audit.jsonl")
    router = ToolRouter(mock_laya, cfg)

    tools = _sample_tools()
    effective, dec = router.route_tools(tools, "USER: Привет, как дела?")

    assert dec["decision"] == "direct_chat"
    assert effective == []  # Stripped in active mode!
    assert dec["selected_tools"] == []


def test_router_direct_chat_shadow_mode(tmp_path):
    mock_laya = MagicMock()
    mock_laya.ask.return_value = {
        "ok": True,
        "latency_ms": 25.0,
        "answers": {
            "tool_decision": {"choice": "direct_chat", "probabilities": {"direct_chat": 0.97}},
            "category": {"choice": "none"},
        },
    }

    log_file = tmp_path / "audit.jsonl"
    cfg = LayaConfig(mode="shadow", log_path=log_file)
    router = ToolRouter(mock_laya, cfg)

    tools = _sample_tools()
    effective, dec = router.route_tools(tools, "USER: Привет, как дела?")

    assert dec["decision"] == "direct_chat"
    assert dec["shadow"] is True
    # CRITICAL REQUIREMENT: In shadow mode, output toolset MUST NOT BE ALTERED!
    assert effective == tools
    assert len(effective) == 5

    # Check that audit log recorded simulated stripping
    content = log_file.read_text(encoding="utf-8")
    assert '"shadow": true' in content
    assert '"decision": "direct_chat"' in content


def test_router_execute_tools_active_mode(tmp_path):
    mock_laya = MagicMock()
    mock_laya.ask.return_value = {
        "ok": True,
        "latency_ms": 40.0,
        "answers": {
            "tool_decision": {"choice": "execute_tools"},
            "category": {
                "choice": "filesystem",
                "probabilities": {"filesystem": 0.95, "terminal_execution": 0.05},
            },
        },
    }

    cfg = LayaConfig(mode="active", max_tools=3, log_path=tmp_path / "audit.jsonl")
    router = ToolRouter(mock_laya, cfg)

    tools = _sample_tools()
    effective, dec = router.route_tools(tools, "USER: read file main.py")

    assert dec["decision"] == "execute_tools"
    assert dec["category"] == "filesystem"
    # Candidate pool had fs_read, fs_write, fs_list (3 tools <= max_tools 3)
    names = [extract_tool_name(t) for t in effective]
    assert "fs_read" in names
    assert "fs_write" in names
    assert "fs_list" in names
    assert "bash" not in names


def test_router_execute_tools_shadow_mode(tmp_path):
    mock_laya = MagicMock()
    mock_laya.ask.return_value = {
        "ok": True,
        "latency_ms": 35.0,
        "answers": {
            "tool_decision": {"choice": "execute_tools"},
            "category": {
                "choice": "filesystem",
                "probabilities": {"filesystem": 0.92},
            },
        },
    }

    log_file = tmp_path / "audit_shadow.jsonl"
    cfg = LayaConfig(mode="shadow", max_tools=3, log_path=log_file)
    router = ToolRouter(mock_laya, cfg)

    tools = _sample_tools()
    effective, dec = router.route_tools(tools, "USER: read file main.py")

    assert dec["decision"] == "execute_tools"
    assert dec["category"] == "filesystem"
    assert dec["shadow"] is True

    # In shadow mode, the returned tool list is the FULL ORIGINAL list!
    assert effective == tools
    assert len(effective) == 5

    # Simulated selection was computed and logged
    assert "simulated_selected_tools" in dec
    assert "fs_read" in dec["simulated_selected_tools"]


def test_router_fallback_on_error(tmp_path):
    mock_laya = MagicMock()
    mock_laya.ask.side_effect = Exception("Laya sidecar timeout")

    # Pass-all fallback in active mode
    cfg_pass = LayaConfig(mode="active", fallback_mode="pass-all", log_path=tmp_path / "pass.jsonl")
    router_pass = ToolRouter(mock_laya, cfg_pass)

    tools = _sample_tools()
    effective, dec = router_pass.route_tools(tools, "USER: run command")
    assert dec["decision"] == "fallback"
    assert effective == tools

    # Strip-all fallback in active mode
    cfg_strip = LayaConfig(mode="active", fallback_mode="strip-all", log_path=tmp_path / "strip.jsonl")
    router_strip = ToolRouter(mock_laya, cfg_strip)
    effective_strip, dec_strip = router_strip.route_tools(tools, "USER: run command")
    assert dec_strip["decision"] == "fallback"
    assert effective_strip == []

    # Strip-all fallback in SHADOW mode: MUST STILL KEEP TOOLS!
    cfg_shadow = LayaConfig(mode="shadow", fallback_mode="strip-all", log_path=tmp_path / "shadow_strip.jsonl")
    router_shadow = ToolRouter(mock_laya, cfg_shadow)
    effective_shadow, dec_shadow = router_shadow.route_tools(tools, "USER: run command")
    assert dec_shadow["decision"] == "fallback"
    assert effective_shadow == tools


def test_router_direct_chat_threshold_safety(tmp_path):
    """Verify that weak confidence (e.g. 53% direct_chat, 0.002 conf from real logs) does NOT strip tools."""
    mock_laya = MagicMock()
    mock_laya.ask.return_value = {
        "ok": True,
        "latency_ms": 25.0,
        "answers": {
            "tool_decision": {
                "choice": "direct_chat",
                "confidence": 0.0027,
                "probabilities": {"direct_chat": 0.5308, "execute_tools": 0.4692},
            },
            "category": {
                "choice": "filesystem",
                "probabilities": {"filesystem": 0.35, "terminal_execution": 0.30},
            },
        },
    }

    cfg = LayaConfig(
        mode="active",
        direct_chat_threshold=0.75,
        min_confidence=0.15,
        log_path=tmp_path / "safety.jsonl",
    )
    router = ToolRouter(mock_laya, cfg)
    tools = _sample_tools()
    # User prompt: "запусти вручную"
    effective, dec = router.route_tools(tools, "USER: запусти вручную")

    # Due to safety threshold, weak direct_chat is rejected; tools are preserved and routed to category!
    assert dec["decision"] == "execute_tools"
    assert dec["category"] == "filesystem"
    assert len(effective) > 0


def test_router_always_include(tmp_path):
    """Verify that tools in always_include are always preserved."""
    mock_laya = MagicMock()
    mock_laya.ask.return_value = {
        "ok": True,
        "latency_ms": 30.0,
        "answers": {
            "tool_decision": {"choice": "execute_tools"},
            "category": {
                "choice": "filesystem",
                "probabilities": {"filesystem": 0.90},
            },
        },
    }

    cfg = LayaConfig(
        mode="active",
        max_tools=2,
        always_include={"web_search"},
        log_path=tmp_path / "always.jsonl",
    )
    router = ToolRouter(mock_laya, cfg)
    tools = _sample_tools()
    effective, dec = router.route_tools(tools, "USER: read file")

    names = [extract_tool_name(t) for t in effective]
    assert "web_search" in names
    assert len(effective) <= 2

