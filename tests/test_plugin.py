"""Tests for Hermes plugin lifecycle hooks and commands."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
import __init__ as plugin


class FakeContext:
    def __init__(self):
        self.hooks = {}
        self.commands = {}

    def register_hook(self, name, handler):
        self.hooks[name] = handler

    def register_command(self, name, handler, **kwargs):
        self.commands[name] = handler


class FakeAgent:
    def __init__(self, session_id="s1"):
        self.session_id = session_id
        self.tools = [
            {"function": {"name": "fs_read", "description": "Read file"}},
            {"function": {"name": "bash", "description": "Run bash"}},
        ]


def test_plugin_registration(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_LOG_PATH", str(tmp_path / "test.jsonl"))
    ctx = FakeContext()
    plugin.register(ctx)

    assert "pre_llm_call" in ctx.hooks
    assert "pre_api_request" in ctx.hooks
    assert "on_session_end" in ctx.hooks
    assert "laya" in ctx.commands


def test_pre_api_request_fallback(tmp_path, monkeypatch):
    log_path = tmp_path / "test_api_req.jsonl"
    monkeypatch.setenv("LAYA_MODE", "shadow")
    monkeypatch.setenv("LAYA_LOG_PATH", str(log_path))
    ctx = FakeContext()
    plugin.register(ctx)

    tools = [
        {"function": {"name": "read_file", "description": "Read file"}},
        {"function": {"name": "write_file", "description": "Write file"}},
    ]

    with patch.object(
        plugin._client,
        "ask",
        return_value={
            "ok": True,
            "latency_ms": 12.0,
            "answers": {
                "tool_decision": {"choice": "execute_tools"},
                "category": {"choice": "filesystem"},
            },
        },
    ):
        ctx.hooks["pre_api_request"](
            session_id="sess_fallback",
            turn_id="turn_fallback",
            user_message="Please read the file test.txt",
            request={"body": {"tools": tools}},
        )

    assert log_path.is_file()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "sess_fallback" in lines[0]


def test_pre_llm_call_active_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_MODE", "active")
    monkeypatch.setenv("LAYA_LOG_PATH", str(tmp_path / "test.jsonl"))
    ctx = FakeContext()
    plugin.register(ctx)

    agent = FakeAgent()

    # Mock Laya to return direct_chat
    with patch.object(
        plugin._client,
        "ask",
        return_value={
            "ok": True,
            "latency_ms": 10.0,
            "answers": {
                "tool_decision": {"choice": "direct_chat"},
                "category": {"choice": "none"},
            },
        },
    ):
        ctx.hooks["pre_llm_call"](
            session_id="s1",
            agent=agent,
            user_message="Hello!",
        )

        # In active mode, tools should be stripped!
        assert agent.tools == []


def test_pre_llm_call_shadow_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_MODE", "shadow")
    monkeypatch.setenv("LAYA_LOG_PATH", str(tmp_path / "test_shadow.jsonl"))
    ctx = FakeContext()
    plugin.register(ctx)

    agent = FakeAgent()
    orig_tools = list(agent.tools)

    with patch.object(
        plugin._client,
        "ask",
        return_value={
            "ok": True,
            "latency_ms": 10.0,
            "answers": {
                "tool_decision": {"choice": "direct_chat"},
                "category": {"choice": "none"},
            },
        },
    ):
        res = ctx.hooks["pre_llm_call"](
            session_id="s1",
            agent=agent,
            user_message="Hello!",
        )

        # In shadow mode, tools MUST remain exactly as original!
        assert agent.tools == orig_tools
        # Return value must be None (no prompt/context modification)
        assert res is None


def test_laya_command_switching(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_MODE", "active")
    monkeypatch.setenv("LAYA_LOG_PATH", str(tmp_path / "test.jsonl"))
    ctx = FakeContext()
    plugin.register(ctx)

    assert plugin._config.is_shadow is False

    # Switch to shadow
    resp = ctx.commands["laya"]("shadow")
    assert "SHADOW" in resp
    assert plugin._config.is_shadow is True

    # Switch back to active
    resp = ctx.commands["laya"]("active")
    assert "ACTIVE" in resp
    assert plugin._config.is_shadow is False

    # Status
    with patch.object(plugin._client, "health", return_value={"ok": True}):
        resp = ctx.commands["laya"]("status")
        assert "online" in resp
