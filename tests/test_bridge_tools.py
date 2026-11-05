"""Tests for Hermes progressive tool disclosure support.

Hermes hides deferred/MCP/plugin tools behind the bridge tools ``tool_search`` / ``tool_describe`` /
``tool_call``. The plugin must always keep them, verify they exist, keep ``agent.valid_tool_names``
in sync with ``agent.tools`` and open the full tool list once the model used discovery tools.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
import __init__ as plugin
import router as router_mod
import tool_collector
from categorizer import ToolCategorizer, extract_tool_name
from config import LayaConfig
from router import ToolRouter
from tool_collector import (
    BRIDGE_TOOL_NAMES,
    get_bridge_tools,
    messages_used_bridge_tools,
    verify_bridge_tools,
)


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": f"Description for {name}"}}


BRIDGE = ["tool_search", "tool_describe", "tool_call"]
CORE = ["read_file", "terminal", "web_search"]
DEFERRED = ["todo_list", "mcp__marketplace__compare_sources", "gui_tour"]


def _names(tools) -> list:
    return [extract_tool_name(t) for t in tools]


def _hermes_tools() -> list:
    """What Hermes hands the agent: core tools + the 3 bridge tools (deferred ones are hidden)."""
    return [_tool(n) for n in CORE + BRIDGE]


def _fake_collect(agent=None, extra_tools=None, include_bridge_tools=False):
    """Stand-in for the uncollapsed catalog: core + deferred (+ bridge on request)."""
    tools = [_tool(n) for n in CORE + DEFERRED] + list(extra_tools or [])
    seen, out = set(), []
    for t in tools:
        n = extract_tool_name(t)
        if n in seen or (n in BRIDGE_TOOL_NAMES and not include_bridge_tools):
            continue
        seen.add(n)
        out.append(t)
    if include_bridge_tools:
        for n in BRIDGE:
            if n not in seen:
                out.append(_tool(n))
    return out


class StubLaya:
    """Laya stand-in: prefers the tools in ``prefer`` and (optionally) answers direct_chat."""

    def __init__(self, prefer=None, direct_chat=False, category="general"):
        self.prefer = prefer or {}
        self.direct_chat = direct_chat
        self.category = category
        self.calls: list = []

    def ask(self, state_text, questions):
        self.calls.append(questions)
        answers = {}
        for q_id, q in questions.items():
            criteria = q.get("criteria", {})
            if q_id == "tool_decision":
                if self.direct_chat:
                    answers[q_id] = {
                        "choice": "direct_chat",
                        "confidence": 0.9,
                        "probabilities": {"direct_chat": 0.95, "execute_tools": 0.05},
                    }
                else:
                    answers[q_id] = {"choice": "execute_tools", "probabilities": {"execute_tools": 0.9, "direct_chat": 0.1}}
            elif q_id == "category":
                cat = self.category if self.category in criteria else next(iter(criteria))
                answers[q_id] = {"choice": cat, "probabilities": {cat: 0.9}}
            else:
                probs = {name: self.prefer.get(name, 0.01) for name in criteria}
                answers[q_id] = {"choice": max(probs, key=probs.get), "probabilities": probs}
        return {"answers": answers, "latency_ms": 1.0}

    def health(self):
        return {"ok": True}


def _router(tmp_path, laya, mode="active", hierarchical=False, max_tools=2, **cfg):
    config = LayaConfig(
        mode=mode,
        hierarchical=hierarchical,
        max_tools=max_tools,
        log_path=tmp_path / "audit.jsonl",
        cache_path=tmp_path / "cats.json",
        **cfg,
    )
    categorizer = ToolCategorizer(laya_client=laya, cache_path=tmp_path / "cats.json")
    return ToolRouter(laya_client=laya, config=config, categorizer=categorizer), config


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list = []

    def emit(self, record):
        self.records.append(record)


def _capture(name="hermes.plugin.laya.router"):
    handler = _LogCapture()
    lg = logging.getLogger(name)
    lg.addHandler(handler)
    return lg, handler


def _inside_hermes(builder=None, declared=None):
    """Patches simulating a running Hermes whose bridge API answers like ``builder``/``declared``."""
    built = builder if builder is not None else [_tool(n) for n in BRIDGE]
    names = frozenset(declared) if declared is not None else frozenset(BRIDGE)
    return (
        patch.object(tool_collector, "running_inside_hermes", lambda: True),
        patch.object(tool_collector, "hermes_bridge_names", lambda: names),
        patch.object(tool_collector, "_build_hermes_bridge_schemas", lambda deferred_count=1: list(built)),
    )


# ------------------------------------------------------------------ trivial phrases in the router


def test_router_trivial_bypass_triggers_for_real_user_prefix(tmp_path):
    laya = StubLaya()
    router, _ = _router(tmp_path, laya, hierarchical=True)
    tools = _hermes_tools()

    for phrase in ("ок", "спасибо", "Да, спасибо"):
        effective, dec = router.route_tools(tools, f"USER: {phrase}")
        assert dec["decision"] == "direct_chat" and dec["trivial_bypass"] is True, phrase
        assert sorted(_names(effective)) == sorted(BRIDGE), phrase
    assert laya.calls == []


def test_router_trivial_bypass_with_hermes_context_dict(tmp_path):
    laya = StubLaya()
    router, _ = _router(tmp_path, laya, hierarchical=True)
    context = {
        "user_message": "ок",
        "conversation_history": [
            {"role": "user", "content": "сделай бекап базы"},
            {"role": "assistant", "content": "Готово."},
            {"role": "user", "content": "ок"},
        ],
    }
    _, dec = router.route_tools(_hermes_tools(), context)
    assert dec["decision"] == "direct_chat" and dec["trivial_bypass"] is True
    assert laya.calls == []


# ------------------------------------------------------------------ bridge tools are always kept


def test_bridge_tools_added_in_flat_mode(tmp_path):
    laya = StubLaya(prefer={"todo_list": 0.9, "gui_tour": 0.5})
    router, _ = _router(tmp_path, laya, hierarchical=False, max_tools=2)
    tools = _hermes_tools()

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        effective, dec = router.route_tools(tools, "USER: составь список дел и покажи тур по интерфейсу")

    names = _names(effective)
    assert set(BRIDGE) <= set(names)
    assert len([n for n in names if n not in BRIDGE]) == 2  # bridge tools don't eat Laya's slots
    assert dec["selected_tools"][0] == "todo_list"
    assert set(BRIDGE) <= set(dec["exposed_tools"])


def test_bridge_tools_added_in_hierarchical_direct_chat(tmp_path):
    laya = StubLaya(direct_chat=True)
    router, _ = _router(tmp_path, laya, hierarchical=True, max_tools=2)

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        effective, dec = router.route_tools(_hermes_tools(), "USER: объясни теорию относительности простыми словами")

    assert dec["decision"] == "direct_chat"
    assert sorted(_names(effective)) == sorted(BRIDGE)


def test_bridge_tools_added_in_enrich_mode(tmp_path):
    laya = StubLaya(prefer={"todo_list": 0.9})
    router, _ = _router(tmp_path, laya, mode="enrich_top3", hierarchical=False)
    # standard list lost the bridge tools (e.g. rewritten on an earlier turn)
    base = [_tool(n) for n in CORE]

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        effective, _ = router.route_tools(
            base, "USER: составь список дел", original_tools=_hermes_tools()
        )
    assert set(BRIDGE) <= set(_names(effective))
    assert set(CORE) <= set(_names(effective))


def test_bridge_tools_restored_from_original_list_after_rewrite(tmp_path):
    """Turn 2 sees an agent.tools that an earlier turn rewrote without bridge tools."""
    laya = StubLaya(prefer={"todo_list": 0.9})
    router, _ = _router(tmp_path, laya, hierarchical=False, max_tools=2)
    rewritten = [_tool("todo_list")]  # no bridge tools

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        effective, _ = router.route_tools(
            rewritten, "USER: составь список дел", original_tools=_hermes_tools()
        )
    assert set(BRIDGE) <= set(_names(effective))


def test_bridge_tools_remembered_per_session(tmp_path):
    laya = StubLaya(prefer={"todo_list": 0.9})
    router, _ = _router(tmp_path, laya, hierarchical=False, max_tools=2)

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        router.route_tools(_hermes_tools(), "USER: составь список дел", session_id="sA")
        # next turn: neither current tools nor an original list carry them
        effective, _ = router.route_tools([_tool("todo_list")], "USER: ещё список", session_id="sA")
        other, _ = router.route_tools([_tool("todo_list")], "USER: ещё список", session_id="sB")
    assert set(BRIDGE) <= set(_names(effective))
    assert not set(BRIDGE) & set(_names(other))  # sessions don't leak into each other


def test_shadow_mode_never_touches_tools(tmp_path):
    laya = StubLaya(prefer={"todo_list": 0.9})
    router, _ = _router(tmp_path, laya, mode="shadow", hierarchical=False)
    tools = [_tool(n) for n in CORE]  # no bridge tools at all

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        effective, _ = router.route_tools(tools, "USER: составь список дел", original_tools=_hermes_tools())
    assert effective is tools


# ------------------------------------------------------------------ presence check


def test_verify_bridge_tools_outside_hermes():
    status = verify_bridge_tools([_tool("read_file")])
    assert status["unresolved"] == sorted(BRIDGE)
    assert status["hermes_available"] is False
    assert status["api_issues"] == []


def test_verify_bridge_tools_ok_and_rebuilt_from_hermes():
    p1, p2, p3 = _inside_hermes()
    with p1, p2, p3:
        present = verify_bridge_tools(_hermes_tools())
        rebuilt = verify_bridge_tools([_tool("read_file")])
    assert present["ok"] is True and present["rebuilt"] == []
    assert rebuilt["ok"] is True and rebuilt["rebuilt"] == sorted(BRIDGE)


def test_verify_bridge_tools_detects_renamed_tools():
    # Hermes added a 4th bridge tool the plugin has not seen and no longer builds a known one
    p1, p2, p3 = _inside_hermes(
        builder=[_tool("tool_search"), _tool("tool_describe"), _tool("tool_invoke")],
        declared={"tool_search", "tool_describe", "tool_invoke"},
    )
    with p1, p2, p3:
        status = verify_bridge_tools([_tool("read_file")])
    assert status["ok"] is False
    assert any("differ" in i for i in status["api_issues"])
    assert "tool_call" in status["unresolved"]


def test_missing_bridge_tools_inside_hermes_logs_error_and_fails_open(tmp_path):
    laya = StubLaya(prefer={"todo_list": 0.9})
    router, _ = _router(tmp_path, laya, hierarchical=False, max_tools=2)
    tools = [_tool(n) for n in CORE]  # Hermes did not provide the bridge tools
    lg, handler = _capture()
    p1, p2, p3 = _inside_hermes(builder=[])  # ... and cannot rebuild them either
    try:
        with p1, p2, p3, patch.object(router_mod, "collect_all_tools", _fake_collect):
            effective, dec = router.route_tools(tools, "USER: составь список дел", session_id="s_missing")
            router.route_tools(tools, "USER: ещё", session_id="s_missing")
    finally:
        lg.removeHandler(handler)

    errors = [r for r in handler.records if r.levelno == logging.ERROR]
    assert len(errors) == 1  # reported once per session
    assert "bridge" in errors[0].getMessage().lower()
    # fail-open: nothing hidden from the model
    assert dec["decision"] == "bridge_unlocked" and dec["fallback_reason"] == "bridge_tools_missing"
    assert set(CORE + DEFERRED) <= set(_names(effective))
    assert laya.calls == []

    audit = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["decision"] == "bridge_tools_missing" for e in audit)
    assert router.bridge_status["unresolved"] == sorted(BRIDGE)


def test_missing_bridge_tools_outside_hermes_is_not_an_error(tmp_path):
    laya = StubLaya(prefer={"todo_list": 0.9})
    router, _ = _router(tmp_path, laya, hierarchical=False, max_tools=2)
    lg, handler = _capture()
    try:
        with patch.object(router_mod, "collect_all_tools", _fake_collect):
            effective, dec = router.route_tools([_tool(n) for n in CORE], "USER: составь список дел")
    finally:
        lg.removeHandler(handler)
    assert not [r for r in handler.records if r.levelno >= logging.ERROR]
    assert dec["decision"] == "execute_tools"
    assert not set(BRIDGE) & set(_names(effective))


# ------------------------------------------------------------------ unlocking the full list


def test_messages_used_bridge_tools():
    assert messages_used_bridge_tools(
        [{"role": "assistant", "tool_calls": [{"function": {"name": "tool_search", "arguments": "{}"}}]}]
    )
    assert messages_used_bridge_tools([{"role": "tool", "name": "tool_describe", "content": "{}"}])
    assert messages_used_bridge_tools(
        [{"role": "assistant", "tool_calls": [{"function": {"name": "manage_connections"}}]}]
    )
    assert not messages_used_bridge_tools(
        [{"role": "assistant", "tool_calls": [{"function": {"name": "terminal"}}]}, {"role": "user", "content": "hi"}]
    )
    assert not messages_used_bridge_tools(None)
    assert not messages_used_bridge_tools([])


def test_history_with_bridge_call_unlocks_full_list(tmp_path):
    laya = StubLaya(prefer={"todo_list": 0.9})
    router, _ = _router(tmp_path, laya, hierarchical=False, max_tools=2)
    context = {
        "user_message": "а теперь проверь терминал",
        "conversation_history": [
            {"role": "user", "content": "найди инструмент для задач"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "tool_search", "arguments": "{}"}}]},
            {"role": "tool", "name": "tool_search", "content": "{}"},
            {"role": "user", "content": "а теперь проверь терминал"},
        ],
    }
    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        effective, dec = router.route_tools(_hermes_tools(), context, session_id="s_hist")
        # ... and it sticks for the following turns of the session
        effective2, dec2 = router.route_tools(_hermes_tools(), "USER: ещё вопрос", session_id="s_hist")

    assert dec["decision"] == "bridge_unlocked" and dec["fallback_reason"] == "bridge_tool_called"
    assert set(CORE + DEFERRED + BRIDGE) <= set(_names(effective))
    assert dec2["decision"] == "bridge_unlocked"
    assert set(CORE + DEFERRED + BRIDGE) <= set(_names(effective2))
    assert laya.calls == []

    router.forget_session("s_hist")
    assert router.is_unlocked("s_hist") is False


def test_get_bridge_tools_prefers_current_list_then_fallbacks():
    current = [_tool("tool_search")]
    original = [_tool("tool_search"), _tool("tool_describe"), _tool("tool_call")]
    got = get_bridge_tools(current, fallback_sources=[original])
    assert _names(got) == BRIDGE
    assert got[0] is current[0]
    assert get_bridge_tools([_tool("read_file")]) == []


# ------------------------------------------------------------------ plugin: valid_tool_names + mid-turn unlock


class FakeHermesAgent:
    """Mimics the parts of the Hermes agent the plugin touches (tools + valid_tool_names)."""

    def __init__(self, session_id):
        self.session_id = session_id
        self.tools = _hermes_tools()
        self.valid_tool_names = set(_names(self.tools))


class FakeContext:
    def __init__(self):
        self.hooks = {}
        self.commands = {}

    def register_hook(self, name, handler):
        self.hooks[name] = handler

    def register_command(self, name, handler, **kwargs):
        self.commands[name] = handler


def _setup_plugin(tmp_path, monkeypatch, mode="active", session_id="s_plugin", prefer=None, direct_chat=False):
    monkeypatch.setenv("LAYA_MODE", mode)
    monkeypatch.setenv("LAYA_HIERARCHICAL", "false")
    monkeypatch.setenv("LAYA_MAX_TOOLS", "2")
    monkeypatch.setenv("LAYA_LOG_PATH", str(tmp_path / "plugin.jsonl"))
    monkeypatch.setenv("LAYA_CACHE_PATH", str(tmp_path / "cats.json"))
    ctx = FakeContext()
    plugin.register(ctx)
    plugin._saved_agent_tools.pop(session_id, None)
    stub = StubLaya(prefer=prefer or {"todo_list": 0.9, "gui_tour": 0.5}, direct_chat=direct_chat)
    plugin._client.ask = stub.ask
    return ctx, stub


def test_valid_tool_names_follow_agent_tools(tmp_path, monkeypatch):
    """Reproduces the Hermes session where Laya-selected tools were rejected as 'does not exist'."""
    ctx, _ = _setup_plugin(tmp_path, monkeypatch, session_id="s_valid")
    agent = FakeHermesAgent("s_valid")
    original_valid = set(agent.valid_tool_names)

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        ctx.hooks["pre_llm_call"](
            session_id="s_valid", turn_id="t1", agent=agent, user_message="составь список дел и покажи тур"
        )

    exposed = set(_names(agent.tools))
    assert "todo_list" in exposed and set(BRIDGE) <= exposed
    assert exposed <= agent.valid_tool_names  # everything shown to the model is valid for Hermes
    assert original_valid <= agent.valid_tool_names  # nothing that was valid becomes invalid


def test_bridge_tools_survive_direct_chat_turn(tmp_path, monkeypatch):
    ctx, _ = _setup_plugin(tmp_path, monkeypatch, session_id="s_direct")
    agent = FakeHermesAgent("s_direct")
    plugin._router.config.hierarchical = True
    plugin._client.ask = StubLaya(direct_chat=True).ask
    monkeypatch.setenv("LAYA_HIERARCHICAL", "true")

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        ctx.hooks["pre_llm_call"](
            session_id="s_direct", turn_id="t1", agent=agent, user_message="объясни теорию относительности"
        )
        assert sorted(_names(agent.tools)) == sorted(BRIDGE)  # was [] before the fix
        ctx.hooks["pre_llm_call"](
            session_id="s_direct", turn_id="t2", agent=agent, user_message="объясни теорию относительности ещё раз"
        )
    assert set(BRIDGE) <= set(_names(agent.tools))


def test_discovery_tool_call_opens_full_list_mid_turn(tmp_path, monkeypatch):
    ctx, _ = _setup_plugin(tmp_path, monkeypatch, session_id="s_unlock")
    agent = FakeHermesAgent("s_unlock")

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        ctx.hooks["pre_llm_call"](session_id="s_unlock", turn_id="t1", agent=agent, user_message="составь список дел")
        narrowed = len(agent.tools)
        assert "mcp__marketplace__compare_sources" not in _names(agent.tools)

        # Iteration 1 ends: the model called tool_search. pre_llm_call does not fire again within the
        # turn, so post_tool_call must widen agent.tools for the next LLM request.
        ctx.hooks["post_tool_call"](tool_name="tool_search", args={"queries": ["terminal"]}, session_id="s_unlock", turn_id="t1")

    full = set(CORE + DEFERRED + BRIDGE)
    assert full <= set(_names(agent.tools))
    assert len(agent.tools) > narrowed
    assert full <= agent.valid_tool_names
    assert plugin._router.is_unlocked("s_unlock")

    # Next user turn: still the full list, no Laya call
    stub = StubLaya()
    plugin._client.ask = stub.ask
    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        ctx.hooks["pre_llm_call"](session_id="s_unlock", turn_id="t2", agent=agent, user_message="проверь терминал")
    assert full <= set(_names(agent.tools))
    assert stub.calls == []

    ctx.hooks["on_session_end"](session_id="s_unlock")
    assert plugin._router.is_unlocked("s_unlock") is False


def test_hidden_tool_reached_through_tool_call_opens_full_list(tmp_path, monkeypatch):
    """Hermes unwraps tool_call: post_tool_call sees the target's own name, not 'tool_call'."""
    ctx, _ = _setup_plugin(tmp_path, monkeypatch, session_id="s_hidden")
    agent = FakeHermesAgent("s_hidden")

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        ctx.hooks["pre_llm_call"](session_id="s_hidden", turn_id="t1", agent=agent, user_message="составь список дел")
        exposed_before = set(_names(agent.tools))
        assert "mcp__marketplace__compare_sources" not in exposed_before

        # a tool that is exposed: no unlock
        ctx.hooks["post_tool_call"](tool_name="todo_list", session_id="s_hidden", turn_id="t1")
        assert plugin._router.is_unlocked("s_hidden") is False

        ctx.hooks["post_tool_call"](tool_name="mcp__marketplace__compare_sources", session_id="s_hidden", turn_id="t1")
    assert plugin._router.is_unlocked("s_hidden") is True
    assert set(CORE + DEFERRED + BRIDGE) <= set(_names(agent.tools))


def test_shadow_mode_does_not_unlock_or_mutate(tmp_path, monkeypatch):
    ctx, _ = _setup_plugin(tmp_path, monkeypatch, mode="shadow", session_id="s_shadow")
    agent = FakeHermesAgent("s_shadow")
    tools_before, valid_before = list(agent.tools), set(agent.valid_tool_names)

    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        res = ctx.hooks["pre_llm_call"](session_id="s_shadow", turn_id="t1", agent=agent, user_message="составь список дел")
        ctx.hooks["post_tool_call"](tool_name="tool_search", session_id="s_shadow", turn_id="t1")

    assert res is None
    assert agent.tools == tools_before
    assert agent.valid_tool_names == valid_before
    assert plugin._router.is_unlocked("s_shadow") is False


def test_laya_status_reports_bridge_state(tmp_path, monkeypatch):
    ctx, _ = _setup_plugin(tmp_path, monkeypatch, session_id="s_status")
    agent = FakeHermesAgent("s_status")
    with patch.object(router_mod, "collect_all_tools", _fake_collect):
        ctx.hooks["pre_llm_call"](session_id="s_status", turn_id="t1", agent=agent, user_message="составь список дел")
    with patch.object(plugin._client, "health", return_value={"ok": True}):
        outside = ctx.commands["laya"]("status")
    assert "Hermes bridge tools: n/a" in outside

    p1, p2, p3 = _inside_hermes()
    with p1, p2, p3, patch.object(plugin._client, "health", return_value={"ok": True}):
        agent2 = FakeHermesAgent("s_status2")
        agent2.tools = [_tool(n) for n in CORE]
        agent2.valid_tool_names = set(CORE)
        plugin._saved_agent_tools.pop("s_status2", None)
        with patch.object(router_mod, "collect_all_tools", _fake_collect):
            ctx.hooks["pre_llm_call"](session_id="s_status2", turn_id="t1", agent=agent2, user_message="составь список дел")
        inside = ctx.commands["laya"]("status")
    assert "Hermes bridge tools: OK" in inside
    assert set(BRIDGE) <= set(_names(agent2.tools))  # rebuilt from Hermes and exposed
    assert set(BRIDGE) <= agent2.valid_tool_names
