"""Tests for the "Добавление топ 3" (enrich_top3) mode: standard tool list + Laya's top 3."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
import __init__ as plugin
import router as router_mod
from categorizer import ToolCategorizer, extract_tool_name
from config import ENRICH_TOP3_MODE, LayaConfig
from router import ToolRouter


def _tool(name: str, desc: str = "") -> dict:
    return {"type": "function", "function": {"name": name, "description": desc or f"Description for {name}"}}


STANDARD_NAMES = ["read_file", "terminal", "web_search"]
MARKET_NAMES = [
    "mcp__marketplace__ozon_search",
    "mcp__marketplace__ozon_card",
    "mcp__marketplace__ozon_reviews",
    "mcp__marketplace__wb_search",
    "mcp__marketplace__wb_card",
]
# Laya's preference inside the marketplace category.
MARKET_SCORES = {
    "mcp__marketplace__ozon_search": 0.60,
    "mcp__marketplace__ozon_card": 0.25,
    "mcp__marketplace__ozon_reviews": 0.10,
    "mcp__marketplace__wb_search": 0.03,
    "mcp__marketplace__wb_card": 0.02,
}
TOP3 = ["mcp__marketplace__ozon_search", "mcp__marketplace__ozon_card", "mcp__marketplace__ozon_reviews"]


def _standard() -> list:
    return [_tool(n) for n in STANDARD_NAMES]


def _catalog(standard: list) -> list:
    return list(standard) + [_tool(n) for n in MARKET_NAMES]


class FakeLaya:
    """Answers level-1 category questions and level-2 tool ranking questions, recording every call."""

    def __init__(self, category: str = "marketplace_and_commerce"):
        self.category = category
        self.calls: list[dict] = []

    def ask(self, state_text, questions):
        self.calls.append(questions)
        answers = {}
        for q_id, q in questions.items():
            criteria = q.get("criteria", {})
            if q_id == "category":
                answers[q_id] = {"choice": self.category, "probabilities": {self.category: 0.9}}
            elif q_id == "tool_decision":
                answers[q_id] = {"choice": "execute_tools", "probabilities": {"execute_tools": 0.9, "direct_chat": 0.1}}
            else:
                probs = {name: MARKET_SCORES.get(name, 0.01) for name in criteria}
                answers[q_id] = {"choice": max(probs, key=probs.get), "probabilities": probs}
        return {"answers": answers, "latency_ms": 1.0}


def _router(tmp_path, laya, hierarchical=True, **cfg_kwargs):
    cfg = LayaConfig(
        mode=ENRICH_TOP3_MODE,
        hierarchical=hierarchical,
        log_path=tmp_path / "audit.jsonl",
        cache_path=tmp_path / "cats.json",
        **cfg_kwargs,
    )
    categorizer = ToolCategorizer(laya_client=laya, cache_path=tmp_path / "cats.json")
    return ToolRouter(laya_client=laya, config=cfg, categorizer=categorizer), cfg


def _agent(tools):
    agent = MagicMock()
    agent.tools = tools
    return agent


# --------------------------------------------------------------------------- config


class _Ctx:
    def __init__(self, settings):
        self._settings = settings

    def get_config(self, key, default=None):
        return self._settings.get(key, default)


def test_mode_name_from_ui_is_parsed():
    cfg = LayaConfig.from_ctx_or_env(ctx=_Ctx({"mode": "Добавление топ 3"}))
    assert cfg.mode == ENRICH_TOP3_MODE
    assert cfg.is_enrich_top3 is True
    assert cfg.is_shadow is False


def test_mode_aliases_from_env(monkeypatch):
    monkeypatch.delenv("LAYA_SHADOW_MODE", raising=False)
    for alias in ("top3", "enrich_top3", "Добавление топ 3"):
        monkeypatch.setenv("LAYA_MODE", alias)
        cfg = LayaConfig.from_env()
        assert cfg.mode == ENRICH_TOP3_MODE, alias
        assert cfg.is_enrich_top3 is True


def test_other_modes_are_not_enrich():
    assert LayaConfig.from_ctx_or_env(ctx=_Ctx({"mode": "active"})).is_enrich_top3 is False
    assert LayaConfig.from_ctx_or_env(ctx=_Ctx({"mode": "shadow"})).is_enrich_top3 is False


def test_mode_reload_switches_to_enrich():
    ctx = _Ctx({"mode": "active"})
    cfg = LayaConfig.from_ctx_or_env(ctx=ctx)
    assert cfg.is_enrich_top3 is False
    ctx._settings["mode"] = "Добавление топ 3"
    cfg.reload(ctx=ctx)
    assert cfg.is_enrich_top3 is True


# --------------------------------------------------------------------------- router


def test_two_level_category_first_then_top3_added(tmp_path):
    laya = FakeLaya()
    router, _ = _router(tmp_path, laya, hierarchical=True)
    standard = _standard()

    effective, dec = router.route_tools(
        standard, "USER: найди наушники на Ozon", agent=_agent(_catalog(standard))
    )

    # Level 1 asked ONLY about the category; level 2 ranked tools afterwards.
    assert list(laya.calls[0].keys()) == ["category"]
    assert all(k.startswith("batch_") for k in laya.calls[1].keys())

    assert dec["decision"] == "enrich_top3"
    assert dec["category"] == "marketplace_and_commerce"
    assert dec["selected_tools"] == TOP3
    assert dec["added_tools"] == TOP3

    names = [extract_tool_name(t) for t in effective]
    # Standard list is untouched and first; the top 3 are appended.
    assert names == STANDARD_NAMES + TOP3


def test_tools_already_in_standard_list_are_not_added_twice(tmp_path):
    laya = FakeLaya()
    router, _ = _router(tmp_path, laya)
    standard = _standard() + [_tool("mcp__marketplace__ozon_search")]

    effective, dec = router.route_tools(
        standard, "USER: найди наушники на Ozon", agent=_agent(_catalog(standard))
    )

    names = [extract_tool_name(t) for t in effective]
    assert dec["selected_tools"] == TOP3
    assert dec["added_tools"] == TOP3[1:]  # ozon_search was already there
    assert len(names) == len(set(names))
    assert names == STANDARD_NAMES + ["mcp__marketplace__ozon_search"] + TOP3[1:]


def test_nothing_added_when_whole_top3_is_in_standard_list(tmp_path):
    laya = FakeLaya()
    router, _ = _router(tmp_path, laya)
    standard = _standard() + [_tool(n) for n in TOP3]

    effective, dec = router.route_tools(
        standard, "USER: найди наушники на Ozon", agent=_agent(_catalog(standard))
    )

    assert dec["added_tools"] == []
    assert [extract_tool_name(t) for t in effective] == [extract_tool_name(t) for t in standard]


def test_flat_routing_picks_top3_without_category_question(tmp_path):
    laya = FakeLaya()
    router, _ = _router(tmp_path, laya, hierarchical=False)
    standard = _standard()

    effective, dec = router.route_tools(
        standard, "USER: найди наушники на Ozon", agent=_agent(_catalog(standard))
    )

    assert all(k.startswith("batch_") for call in laya.calls for k in call.keys())
    assert dec["category"] == "flat"
    assert dec["selected_tools"] == TOP3
    assert [extract_tool_name(t) for t in effective] == STANDARD_NAMES + TOP3


def test_always_include_does_not_eat_top3_slots(tmp_path):
    laya = FakeLaya()
    router, _ = _router(tmp_path, laya, always_include={"web_search"})
    standard = _standard()

    _, dec = router.route_tools(
        standard, "USER: найди наушники на Ozon", agent=_agent(_catalog(standard))
    )

    assert len(dec["selected_tools"]) == 3


def test_trivial_message_keeps_standard_list_without_laya_calls(tmp_path):
    laya = FakeLaya()
    router, _ = _router(tmp_path, laya)
    standard = _standard()

    # No patching: the real looks_trivial must recognise a short phrase behind the "USER:" prefix.
    effective, dec = router.route_tools(standard, "USER: ок", agent=_agent(_catalog(standard)))

    assert effective == standard
    assert laya.calls == []
    assert dec["decision"] == "enrich_top3"
    assert dec["added_tools"] == []


def test_laya_failure_never_strips_standard_list(tmp_path):
    laya = MagicMock()
    laya.ask.side_effect = Exception("Laya timeout")
    router, _ = _router(tmp_path, laya, fallback_mode="strip-all")
    standard = _standard()

    effective, dec = router.route_tools(
        standard, "USER: найди наушники на Ozon", agent=_agent(_catalog(standard))
    )

    assert dec["decision"] == "fallback"
    assert effective == standard


def test_decision_is_cached_per_standard_list(tmp_path):
    laya = FakeLaya()
    router, _ = _router(tmp_path, laya)
    standard = _standard()
    agent = _agent(_catalog(standard))

    router.route_tools(standard, "USER: найди наушники на Ozon", agent=agent)
    calls_after_first = len(laya.calls)
    effective, dec = router.route_tools(standard, "USER: найди наушники на Ozon", agent=agent)

    assert len(laya.calls) == calls_after_first
    assert dec.get("cached") is True
    assert [extract_tool_name(t) for t in effective] == STANDARD_NAMES + TOP3


# --------------------------------------------------------------------------- manifest


def test_manifest_offers_mode_and_has_no_recategorize_switch():
    text = (Path(__file__).parent.parent / "plugin.yaml").read_text(encoding="utf-8")
    assert "Добавление топ 3" in text
    # The action is a button on the desktop page, not a switch field in the schema form.
    assert "recategorize_now:" not in text


# --------------------------------------------------------------------------- plugin


class _FakeContext:
    def __init__(self):
        self.hooks = {}
        self.commands = {}

    def register_hook(self, name, handler):
        self.hooks[name] = handler

    def register_command(self, name, handler, **kwargs):
        self.commands[name] = handler


class _FakeAgent:
    def __init__(self, tools, session_id="s_enrich"):
        self.session_id = session_id
        self.tools = tools


def test_laya_top3_command_switches_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_MODE", "active")
    monkeypatch.setenv("LAYA_LOG_PATH", str(tmp_path / "t.jsonl"))
    ctx = _FakeContext()
    plugin.register(ctx)

    resp = ctx.commands["laya"]("top3")

    assert "топ 3" in resp
    assert plugin._config.is_enrich_top3 is True


def test_pre_llm_call_enriches_agent_tools_and_does_not_pile_up(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_MODE", "Добавление топ 3")
    monkeypatch.setenv("LAYA_LOG_PATH", str(tmp_path / "t.jsonl"))
    monkeypatch.setenv("LAYA_CACHE_PATH", str(tmp_path / "cats.json"))
    ctx = _FakeContext()
    plugin.register(ctx)

    standard = _standard()
    agent = _FakeAgent(list(standard))
    laya = FakeLaya()

    with patch.object(plugin._client, "ask", side_effect=laya.ask), patch.object(
        router_mod, "collect_all_tools", return_value=_catalog(standard)
    ):
        res1 = ctx.hooks["pre_llm_call"](session_id="s_enrich", turn_id="t1", agent=agent, user_message="найди наушники на Ozon")
        names_after_first = [extract_tool_name(t) for t in agent.tools]
        ctx.hooks["pre_llm_call"](session_id="s_enrich", turn_id="t2", agent=agent, user_message="а теперь найди ещё чехол на Ozon")
        names_after_second = [extract_tool_name(t) for t in agent.tools]

    assert names_after_first == STANDARD_NAMES + TOP3
    assert names_after_second == STANDARD_NAMES + TOP3  # enriched from the saved standard list, no growth
    assert res1 is not None and "Топ-3" in res1["context"]
    assert plugin._saved_agent_tools["s_enrich"] and len(plugin._saved_agent_tools["s_enrich"]) == 3
