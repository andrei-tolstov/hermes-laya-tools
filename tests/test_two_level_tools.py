"""Tests for Two-Level Full Catalog (88 tools) Distribution and Routing in Laya."""

from __future__ import annotations

from unittest.mock import MagicMock

from categorizer import ToolCategorizer
from config import LayaConfig
from router import ToolRouter
from tool_collector import (
    BRIDGE_TOOL_NAMES,
    collect_all_tools,
    extract_tool_name,
    get_bridge_tools,
    is_bridge_tool,
)


def _make_tool(name: str, desc: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc or f"Description for {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_sample_88_tools() -> list[dict]:
    """Generate realistic mock dataset of 88 Hermes tools (hot + catalog/MCP)."""
    tools = []

    # 1. 21 Hot tools (core + active)
    hot_names = [
        ("read_file", "Read file contents from disk"),
        ("write_file", "Write text to file on disk"),
        ("search_files", "Search workspace files for regex patterns"),
        ("patch", "Apply diff patch to file"),
        ("terminal", "Execute command line in terminal"),
        ("execute_code", "Execute Python sandbox code"),
        ("web_search", "Search the internet via search engine"),
        ("web_extract", "Extract content from web URL"),
        ("browser_exec", "Execute browser action in web page"),
        ("delegate_task", "Delegate complex subtask to subagent"),
        ("memory", "Store or retrieve memory facts"),
        ("clarify", "Ask user clarifying question"),
        ("skill_manage", "Manage installed agent skills"),
        ("skill_view", "View specific skill instructions"),
        ("skills_list", "List available skills"),
        ("vision_analyze", "Analyze image with vision model"),
        ("text_to_speech", "Synthesize audio voice TTS"),
        ("manage_connections", "Manage gateway connection secrets"),
    ]
    for name, desc in hot_names:
        tools.append(_make_tool(name, desc))

    # 3 Bridge tools in hot set (total hot = 21)
    for b_name in sorted(list(BRIDGE_TOOL_NAMES)):
        tools.append(_make_tool(b_name, f"Bridge tool {b_name}"))

    # 2. 10 Additional Filesystem & Code tools (deferred catalog)
    fs_code_names = [
        ("list_dir", "List files and subdirectories in path"),
        ("edit_file", "Make block replacements in file"),
        ("grep_search", "Ripgrep search pattern across directory"),
        ("view_file", "View text or binary file slice"),
        ("find_files", "Find files by glob pattern"),
        ("git_status", "Check git working tree status"),
        ("git_diff", "Inspect git uncommitted changes"),
        ("git_commit", "Create git commit with message"),
        ("git_branch", "Manage git branches"),
        ("git_log", "View commit history"),
    ]
    for name, desc in fs_code_names:
        tools.append(_make_tool(name, desc))

    # 3. 10 Desktop & Computer Use tools (deferred catalog)
    desktop_names = [
        ("computer_use", "Take screenshot and click/type native OS UI"),
        ("desktop_preview", "Preview GUI desktop surface"),
        ("desktop_project", "Project native desktop window"),
        ("focus_pane", "Focus specific application pane"),
        ("read_window_below", "Read window text below cursor"),
        ("gui_tour", "Run interactive GUI tour"),
        ("set_theme", "Change desktop UI theme"),
        ("show_tip", "Display UI tooltip helper"),
        ("annotate_preview", "Annotate preview image"),
        ("drive_preview", "Preview Google Drive / cloud file"),
    ]
    for name, desc in desktop_names:
        tools.append(_make_tool(name, desc))

    # 4. 7 System, Task & Jev tools (deferred catalog)
    system_names = [
        ("cronjob_manage", "Schedule recurring cron jobs"),
        ("browser_vault", "Secure credential storage vault"),
        ("session_search", "Search past session history"),
        ("process_manage", "List and kill background processes"),
        ("todo_list", "Track and manage project todo list"),
        ("video_analyze", "Analyze video stream or recording"),
        ("jev_decide", "Jev fast lane decision model"),
    ]
    for name, desc in system_names:
        tools.append(_make_tool(name, desc))

    # 5. 40 Marketplace / E-commerce MCP tools (deferred catalog)
    mcp_market_names = [
        ("mcp__marketplace__marketplace_sources", "List supported Russian and global marketplace platforms"),
        ("mcp__marketplace__ozon_search", "Search products on Ozon marketplace"),
        ("mcp__marketplace__ozon_card", "Get product card details on Ozon"),
        ("mcp__marketplace__ozon_reviews", "Fetch customer reviews on Ozon"),
        ("mcp__marketplace__wb_search", "Search products on Wildberries marketplace"),
        ("mcp__marketplace__wb_card", "Get product card details on Wildberries"),
        ("mcp__marketplace__wb_reviews", "Fetch customer reviews on Wildberries"),
        ("mcp__marketplace__yandex_search", "Search goods on Yandex Market"),
        ("mcp__marketplace__yandex_card", "Get product card details on Yandex Market"),
        ("mcp__marketplace__avito_search", "Search classified listings on Avito"),
        ("mcp__marketplace__avito_card", "Get classified ad details on Avito"),
        ("mcp__marketplace__avito_seller", "Get seller profile on Avito"),
        ("mcp__marketplace__taobao_search", "Search items on Taobao"),
        ("mcp__marketplace__taobao_card", "Get item details on Taobao"),
        ("mcp__marketplace__megamarket_search", "Search goods on MegaMarket"),
        ("mcp__marketplace__megamarket_card", "Get product details on MegaMarket"),
        ("mcp__marketplace__lamoda_search", "Search fashion clothing on Lamoda"),
        ("mcp__marketplace__lamoda_card", "Get product details on Lamoda"),
        ("mcp__marketplace__detmir_search", "Search toys and children goods on Detsky Mir"),
        ("mcp__marketplace__detmir_card", "Get product details on Detsky Mir"),
        ("mcp__marketplace__drom_search", "Search auto car listings on Drom"),
        ("mcp__marketplace__drom_card", "Get vehicle details on Drom"),
        ("mcp__marketplace__citilink_search", "Search electronics on Citilink"),
        ("mcp__marketplace__citilink_card", "Get tech product details on Citilink"),
        ("mcp__marketplace__dns_search", "Search digital appliances on DNS"),
        ("mcp__marketplace__dns_card", "Get appliances details on DNS"),
        ("mcp__marketplace__aliexpress_search", "Search products on AliExpress global"),
        ("mcp__marketplace__aliexpress_card", "Get product card details on AliExpress"),
        ("mcp__marketplace__cian_search", "Search real estate on Cian"),
        ("mcp__marketplace__cian_card", "Get property details on Cian"),
        ("mcp__marketplace__mpstats_search", "Analyze marketplace product analytics on MPStats"),
        ("mcp__marketplace__mpstats_card", "Get product sales analytics on MPStats"),
        ("mcp__marketplace__compare_prices", "Compare product prices across all marketplaces"),
        ("mcp__marketplace__compare_specs", "Compare technical specs of marketplace items"),
        ("mcp__marketplace__order_status", "Check delivery and order status on marketplace"),
        ("mcp__marketplace__cart_view", "View items in user shopping cart"),
        ("mcp__marketplace__cart_add", "Add item to marketplace shopping cart"),
        ("mcp__marketplace__seller_feedback", "Fetch seller ratings and feedback"),
        ("mcp__marketplace__price_history", "Get historical price chart for product"),
        ("mcp__marketplace__category_tree", "Browse marketplace catalog category hierarchy"),
    ]
    for name, desc in mcp_market_names:
        tools.append(_make_tool(name, desc))

    return tools


def test_sample_dataset_count_and_bridge_filtering():
    dataset = _make_sample_88_tools()
    # 21 hot (including 3 bridge) + 10 fs/code + 10 desktop + 7 system + 40 market = exactly 88
    assert len(dataset) == 88

    # Bridge tools detection
    bridges = get_bridge_tools(dataset)
    assert len(bridges) == 3
    assert all(is_bridge_tool(b) for b in bridges)

    # collect_all_tools filters out bridge tools so genuine capabilities are routed
    filtered = collect_all_tools(extra_tools=dataset, include_bridge_tools=False)
    assert len(filtered) == 85
    assert not any(is_bridge_tool(t) for t in filtered)


def test_categorizer_partitions_all_88_tools(tmp_path):
    dataset = _make_sample_88_tools()
    mock_laya = MagicMock()
    mock_laya.ask.return_value = {"answers": {}}

    cache_file = tmp_path / "cat_test.json"
    categorizer = ToolCategorizer(
        laya_client=mock_laya,
        cache_path=cache_file,
        interval_ms=1000,
    )

    categories = categorizer.get_categories(dataset, force_refresh=True)
    summary = categorizer.format_distribution_summary(categories)

    # Ensure summary reflects total tools
    assert "Всего тулзов: 88" in summary
    assert "marketplace_and_commerce" in summary
    assert "desktop_and_computer_use" in summary

    # Marketplace category has 40 tools
    mkt_tools = categories["marketplace_and_commerce"]["tools"]
    assert len(mkt_tools) == 40

    # Desktop category has 10 tools
    desk_tools = categories["desktop_and_computer_use"]["tools"]
    assert len(desk_tools) == 10


def test_router_two_level_marketplace_selection(tmp_path):
    dataset = _make_sample_88_tools()
    mock_laya = MagicMock()

    # Level 1 response: execute_tools, category = marketplace_and_commerce
    # Level 2 response: rank tools
    def fake_ask(state_text, questions):
        if "category" in questions:
            return {
                "answers": {
                    "tool_decision": {"choice": "execute_tools", "confidence": 0.98},
                    "category": {"choice": "marketplace_and_commerce", "confidence": 0.99},
                },
                "latency_ms": 2.5,
            }
        # Level 2 batch ranking
        answers = {}
        for q_id, q_data in questions.items():
            criteria = q_data.get("criteria", {})
            probs = {}
            for tool_name in criteria:
                probs[tool_name] = 0.95 if "ozon" in tool_name else 0.05
            answers[q_id] = {"choice": "mcp__marketplace__ozon_search", "probabilities": probs}
        return {"answers": answers, "latency_ms": 3.0}

    mock_laya.ask.side_effect = fake_ask

    mock_audit = MagicMock()
    cfg = LayaConfig(enabled=True, mode="active", max_tools=5, hierarchical=True)
    categorizer = ToolCategorizer(laya_client=mock_laya, cache_path=tmp_path / "cat1.json")

    router = ToolRouter(
        laya_client=mock_laya,
        config=cfg,
        categorizer=categorizer,
        audit_logger=mock_audit,
    )

    # Mock agent holding full uncollapsed catalog
    mock_agent = MagicMock()
    mock_agent.tools = dataset

    hot_21 = dataset[:21]
    effective_tools, decision = router.route_tools(
        tools=hot_21,
        context="Купи на Ozon наушники Sony WH-1000XM5",
        session_id="s_mkt_1",
        turn_id="t_1",
        agent=mock_agent,
    )

    # Winning category must be marketplace_and_commerce
    assert decision["category"] == "marketplace_and_commerce"
    assert decision["decision"] == "execute_tools"

    # Top tools selected from marketplace category
    selected_names = [extract_tool_name(t) for t in effective_tools]
    assert any("ozon" in name for name in selected_names)

    # Check audit log received all 88 available tools
    mock_audit.log_interaction.assert_called_once()
    audit_kwargs = mock_audit.log_interaction.call_args[1]
    assert len(audit_kwargs["all_available_tools"]) >= 80
    assert audit_kwargs["winning_category"] == "marketplace_and_commerce"


def test_router_shadow_mode_leaves_hot_tools_intact_while_auditing_full_catalog(tmp_path):
    dataset = _make_sample_88_tools()
    mock_laya = MagicMock()
    def fake_shadow_ask(state_text, questions):
        if "category" in questions:
            return {
                "answers": {
                    "tool_decision": {"choice": "execute_tools", "confidence": 0.95},
                    "category": {"choice": "desktop_and_computer_use", "confidence": 0.90},
                },
                "latency_ms": 2.0,
            }
        answers = {}
        for q_id, q_data in questions.items():
            criteria = q_data.get("criteria", {})
            probs = {tool_name: 0.9 if "computer_use" in tool_name else 0.1 for tool_name in criteria}
            answers[q_id] = {"choice": "computer_use", "probabilities": probs}
        return {"answers": answers, "latency_ms": 2.5}

    mock_laya.ask.side_effect = fake_shadow_ask

    mock_audit = MagicMock()
    cfg = LayaConfig(enabled=True, mode="shadow", max_tools=5, hierarchical=True)
    categorizer = ToolCategorizer(laya_client=mock_laya, cache_path=tmp_path / "cat2.json")

    router = ToolRouter(
        laya_client=mock_laya,
        config=cfg,
        categorizer=categorizer,
        audit_logger=mock_audit,
    )

    mock_agent = MagicMock()
    mock_agent.tools = dataset

    hot_21 = dataset[:21]
    effective_tools, decision = router.route_tools(
        tools=hot_21,
        context="Сделай клик по окну приложения и перемести курсор мыши",
        session_id="s_shadow",
        turn_id="t_1",
        agent=mock_agent,
    )

    # In shadow mode, effective_tools MUST remain identical to the passed hot_21
    assert effective_tools == hot_21
    assert decision["shadow"] is True
    assert decision["category"] == "desktop_and_computer_use"

    # Simulated selected tools were chosen from desktop category
    simulated = decision["simulated_selected_tools"]
    assert any("computer_use" in name or "desktop" in name for name in simulated)
