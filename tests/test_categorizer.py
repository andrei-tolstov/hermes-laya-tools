"""Tests for ToolCategorizer."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from categorizer import ToolCategorizer, extract_tool_desc, extract_tool_name


def test_extract_tool_name_and_desc():
    t1 = {
        "type": "function",
        "function": {"name": "read_file", "description": "Reads a local file."},
    }
    assert extract_tool_name(t1) == "read_file"
    assert extract_tool_desc(t1) == "Reads a local file."

    t2 = {"name": "bash", "description": "Runs bash script."}
    assert extract_tool_name(t2) == "bash"
    assert extract_tool_desc(t2) == "Runs bash script."


def test_pattern_categorization(tmp_path):
    mock_laya = MagicMock()
    cache_path = tmp_path / "cache.json"
    categorizer = ToolCategorizer(mock_laya, cache_path=cache_path)

    tools = [
        {"function": {"name": "fs_read", "description": "Read file contents"}},
        {"function": {"name": "run_bash", "description": "Execute terminal shell command"}},
        {"function": {"name": "web_search", "description": "Search the web"}},
        {"function": {"name": "git_commit", "description": "Git commit changes"}},
        {"function": {"name": "docker_build", "description": "Build container"}},
        {"function": {"name": "send_telegram", "description": "Send alert message"}},
        {"function": {"name": "calculate_math", "description": "Math expression"}},
    ]

    cats = categorizer.get_categories(tools, force_refresh=True)

    fs_names = [extract_tool_name(t) for t in cats["filesystem"]["tools"]]
    assert "fs_read" in fs_names

    term_names = [extract_tool_name(t) for t in cats["terminal_execution"]["tools"]]
    assert "run_bash" in term_names

    web_names = [extract_tool_name(t) for t in cats["web_and_search"]["tools"]]
    assert "web_search" in web_names

    git_names = [extract_tool_name(t) for t in cats["code_and_git"]["tools"]]
    assert "git_commit" in git_names

    # Ensure no call to Laya was needed because patterns matched all tools
    mock_laya.ask.assert_not_called()


def test_laya_fallback_for_unknown_tools(tmp_path):
    mock_laya = MagicMock()
    mock_laya.ask.return_value = {
        "ok": True,
        "answers": {
            "tool_0": {"choice": "code_and_git"},
        },
    }
    cache_path = tmp_path / "cache.json"
    categorizer = ToolCategorizer(mock_laya, cache_path=cache_path)

    tools = [
        {"function": {"name": "xyz_widget", "description": "custom specialized gadget"}},
    ]

    cats = categorizer.get_categories(tools, force_refresh=True)
    git_names = [extract_tool_name(t) for t in cats["code_and_git"]["tools"]]
    assert "xyz_widget" in git_names
    mock_laya.ask.assert_called_once()


def test_caching_persistence(tmp_path):
    mock_laya = MagicMock()
    cache_path = tmp_path / "cache.json"
    cat1 = ToolCategorizer(mock_laya, cache_path=cache_path, interval_ms=100_000)

    tools = [{"function": {"name": "fs_read", "description": "Read file"}}]
    cat1.get_categories(tools)
    assert cache_path.is_file()

    # Create new instance pointing to same file
    cat2 = ToolCategorizer(mock_laya, cache_path=cache_path, interval_ms=100_000)
    cats = cat2.get_categories(tools)
    fs_names = [extract_tool_name(t) for t in cats["filesystem"]["tools"]]
    assert "fs_read" in fs_names
