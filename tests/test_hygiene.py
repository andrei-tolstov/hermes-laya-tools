"""Tests for hygiene module, fast-bypass, repeat-ask cache, kill-switch and recategorization."""

from pathlib import Path
from unittest.mock import MagicMock

from categorizer import ToolCategorizer
from config import LayaConfig
from hygiene import looks_trivial, redact_sensitive, unwrap
from router import ToolRouter


def test_looks_trivial_russian_and_english():
    # Trivial acknowledgments should be True
    assert looks_trivial("ок") is True
    assert looks_trivial("Да, спасибо") is True
    assert looks_trivial("Хорошо, продолжай") is True
    assert looks_trivial("понял, отлично") is True
    assert looks_trivial("ok") is True
    assert looks_trivial("yes thanks") is True
    assert looks_trivial("got it, proceed") is True
    assert looks_trivial("hello") is True
    assert looks_trivial("привет") is True

    # Action phrases must NOT be trivial (return False)
    assert looks_trivial("запусти бекап") is False
    assert looks_trivial("проверь сервер") is False
    assert looks_trivial("покажи файлы") is False
    assert looks_trivial("run git status") is False
    assert looks_trivial("delete this table") is False
    assert looks_trivial("curl https://api.com") is False
    assert looks_trivial("напиши скрипт на питоне") is False


def test_unwrap_markdown_envelopes():
    # Envelope with ## Prompt
    contract = """
# Scheduled Task: Daily Maintenance
Contract: Do not alter state without lock.

## Prompt
Проверь свободное место на диске и очисти старые логи

## Script Output
Error in line 44: disk full
"""
    unwrapped = unwrap(contract)
    assert unwrapped == "Проверь свободное место на диске и очисти старые логи"

    # Plain text without envelopes remains unchanged
    plain = "Обычный запрос пользователя"
    assert unwrap(plain) == plain


def test_redact_sensitive():
    text = (
        "Key: github_pat_dummy_sample_token_for_testing_purposes_1234567890 "
        "and sk-abcdef1234567890abcdef1234567890 and email test@company.org"
    )
    redacted = redact_sensitive(text)
    assert "github_pat_" not in redacted
    assert "[REDACTED_GH_TOKEN]" in redacted
    assert "sk-" not in redacted
    assert "[REDACTED_API_KEY]" in redacted
    assert "test@company.org" not in redacted
    assert "[REDACTED_EMAIL]" in redacted


def test_hardware_kill_switch(tmp_path, monkeypatch):
    marker = tmp_path / "laya" / "LAYA_OFF"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("1")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = LayaConfig.from_env()
    assert cfg.is_killed is True
    assert cfg.enabled is False

    mock_client = MagicMock()
    router = ToolRouter(laya_client=mock_client, config=cfg)
    tools = [{"name": "bash"}, {"name": "read_file"}]
    effective, dec = router.route_tools(tools, "запусти команду")
    assert dec["decision"] == "killed"
    assert len(effective) == 2  # Fail-open


def test_repeat_ask_lru_cache():
    mock_client = MagicMock()
    mock_client.ask.return_value = {
        "answers": {
            "tool_decision": {
                "choice": "execute_tools",
                "confidence": 0.95,
                "probabilities": {"execute_tools": 0.95, "direct_chat": 0.05},
            },
            "category": {
                "choice": "terminal_execution",
                "probabilities": {"terminal_execution": 0.90},
            },
        },
        "latency_ms": 25.0,
    }
    cfg = LayaConfig(cache_repeat_asks=True, cache_size=10, mode="active")
    router = ToolRouter(laya_client=mock_client, config=cfg)
    tools = [{"name": "bash"}, {"name": "read_file"}]

    # First turn: calls Laya
    eff1, dec1 = router.route_tools(tools, "запусти проверку диска")
    assert dec1.get("cached") is not True
    assert mock_client.ask.call_count >= 1

    initial_calls = mock_client.ask.call_count

    # Second turn with identical instruction: returns from cache (0 calls to Laya)
    eff2, dec2 = router.route_tools(tools, "запусти проверку диска")
    assert dec2.get("cached") is True
    assert dec2.get("latency_ms") == 0.0
    assert mock_client.ask.call_count == initial_calls


def test_recategorize_summary_report(tmp_path):
    mock_client = MagicMock()
    cat = ToolCategorizer(
        laya_client=mock_client,
        cache_path=tmp_path / "cache.json",
    )
    tools = [
        {"name": "read_file", "description": "Read file content"},
        {"name": "write_file", "description": "Write file content"},
        {"name": "bash", "description": "Execute terminal commands"},
    ]
    report = cat.recategorize(tools)
    assert report["status"] == "ok"
    assert report["total_tools"] == 3
    assert "filesystem" in report["counts"]
    assert "terminal_execution" in report["counts"]
    assert "Разбито 3 инструментов" in report["message"]
