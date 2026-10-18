"""Tests for LayaAuditLogger."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import LayaAuditLogger


def test_audit_logger_active_mode(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    logger = LayaAuditLogger(log_file)

    entry = logger.log_interaction(
        session_id="sess-123",
        turn_id="turn-1",
        mode="active",
        decision="execute_tools",
        winning_category="filesystem",
        probability=0.9876,
        selected_tools=["read_file", "write_file"],
        all_available_tools=["read_file", "write_file", "bash", "calc"],
        state_snippet="USER: read config.json",
        latency_ms=35.2,
    )

    assert entry["session_id"] == "sess-123"
    assert entry["mode"] == "active"
    assert entry["shadow"] is False
    assert entry["decision"] == "execute_tools"
    assert entry["winning_category"] == "filesystem"
    assert entry["selected_tools"] == ["read_file", "write_file"]
    assert entry["available_tools_count"] == 4

    # Verify disk content
    assert log_file.is_file()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    disk_data = json.loads(lines[0])
    assert disk_data["session_id"] == "sess-123"
    assert disk_data["probability"] == 0.9876
    assert disk_data["shadow"] is False


def test_audit_logger_shadow_mode(tmp_path):
    log_file = tmp_path / "audit_shadow.jsonl"
    logger = LayaAuditLogger(log_file)

    entry = logger.log_interaction(
        session_id="sess-456",
        turn_id="turn-2",
        mode="shadow",
        decision="direct_chat",
        winning_category=None,
        probability=0.9912,
        selected_tools=[],
        all_available_tools=["read_file", "bash"],
        state_snippet="USER: hello there",
        latency_ms=21.0,
    )

    assert entry["shadow"] is True
    assert entry["mode"] == "shadow"
    assert entry["decision"] == "direct_chat"
    assert entry["selected_tools"] == []

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    disk_data = json.loads(lines[0])
    assert disk_data["shadow"] is True
    assert disk_data["mode"] == "shadow"


def test_audit_logger_error(tmp_path):
    log_file = tmp_path / "audit_err.jsonl"
    logger = LayaAuditLogger(log_file)

    logger.log_interaction(
        session_id="sess-err",
        turn_id="turn-err",
        mode="active",
        decision="fallback",
        error="Laya connection timeout",
        latency_ms=5000.0,
    )

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    disk_data = json.loads(lines[0])
    assert disk_data["decision"] == "fallback"
    assert disk_data["error"] == "Laya connection timeout"
