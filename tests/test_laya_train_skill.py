"""Unit tests for skills/laya-train/scripts (extract_and_label.py & evaluate_accuracy.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

skill_scripts = Path(__file__).resolve().parent.parent / "skills" / "laya-train" / "scripts"
if str(skill_scripts) not in sys.path:
    sys.path.insert(0, str(skill_scripts))

import extract_and_label as el
import evaluate_accuracy as ea
import train_pipeline as tp


def test_clean_user_state_strips_context_and_rejects_system():
    dirty = "Найди смартфон на Ozon\n--- Attached Context ---\nHuge 100KB dump here"
    assert el.clean_user_state(dirty) == "Найди смартфон на Ozon"
    assert el.clean_user_state("[System: background task]") is None
    assert el.clean_user_state("executed=terminal hit=True") is None
    assert el.clean_user_state("For this turn, you are in PLAN MODE — planning only.") is None
    assert el.clean_user_state("Interaction style:\n- If the request is clear enough") is None
    assert el.clean_user_state("продолжай") is None


def test_extract_from_logs_pairs_tool_telemetry(tmp_path: Path):
    log_file = tmp_path / "laya_requests.jsonl"
    rows = [
        {
            "session_id": "s1",
            "turn_id": "t1",
            "state_snippet": "Покажи файлы в текущем каталоге",
            "decision": "execute_tools",
            "winning_category": "terminal_execution",
            "confidence": 0.65,
            "raw_answers": {
                "tool_decision": {"choice": "execute_tools", "probabilities": {"execute_tools": 0.65, "direct_chat": 0.35}},
                "category": {"choice": "terminal_execution", "probabilities": {"terminal_execution": 0.55}},
            },
        },
        {
            "session_id": "s1",
            "turn_id": "t1",
            "state_snippet": "executed=read_file hit=False",
            "decision": "tool_exec:miss",
            "winning_category": "read_file",
            "confidence": 1.0,
            "raw_answers": {"kind": "tool_telemetry", "tool_name": "read_file", "hit": False},
        },
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    records = el.extract_from_logs(log_file)
    assert len(records) == 1
    rec = records[0]
    assert rec["state"] == "Покажи файлы в текущем каталоге"
    assert rec["laya_choice"]["tool_decision"] == "execute_tools"
    assert rec["laya_choice"]["category"] == "terminal_execution"
    assert rec["primary_choice"]["tool_decision"] == "execute_tools"
    assert rec["primary_choice"]["category"] == "filesystem"


def test_format_laya_training_item_expected_schema():
    # Scenario 1: execute_tools includes both tool_decision and category in expected
    item_exec = el.format_laya_training_item(
        state="Проверь git status в репозитории",
        primary_choice={"tool_decision": "execute_tools", "category": "terminal_execution", "source": "primary_observed_telemetry"},
        laya_choice={"tool_decision": "direct_chat", "category": "general_utility"},
        match=False,
        is_hard_negative=True,
    )
    assert item_exec["expected"] == {"tool_decision": "execute_tools", "category": "terminal_execution"}
    assert item_exec["label"] == {"tool_decision": "execute_tools", "category": "terminal_execution"}
    assert item_exec["metadata"]["hard_negative"] is True

    # Scenario 2: direct_chat omits category in expected so category head isn't polluted
    item_chat = el.format_laya_training_item(
        state="Привет! Как твои дела?",
        primary_choice={"tool_decision": "direct_chat", "category": "general_utility", "source": "primary_model_queried"},
        laya_choice={"tool_decision": "execute_tools", "category": "web_and_search"},
        match=False,
        is_hard_negative=True,
    )
    assert item_chat["expected"] == {"tool_decision": "direct_chat"}
    assert item_chat["label"] == {"tool_decision": "direct_chat", "category": "general_utility"}


def test_evaluate_accuracy_metrics_and_expected_schema():
    results = [
        {
            "expected_decision": "execute_tools",
            "predicted_decision": "execute_tools",
            "tool_decision_match": True,
            "expected_category": "terminal_execution",
            "predicted_category": "terminal_execution",
            "category_match": True,
            "strict_match": True,
            "predicted_td_prob": 0.95,
            "predicted_td_conf": 0.95,
            "predicted_cat_conf": 0.89,
            "combined_conf": 0.89,
            "latency_ms": 12.5,
        },
        {
            "expected_decision": "direct_chat",
            "predicted_decision": "direct_chat",
            "tool_decision_match": True,
            "expected_category": "general_utility",
            "predicted_category": "general_utility",
            "category_match": True,
            "strict_match": True,
            "predicted_td_prob": 0.08,
            "predicted_td_conf": 0.92,
            "predicted_cat_conf": 0.92,
            "combined_conf": 0.92,
            "latency_ms": 11.0,
        },
    ]
    metrics = ea.calculate_metrics(results)
    assert metrics["tool_decision_accuracy"] == 1.0
    assert metrics["category_accuracy"] == 1.0
    assert metrics["combined_strict_accuracy"] == 1.0
    assert metrics["brier_score"] < 0.01


def test_gpu0_vram_cpu_selection():
    from unittest.mock import patch, MagicMock
    mock_proc = MagicMock(returncode=0, stdout="3100\n")
    with patch("subprocess.run", return_value=mock_proc):
        free_mib = tp.get_gpu0_free_vram_mib()
        assert free_mib == 3100
        assert free_mib < 4500


def test_backup_before_replace_file_dir_and_retention(tmp_path: Path):
    gguf = tmp_path / "laya_multilingual_f16.gguf"
    gguf.write_bytes(b"v1")
    manual_bak = tmp_path / "laya_multilingual_f16.gguf.bak"  # ручной бэкап без метки времени
    manual_bak.write_bytes(b"manual")

    b1 = tp.backup_before_replace(gguf, "20260101-000001", keep=2)
    assert b1 is not None and b1.read_bytes() == b"v1"
    gguf.write_bytes(b"v2")
    tp.backup_before_replace(gguf, "20260101-000002", keep=2)
    gguf.write_bytes(b"v3")
    b3 = tp.backup_before_replace(gguf, "20260101-000003", keep=2)

    assert not b1.exists()  # самая старая удалена по keep=2
    assert b3.read_bytes() == b"v3"
    assert manual_bak.read_bytes() == b"manual"  # ручной бэкап не тронут

    ckpt = tmp_path / "finetune-checkpoint"
    (ckpt / "tokenizer").mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"w")
    cb = tp.backup_before_replace(ckpt, "20260101-000004", keep=3)
    assert (cb / "model.safetensors").read_bytes() == b"w"
    assert (cb / "tokenizer").is_dir()

    assert tp.backup_before_replace(tmp_path / "missing.gguf", "20260101-000005") is None


def test_query_primary_model_uses_enable_thinking_true_and_xhigh():
    from unittest.mock import patch, MagicMock
    captured_req = {}

    class DummyResp:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {
                        "content": '{"tool_decision": "execute_tools", "category": "marketplace_and_commerce", "reasoning": "Ozon search"}',
                        "reasoning_content": "Thinking deeply at xhigh effort..."
                    }
                }]
            }).encode("utf-8")

    def fake_urlopen(req, timeout=90):
        captured_req["body"] = json.loads(req.data.decode("utf-8"))
        return DummyResp()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        res = el.query_primary_model("Найди наушники на Ozon")
        assert res is not None
        assert res["tool_decision"] == "execute_tools"
        assert res["category"] == "marketplace_and_commerce"
        assert captured_req["body"]["reasoning_effort"] == "xhigh"
        assert captured_req["body"]["chat_template_kwargs"] == {"enable_thinking": True}
