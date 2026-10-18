"""Tests for LayaClient."""

import io
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from client import LayaClient, LayaClientError


def test_truncate_state():
    client = LayaClient()
    short = "Hello world"
    assert client.truncate_state(short, max_chars=100) == short

    long_text = "A" * 1000 + "MID" + "Z" * 1000
    truncated = client.truncate_state(long_text, max_chars=200)
    assert len(truncated) <= 220
    assert "...[truncated]..." in truncated
    assert truncated.startswith("A" * 50)
    assert truncated.endswith("Z" * 50)


def test_health_success():
    client = LayaClient(sidecar_url="http://127.0.0.1:8787")
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"ok": True, "model": "laya"}).encode()
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.health()
        assert res is not None
        assert res["ok"] is True
        assert res["model"] == "laya"


def test_health_failure():
    client = LayaClient(sidecar_url="http://127.0.0.1:8787")
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        res = client.health()
        assert res is None


def test_ask_success():
    client = LayaClient(sidecar_url="http://127.0.0.1:8787")
    mock_resp = MagicMock()
    mock_resp.status = 200
    fake_payload = {
        "ok": True,
        "latency_ms": 42.5,
        "answers": {
            "tool_decision": {"choice": "direct_chat", "probabilities": {"direct_chat": 0.98}}
        },
    }
    mock_resp.read.return_value = json.dumps(fake_payload).encode()
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.ask("Hello", {"tool_decision": {"type": "choice"}})
        assert res["ok"] is True
        assert res["answers"]["tool_decision"]["choice"] == "direct_chat"
        assert res["latency_ms"] == 42.5


def test_ask_http_error():
    client = LayaClient()
    http_err = urllib.error.HTTPError(
        url="http://127.0.0.1:8787/ask",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=io.BytesIO(b'{"ok": false, "error": "invalid_format"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        try:
            client.ask("test", {})
            assert False, "Should have raised LayaClientError"
        except LayaClientError as err:
            assert err.status_code == 400
            assert "invalid_format" in str(err)
