import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LayaConfig, load_hermes_env


def test_default_config(monkeypatch, tmp_path):
    for k in list(os.environ.keys()):
        if k.startswith("LAYA_"):
            monkeypatch.delenv(k, raising=False)

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    cfg = LayaConfig.from_env(env_path=empty_env)
    assert cfg.sidecar_url == "http://127.0.0.1:8787"
    assert cfg.enabled is True
    assert cfg.mode == "active"
    assert cfg.is_shadow is False
    assert cfg.hierarchical is True
    assert cfg.max_tools == 10
    assert cfg.fallback_mode == "pass-all"


def test_shadow_mode_env(monkeypatch):
    monkeypatch.setenv("LAYA_MODE", "shadow")
    cfg = LayaConfig.from_env()
    assert cfg.mode == "shadow"
    assert cfg.is_shadow is True
    assert cfg.notify_in_chat is False  # In shadow mode, notify_in_chat must be False


def test_shadow_mode_flag(monkeypatch):
    monkeypatch.delenv("LAYA_MODE", raising=False)
    monkeypatch.setenv("LAYA_SHADOW_MODE", "true")
    cfg = LayaConfig.from_env()
    assert cfg.mode == "shadow"
    assert cfg.is_shadow is True


def test_custom_env_values(monkeypatch, tmp_path):
    monkeypatch.setenv("LAYA_URL", "http://custom-laya:9000/")
    monkeypatch.setenv("LAYA_MAX_TOOLS", "5")
    monkeypatch.setenv("LAYA_TIMEOUT_MS", "3000")
    monkeypatch.setenv("LAYA_ALWAYS_INCLUDE", "execute_code, bash")
    monkeypatch.setenv("LAYA_FALLBACK_MODE", "strip-all")

    cfg = LayaConfig.from_env()
    assert cfg.sidecar_url == "http://custom-laya:9000"
    assert cfg.max_tools == 5
    assert cfg.timeout_ms == 3000
    assert cfg.always_include == {"execute_code", "bash"}
    assert cfg.fallback_mode == "strip-all"


def test_load_hermes_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LAYA_MODE=shadow\nLAYA_MAX_TOOLS=7\n# Comment\n", encoding="utf-8")

    monkeypatch.delenv("LAYA_MODE", raising=False)
    monkeypatch.delenv("LAYA_MAX_TOOLS", raising=False)

    loaded = load_hermes_env(env_file)
    assert loaded["LAYA_MODE"] == "shadow"
    assert loaded["LAYA_MAX_TOOLS"] == "7"

    cfg = LayaConfig.from_env(env_file)
    assert cfg.is_shadow is True
    assert cfg.max_tools == 7


class FakePluginContext:
    def __init__(self, settings: dict):
        self._settings = settings

    def get_config(self, key: str, default=None):
        return self._settings.get(key, default)


def test_config_from_ctx():
    fake_ctx = FakePluginContext({
        "mode": "shadow",
        "max_tools": 8,
        "sidecar_url": "http://hermes-ui-sidecar:8787",
        "always_include": ["clarify", "delegate_task"],
        "direct_chat_threshold": 0.80,
    })
    cfg = LayaConfig.from_ctx_or_env(ctx=fake_ctx)
    assert cfg.mode == "shadow"
    assert cfg.is_shadow is True
    assert cfg.max_tools == 8
    assert cfg.sidecar_url == "http://hermes-ui-sidecar:8787"
    assert cfg.always_include == {"clarify", "delegate_task"}
    assert cfg.direct_chat_threshold == 0.80


def test_config_reload():
    fake_ctx = FakePluginContext({"mode": "active", "max_tools": 10})
    cfg = LayaConfig.from_ctx_or_env(ctx=fake_ctx)
    assert cfg.mode == "active"
    assert cfg.max_tools == 10

    # Simulate user changing settings in Hermes UI
    fake_ctx._settings["mode"] = "shadow"
    fake_ctx._settings["max_tools"] = 3
    cfg.reload(ctx=fake_ctx)

    assert cfg.mode == "shadow"
    assert cfg.is_shadow is True
    assert cfg.max_tools == 3

