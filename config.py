"""Configuration loader for hermes-laya-tools.

All settings are read from Hermes environment variables (.env / os.environ).
Supports automatic discovery of Hermes .env from $HERMES_HOME/.env or ~/.hermes/.env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set


def _find_hermes_env_file() -> Path | None:
    """Find Hermes .env file path."""
    candidates = []
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        candidates.append(Path(hermes_home) / ".env")
    candidates.append(Path.home() / ".hermes" / ".env")
    candidates.append(Path.cwd() / ".env")

    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def load_hermes_env(env_path: Path | str | None = None) -> dict[str, str]:
    """Parse a .env file and inject variables into os.environ if not already present.

    Returns the loaded key-value dictionary.
    """
    loaded: dict[str, str] = {}
    target_path = Path(env_path) if env_path else _find_hermes_env_file()
    if not target_path or not target_path.is_file():
        return loaded

    try:
        with open(target_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip wrapping quotes
                if len(value) >= 2 and (
                    (value.startswith('"') and value.endswith('"'))
                    or (value.startswith("'") and value.endswith("'"))
                ):
                    value = value[1:-1]
                loaded[key] = value
                if key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass
    return loaded


# Load Hermes environment upon import
load_hermes_env()


def _bool_from_env(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


def _int_from_env(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _path_from_env(key: str, default_subpath: str) -> Path:
    val = os.environ.get(key)
    if val:
        return Path(val.strip()).expanduser()
    hermes_home = os.environ.get("HERMES_HOME")
    base = Path(hermes_home) if hermes_home else Path.home() / ".hermes"
    return base / default_subpath


@dataclass
class LayaConfig:
    """Runtime configuration for Laya System 1 Decision Engine plugin."""

    sidecar_url: str = "http://127.0.0.1:8787"
    enabled: bool = True
    mode: str = "active"  # "active" | "shadow"
    hierarchical: bool = True
    max_tools: int = 10
    timeout_ms: int = 5000
    categorization_interval_ms: int = 86_400_000  # 24h
    cache_path: Path = field(default_factory=lambda: Path.home() / ".hermes" / "cache" / "laya_categories.json")
    always_include: Set[str] = field(default_factory=set)
    fallback_mode: str = "pass-all"  # "pass-all" | "strip-all"
    notify_in_chat: bool = True
    log_path: Path = field(default_factory=lambda: Path.home() / ".hermes" / "logs" / "laya_requests.jsonl")
    log_level: str = "INFO"

    @property
    def is_shadow(self) -> bool:
        """True if running in shadow mode (evaluation and logging only, no tool mutation)."""
        return self.mode.lower() == "shadow"

    @classmethod
    def from_env(cls, env_path: Path | str | None = None) -> LayaConfig:
        """Create configuration instance from environment variables (.env)."""
        load_hermes_env(env_path)

        url = os.environ.get("LAYA_SIDECAR_URL") or os.environ.get("LAYA_URL") or "http://127.0.0.1:8787"
        url = url.rstrip("/")

        enabled = _bool_from_env("LAYA_ENABLED", True)

        # Check mode: LAYA_MODE="shadow" or LAYA_SHADOW_MODE="true"
        raw_mode = os.environ.get("LAYA_MODE", "active").strip().lower()
        if _bool_from_env("LAYA_SHADOW_MODE", False) or raw_mode == "shadow":
            mode = "shadow"
        else:
            mode = "active"

        hierarchical = _bool_from_env("LAYA_HIERARCHICAL", True)
        max_tools = max(1, _int_from_env("LAYA_MAX_TOOLS", 10))
        timeout_ms = max(500, _int_from_env("LAYA_TIMEOUT_MS", 5000))
        cat_interval = max(60_000, _int_from_env("LAYA_CATEGORIZATION_INTERVAL_MS", 86_400_000))

        cache_path = _path_from_env("LAYA_CACHE_PATH", "cache/laya_categories.json")

        always_raw = os.environ.get("LAYA_ALWAYS_INCLUDE", "")
        always_include = {t.strip() for t in always_raw.split(",") if t.strip()}

        fallback = os.environ.get("LAYA_FALLBACK_MODE", "pass-all").strip().lower()
        if fallback not in ("pass-all", "strip-all"):
            fallback = "pass-all"

        # In shadow mode notify_in_chat defaults to False so output is never modified
        default_notify = False if mode == "shadow" else True
        notify_in_chat = _bool_from_env("LAYA_NOTIFY_IN_CHAT", default_notify)
        if mode == "shadow":
            notify_in_chat = False  # Strict guarantee: shadow mode never touches chat output

        log_path = _path_from_env("LAYA_LOG_PATH", "logs/laya_requests.jsonl")
        log_level = os.environ.get("LAYA_LOG_LEVEL", "INFO").strip().upper()

        return cls(
            sidecar_url=url,
            enabled=enabled,
            mode=mode,
            hierarchical=hierarchical,
            max_tools=max_tools,
            timeout_ms=timeout_ms,
            categorization_interval_ms=cat_interval,
            cache_path=cache_path,
            always_include=always_include,
            fallback_mode=fallback,
            notify_in_chat=notify_in_chat,
            log_path=log_path,
            log_level=log_level,
        )
