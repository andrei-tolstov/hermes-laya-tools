"""Configuration loader for hermes-laya-tools.

Settings hierarchy (highest priority first):
1. Hermes Context (`ctx.get_config(...)`) from WebUI / Desktop settings UI
2. Hermes config.yaml (`plugins.entries.hermes-laya-tools.settings.*`)
3. Environment variables (.env / os.environ with LAYA_* prefix)
4. Sane default values
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Set

try:
    import yaml
except ImportError:
    try:
        import hermes_yaml as yaml  # type: ignore[no-redef]
    except ImportError:
        yaml = None


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


def _find_hermes_config_file(config_path: Path | str | None = None) -> Path | None:
    """Find Hermes config.yaml file path."""
    if config_path is not None:
        p = Path(config_path)
        return p if p.is_file() else None

    config_env = os.environ.get("HERMES_CONFIG_FILE") or os.environ.get("HERMES_CONFIG_PATH")
    if config_env:
        p = Path(config_env)
        return p if p.is_file() else None

    candidates = []
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        candidates.append(Path(hermes_home) / "config.yaml")
    candidates.append(Path.home() / ".hermes" / "config.yaml")
    candidates.append(Path.cwd() / "config.yaml")

    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def load_hermes_env(env_path: Path | str | None = None) -> dict[str, str]:
    """Parse a .env file and inject variables into os.environ if not already present."""
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


def _read_config_yaml_settings(
    plugin_id: str = "hermes-laya-tools",
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Read plugin settings from config.yaml if available."""
    if yaml is None:
        return {}
    cfg_path = _find_hermes_config_file(config_path=config_path)
    if not cfg_path or not cfg_path.is_file():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            entries = data.get("plugins", {}).get("entries", {})
            plugin_entry = entries.get(plugin_id, {})
            settings = plugin_entry.get("settings") or plugin_entry.get("config") or {}
            if isinstance(settings, dict):
                return settings
    except Exception:
        pass
    return {}


def clear_yaml_setting(
    key: str,
    plugin_id: str = "hermes-laya-tools",
    config_path: Path | str | None = None,
) -> None:
    """Clear or reset a transient plugin setting (e.g. recategorize_now: true -> false) in config.yaml."""
    cfg_path = _find_hermes_config_file(config_path=config_path)
    if not cfg_path or not cfg_path.is_file():
        return
    try:
        import re
        text = cfg_path.read_text(encoding="utf-8")
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*)(?:true|yes|1)\b", re.IGNORECASE | re.MULTILINE)
        new_text, count = pattern.subn(r"\g<1>false", text)
        if count > 0:
            cfg_path.write_text(new_text, encoding="utf-8")
    except Exception:
        pass


def _resolve_val(
    ctx: Any,
    yaml_settings: Mapping[str, Any],
    key: str,
    env_vars: tuple[str, ...],
    default: Any,
) -> Any:
    """Resolve config value respecting hierarchy: ctx -> env -> yaml -> default."""
    if ctx is not None and hasattr(ctx, "get_config"):
        try:
            val = ctx.get_config(key, None)
            if val is not None:
                return val
        except Exception:
            pass

    for ev in env_vars:
        if ev in os.environ:
            return os.environ[ev]

    if key in yaml_settings and yaml_settings[key] is not None:
        return yaml_settings[key]

    return default


def _to_bool(val: Any, default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def _to_int(val: Any, default: int) -> int:
    if val is None:
        return default
    if isinstance(val, int):
        return val
    try:
        return int(str(val).strip())
    except ValueError:
        return default


def _to_float(val: Any, default: float) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).strip())
    except ValueError:
        return default


def _to_set(val: Any, default: Set[str]) -> Set[str]:
    if val is None:
        return default
    if isinstance(val, (set, list, tuple)):
        return {str(x).strip() for x in val if str(x).strip()}
    if isinstance(val, str):
        return {x.strip() for x in val.split(",") if x.strip()}
    return default


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
    direct_chat_threshold: float = 0.75  # Safety threshold: only strip tools if direct_chat prob >= 75%
    min_confidence: float = 0.15  # Minimum confidence required to strip tools
    log_path: Path = field(default_factory=lambda: Path.home() / ".hermes" / "logs" / "laya_requests.jsonl")
    log_level: str = "INFO"
    cache_repeat_asks: bool = True  # LRU cache for identical instructions
    cache_size: int = 512  # Maximum number of cached decisions
    trivial_bypass: bool = True  # Instant local bypass for acknowledgements/greetings (<1ms)
    privacy_mode: bool = True  # Mask API keys, tokens and sensitive credentials in logs
    recategorize_now: bool = False  # Trigger flag from settings UI for immediate recategorization

    @property
    def is_shadow(self) -> bool:
        """True if running in shadow mode (evaluation and logging only, no tool mutation)."""
        return self.mode.lower() == "shadow"

    @property
    def is_killed(self) -> bool:
        """Check hardware kill-switch: env LAYA_OFF or marker file <hermes_home>/laya/LAYA_OFF."""
        if os.environ.get("LAYA_OFF", "").strip().lower() in ("1", "true", "yes", "on"):
            return True
        hermes_home = os.environ.get("HERMES_HOME")
        base = Path(hermes_home) if hermes_home else Path.home() / ".hermes"
        if (base / "laya" / "LAYA_OFF").is_file() or (base / "LAYA_OFF").is_file():
            return True
        return False

    @classmethod
    def from_ctx_or_env(
        cls,
        ctx: Any = None,
        env_path: Path | str | None = None,
        config_path: Path | str | None = None,
    ) -> LayaConfig:
        """Create configuration instance reading from Hermes ctx, config.yaml, or .env."""
        load_hermes_env(env_path)
        if env_path is not None and config_path is None:
            yaml_settings = {}
        else:
            yaml_settings = _read_config_yaml_settings("hermes-laya-tools", config_path=config_path)

        raw_url = _resolve_val(
            ctx, yaml_settings, "sidecar_url", ("LAYA_SIDECAR_URL", "LAYA_URL"), "http://127.0.0.1:8787"
        )
        url = str(raw_url).rstrip("/")

        enabled = _to_bool(
            _resolve_val(ctx, yaml_settings, "enabled", ("LAYA_ENABLED",), True),
            default=True,
        )

        # Mode: "active" | "shadow" (also support LAYA_SHADOW_MODE=true)
        raw_mode = str(
            _resolve_val(ctx, yaml_settings, "mode", ("LAYA_MODE",), "active")
        ).strip().lower()
        if _to_bool(os.environ.get("LAYA_SHADOW_MODE"), False) or raw_mode == "shadow":
            mode = "shadow"
        else:
            mode = "active"

        hierarchical = _to_bool(
            _resolve_val(ctx, yaml_settings, "hierarchical", ("LAYA_HIERARCHICAL",), True),
            default=True,
        )

        max_tools = max(
            1,
            _to_int(
                _resolve_val(ctx, yaml_settings, "max_tools", ("LAYA_MAX_TOOLS",), 10),
                default=10,
            ),
        )

        timeout_ms = max(
            500,
            _to_int(
                _resolve_val(ctx, yaml_settings, "timeout_ms", ("LAYA_TIMEOUT_MS",), 5000),
                default=5000,
            ),
        )

        cat_interval = max(
            60_000,
            _to_int(
                _resolve_val(
                    ctx,
                    yaml_settings,
                    "categorization_interval_ms",
                    ("LAYA_CATEGORIZATION_INTERVAL_MS",),
                    86_400_000,
                ),
                default=86_400_000,
            ),
        )

        fallback = str(
            _resolve_val(ctx, yaml_settings, "fallback_mode", ("LAYA_FALLBACK_MODE",), "pass-all")
        ).strip().lower()
        if fallback not in ("pass-all", "strip-all"):
            fallback = "pass-all"

        # In shadow mode notify_in_chat is strictly False
        default_notify = False if mode == "shadow" else True
        notify_in_chat = _to_bool(
            _resolve_val(ctx, yaml_settings, "notify_in_chat", ("LAYA_NOTIFY_IN_CHAT",), default_notify),
            default=default_notify,
        )
        if mode == "shadow":
            notify_in_chat = False

        direct_chat_threshold = _to_float(
            _resolve_val(
                ctx,
                yaml_settings,
                "direct_chat_threshold",
                ("LAYA_DIRECT_CHAT_THRESHOLD",),
                0.75,
            ),
            default=0.75,
        )

        min_confidence = _to_float(
            _resolve_val(
                ctx,
                yaml_settings,
                "min_confidence",
                ("LAYA_MIN_CONFIDENCE",),
                0.15,
            ),
            default=0.15,
        )

        always_raw = _resolve_val(
            ctx, yaml_settings, "always_include", ("LAYA_ALWAYS_INCLUDE",), set()
        )
        always_include = _to_set(always_raw, default=set())

        log_level = str(
            _resolve_val(ctx, yaml_settings, "log_level", ("LAYA_LOG_LEVEL",), "INFO")
        ).strip().upper()

        hermes_home = os.environ.get("HERMES_HOME")
        base = Path(hermes_home) if hermes_home else Path.home() / ".hermes"

        cache_raw = _resolve_val(
            ctx, yaml_settings, "cache_path", ("LAYA_CACHE_PATH",), base / "cache" / "laya_categories.json"
        )
        cache_path = Path(str(cache_raw)).expanduser()

        log_raw = _resolve_val(
            ctx, yaml_settings, "log_path", ("LAYA_LOG_PATH",), base / "logs" / "laya_requests.jsonl"
        )
        log_path = Path(str(log_raw)).expanduser()

        cache_repeat_asks = _to_bool(
            _resolve_val(ctx, yaml_settings, "cache_repeat_asks", ("LAYA_CACHE_REPEAT_ASKS",), True),
            default=True,
        )

        cache_size = max(
            16,
            _to_int(
                _resolve_val(ctx, yaml_settings, "cache_size", ("LAYA_CACHE_SIZE",), 512),
                default=512,
            ),
        )

        trivial_bypass = _to_bool(
            _resolve_val(ctx, yaml_settings, "trivial_bypass", ("LAYA_TRIVIAL_BYPASS",), True),
            default=True,
        )

        privacy_mode = _to_bool(
            _resolve_val(ctx, yaml_settings, "privacy_mode", ("LAYA_PRIVACY_MODE",), True),
            default=True,
        )

        recategorize_now = _to_bool(
            _resolve_val(ctx, yaml_settings, "recategorize_now", ("LAYA_RECATEGORIZE_NOW",), False),
            default=False,
        )

        config_obj = cls(
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
            direct_chat_threshold=direct_chat_threshold,
            min_confidence=min_confidence,
            log_path=log_path,
            log_level=log_level,
            cache_repeat_asks=cache_repeat_asks,
            cache_size=cache_size,
            trivial_bypass=trivial_bypass,
            privacy_mode=privacy_mode,
            recategorize_now=recategorize_now,
        )

        # If hardware kill-switch is active, forcefully disable
        if config_obj.is_killed:
            config_obj.enabled = False

        return config_obj

    @classmethod
    def from_env(
        cls,
        env_path: Path | str | None = None,
        config_path: Path | str | None = None,
    ) -> LayaConfig:
        """Create configuration instance from environment variables (.env)."""
        return cls.from_ctx_or_env(ctx=None, env_path=env_path, config_path=config_path)

    def reload(self, ctx: Any = None) -> None:
        """Reload configuration values in-place from ctx, config.yaml, or environment."""
        new_cfg = self.from_ctx_or_env(ctx=ctx)
        self.sidecar_url = new_cfg.sidecar_url
        self.enabled = new_cfg.enabled
        self.mode = new_cfg.mode
        self.hierarchical = new_cfg.hierarchical
        self.max_tools = new_cfg.max_tools
        self.timeout_ms = new_cfg.timeout_ms
        self.categorization_interval_ms = new_cfg.categorization_interval_ms
        self.cache_path = new_cfg.cache_path
        self.always_include = new_cfg.always_include
        self.fallback_mode = new_cfg.fallback_mode
        self.notify_in_chat = new_cfg.notify_in_chat
        self.direct_chat_threshold = new_cfg.direct_chat_threshold
        self.min_confidence = new_cfg.min_confidence
        self.log_path = new_cfg.log_path
        self.log_level = new_cfg.log_level
        self.cache_repeat_asks = new_cfg.cache_repeat_asks
        self.cache_size = new_cfg.cache_size
        self.trivial_bypass = new_cfg.trivial_bypass
        self.privacy_mode = new_cfg.privacy_mode
        self.recategorize_now = new_cfg.recategorize_now
