"""Laya HTTP client for System 1 Decision Engine.

Communicates with Laya's REST API (/ask, /health, /reload) using Python stdlib.
Zero external dependencies required.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugin.laya.client")


class LayaClientError(Exception):
    """Raised when a request to Laya Decision Engine fails."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class LayaClient:
    """Client for Laya System 1 REST API."""

    def __init__(
        self,
        sidecar_url: str = "http://127.0.0.1:8787",
        timeout_ms: int = 5000,
        custom_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.sidecar_url = sidecar_url.rstrip("/")
        self.timeout_sec = max(0.5, timeout_ms / 1000.0)
        self.logger = custom_logger or logger

    def health(self) -> Optional[Dict[str, Any]]:
        """Check if Laya engine is healthy.

        Returns health dictionary or None on failure.
        """
        url = f"{self.sidecar_url}/health"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        timeout = min(self.timeout_sec, 3.0)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("ok") is True or data.get("status") == "ok":
                        return data
            return None
        except Exception as exc:
            self.logger.debug("Laya health check failed: %s", exc)
            return None

    def ask(
        self,
        state: str,
        questions: Dict[str, Any],
        strict: bool = False,
        lang: str = "auto",
        max_state_chars: int = 1800,
    ) -> Dict[str, Any]:
        """Send typed questions to Laya System 1 decision engine.

        Args:
            state: Context or user prompt string.
            questions: Dictionary of questions (type: choice, criteria, instructions).
            strict: Strict parsing flag.
            lang: Language ('auto', 'ru', 'en').
            max_state_chars: Truncate state to fit model budget.

        Returns:
            Parsed JSON response from Laya with 'answers', 'latency_ms', etc.
        """
        url = f"{self.sidecar_url}/ask"
        truncated_state = self.truncate_state(state, max_state_chars)
        payload = {
            "state": truncated_state,
            "questions": questions,
            "strict": strict,
            "lang": lang,
        }

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                content = resp.read().decode("utf-8")
                data = json.loads(content)
                if not data.get("ok") and data.get("error"):
                    raise LayaClientError(
                        f"Laya error: {data.get('error')} ({data.get('hint', '')})",
                        status_code=resp.status,
                        details=data,
                    )
                return data
        except urllib.error.HTTPError as http_err:
            try:
                err_body = http_err.read().decode("utf-8")
            except Exception:
                err_body = str(http_err)
            raise LayaClientError(
                f"Laya HTTP {http_err.code}: {err_body}",
                status_code=http_err.code,
                details=err_body,
            ) from http_err
        except urllib.error.URLError as url_err:
            raise LayaClientError(f"Laya connection error: {url_err.reason}") from url_err
        except Exception as exc:
            raise LayaClientError(f"Laya request failed: {exc}") from exc

    def truncate_state(self, text: str, max_chars: int = 1800) -> str:
        """Safely truncate text preserving head and tail portions."""
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        head_len = int(max_chars * 0.3)
        tail_len = max_chars - head_len - 20
        return f"{text[:head_len]}\n...[truncated]...\n{text[-tail_len:]}"

    def reload_model(self, timeout_sec: float = 25.0) -> Dict[str, Any]:
        """Hot-reload model weights in Laya inference engine."""
        url = f"{self.sidecar_url}/reload"
        req = urllib.request.Request(
            url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise LayaClientError(f"Model reload failed: {exc}") from exc
