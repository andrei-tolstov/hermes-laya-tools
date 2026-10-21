"""Context hygiene, fast-bypass heuristics, and PII/secret redaction.

Implements battle-tested operational patterns inspired by hermes-jev-skills:
1. looks_trivial: Local instant bypass for acknowledgements, greetings, and trivial replies (<1ms).
2. unwrap: Stripping markdown envelopes and previous run logs from cron/subagent prompts.
3. redact_sensitive: Masking API keys, tokens, credentials, and emails before logging.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

# Punctuation splitter
_WORD_SPLIT = re.compile(r"[^\w]+", re.UNICODE)

# Common acknowledgment, greeting, and confirmation words (RU + EN)
_ACK_WORDS: Set[str] = {
    # Russian
    "ок", "окей", "да", "ага", "угу", "хорошо", "понял", "понятно", "ясно",
    "спасибо", "спс", "благодарю", "привет", "здравствуйте", "хай", "добрый",
    "день", "вечер", "утро", "продолжай", "давай", "вперед", "отлично",
    "ладно", "супер", "принято", "точно", "согласен", "верно",
    # English
    "ok", "okay", "yes", "yeah", "yep", "sure", "thanks", "thx", "thank",
    "you", "got", "it", "understood", "hello", "hi", "hey", "good",
    "morning", "evening", "proceed", "continue", "ahead", "great",
    "fine", "acknowledged", "agreed", "correct", "perfect",
}

# Words that explicitly indicate command/tool intent even if short
_ACTION_WORDS: Set[str] = {
    "запусти", "выполни", "сделай", "удали", "создай", "открой", "покажи",
    "найди", "проверь", "напиши", "скачай", "исправь", "перезапусти", "клонируй",
    "run", "exec", "execute", "start", "stop", "delete", "remove", "create",
    "open", "show", "find", "search", "check", "write", "download", "fix",
    "git", "bash", "curl", "python", "cat", "grep", "ls", "cd", "mkdir",
}

# Markdown section headers for task unwrapping
_H2_OR_H3 = re.compile(r"(?m)^[ \t]*#{2,3}[ \t]+(\S[^\n]*?)[ \t]*$")
_WANTED_SECTIONS: List[str] = ["prompt", "task", "request", "instruction", "objective", "goal"]
_UNWANTED_SECTIONS: List[str] = [
    "previous output", "prior run", "last run", "script output",
    "output from job", "your previous run's output", "context",
]

# Sensitive patterns for redaction
_SECRET_PATTERNS = [
    # GitHub PAT / tokens
    (re.compile(r"github_pat_[a-zA-Z0-9_]{30,}", re.IGNORECASE), "[REDACTED_GH_TOKEN]"),
    # Generic API keys / Bearer tokens
    (re.compile(r"(?i)(bearer\s+|api[_-]?key[\s:=]+|token[\s:=]+)[a-zA-Z0-9_\-\.]{16,}"), r"\1[REDACTED_TOKEN]"),
    # OpenAI / Anthropic / TypeSafe / AI keys
    (re.compile(r"\b(sk-[a-zA-Z0-9_-]{20,})\b"), "[REDACTED_API_KEY]"),
    # SSH / RSA / EC private keys
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
    # Passwords in URLs (http://user:pass@host)
    (re.compile(r"(https?://[^:]+:)[^@]+(@)"), r"\1[REDACTED_PASS]\2"),
    # Emails
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
]


def looks_trivial(text: str) -> bool:
    """Check if the user turn is purely conversational acknowledgment or greeting.

    Returns True if the turn contains <= 6 words, all of which are common acknowledgements
    or greetings, and does not contain any tool/action verbs.
    """
    if not text:
        return True

    words = [w.lower() for w in _WORD_SPLIT.split(text.strip()) if w]
    if not words or len(words) > 6:
        return False

    # Any explicit action word forbids trivial classification
    if any(w in _ACTION_WORDS for w in words):
        return False

    # All words must be in known ack/greeting dictionary
    return all(w in _ACK_WORDS for w in words)


def unwrap(text: str) -> str:
    """Unwrap the real instruction from structured cron/agent envelopes.

    Scheduled cron jobs or queued tasks often wrap a short instruction in a large
    standard contract (anti-stall, status formats, previous run's output).
    If a section explicitly names the request (## Prompt, ## Task, ## Instruction),
    we extract only that section. If unwanted previous outputs are present, we strip them.
    """
    if not text or "#" not in text:
        return text

    heads = list(_H2_OR_H3.finditer(text))
    if not heads:
        return text

    # Check for wanted instruction section
    for idx, match in enumerate(heads):
        title = match.group(1).strip().lower().rstrip(":")
        if any(title == w or title.startswith(w + " ") for w in _WANTED_SECTIONS):
            end = heads[idx + 1].start() if idx + 1 < len(heads) else len(text)
            body = text[match.end():end].strip()
            if len(body) >= 10:
                return body

    # Otherwise strip unwanted sections (previous logs/outputs)
    keep: List[str] = []
    cut = 0
    trimmed_any = False
    for idx, match in enumerate(heads):
        title = match.group(1).strip().lower().rstrip(":")
        if any(title.startswith(w) for w in _UNWANTED_SECTIONS):
            end = heads[idx + 1].start() if idx + 1 < len(heads) else len(text)
            keep.append(text[cut:match.start()])
            cut = end
            trimmed_any = True

    if trimmed_any:
        keep.append(text[cut:])
        candidate = "".join(keep).strip()
        if len(candidate) >= 10:
            return candidate

    return text


def redact_sensitive(text: str) -> str:
    """Mask tokens, private keys, passwords, and sensitive PII from log output."""
    if not text:
        return ""

    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted
