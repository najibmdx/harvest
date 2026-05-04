"""Sanitization utilities for Hunter Phase 0 audit outputs."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

REDACTED = "[REDACTED]"

SENSITIVE_KEY_PATTERNS = [
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "token",
    "bearer",
    "cookie",
    "set-cookie",
    "secret",
    "password",
]

EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
ETH_ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")
B64ISH_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{10,}\.?[A-Za-z0-9_-]*\b")
HEX_LONG_RE = re.compile(r"\b[a-fA-F0-9]{32,}\b")


def mask_wallet_address(value: str) -> str:
    if ETH_ADDR_RE.fullmatch(value):
        return f"{value[:6]}...{value[-4:]}"
    if len(value) >= 20 and re.fullmatch(r"[A-Za-z0-9]+", value):
        return f"{value[:6]}...{value[-4:]}"
    return value


def sanitize_string(value: str) -> str:
    text = value
    text = EMAIL_RE.sub(REDACTED, text)
    text = IPV4_RE.sub(REDACTED, text)
    text = ETH_ADDR_RE.sub(lambda m: mask_wallet_address(m.group(0)), text)
    text = B64ISH_TOKEN_RE.sub(REDACTED, text)
    text = HEX_LONG_RE.sub(REDACTED, text)

    lowered = text.lower()
    if lowered.startswith("bearer "):
        return "Bearer [REDACTED]"
    return text


def _is_sensitive_key(key: str) -> bool:
    k = key.lower()
    return any(pattern in k for pattern in SENSITIVE_KEY_PATTERNS)


def sanitize_data(obj: Any, parent_key: str = "") -> Any:
    """Recursively sanitize data structures for safe storage."""
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            if _is_sensitive_key(key):
                new_obj[key] = REDACTED
            else:
                new_obj[key] = sanitize_data(value, parent_key=key)
        return new_obj

    if isinstance(obj, list):
        return [sanitize_data(item, parent_key=parent_key) for item in obj]

    if isinstance(obj, str):
        if _is_sensitive_key(parent_key):
            return REDACTED
        return sanitize_string(mask_wallet_address(obj))

    return deepcopy(obj)


def sanitize_headers(headers: dict[str, str] | None) -> dict[str, str]:
    if not headers:
        return {}
    return {k: (REDACTED if _is_sensitive_key(k) else sanitize_string(v)) for k, v in headers.items()}
