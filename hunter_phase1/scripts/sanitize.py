#!/usr/bin/env python3
"""Sanitization helpers for Hunter Phase 1 evidence capture."""

from __future__ import annotations

import copy
from typing import Any

SENSITIVE_KEYWORDS = {
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "token",
    "secret",
    "password",
}


def redact_address(address: str) -> str:
    if not address:
        return ""
    if len(address) <= 10:
        return "***REDACTED***"
    return f"{address[:6]}...{address[-4:]}"


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize raw API payloads for safe-at-rest storage."""
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in SENSITIVE_KEYWORDS:
                clean[key] = "***REDACTED***"
            else:
                clean[key] = sanitize_value(item)
        return clean

    if isinstance(value, list):
        return [sanitize_value(item) for item in value]

    if isinstance(value, str):
        # Avoid accidentally persisting any header-like key material
        lowered = value.lower()
        if "api-key" in lowered or "bearer " in lowered:
            return "***REDACTED***"

    return copy.deepcopy(value)
