#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlencode

import requests


def load_env() -> dict[str, str]:
    return {
        "api_key": os.getenv("ARKHAM_API_KEY", "").strip(),
        "base_url": os.getenv("ARKHAM_API_BASE_URL", "").strip().rstrip("/"),
        "auth_mode": (os.getenv("ARKHAM_AUTH_MODE", "api-key").strip().lower() or "api-key"),
        "openapi_file": os.getenv("ARKHAM_OPENAPI_FILE", "").strip(),
    }


def build_headers(auth_mode: str, api_key: str) -> tuple[dict[str, str], dict[str, bool]]:
    if auth_mode != "api-key":
        raise ValueError("ARKHAM_AUTH_MODE must be api-key")
    headers = {"API-Key": api_key, "Accept": "application/json", "User-Agent": "HunterPhase1EvidenceCapture/1.0"}
    sent = {"authorization": False, "api-key": True, "x-api-key": False, "accept": True, "user-agent": True}
    return headers, sent


def sanitize_response(payload: Any) -> Any:
    if isinstance(payload, (dict, list)):
        return payload
    return {"raw_text": str(payload)[:4000]}


def redact_url(url: str, address: str) -> str:
    if not address:
        return url
    return url.replace(address, f"{address[:6]}...{address[-4:]}")


def get_json(base_url: str, path: str, headers: dict[str, str], path_params: dict[str, str] | None = None, query_params: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any]:
    path_params = path_params or {}
    query_params = query_params or {}
    url = base_url.rstrip("/") + path
    for k, v in path_params.items():
        url = url.replace("{" + k + "}", v)
    response = requests.get(url, headers=headers, timeout=timeout, params=query_params if query_params else None)
    try:
        body = response.json()
    except Exception:
        body = {"raw_text": response.text[:4000], "parse": "non-json"}

    redacted = url
    if query_params:
        redacted = redacted + "?" + urlencode(query_params)
    return {"status_code": response.status_code, "body": sanitize_response(body), "url": url, "url_redacted": redacted}
