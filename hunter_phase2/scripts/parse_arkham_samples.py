#!/usr/bin/env python3
"""Utility parsers for Hunter Phase 2 Arkham raw samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

CHUNK_SIZE = 1024 * 1024


def safe_json_load(path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Load JSON with explicit guardrails for malformed and huge files."""
    meta: Dict[str, Any] = {
        "file": str(path),
        "status": "ok",
        "errors": [],
        "bytes": 0,
    }
    if not path.exists():
        meta["status"] = "error"
        meta["errors"].append("file_missing")
        return None, meta

    try:
        size = path.stat().st_size
        meta["bytes"] = size
        text_parts: List[str] = []
        with path.open("r", encoding="utf-8") as fh:
            while True:
                chunk = fh.read(CHUNK_SIZE)
                if not chunk:
                    break
                text_parts.append(chunk)
        text = "".join(text_parts).strip()
        if not text:
            meta["status"] = "empty"
            return None, meta
        payload = json.loads(text)
        return payload, meta
    except json.JSONDecodeError as exc:
        meta["status"] = "error"
        meta["errors"].append(f"malformed_json:{exc.msg}@{exc.lineno}:{exc.colno}")
        return None, meta
    except OSError as exc:
        meta["status"] = "error"
        meta["errors"].append(f"io_error:{exc}")
        return None, meta


def _iter_records(payload: Any, chain_hint: str | None = None) -> Iterable[Dict[str, Any]]:
    if payload is None:
        return
    if isinstance(payload, dict):
        if any(isinstance(v, list) for v in payload.values()):
            for key, val in payload.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            if chain_hint and "chain" not in item:
                                item = {**item, "chain": chain_hint}
                            yield item
                elif isinstance(val, dict):
                    yield from _iter_records(val, key)
        else:
            if chain_hint and "chain" not in payload:
                payload = {**payload, "chain": chain_hint}
            yield payload
            for key, val in payload.items():
                if isinstance(val, (dict, list)):
                    yield from _iter_records(val, key)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                if chain_hint and "chain" not in item:
                    item = {**item, "chain": chain_hint}
                yield item
            elif isinstance(item, list):
                yield from _iter_records(item, chain_hint)


def parse_sample_file(path: Path, max_records: int = 10000) -> Dict[str, Any]:
    payload, meta = safe_json_load(path)
    result: Dict[str, Any] = {
        "file": str(path),
        "meta": meta,
        "records": [],
        "record_count": 0,
        "truncated": False,
    }
    if meta["status"] in {"error", "empty"}:
        return result

    count = 0
    for rec in _iter_records(payload):
        result["records"].append(rec)
        count += 1
        if count >= max_records:
            result["truncated"] = True
            break
    result["record_count"] = len(result["records"])
    if result["record_count"] == 0 and isinstance(payload, dict):
        result["records"].append(payload)
        result["record_count"] = 1
    return result
