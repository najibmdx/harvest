#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError, URLError

from sanitize import redact_address, sanitize_value

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "wallet_inputs.json"
OUTPUT_DIR = ROOT / "output"
RAW_DIR = OUTPUT_DIR / "raw_samples"
INDEX_PATH = OUTPUT_DIR / "wallet_evidence_index.json"
MATRIX_PATH = OUTPUT_DIR / "endpoint_success_matrix.md"
MISSING_PATH = OUTPUT_DIR / "missing_evidence_report.md"
RECOMMEND_PATH = OUTPUT_DIR / "phase1_recommendation.md"
AUTH_DIAG_PATH = OUTPUT_DIR / "phase1_auth_diagnostics.md"
REQUEST_DIAG_PATH = OUTPUT_DIR / "phase1_request_diagnostics.json"
SANITY_SAMPLE_PATH = RAW_DIR / "auth_sanity_check.json"

ENDPOINTS = {
    "balances": "/balances/address/{address}",
    "history": "/history/address/{address}",
    "portfolio": "/portfolio/address/{address}",
    "flow": "/flow/address/{address}",
    "counterparties": "/counterparties/address/{address}",
    "intelligence": "/intelligence/address/{address}",
    "intelligence_enriched": "/intelligence/address_enriched/{address}",
    "transfers": "/transfers",
}

TIME_FILTER_KEYS = {"timeGte", "timeLte", "timeLast", "since", "from", "to"}


@dataclass
class ApiResult:
    endpoint: str
    status: int
    body: Any
    error: str | None
    method: str
    resolved_url_template: str
    final_url_redacted: str
    path_params_used: dict[str, str]
    query_params_sent: dict[str, Any]


def load_env() -> dict[str, str]:
    return {
        "api_key": os.getenv("ARKHAM_API_KEY", "").strip(),
        "base_url": os.getenv("ARKHAM_API_BASE_URL", "").strip().rstrip("/"),
        "auth_mode": os.getenv("ARKHAM_AUTH_MODE", "api-key").strip().lower() or "api-key",
        "openapi_file": os.getenv("ARKHAM_OPENAPI_FILE", "").strip(),
    }


def classify_403(payload: Any) -> str:
    text = json.dumps(payload).lower() if isinstance(payload, (dict, list)) else str(payload).lower()
    if any(x in text for x in ["api key", "apikey", "auth", "sign up"]):
        return "auth failure"
    if any(x in text for x in ["forbidden", "permission", "tier"]):
        return "permission or tier restriction"
    if "timestamp" in text:
        return "timestamp failure"
    return "forbidden unknown"


def result_category(status: int, payload: Any) -> str:
    if status == 403:
        return classify_403(payload)
    if 200 <= status < 300:
        return "endpoint working"
    if status in {401}:
        return "auth failure"
    if status == 0:
        return "request transport failure"
    return "other failure"


def pick_sanity_endpoint(openapi_file: str) -> str | None:
    if not openapi_file:
        return None
    p = Path(openapi_file)
    if not p.exists():
        p = (ROOT / openapi_file).resolve()
    if not p.exists():
        return None
    try:
        spec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    for path, methods in (spec.get("paths") or {}).items():
        if not isinstance(methods, dict) or "get" not in methods:
            continue
        m = methods.get("get") or {}
        params = m.get("parameters") or []
        if "{" in path or "}" in path:
            continue
        has_required = any((param or {}).get("required") for param in params)
        if has_required:
            continue
        return path
    return None


def call_api(base_url: str, api_key: str, endpoint_name: str, address: str, extra_query: dict[str, Any] | None = None) -> ApiResult:
    path = ENDPOINTS[endpoint_name]
    path_params_used: dict[str, str] = {}
    query_params_sent = extra_query.copy() if extra_query else {}

    if endpoint_name == "transfers":
        query_params_sent["address"] = address
        url = f"{base_url}{path}?{parse.urlencode(query_params_sent)}"
    else:
        path_params_used = {"address": redact_address(address)}
        url = f"{base_url}{path.format(address=address)}"
        if query_params_sent:
            url = f"{url}?{parse.urlencode(query_params_sent)}"

    req = request.Request(url, method="GET")
    req.add_header("API-Key", api_key)

    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
            status = resp.status
            err = None
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = {"raw": raw}
        status = e.code
        err = str(e)
    except URLError as e:
        data = {"error": str(e)}
        status = 0
        err = str(e)

    return ApiResult(endpoint_name, status, data, err, "GET", path, url.replace(address, redact_address(address)), path_params_used, query_params_sent)


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    env = load_env()

    if env["auth_mode"] != "api-key":
        raise SystemExit("ARKHAM_AUTH_MODE must be api-key for Hunter Phase 1")
    if not env["api_key"] or not env["base_url"]:
        raise SystemExit("ARKHAM_API_KEY and ARKHAM_API_BASE_URL are required")
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing required input file: {CONFIG_PATH}")

    api_key = env["api_key"]
    key_prefix = f"{api_key[:4]}***" if api_key else "N/A"
    auth_diag = [
        "# Phase 1 Auth Diagnostics",
        f"- ARKHAM_API_KEY exists: {'yes' if bool(api_key) else 'no'}",
        f"- API key length: {len(api_key)}",
        f"- API key prefix first 4 chars only: {key_prefix}",
        f"- ARKHAM_API_BASE_URL value: {env['base_url']}",
        f"- ARKHAM_AUTH_MODE value: {env['auth_mode']}",
        "- Whether API-Key header was sent: yes",
        "- Whether Authorization header was sent: no",
        "- Whether X-API-Key header was sent: no",
    ]
    write_md(AUTH_DIAG_PATH, auth_diag)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    wallets = config.get("wallets", [])

    sanity_path = pick_sanity_endpoint(env.get("openapi_file", ""))
    sanity = {"status_code": "not_run", "auth_worked": False, "result_category": "no sanity endpoint available"}
    if sanity_path:
        req = request.Request(f"{env['base_url']}{sanity_path}", method="GET")
        req.add_header("API-Key", api_key)
        try:
            with request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8") or "{}")
                sanity = {"status_code": resp.status, "auth_worked": 200 <= resp.status < 300, "result_category": result_category(resp.status, body), "endpoint": sanity_path, "body": sanitize_value(body)}
        except HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
            payload = json.loads(raw) if raw.startswith("{") else {"raw": raw}
            sanity = {"status_code": e.code, "auth_worked": False, "result_category": result_category(e.code, payload), "endpoint": sanity_path, "body": sanitize_value(payload)}
    SANITY_SAMPLE_PATH.write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    index_items = []
    request_diags = []
    stats = {name: {"tested": 0, "success": 0, "fail": 0, "errors": Counter()} for name in ENDPOINTS}

    for wallet in wallets:
        label = wallet.get("label", "unlabeled_wallet")
        address = wallet.get("address", "")
        chain = wallet.get("chain", "unknown")
        wallet_filters = {k: v for k, v in wallet.items() if k in TIME_FILTER_KEYS}

        item = {"wallet_label": label, "chain": chain, "address_redacted": redact_address(address), "endpoints_attempted": []}
        per_endpoint = {}
        avail, miss, errors = [], [], []

        for endpoint_name in ENDPOINTS:
            stats[endpoint_name]["tested"] += 1
            result = call_api(env["base_url"], api_key, endpoint_name, address, wallet_filters)
            safe_body = sanitize_value(result.body)
            sample = RAW_DIR / f"{label}__{endpoint_name}.json"
            sample.write_text(json.dumps({"endpoint": endpoint_name, "status": result.status, "data": safe_body}, indent=2), encoding="utf-8")
            rc = result_category(result.status, safe_body)

            per_endpoint[endpoint_name] = {
                "resolved_url_template": result.resolved_url_template,
                "final_url_redacted": result.final_url_redacted,
                "path_params_used": result.path_params_used,
                "query_params_sent": result.query_params_sent,
                "auth_mode_used": env["auth_mode"],
                "request_had_api_key_header": True,
                "status_code": result.status,
                "sanitized_error_body": safe_body if result.status >= 400 else {},
                "sample_file_path": str(sample.relative_to(ROOT)),
            }
            if 200 <= result.status < 300:
                avail.append(endpoint_name)
                stats[endpoint_name]["success"] += 1
            else:
                miss.append(endpoint_name)
                stats[endpoint_name]["fail"] += 1
                if result.error:
                    errors.append(f"{endpoint_name}: {result.error}")
                    stats[endpoint_name]["errors"][rc] += 1

            request_diags.append({
                "wallet_label": label,
                "category": endpoint_name,
                "method": result.method,
                "url_redacted": result.final_url_redacted,
                "path_params_used": result.path_params_used,
                "query_params_sent": result.query_params_sent,
                "auth_mode_used": env["auth_mode"],
                "status_code": result.status,
                "result_category": rc,
                "sanitized_response_body": safe_body,
                "sample_file": str(sample.relative_to(ROOT)),
            })

        item["endpoints_attempted"] = list(ENDPOINTS.keys())
        item["endpoint_request_diagnostics"] = per_endpoint
        item["available_evidence_categories"] = avail
        item["missing_evidence_categories"] = miss
        item["errors"] = errors
        index_items.append(item)

    INDEX_PATH.write_text(json.dumps({"wallets": index_items}, indent=2), encoding="utf-8")
    REQUEST_DIAG_PATH.write_text(json.dumps(request_diags, indent=2), encoding="utf-8")

    lines = ["# Endpoint Success Matrix", "", "| endpoint | tested count | success count | failure count | common error | usable for Phase 2 |", "|---|---:|---:|---:|---|---|"]
    for ep, row in stats.items():
        ce = row["errors"].most_common(1)[0][0] if row["errors"] else "n/a"
        usable = "yes" if row["fail"] == 0 else ("partial" if row["success"] > 0 else "no")
        lines.append(f"| {ep} | {row['tested']} | {row['success']} | {row['fail']} | {ce} | {usable} |")
    write_md(MATRIX_PATH, lines)

    miss_lines = ["# Missing Evidence Report", ""]
    for wi in index_items:
        miss_lines.append(f"## {wi['wallet_label']}")
        if wi["missing_evidence_categories"]:
            for c in wi["missing_evidence_categories"]:
                miss_lines.append(f"- {c}: blocked/missing")
        else:
            miss_lines.append("- None")
        miss_lines.append("")
    write_md(MISSING_PATH, miss_lines)

    any_wallet_success = any(w["available_evidence_categories"] for w in index_items)
    if sanity.get("result_category") == "auth failure":
        rec = "C) Phase 1 evidence capture is insufficient because auth/request construction failed."
    elif sanity.get("auth_worked") and not any_wallet_success:
        rec = "B) Phase 1 evidence capture is partially sufficient but wallet endpoints are blocked by permission/tier/fixture issue."
    elif any_wallet_success:
        rec = "A) Phase 1 evidence capture is sufficient to proceed to wallet performance reconstruction."
    else:
        rec = "C) Phase 1 evidence capture is insufficient"
    write_md(RECOMMEND_PATH, ["# Phase 1 Recommendation", "", rec])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
