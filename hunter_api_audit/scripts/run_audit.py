#!/usr/bin/env python3
"""Hunter Phase 0: Arkham API Capability Auditor."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

from sanitize import sanitize_data, sanitize_headers

CAPABILITIES = [
    "wallet/address lookup",
    "entity labels",
    "tags/categories",
    "transaction history",
    "token transfers",
    "balance history",
    "CEX wallet identification",
    "deposit/withdrawal detection",
    "historical timestamps",
    "pagination",
    "rate limits",
    "PnL data",
    "realized PnL directly available",
    "price history",
]

FIXTURE_PARAM_HINTS = {
    "address": [("addresses", "evm"), ("addresses", "btc"), ("addresses", "solana")],
    "entity": [("entities", "exchange"), ("entities", "fund"), ("entities", "project")],
    "hash": [("transactions", "evm_hash"), ("transactions", "btc_hash"), ("transactions", "solana_hash")],
    "token": [("tokens", "evm_token_address"), ("tokens", "solana_token_address"), ("tokens", "pricing_id")],
    "chain": [("tokens", "evm_chain")],
    "tag": [("tags", "tag_id")],
}


def ensure_dirs(base: Path) -> dict[str, Path]:
    out = base / "output"
    sample = out / "sample_responses"
    sample.mkdir(parents=True, exist_ok=True)
    return {"output": out, "sample": sample}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_md(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def load_docs(url: str | None) -> tuple[dict[str, Any] | None, str]:
    if not url:
        return None, "ARKHAM_DOCS_URL not provided"

    try:
        resp = requests.get(url, timeout=20)
    except Exception as exc:
        return None, f"Failed to fetch docs/spec: {type(exc).__name__}"

    if resp.status_code != 200:
        return None, f"Docs/spec request returned HTTP {resp.status_code}"

    try:
        data = resp.json()
    except Exception:
        return None, "Docs/spec response is not JSON; automatic endpoint discovery unavailable"

    return data, "Docs/spec loaded"


def load_local_openapi(path_value: str | None, base: Path) -> tuple[dict[str, Any] | None, str]:
    if not path_value:
        return None, "ARKHAM_OPENAPI_FILE not provided"
    p = Path(path_value)
    if not p.is_absolute():
        p = base / p
    if not p.exists():
        return None, f"ARKHAM_OPENAPI_FILE not found: {p}"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to parse ARKHAM_OPENAPI_FILE: {type(exc).__name__}"
    return data, f"Loaded local OpenAPI/spec: {p}"


def load_docs_raw(url: str | None) -> tuple[str | None, str, str | None]:
    if not url:
        return None, "ARKHAM_DOCS_URL not provided", None
    try:
        resp = requests.get(url, timeout=20)
    except Exception as exc:
        return None, f"Failed to fetch docs page: {type(exc).__name__}", None
    ctype = resp.headers.get("content-type", "")
    return resp.text, f"Docs page HTTP {resp.status_code}", ctype


def extract_endpoints_from_html(html: str, docs_url: str) -> tuple[list[dict[str, Any]], list[str]]:
    found: dict[tuple[str, str], dict[str, Any]] = {}
    spec_links: list[str] = []

    for href in re.findall(r"""href=["']([^"'#]+)["']""", html, flags=re.IGNORECASE):
        candidate = href.strip()
        lower = candidate.lower()
        if any(x in lower for x in ["openapi", "swagger", "api-docs", "spec"]) and (
            lower.endswith(".json") or "json" in lower
        ):
            resolved = urljoin(docs_url, candidate)
            spec_links.append(resolved)

    path_re = re.compile(r"""(?:"|')(/api/[A-Za-z0-9_./{}:-]*)(?:"|')|(?:\b)(/api/[A-Za-z0-9_./{}:-]*)""")
    for m in path_re.finditer(html):
        path = (m.group(1) or m.group(2) or "").strip()
        if not path or "//" in path:
            continue
        key = ("GET", path)
        found[key] = {
            "name": f"GET {path}",
            "method": "GET",
            "path": path,
            "purpose": "UNKNOWN",
            "params": {},
            "enabled": False,
            "discovered_from": "docs_html",
            "unknowns": ["Method not explicitly verified in HTML; defaulted to GET for safe read-only audit handling."],
        }

    return list(found.values()), sorted(set(spec_links))


def extract_get_endpoints(spec: dict[str, Any]) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return endpoints

    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        get_op = methods.get("get")
        if not isinstance(get_op, dict):
            continue
        params = get_op.get("parameters", [])
        required_params = [p.get("name") for p in params if isinstance(p, dict) and p.get("required")]
        optional_params = [p.get("name") for p in params if isinstance(p, dict) and not p.get("required")]
        endpoints.append(
            {
                "name": get_op.get("operationId") or f"GET {path}",
                "method": "GET",
                "path": path,
                "purpose": get_op.get("summary") or "UNKNOWN",
                "parameters": params,
                "required_params": required_params,
                "optional_params": optional_params,
            }
        )
    return endpoints


def summarize_schema(data: Any) -> dict[str, Any]:
    summary = {
        "top_level_fields": [],
        "nested_field_summary": {},
        "timestamp_fields": [],
        "wallet/address_fields": [],
        "entity/label/tag_fields": [],
        "transfer/transaction_fields": [],
        "balance_fields": [],
        "pagination_fields": [],
        "error_fields": [],
    }

    if isinstance(data, dict):
        summary["top_level_fields"] = sorted(data.keys())

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else k
                lk = k.lower()
                if any(x in lk for x in ["time", "date", "timestamp", "block"]):
                    summary["timestamp_fields"].append(p)
                if any(x in lk for x in ["address", "wallet"]):
                    summary["wallet/address_fields"].append(p)
                if any(x in lk for x in ["entity", "label", "tag", "category"]):
                    summary["entity/label/tag_fields"].append(p)
                if any(x in lk for x in ["transfer", "transaction", "tx", "hash"]):
                    summary["transfer/transaction_fields"].append(p)
                if "balance" in lk:
                    summary["balance_fields"].append(p)
                if any(x in lk for x in ["page", "cursor", "next", "limit", "offset", "has_more"]):
                    summary["pagination_fields"].append(p)
                if any(x in lk for x in ["error", "message", "code", "detail"]):
                    summary["error_fields"].append(p)
                walk(v, p)
        elif isinstance(obj, list) and obj:
            walk(obj[0], f"{prefix}[]" if prefix else "[]")

    walk(data)
    summary["nested_field_summary"] = {
        "detected_paths_sample": sorted(
            set(
                summary["timestamp_fields"]
                + summary["wallet/address_fields"]
                + summary["entity/label/tag_fields"]
                + summary["transfer/transaction_fields"]
                + summary["balance_fields"]
                + summary["pagination_fields"]
                + summary["error_fields"]
            )
        )[:200]
    }
    for k in list(summary.keys()):
        if isinstance(summary[k], list):
            summary[k] = sorted(set(summary[k]))
    return summary


def load_test_fixtures(base: Path) -> tuple[dict[str, Any], str]:
    fixture_path = base / "config" / "test_fixtures.json"
    if not fixture_path.exists():
        return {}, "config/test_fixtures.json not provided"
    try:
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"Failed to parse config/test_fixtures.json: {type(exc).__name__}"
    cleaned: dict[str, Any] = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            cleaned[section] = {k: v for k, v in values.items() if isinstance(v, str) and v.strip()}
    return cleaned, "Loaded config/test_fixtures.json"


def resolve_param_with_fixtures(param_name: str, fixtures: dict[str, Any]) -> tuple[str | None, str]:
    pname = param_name.lower()
    for needle, candidates in FIXTURE_PARAM_HINTS.items():
        if needle in pname:
            for section, key in candidates:
                value = fixtures.get(section, {}).get(key)
                if value:
                    return value, f"{section}.{key}"
    return None, ""


def categorize_status(status_code: int, payload: Any) -> str:
    text = json.dumps(payload).lower() if isinstance(payload, (dict, list)) else str(payload).lower()
    if "api key" in text or "sign up" in text:
        return "auth failure"
    if "timestamp" in text:
        return "timestamp format failure"
    if status_code == 401:
        return "auth failure"
    if status_code == 403:
        return "permission/tier restriction"
    if status_code == 400:
        if any(x in text for x in ["param", "invalid", "missing", "required"]):
            return "missing/invalid params"
    if 200 <= status_code < 300:
        return "endpoint working"
    return "unknown error"


def build_auth_headers(auth_mode: str, api_key: str) -> tuple[dict[str, str], dict[str, bool]]:
    mode = auth_mode.lower().strip()
    if mode not in {"bearer", "api-key", "x-api-key"}:
        mode = "bearer"
    headers: dict[str, str] = {}
    sent = {"authorization": False, "api-key": False, "x-api-key": False}
    if mode == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"
        sent["authorization"] = True
    elif mode == "api-key":
        headers["API-Key"] = api_key
        sent["api-key"] = True
    else:
        headers["X-API-Key"] = api_key
        sent["x-api-key"] = True
    return headers, sent


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    base = script_dir.parent
    load_dotenv(base / ".env")

    paths = ensure_dirs(base)
    output_dir = paths["output"]
    sample_dir = paths["sample"]

    api_base = os.getenv("ARKHAM_API_BASE_URL", "").strip()
    api_key = os.getenv("ARKHAM_API_KEY", "").strip()
    docs_url = os.getenv("ARKHAM_DOCS_URL", "").strip() or None
    local_openapi_file = os.getenv("ARKHAM_OPENAPI_FILE", "").strip() or None
    auth_mode = os.getenv("ARKHAM_AUTH_MODE", "").strip().lower() or "bearer"

    unknowns: list[str] = []
    notes: list[str] = []

    if not api_base:
        unknowns.append("ARKHAM_API_BASE_URL missing")
    if not api_key:
        unknowns.append("ARKHAM_API_KEY missing")

    local_spec, local_spec_note = load_local_openapi(local_openapi_file, base)
    notes.append(local_spec_note)
    spec = local_spec
    docs_note = "Skipped remote docs fetch because local OpenAPI/spec was used."
    remote_docs_http_403 = False
    if spec is None:
        spec, docs_note = load_docs(docs_url)
        if "HTTP 403" in docs_note:
            remote_docs_http_403 = True
    notes.append(docs_note)
    docs_html_found = False
    html_discovered: list[dict[str, Any]] = []
    discovered_spec_links: list[str] = []

    if docs_url and not spec:
        raw, raw_note, ctype = load_docs_raw(docs_url)
        notes.append(raw_note)
        if "HTTP 403" in raw_note:
            remote_docs_http_403 = True
        if raw and ("text/html" in (ctype or "").lower() or "<html" in raw[:500].lower()):
            docs_html_found = True
            sanitized_html = sanitize_data(raw)
            write_md(output_dir / "docs_snapshot.html", sanitized_html)
            html_discovered, discovered_spec_links = extract_endpoints_from_html(raw, docs_url)

    endpoint_inventory: list[dict[str, Any]] = []
    schema_summary: dict[str, Any] = {}
    fixtures, fixture_note = load_test_fixtures(base)
    notes.append(fixture_note)

    discovered = extract_get_endpoints(spec) if spec else []
    for ep in discovered:
        ep["params"] = {}
        ep["enabled"] = True
        ep["discovered_from"] = "local_openapi" if local_spec is not None else "docs_json"
    if html_discovered:
        discovered.extend(html_discovered)

    seed_path = base / "config" / "endpoints_seed.json"
    seed_discovered: list[dict[str, Any]] = []
    if seed_path.exists():
        try:
            seed_data = json.loads(seed_path.read_text(encoding="utf-8"))
            for ep in seed_data.get("endpoints", []):
                if not isinstance(ep, dict):
                    continue
                seed_discovered.append(
                    {
                        "name": ep.get("name", "UNKNOWN"),
                        "method": ep.get("method", "GET"),
                        "path": ep.get("path", "UNKNOWN"),
                        "purpose": ep.get("purpose", "UNKNOWN"),
                        "params": ep.get("params", {}),
                        "enabled": bool(ep.get("enabled", False)),
                        "discovered_from": "seed",
                        "unknowns": [],
                    }
                )
        except Exception as exc:
            notes.append(f"Failed to parse config/endpoints_seed.json: {type(exc).__name__}")
    discovered.extend(seed_discovered)
    if not discovered:
        notes.append("No discoverable GET endpoints from docs/spec")

    for ep in discovered:
        inv = {
            "name": ep["name"],
            "method": ep["method"],
            "path": ep["path"],
            "purpose": ep["purpose"],
            "discovered_from": ep.get("discovered_from", "unknown"),
            "enabled": bool(ep.get("enabled", False)),
            "params": ep.get("params", {}),
            "tested": False,
            "testable": False,
            "required_params": ep.get("required_params", []),
            "optional_params": ep.get("optional_params", []),
            "reason_not_tested": "",
            "status": "UNKNOWN",
            "status_code": "UNKNOWN",
            "error": "",
            "notes": [],
            "unknowns": ep.get("unknowns", []).copy(),
        }

        required_params = [p.get("name") for p in ep.get("parameters", []) if isinstance(p, dict) and p.get("required")]
        resolved_params = {}
        unresolved = []
        fixture_sources = {}
        timestamp_param_names = {"timegte", "timelte", "timelast", "since", "from", "to"}
        for rp in required_params:
            value, source = resolve_param_with_fixtures(rp, fixtures)
            if value:
                resolved_params[rp] = value
                fixture_sources[rp] = source
            else:
                unresolved.append(rp)
        inv["params"] = sanitize_data(resolved_params)
        query_params_to_send = {}
        if isinstance(fixtures.get("query_params"), dict):
            for k, v in fixtures["query_params"].items():
                if v in ("", None):
                    continue
                if k.lower() in timestamp_param_names:
                    query_params_to_send[k] = v
                elif k.lower() in [n.lower() for n in inv.get("optional_params", [])]:
                    query_params_to_send[k] = v
        omitted_timestamps = [p for p in inv.get("optional_params", []) if p.lower() in timestamp_param_names and p not in query_params_to_send]
        if omitted_timestamps:
            inv["notes"].append({"timestamp_params_omitted": omitted_timestamps})
        if fixture_sources:
            inv["notes"].append({"fixture_source_used": fixture_sources})
        if unresolved:
            inv["unknowns"].append(f"Required params unresolved: {unresolved}")
            inv["reason_not_tested"] = "Missing required params; params were not invented."
            endpoint_inventory.append(inv)
            continue

        if not inv["enabled"]:
            inv["notes"].append("Endpoint not enabled for testing.")
            inv["reason_not_tested"] = "Endpoint disabled."
            endpoint_inventory.append(inv)
            continue
        if inv["method"].upper() != "GET":
            inv["notes"].append("Non-GET endpoint skipped (safe read-only policy).")
            inv["reason_not_tested"] = "Non-GET endpoint."
            endpoint_inventory.append(inv)
            continue
        if not api_base or not api_key:
            inv["unknowns"].append("Missing required environment variables for live call")
            inv["reason_not_tested"] = "Missing ARKHAM_API_BASE_URL or ARKHAM_API_KEY."
            endpoint_inventory.append(inv)
            continue

        url = api_base.rstrip("/") + ep["path"]
        for key, value in resolved_params.items():
            url = url.replace("{" + key + "}", value)
        headers, auth_sent = build_auth_headers(auth_mode, api_key)
        inv["tested"] = True
        inv["testable"] = True
        inv["resolved_url_template"] = ep["path"]
        inv["query_params_sent"] = sanitize_data(query_params_to_send)
        inv["path_params_used"] = sanitize_data(resolved_params)
        inv["auth_mode_used"] = auth_mode
        inv["request_had_auth_header"] = any(auth_sent.values())

        try:
            r = requests.get(url, headers=headers, timeout=20, params=query_params_to_send if query_params_to_send else None)
            inv["status"] = f"HTTP {r.status_code}"
            inv["status_code"] = r.status_code
            inv["notes"].append({"response_headers": sanitize_headers(dict(r.headers))})
            payload: Any
            try:
                payload = r.json()
            except Exception:
                payload = {"raw_text": r.text[:2000], "parse": "non-json"}
            spayload = sanitize_data(payload)
            fname = f"{ep['name'].replace('/', '_').replace(' ', '_')}.json"
            sample_path = sample_dir / fname
            write_json(sample_path, spayload)
            inv["notes"].append({"sample_file": str(sample_path.relative_to(base))})
            result_category = categorize_status(r.status_code, spayload)
            inv["notes"].append({"result_category": result_category})
            if isinstance(spayload, dict):
                inv["sanitized_error_message"] = sanitize_data(spayload.get("message") or spayload.get("error") or "")
            else:
                inv["sanitized_error_message"] = ""
            schema_summary[ep["name"]] = summarize_schema(spayload)
            if r.status_code in (400, 401, 403):
                err_path = sample_dir / f"{ep['name'].replace('/', '_').replace(' ', '_')}_error.json"
                write_json(err_path, {"status_code": r.status_code, "body": spayload})
        except Exception as exc:
            inv["status"] = "ERROR"
            inv["error"] = type(exc).__name__
            inv["unknowns"].append(f"Request failed: {type(exc).__name__}")
            inv["sanitized_error_message"] = type(exc).__name__

        endpoint_inventory.append(inv)

    capability_rules = {
        "wallet/address lookup": ["intelligence/address", "address_enriched", "balances/address", "history/address"],
        "entity labels": ["intelligence/entity", "entities/updates", "address_enriched"],
        "tags/categories": ["address_tags/updates", "tags/updates", "includetags"],
        "transaction history": ["/tx/", "transfers/tx", "transfers"],
        "token transfers": ["transfers", "histogram"],
        "balance history": ["balances/address", "balances/entity", "portfolio/address", "timeseries", "history/address"],
        "CEX wallet identification": ["intelligence", "entitytype", "depositexchangeid", "service", "label", "tag"],
        "deposit/withdrawal detection": ["transfers", "counterpart", "flow", "exchange", "entity"],
        "historical timestamps": ["timegte", "timelte", "timelast", "since", "from", "to"],
        "pagination": ["limit", "offset", "pagetoken", "cursor"],
        "rate limits": ["x-ratelimit", "rate"],
        "PnL data": ["pnl", "performance"],
        "realized PnL directly available": ["realized_pnl", "realizedpnl"],
        "price history": ["price/history", "token/price"],
    }
    capability_rows = []
    inv_text = json.dumps(endpoint_inventory).lower()
    for cap in CAPABILITIES:
        needles = capability_rules.get(cap, [])
        matched = [e for e in endpoint_inventory if any(n in (e.get("path", "") + e.get("name", "")).lower() for n in needles)]
        available = "unknown"
        notes_msg = "No endpoint/schema evidence."
        confidence = "low"
        sample_file = "UNKNOWN"
        evidence = "UNKNOWN"
        if matched:
            available = "partial"
            evidence = matched[0].get("path", "UNKNOWN")
            for n in matched[0].get("notes", []):
                if isinstance(n, dict) and "sample_file" in n:
                    sample_file = n["sample_file"]
            if any(str(m.get("status_code")) == "200" for m in matched):
                available = "yes"
                confidence = "high"
                notes_msg = "Endpoint tested successfully with evidence."
            else:
                notes_msg = "OpenAPI evidence exists; full verification incomplete."
                confidence = "medium"
        if cap in ["PnL data", "realized PnL directly available"] and not any(k in inv_text for k in needles):
            available = "no"
            notes_msg = "No explicit PnL evidence in discovered endpoints/schemas."
        capability_rows.append({"capability": cap, "available": available, "evidence_endpoint": evidence, "sample_file": sample_file, "confidence": confidence, "notes": notes_msg})

    missing_lines = ["# Missing Data Report", "", "## Unavailable or Unverified Capabilities", ""]
    for row in capability_rows:
        missing_lines.append(
            f"- {row['capability']}: UNVERIFIED; blocks Hunter Phase 1: UNKNOWN; workaround later: UNKNOWN."
        )

    verdict = "C) Arkham API is insufficient for Hunter Phase 1"
    if local_spec is not None and any(row["available"] in ("yes", "partial") for row in capability_rows):
        verdict = "B) Arkham API is partially sufficient but needs fixture verification and possibly external data source."

    capability_md = [
        "# Capability Matrix",
        "",
        "| capability | available | evidence endpoint | sample file | confidence | notes |",
        "|---|---|---|---|---|---|",
    ]
    for row in capability_rows:
        capability_md.append(
            f"| {row['capability']} | {row['available']} | {row['evidence_endpoint']} | {row['sample_file']} | {row['confidence']} | {row['notes']} |"
        )

    next_md = [
        "# Next Phase Recommendation",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Evidence Summary",
        f"- Docs/spec status: {docs_note}",
        f"- Docs HTML detected: {'yes' if docs_html_found else 'no'}",
        f"- HTML-discovered endpoint candidates: {len(html_discovered)}",
        f"- HTML-discovered spec links: {len(discovered_spec_links)}",
        (f"- Spec link candidates: {', '.join(discovered_spec_links)}" if discovered_spec_links else "- Spec link candidates: NONE"),
        f"- Discovered GET endpoints: {len(discovered)}",
        f"- Tested endpoints: {sum(1 for e in endpoint_inventory if e.get('tested'))}",
        "",
        "## Blockers",
        "- Capability evidence remains unverified where endpoint docs/spec or executable test cases are unavailable.",
        "",
        "## Safe next step",
        "- Populate config/test_fixtures.json with known safe public addresses/entities/token IDs/transaction hashes, then rerun Phase 0B.",
    ]

    if docs_html_found and not discovered and not seed_discovered:
        next_md.insert(
            9,
            "- Docs page reachable, but no machine-readable or safely executable endpoint inventory was discovered.",
        )
    if remote_docs_http_403 and local_spec is None and not seed_discovered:
        next_md.insert(
            9,
            "- Remote docs/spec fetch was blocked by HTTP 403. Provide ARKHAM_OPENAPI_FILE or config/endpoints_seed.json.",
        )

    if not endpoint_inventory:
        endpoint_inventory = [
            {
                "name": "UNKNOWN",
                "method": "UNKNOWN",
                "path": "UNKNOWN",
                "purpose": "UNKNOWN",
                "discovered_from": "unknown",
                "enabled": False,
                "params": {},
                "tested": False,
                "testable": False,
                "required_params": [],
                "optional_params": [],
                "reason_not_tested": "No endpoint inventory source available.",
                "status": "UNKNOWN",
                "status_code": "UNKNOWN",
                "error": "",
                "notes": notes,
                "unknowns": unknowns + ["No endpoint inventory source available"],
            }
        ]

    write_json(output_dir / "endpoint_inventory.json", endpoint_inventory)
    write_json(output_dir / "schema_summary.json", schema_summary)
    write_md(output_dir / "capability_matrix.md", "\n".join(capability_md))
    write_md(output_dir / "missing_data_report.md", "\n".join(missing_lines))
    write_md(output_dir / "next_phase_recommendation.md", "\n".join(next_md))
    key_prefix = (api_key[:4] + "***") if api_key else "N/A"
    auth_headers, sent_flags = build_auth_headers(auth_mode, api_key if api_key else "")
    auth_md = [
        "# Auth Diagnostics",
        f"- ARKHAM_API_KEY exists: {'yes' if bool(api_key) else 'no'}",
        f"- API key length: {len(api_key)}",
        f"- API key prefix: {key_prefix}",
        f"- Auth header mode used: {auth_mode}",
        f"- Whether Authorization header was sent: {'yes' if sent_flags['authorization'] else 'no'}",
        f"- Whether API-Key header was sent: {'yes' if sent_flags['api-key'] else 'no'}",
        f"- Whether X-API-Key header was sent: {'yes' if sent_flags['x-api-key'] else 'no'}",
    ]
    write_md(output_dir / "auth_diagnostics.md", "\n".join(auth_md))


if __name__ == "__main__":
    main()
