#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from arkham_client import build_headers, get_json, load_env, redact_url
from sanitize import redact_address, sanitize_value

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "wallet_inputs.json"
OUTPUT_DIR = ROOT / "output"
RAW_DIR = OUTPUT_DIR / "raw_samples"

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


def classify(status: int, payload: Any) -> str:
    text = json.dumps(payload).lower() if isinstance(payload, (dict, list)) else str(payload).lower()
    if "error code: 1010" in text:
        return "cloudflare_or_edge_block_1010"
    if status == 403 and any(x in text for x in ["api key", "auth", "sign up"]):
        return "auth failure"
    if status == 403 and any(x in text for x in ["forbidden", "permission", "tier"]):
        return "permission or tier restriction"
    if status == 403 and "timestamp" in text:
        return "timestamp failure"
    if 200 <= status < 300:
        return "endpoint working"
    return "other failure"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    env = load_env()
    if env["auth_mode"] != "api-key":
        raise SystemExit("ARKHAM_AUTH_MODE must be api-key")
    if not env["api_key"] or not env["base_url"]:
        raise SystemExit("ARKHAM_API_KEY and ARKHAM_API_BASE_URL are required")
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing input: {CONFIG_PATH}")

    headers, sent = build_headers(env["auth_mode"], env["api_key"])
    wallets = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("wallets", [])

    sanity = get_json(env["base_url"], "/arkm/circulating", headers)
    sanity_cat = classify(sanity["status_code"], sanity["body"])
    sanity_file = RAW_DIR / "auth_sanity_check.json"
    write(sanity_file, json.dumps({"status_code": sanity["status_code"], "auth_worked": 200 <= sanity["status_code"] < 300, "result_category": sanity_cat, "body": sanitize_value(sanity["body"])}, indent=2))

    req_diags, index_wallets = [], []
    stats = {k: {"tested": 0, "success": 0, "fail": 0, "errors": Counter()} for k in ENDPOINTS}

    any_success = False
    any_1010 = False
    for w in wallets:
        label, chain, address = w.get("label", "unlabeled"), w.get("chain", "unknown"), w.get("address", "")
        filters = {k: v for k, v in w.items() if k in TIME_FILTER_KEYS}
        endpoint_diag = {}
        avail, miss = [], []
        for cat, path in ENDPOINTS.items():
            stats[cat]["tested"] += 1
            q = filters.copy()
            p = {"address": address}
            if cat == "transfers":
                q["address"] = address
                p = {}
            res = get_json(env["base_url"], path, headers, path_params=p, query_params=q)
            safe = sanitize_value(res["body"])
            status = res["status_code"]
            rc = classify(status, safe)
            any_1010 = any_1010 or (rc == "cloudflare_or_edge_block_1010")
            sample = RAW_DIR / f"{label}__{cat}.json"
            write(sample, json.dumps({"status": status, "data": safe}, indent=2))
            endpoint_diag[cat] = {
                "resolved_url_template": path,
                "final_url_redacted": redact_url(res["url_redacted"], address),
                "path_params_used": {"address": redact_address(address)} if p else {},
                "query_params_sent": {k: (redact_address(v) if k == "address" else v) for k, v in q.items()},
                "auth_mode_used": env["auth_mode"],
                "request_had_api_key_header": sent["api-key"],
                "status_code": status,
                "sanitized_error_body": safe if status >= 400 else {},
                "sample_file_path": str(sample.relative_to(ROOT)),
            }
            req_diags.append({"wallet_label": label, "category": cat, "method": "GET", "url_redacted": endpoint_diag[cat]["final_url_redacted"], "path_params_used": endpoint_diag[cat]["path_params_used"], "query_params_sent": endpoint_diag[cat]["query_params_sent"], "auth_mode_used": env["auth_mode"], "status_code": status, "result_category": rc, "sanitized_response_body": safe, "sample_file": str(sample.relative_to(ROOT))})
            if 200 <= status < 300:
                any_success = True
                avail.append(cat)
                stats[cat]["success"] += 1
            else:
                miss.append(cat)
                stats[cat]["fail"] += 1
                stats[cat]["errors"][rc] += 1
        index_wallets.append({"wallet_label": label, "chain": chain, "address_redacted": redact_address(address), "endpoints_attempted": list(ENDPOINTS.keys()), "endpoint_request_diagnostics": endpoint_diag, "available_evidence_categories": avail, "missing_evidence_categories": miss, "errors": []})

    write(OUTPUT_DIR / "wallet_evidence_index.json", json.dumps({"wallets": index_wallets}, indent=2))
    write(OUTPUT_DIR / "phase1_request_diagnostics.json", json.dumps(req_diags, indent=2))
    matrix = ["# Endpoint Success Matrix", "", "| endpoint | tested count | success count | failure count | common error | usable for Phase 2 |", "|---|---:|---:|---:|---|---|"]
    for ep, row in stats.items():
        common = row["errors"].most_common(1)[0][0] if row["errors"] else "n/a"
        usable = "yes" if row["fail"] == 0 else ("partial" if row["success"] > 0 else "no")
        matrix.append(f"| {ep} | {row['tested']} | {row['success']} | {row['fail']} | {common} | {usable} |")
    write(OUTPUT_DIR / "endpoint_success_matrix.md", "\n".join(matrix) + "\n")
    missing = ["# Missing Evidence Report", ""]
    for wallet in index_wallets:
        missing.append(f"## {wallet['wallet_label']}")
        if wallet["missing_evidence_categories"]:
            for cat in wallet["missing_evidence_categories"]:
                missing.append(f"- {cat}")
        else:
            missing.append("- None")
        missing.append("")
    write(OUTPUT_DIR / "missing_evidence_report.md", "\n".join(missing) + "\n")

    md = ["# Phase 1 Auth Diagnostics", f"- ARKHAM_API_KEY exists: {'yes' if env['api_key'] else 'no'}", f"- API key length only: {len(env['api_key'])}", f"- API key prefix first 4 chars only: {env['api_key'][:4]}***", f"- ARKHAM_API_BASE_URL value: {env['base_url']}", f"- ARKHAM_AUTH_MODE value: {env['auth_mode']}", f"- Whether API-Key header was sent: {'yes' if sent['api-key'] else 'no'}", f"- Whether Authorization header was sent: {'yes' if sent['authorization'] else 'no'}", f"- Whether X-API-Key header was sent: {'yes' if sent['x-api-key'] else 'no'}", f"- sanity endpoint: /arkm/circulating", f"- sanity status code: {sanity['status_code']}", f"- sanity result category: {sanity_cat}", f"- sanity sample file: hunter_phase1/output/raw_samples/auth_sanity_check.json"]
    write(OUTPUT_DIR / "phase1_auth_diagnostics.md", "\n".join(md) + "\n")

    comp = ["# Phase0B vs Phase1 Request Comparison", f"- Phase 1 base URL: {env['base_url']}", f"- Phase 1 auth mode: {env['auth_mode']}", f"- Whether API-Key header was sent: {'yes' if sent['api-key'] else 'no'}", f"- Whether User-Agent was sent: {'yes' if sent['user-agent'] else 'no'}", f"- Whether Accept was sent: {'yes' if sent['accept'] else 'no'}", "- Whether requests are using the shared Arkham client: yes", f"- Whether sanity endpoint passed: {'yes' if 200 <= sanity['status_code'] < 300 else 'no'}", f"- Whether wallet endpoints still 403: {'yes' if any(d['status_code']==403 for d in req_diags) else 'no'}"]
    write(OUTPUT_DIR / "phase1_request_comparison.md", "\n".join(comp) + "\n")

    if sanity["status_code"] < 200 or sanity["status_code"] >= 300:
        rec = "C) Phase 1 evidence capture is insufficient because the shared Arkham client cannot pass sanity auth."
    elif any_1010:
        rec = "C) Phase 1 evidence capture is insufficient because wallet endpoints are blocked by Cloudflare/edge rule or request policy."
    elif any_success:
        rec = "A) Phase 1 evidence capture is sufficient to proceed to wallet performance reconstruction."
    else:
        rec = "C) Phase 1 evidence capture is insufficient"
    write(OUTPUT_DIR / "phase1_recommendation.md", f"# Phase 1 Recommendation\n\n{rec}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
