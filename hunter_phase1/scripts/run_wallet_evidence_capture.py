#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
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


@dataclass
class ApiResult:
    endpoint: str
    status: int
    body: Any
    error: str | None = None


def load_env() -> dict[str, str]:
    auth_mode = os.getenv("ARKHAM_AUTH_MODE", "api-key")
    return {
        "api_key": os.getenv("ARKHAM_API_KEY", ""),
        "base_url": os.getenv("ARKHAM_API_BASE_URL", "").rstrip("/"),
        "auth_mode": auth_mode,
    }


def call_api(base_url: str, api_key: str, endpoint_name: str, address: str) -> ApiResult:
    path = ENDPOINTS[endpoint_name]
    if endpoint_name == "transfers":
        query = parse.urlencode({"address": address})
        url = f"{base_url}{path}?{query}"
    else:
        url = f"{base_url}{path.format(address=address)}"

    req = request.Request(url)
    req.add_header("API-Key", api_key)
    req.add_header("Accept", "application/json")

    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
            return ApiResult(endpoint=endpoint_name, status=resp.status, body=data)
    except HTTPError as err:
        raw = err.read().decode("utf-8", errors="replace") if err.fp else ""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"raw": raw}
        return ApiResult(endpoint=endpoint_name, status=err.code, body=payload, error=str(err))
    except URLError as err:
        return ApiResult(endpoint=endpoint_name, status=0, body={}, error=str(err))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    env = load_env()
    errors: list[str] = []
    if not env["base_url"]:
        errors.append("ARKHAM_API_BASE_URL is required")
    if not env["api_key"]:
        errors.append("ARKHAM_API_KEY is required")
    if env["auth_mode"] != "api-key":
        errors.append("ARKHAM_AUTH_MODE must be api-key for Hunter Phase 1")
    if errors:
        raise SystemExit("; ".join(errors))

    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing required input file: {CONFIG_PATH}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    wallets = config.get("wallets", [])

    index_items: list[dict[str, Any]] = []
    stats = {name: {"tested": 0, "success": 0, "fail": 0, "errors": Counter()} for name in ENDPOINTS}

    for wallet in wallets:
        label = wallet.get("label", "unlabeled_wallet")
        chain = wallet.get("chain", "unknown")
        address = wallet.get("address", "")

        item = {
            "wallet_label": label,
            "chain": chain,
            "address_redacted": redact_address(address),
            "endpoints_attempted": [],
            "status_codes": {},
            "sample_file_paths": {},
            "available_evidence_categories": [],
            "missing_evidence_categories": [],
            "errors": [],
        }

        for endpoint_name in ENDPOINTS:
            stats[endpoint_name]["tested"] += 1
            item["endpoints_attempted"].append(endpoint_name)

            result = call_api(env["base_url"], env["api_key"], endpoint_name, address)
            item["status_codes"][endpoint_name] = result.status

            safe_payload = {
                "wallet_label": label,
                "chain": chain,
                "address_redacted": redact_address(address),
                "endpoint": endpoint_name,
                "status": result.status,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "data": sanitize_value(result.body),
            }
            raw_file = RAW_DIR / f"{label}__{endpoint_name}.json"
            save_json(raw_file, safe_payload)
            item["sample_file_paths"][endpoint_name] = str(raw_file.relative_to(ROOT))

            if 200 <= result.status < 300:
                item["available_evidence_categories"].append(endpoint_name)
                stats[endpoint_name]["success"] += 1
            else:
                item["missing_evidence_categories"].append(endpoint_name)
                stats[endpoint_name]["fail"] += 1
                if result.error:
                    item["errors"].append(f"{endpoint_name}: {result.error}")
                    stats[endpoint_name]["errors"][result.error] += 1

        index_items.append(item)

    save_json(INDEX_PATH, {"generated_at": datetime.now(timezone.utc).isoformat(), "wallets": index_items})

    matrix_lines = [
        "# Endpoint Success Matrix",
        "",
        "| endpoint | tested count | success count | failure count | common error | usable for Phase 2 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for endpoint, row in stats.items():
        common_error = row["errors"].most_common(1)[0][0] if row["errors"] else "n/a"
        usable = "yes" if row["fail"] == 0 else ("partial" if row["success"] > 0 else "no")
        matrix_lines.append(
            f"| {endpoint} | {row['tested']} | {row['success']} | {row['fail']} | {common_error} | {usable} |"
        )
    MATRIX_PATH.write_text("\n".join(matrix_lines) + "\n", encoding="utf-8")

    missing_lines = ["# Missing Evidence Report", "", "No speculation. Report is based on attempted endpoint statuses.", ""]
    for wallet_item in index_items:
        missing = wallet_item["missing_evidence_categories"]
        missing_lines.append(f"## {wallet_item['wallet_label']}")
        if missing:
            missing_lines.append("Missing/blocked evidence categories:")
            for m in missing:
                code = wallet_item["status_codes"].get(m)
                missing_lines.append(f"- {m} (status: {code})")
        else:
            missing_lines.append("- None")
        if wallet_item["errors"]:
            missing_lines.append("Errors:")
            for err in wallet_item["errors"]:
                missing_lines.append(f"- {err}")
        missing_lines.append("")
    MISSING_PATH.write_text("\n".join(missing_lines), encoding="utf-8")

    any_missing = any(w["missing_evidence_categories"] for w in index_items)
    recommendation = (
        "B) Phase 1 evidence capture is partially sufficient but needs more fixtures/sources"
        if any_missing
        else "A) Phase 1 evidence capture is sufficient to proceed to wallet performance reconstruction"
    )
    RECOMMEND_PATH.write_text(
        "# Phase 1 Recommendation\n\n" + recommendation + "\n", encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
