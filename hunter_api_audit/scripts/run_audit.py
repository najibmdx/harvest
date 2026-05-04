#!/usr/bin/env python3
"""Hunter Phase 0: Arkham API Capability Auditor."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
        endpoints.append(
            {
                "name": get_op.get("operationId") or f"GET {path}",
                "method": "GET",
                "path": path,
                "purpose": get_op.get("summary") or "UNKNOWN",
                "parameters": get_op.get("parameters", []),
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

    unknowns: list[str] = []
    notes: list[str] = []

    if not api_base:
        unknowns.append("ARKHAM_API_BASE_URL missing")
    if not api_key:
        unknowns.append("ARKHAM_API_KEY missing")

    spec, docs_note = load_docs(docs_url)
    notes.append(docs_note)

    endpoint_inventory: list[dict[str, Any]] = []
    schema_summary: dict[str, Any] = {}

    discovered = extract_get_endpoints(spec) if spec else []
    if not discovered:
        notes.append("No discoverable GET endpoints from docs/spec")

    for ep in discovered:
        inv = {
            "name": ep["name"],
            "method": ep["method"],
            "path": ep["path"],
            "purpose": ep["purpose"],
            "tested": False,
            "status": "UNKNOWN",
            "notes": [],
            "unknowns": [],
        }

        required_params = [p.get("name") for p in ep.get("parameters", []) if p.get("required")]
        if required_params:
            inv["unknowns"].append(f"Required params unresolved: {required_params}")
            endpoint_inventory.append(inv)
            continue

        if not api_base or not api_key:
            inv["unknowns"].append("Missing required environment variables for live call")
            endpoint_inventory.append(inv)
            continue

        url = api_base.rstrip("/") + ep["path"]
        headers = {"Authorization": f"Bearer {api_key}"}
        inv["tested"] = True

        try:
            r = requests.get(url, headers=headers, timeout=20)
            inv["status"] = f"HTTP {r.status_code}"
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
            schema_summary[ep["name"]] = summarize_schema(spayload)
        except Exception as exc:
            inv["status"] = "ERROR"
            inv["unknowns"].append(f"Request failed: {type(exc).__name__}")

        endpoint_inventory.append(inv)

    capability_rows = []
    for cap in CAPABILITIES:
        capability_rows.append(
            {
                "capability": cap,
                "available": "unknown",
                "evidence_endpoint": "UNKNOWN",
                "sample_file": "UNKNOWN",
                "confidence": "low",
                "notes": "No verified mapping yet from tested Arkham endpoint evidence.",
            }
        )

    missing_lines = ["# Missing Data Report", "", "## Unavailable or Unverified Capabilities", ""]
    for row in capability_rows:
        missing_lines.append(
            f"- {row['capability']}: UNVERIFIED; blocks Hunter Phase 1: UNKNOWN; workaround later: UNKNOWN."
        )

    verdict = "C) Arkham API is insufficient for Hunter Phase 1"
    if discovered and any(ep.get("tested") for ep in endpoint_inventory):
        verdict = "B) Arkham API is partially sufficient but needs external data source"

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
        f"- Discovered GET endpoints: {len(discovered)}",
        f"- Tested endpoints: {sum(1 for e in endpoint_inventory if e.get('tested'))}",
        "",
        "## Blockers",
        "- Capability evidence remains unverified where endpoint docs/spec or executable test cases are unavailable.",
        "",
        "## Safe next step",
        "- Provide a machine-readable Arkham API spec or explicit read-only endpoint list and required query parameters, then rerun this auditor.",
    ]

    if not endpoint_inventory:
        endpoint_inventory = [
            {
                "name": "UNKNOWN",
                "method": "UNKNOWN",
                "path": "UNKNOWN",
                "purpose": "UNKNOWN",
                "tested": False,
                "status": "UNKNOWN",
                "notes": notes,
                "unknowns": unknowns + ["No endpoint inventory source available"],
            }
        ]

    write_json(output_dir / "endpoint_inventory.json", endpoint_inventory)
    write_json(output_dir / "schema_summary.json", schema_summary)
    write_md(output_dir / "capability_matrix.md", "\n".join(capability_md))
    write_md(output_dir / "missing_data_report.md", "\n".join(missing_lines))
    write_md(output_dir / "next_phase_recommendation.md", "\n".join(next_md))


if __name__ == "__main__":
    main()
