#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from direction_resolver import resolve_transfer_directions
from evidence_classifiers import classify_transfer_event, cost_basis_candidate_possible, tx_has_fee_gas, tx_has_trade_side
from tx_hash_inventory import build_tx_hash_inventory

try:
    from hunter_phase1.scripts.arkham_client import build_headers, get_json, load_env
except Exception:
    build_headers = get_json = load_env = None


DEFAULT_CONFIG: dict[str, Any] = {
    "phase2_output_dir": "hunter_phase2/output",
    "phase3a_output_dir": "hunter_phase3a/output",
    "wallet_input_candidates": ["config/wallet_inputs.json", "hunter_phase1/config/wallet_inputs.json"],
    "offline_only": False,
    "allow_api_calls": False,
    "allowed_online_endpoint_families": ["/tx/{hash}"],
    "max_tx_hashes_per_wallet": 100,
    "strict_no_pnl": True,
    "strict_no_profitability_labels": True,
    "strict_no_external_price_api": True,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_timeline(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def resolve_repo_root(script_path: Path, cwd: Path) -> Path:
    if (cwd / "hunter_phase3b").exists() or (cwd / "scripts").exists():
        return cwd
    if script_path.parent.name == "scripts":
        return script_path.parents[1]
    return script_path.parents[2]


def main() -> int:
    script_path = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    root = resolve_repo_root(script_path, cwd)
    phase_root = root / "hunter_phase3b"
    out = phase_root / "output"
    raw_tx = out / "raw_tx_details"
    out.mkdir(parents=True, exist_ok=True)
    raw_tx.mkdir(parents=True, exist_ok=True)

    cfg_attempts = [phase_root / "config" / "phase3b_config.json", phase_root / "config" / "phase3b_config.example.json"]
    selected_config_path = next((p for p in cfg_attempts if p.exists()), None)
    cfg = dict(DEFAULT_CONFIG)
    if selected_config_path:
        cfg.update(load_json(selected_config_path))

    capture_allowed_by_endpoint = "/tx/{hash}" in cfg.get("allowed_online_endpoint_families", [])
    capture_enabled = bool((not cfg.get("offline_only", True)) and cfg.get("allow_api_calls", False) and capture_allowed_by_endpoint)
    capture_disable_reason = "enabled"
    if not capture_enabled:
        if cfg.get("offline_only", True):
            capture_disable_reason = "offline_only=true"
        elif not cfg.get("allow_api_calls", False):
            capture_disable_reason = "allow_api_calls=false"
        else:
            capture_disable_reason = "endpoint_not_allowed"

    p2 = root / cfg["phase2_output_dir"]
    p3a = root / cfg["phase3a_output_dir"]
    timeline_file = p2 / "wallet_activity_timeline.jsonl"
    phase2_required = [timeline_file, p2 / "wallet_flow_summary.json", p2 / "wallet_identity_summary.json", p2 / "phase2_consistency_check.md"]
    phase3a_required = [p3a / "pnl_mode_eligibility.json", p3a / "pnl_input_requirements.json", p3a / "phase3a_design_constraints.md"]

    diag_lines = [
        "# Phase 3B Input Diagnostics",
        "",
        f"- script path: {script_path}",
        f"- current working directory: {cwd}",
        f"- resolved repo root: {root}",
        f"- resolved phase3b root: {phase_root}",
        "- config paths attempted:",
    ]
    for p in cfg_attempts:
        diag_lines.append(f"  - {p}")
    diag_lines.extend([
        f"- selected config path: {selected_config_path if selected_config_path else 'in-memory-default'}",
        f"- whether config exists: {'yes' if selected_config_path else 'no'}",
        f"- offline_only value: {cfg.get('offline_only')}",
        f"- allow_api_calls value: {cfg.get('allow_api_calls')}",
        f"- allowed_online_endpoint_families value: {cfg.get('allowed_online_endpoint_families')}",
        f"- API capture enabled: {'yes' if capture_enabled else 'no'}",
        f"- API capture disabled reason: {capture_disable_reason if not capture_enabled else 'n/a'}",
        f"- phase2 output dir resolved: {p2}",
        f"- phase3a output dir resolved: {p3a}",
        "- required Phase 2 files:",
    ])
    for p in phase2_required:
        diag_lines.append(f"  - {'yes' if p.exists() else 'no'} :: {p}")
    diag_lines.append("- required Phase 3A files:")
    for p in phase3a_required:
        diag_lines.append(f"  - {'yes' if p.exists() else 'no'} :: {p}")
    (out / "phase3b_input_diagnostics.md").write_text("\n".join(diag_lines) + "\n")

    wallet_addresses: dict[str, str] = {}
    for candidate in cfg["wallet_input_candidates"]:
        c = root / candidate
        if c.exists():
            data = load_json(c)
            items = data if isinstance(data, list) else data.get("wallets", [])
            for w in items:
                wallet_addresses[w.get("label", "")] = w.get("address", "")
            break

    rows = load_timeline(timeline_file) if timeline_file.exists() else []
    transfers_by_wallet: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        wl = r.get("wallet_label", "unknown_wallet")
        evt = (r.get("event_type") or "").lower()
        if evt == "transfer":
            transfers_by_wallet.setdefault(wl, []).append(r)

    direction_resolution = resolve_transfer_directions(transfers_by_wallet, wallet_addresses)
    tx_inventory = build_tx_hash_inventory(transfers_by_wallet, str(timeline_file))
    total_hashes = sum(i["unique_tx_hash_count"] for i in tx_inventory)
    if sum(len(v) for v in transfers_by_wallet.values()) > 0 and total_hashes == 0:
        first_events: list[dict[str, Any]] = []
        for wl in sorted(transfers_by_wallet.keys()):
            for ev in transfers_by_wallet[wl]:
                first_events.append(ev)
                if len(first_events) >= 3:
                    break
            if len(first_events) >= 3:
                break
        p2_check = p2 / "transfer_field_preservation_check.md"
        p2_pass = "unknown"
        if p2_check.exists():
            text = p2_check.read_text(encoding="utf-8", errors="ignore")
            if "preservation_pass: yes" in text:
                p2_pass = "yes"
            elif "preservation_pass: no" in text:
                p2_pass = "no"
        lines = ["# Phase 3B Transfer Field Diagnostics", ""]
        for i, ev in enumerate(first_events, start=1):
            lines.append(f"## Transfer Event {i}")
            lines.append(f"- keys: {sorted(ev.keys())}")
            lines.append(
                f"- tx fields: tx_hash={ev.get('tx_hash')} txHash={ev.get('txHash')} transactionHash={ev.get('transactionHash')}"
            )
            lines.append(
                f"- from/to fields: from_address={ev.get('from_address')} fromAddress={ev.get('fromAddress')} to_address={ev.get('to_address')} toAddress={ev.get('toAddress')}"
            )
            lines.append("")
        lines.append(f"- Phase 2 preservation check exists: {'yes' if p2_check.exists() else 'no'}")
        lines.append(f"- Phase 2 preservation check passed: {p2_pass}")
        (out / "phase3b_transfer_field_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plan, capture_index = [], []
    allow_calls = capture_enabled
    env = load_env() if (allow_calls and load_env) else {}
    env_missing = False
    headers = None
    if allow_calls and build_headers and env:
        if not env.get("api_key") or not env.get("base_url"):
            env_missing = True
        else:
            headers, _ = build_headers(env.get("auth_mode", "api-key") or "api-key", env.get("api_key", ""))
    elif allow_calls:
        env_missing = True
    if env_missing:
        diag_lines.append("- API environment status: missing Arkham API environment variables")
        (out / "phase3b_input_diagnostics.md").write_text("\n".join(diag_lines) + "\n")

    tx_detail_map: dict[tuple[str, str], dict[str, Any]] = {}
    for inv in tx_inventory:
        wl = inv["wallet_label"]
        for txh in inv["tx_hashes"][: int(cfg.get("max_tx_hashes_per_wallet", 100))]:
            rec = {"wallet_label": wl, "tx_hash": txh, "endpoint": "/tx/{hash}", "allowed_by_config": allow_calls, "capture_status": "planned", "reason": "config_allows" if allow_calls else capture_disable_reason}
            idx = {"wallet_label": wl, "tx_hash": txh, "attempted": False, "status_code": None, "sample_file": None, "sanitized_error": None, "available_tx_fields": [], "possible_trade_evidence_found": False, "possible_fee_gas_fields_found": False}
            if allow_calls and env_missing:
                rec["capture_status"] = "skipped"
                rec["reason"] = "missing_arkham_env_vars"
                idx["sanitized_error"] = "missing Arkham API environment variables"
            elif allow_calls and headers and env.get("base_url") and get_json:
                try:
                    idx["attempted"] = True
                    res = get_json(env["base_url"], "/tx/{hash}", headers=headers, path_params={"hash": txh})
                    idx["status_code"] = res.get("status_code")
                    body = res.get("body", {})
                    idx["available_tx_fields"] = sorted(body.keys()) if isinstance(body, dict) else []
                    idx["possible_trade_evidence_found"] = tx_has_trade_side(body if isinstance(body, dict) else {})
                    idx["possible_fee_gas_fields_found"] = tx_has_fee_gas(body if isinstance(body, dict) else {})[0]
                    if idx["status_code"] == 200:
                        sample = raw_tx / f"{wl}__{txh}.json"
                        sample.write_text(json.dumps(body, indent=2, sort_keys=True))
                        idx["sample_file"] = str(sample.relative_to(root))
                        rec["capture_status"] = "captured"
                        rec["reason"] = "ok"
                        tx_detail_map[(wl, txh)] = body if isinstance(body, dict) else {"raw": body}
                    else:
                        rec["capture_status"] = "failed"
                        rec["reason"] = f"http_{idx['status_code']}"
                except Exception as e:
                    idx["attempted"] = True
                    idx["sanitized_error"] = str(e)[:300]
                    rec["capture_status"] = "failed"
                    rec["reason"] = "request_exception"
            else:
                rec["capture_status"] = "skipped"
            plan.append(rec)
            capture_index.append(idx)

    trade_lines, cost_lines, link_lines, fee_lines = [], [], [], []
    total_confirmed_trade = 0
    total_cost_candidates = 0
    linkage_status = []
    for d in direction_resolution:
        wl = d["wallet_label"]
        events = transfers_by_wallet.get(wl, [])
        cex_count = 0
        transfer_only = 0
        unknown = d["unknown_direction_count"]
        directions = {e["txHash"]: e["direction"] for e in d["resolved_transfer_events"] if e.get("txHash")}
        confirmed = 0
        cost_candidates = 0
        fee_found = False
        fee_fields = set()
        for ev in events:
            cl = classify_transfer_event(ev)
            if cl == "CEX_FLOW_ONLY":
                cex_count += 1
            else:
                transfer_only += 1
            txh = (ev.get("txHash") or "").strip()
            detail = tx_detail_map.get((wl, txh), {})
            if detail and tx_has_trade_side(detail):
                confirmed += 1
            if cost_basis_candidate_possible(directions.get(txh, "unknown"), detail):
                cost_candidates += 1
            ff, fields = tx_has_fee_gas(detail)
            fee_found = fee_found or ff
            fee_fields.update(fields)

        total_confirmed_trade += confirmed
        total_cost_candidates += cost_candidates
        verdict = "LINKAGE_BLOCKED"
        if confirmed > 0 and cost_candidates > 0:
            verdict = "LINKAGE_PARTIAL"
        if confirmed > 0 and cost_candidates >= confirmed and unknown == 0:
            verdict = "LINKAGE_CONFIRMED"
        linkage_status.append(verdict)

        trade_lines.append(f"## {wl}\n- transfer events parsed: {len(events)}\n- direction resolution counts: inbound={d['inbound_count']}, outbound={d['outbound_count']}, unknown={unknown}\n- transaction detail capture status: captured={sum(1 for p in plan if p['wallet_label']==wl and p['capture_status']=='captured')} failed={sum(1 for p in plan if p['wallet_label']==wl and p['capture_status']=='failed')} skipped={sum(1 for p in plan if p['wallet_label']==wl and p['capture_status']=='skipped')}\n- confirmed trade-side evidence found: {'yes' if confirmed>0 else 'no'}\n- transfer-only evidence count: {transfer_only}\n- CEX-flow-only evidence count: {cex_count}\n- unknown-direction count: {unknown}\n- realized trade side status: {'unblocked' if confirmed>0 else 'blocked; no direct trade-side fields in tx detail'}\n")
        cost_lines.append(f"## {wl}\n- cost basis available: {'yes' if cost_candidates>0 else 'no'}\n- cost basis candidate records count: {cost_candidates}\n- required fields found: token/amount/timestamp/tx/value_or_price only when direct tx evidence present\n- required fields missing: deterministic acquisition/disposal linkage where candidate count is 0\n- whether historicalUSD was present: {'yes' if any('historicalUSD' in str(e) for e in events) else 'no'}\n- historicalUSD is not cost basis unless linked to confirmed acquisition/disposal methodology.\n")
        link_lines.append(f"## {wl}\n- confirmed acquisitions found count: {cost_candidates}\n- confirmed disposals found count: {confirmed}\n- transfer-only inbound count: {d['inbound_count']}\n- transfer-only outbound count: {d['outbound_count']}\n- unknown-direction transfer count: {unknown}\n- linkage verdict: {verdict}\n")
        fee_lines.append(f"## {wl}\n- fee/gas fields found: {'yes' if fee_found else 'no'}\n- fee/gas source fields: {sorted(fee_fields) if fee_fields else []}\n- whether fee/gas can be included later: {'yes' if fee_found else 'unknown'}\n- whether fee/gas must be excluded if PnL prototype proceeds: {'no' if fee_found else 'yes'}\n- methodology blocker: {'no' if fee_found else 'yes'}\n")

    (out / "transfer_direction_resolution.json").write_text(json.dumps(direction_resolution, indent=2))
    (out / "transaction_hash_inventory.json").write_text(json.dumps(tx_inventory, indent=2))
    (out / "tx_detail_request_plan.json").write_text(json.dumps(plan, indent=2))
    (out / "tx_detail_capture_index.json").write_text(json.dumps(capture_index, indent=2))
    (out / "trade_side_evidence_report.md").write_text("\n".join(trade_lines))
    (out / "cost_basis_evidence_report.md").write_text("\n".join(cost_lines))
    (out / "acquisition_disposal_linkage_report.md").write_text("\n".join(link_lines))
    (out / "fee_gas_methodology_report.md").write_text("\n".join(fee_lines))

    wallets_processed = len(transfers_by_wallet)
    transfers_processed = sum(len(v) for v in transfers_by_wallet.values())
    directions_resolved = sum(d["inbound_count"] + d["outbound_count"] for d in direction_resolution)
    tx_hashes = total_hashes
    tx_captured = sum(1 for p in plan if p["capture_status"] == "captured")
    realized_allowed = total_confirmed_trade > 0 and total_cost_candidates > 0 and all(v != "LINKAGE_BLOCKED" for v in linkage_status)

    phase3b_report = f"# Phase 3B Evidence Expansion Report\n\n- wallets processed: {wallets_processed}\n- transfer events processed: {transfers_processed}\n- directions resolved: {directions_resolved}\n- tx hashes found: {tx_hashes}\n- tx details captured: {tx_captured}\n- confirmed trade-side evidence found: {total_confirmed_trade}\n- cost-basis evidence found: {total_cost_candidates}\n- acquisition/disposal linkage status: {', '.join(linkage_status) if linkage_status else 'none'}\n- remaining blockers: deterministic trade-side and cost-basis gaps remain unless directly evidenced\n\n## allowed next modes\n- exposure-only: allowed\n- flow-only: allowed\n- realized PnL limited design: {'allowed' if wallets_processed > 0 else 'blocked'}\n- realized PnL computation: {'allowed' if realized_allowed else 'blocked'}\n"
    (out / "phase3b_evidence_expansion_report.md").write_text(phase3b_report)

    phase2_found = all(f.exists() for f in phase2_required)
    phase3a_found = all(f.exists() for f in phase3a_required)
    direction_align = all((d["inbound_count"] + d["outbound_count"] + d["unknown_direction_count"]) == d["transfer_events_seen"] for d in direction_resolution)
    consistency_pass = (not realized_allowed) and direction_align
    attempted_count = sum(1 for i in capture_index if i["attempted"])
    captured_count = sum(1 for p in plan if p["capture_status"] == "captured")
    failed_count = sum(1 for p in plan if p["capture_status"] == "failed")
    skipped_count = sum(1 for p in plan if p["capture_status"] == "skipped")
    consistency = f"# Phase 3B Consistency Check\n\n- Phase 2 files found: {'yes' if phase2_found else 'no'}\n- Phase 3A files found: {'yes' if phase3a_found else 'no'}\n- wallet count: {wallets_processed}\n- transfer event count: {transfers_processed}\n- tx hash count: {tx_hashes}\n- tx detail capture attempted: {'yes' if any(i['attempted'] for i in capture_index) else 'no'}\n- tx detail attempted count: {attempted_count}\n- tx detail captured count: {captured_count}\n- tx detail failed count: {failed_count}\n- tx detail skipped count: {skipped_count}\n- config capture enabled: {'yes' if capture_enabled else 'no'}\n- PnL numbers generated: no\n- profitability labels generated: no\n- realized PnL allowed: {'yes' if realized_allowed else 'no'}\n- direction counts align: {'yes' if direction_align else 'no'}\n- cost basis classification exists: {'yes' if (out/'cost_basis_evidence_report.md').exists() else 'no'}\n- consistency_pass: {'yes' if consistency_pass else 'no'}\n"
    (out / "phase3b_consistency_check.md").write_text(consistency)

    verdict = "B" if phase2_found and phase3a_found else "C"
    rec = f"# Phase 3B Recommendation\n\nVerdict: {verdict}) "
    rec += {
        "A": "Phase 3B evidence expansion is sufficient to proceed to Phase 3C limited PnL prototype design",
        "B": "Phase 3B evidence expansion improved coverage but realized PnL remains blocked",
        "C": "Phase 3B evidence expansion is insufficient due to missing inputs or failed capture",
    }[verdict]
    rec += "\n\nSafe next step: " + ("Proceed to Phase 3C limited PnL prototype design" if verdict == "B" else "Fix Phase 3B evidence extraction first") + "\n"
    (out / "phase3b_recommendation.md").write_text(rec)

    if not phase2_found or not phase3a_found:
        print("[Phase3B] Missing required inputs. Paths checked:")
        for p in phase2_required + phase3a_required:
            print(f" - {'FOUND' if p.exists() else 'MISSING'}: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
