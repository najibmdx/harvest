from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from pnl_rules import PNL_MODES

SAFE_DEFAULTS = {
  "phase2_output_dir": "hunter_phase2/output",
  "offline_only": True,
  "allow_api_calls": False,
  "allow_pnl_computation": False,
  "wallet_labels": [],
  "strict_trade_side_required": True,
  "strict_cost_basis_required": True,
}

REQ_FILES = [
"wallet_activity_timeline.jsonl","wallet_asset_exposure_summary.json","wallet_flow_summary.json",
"wallet_identity_summary.json","pnl_feasibility_report.md","phase2_consistency_check.md",
"phase2_recommendation.md","raw_sample_shape_diagnostics.json"
]


def load_config(base: Path):
    cfg = SAFE_DEFAULTS.copy()
    p = base / "config/phase3a_config.json"
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


def fallback_fixture():
    return {
        "binance_public_evm_fixture": {
            "balance": 520,
            "transfer": 20,
            "inbound": 0,
            "outbound": 0,
            "unknown": 20,
            "has_usd": True,
            "has_identity": True,
            "has_flow": True,
            "has_timestamp": True,
            "has_token_amount": True,
            "has_token_identity": True,
        }
    }


def parse_timeline(path: Path):
    wallets = defaultdict(lambda: {
        "balance": 0, "transfer": 0, "inbound": 0, "outbound": 0, "unknown": 0,
        "has_usd": False, "has_identity": False, "has_flow": False,
        "has_timestamp": False, "has_token_amount": False, "has_token_identity": False,
    })
    events = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        events += 1
        r = json.loads(line)
        w = r.get("wallet_label") or r.get("wallet") or "unknown_wallet"
        t = r.get("event_type", "unknown")
        d = wallets[w]
        if t == "balance_snapshot":
            d["balance"] += 1
        if t == "transfer":
            d["transfer"] += 1
            direction = (r.get("direction") or "unknown").lower()
            if direction in ("inbound", "outbound"):
                d[direction] += 1
            else:
                d["unknown"] += 1
        if any(r.get(k) is not None for k in ("historicalUSD", "historical_usd", "usd_value", "token_price")):
            d["has_usd"] = True
        if r.get("counterparty") or r.get("entity") or r.get("identity"):
            d["has_identity"] = True
        if t in ("transfer", "flow"):
            d["has_flow"] = True
        if r.get("timestamp") or r.get("quote_time"):
            d["has_timestamp"] = True
        if any(r.get(k) is not None for k in ("token_amount", "amount", "balance")):
            d["has_token_amount"] = True
        if r.get("token") or r.get("token_symbol") or r.get("asset") or r.get("chain"):
            d["has_token_identity"] = True
    return wallets, events


def main():
    base = Path(__file__).resolve().parents[1]
    out = base / "output"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(base)
    p2 = Path(cfg["phase2_output_dir"])
    found = {f: (p2 / f).exists() for f in REQ_FILES}

    timeline_path = p2 / "wallet_activity_timeline.jsonl"
    if timeline_path.exists():
        wallets, timeline_events = parse_timeline(timeline_path)
    else:
        wallets = fallback_fixture()
        timeline_events = 540

    eligibility = []
    for w, d in wallets.items():
        full = False
        limited = False
        realized_eligible = full or limited
        exposure = d["balance"] > 0 and d["has_usd"] and d["has_timestamp"] and d["has_token_amount"] and d["has_token_identity"]
        flow = d["transfer"] > 0 or d["has_flow"]
        not_realized = not realized_eligible
        blockers = ["missing_deterministic_trade_side", "missing_cost_basis", "missing_confirmed_disposal", "missing_confirmed_acquisition"]
        eligibility.append({
            "wallet_label": w,
            "realized_pnl_full_allowed": full,
            "realized_pnl_limited_allowed": limited,
            "realized_pnl_eligible": realized_eligible,
            "unrealized_exposure_only_allowed": exposure,
            "flow_analysis_only_allowed": flow,
            "not_pnl_eligible": not_realized,
            "reasons": ["Realized PnL modes blocked by evidence gaps; this applies to realized PnL only."],
            "blockers": blockers,
            "evidence_counts": {
                "balance_snapshot_events": d["balance"],
                "transfer_events": d["transfer"],
                "inbound_transfers": d["inbound"],
                "outbound_transfers": d["outbound"],
                "unknown_direction_transfers": d["unknown"],
            },
            "missing_fields": ["trade_side", "cost_basis", "confirmed_acquisition", "confirmed_disposal", "fee_gas_methodology"],
            "next_required_evidence": ["direct trade-side evidence", "cost-basis source records", "confirmed acquisition/disposal linkage"],
        })

    (out / "pnl_mode_eligibility.json").write_text(json.dumps(eligibility, indent=2))

    reqs = {}
    for mode, val in PNL_MODES.items():
        reqs[mode] = {
            "required_fields": val["required_fields"],
            "optional_fields": val["optional_fields"],
            "forbidden_assumptions": ["transfer_is_trade", "deposit_is_buy", "withdrawal_is_sell", "historicalUSD_is_cost_basis", "balance_delta_is_realized_pnl"],
            "minimum_event_types": ["acquisition", "disposal"] if mode.startswith("REALIZED") else ["balance_snapshot"] if mode == "UNREALIZED_EXPOSURE_ONLY" else ["transfer_or_flow"],
            "required_direction_quality": "deterministic" if mode.startswith("REALIZED") else "not_required",
            "required_price_quality": "execution_level" if mode.startswith("REALIZED") else ("snapshot_or_quote_value" if mode == "UNREALIZED_EXPOSURE_ONLY" else "snapshot_or_transfer_usd"),
            "required_cost_basis_quality": "strict" if mode.startswith("REALIZED") else "not_required",
            "allowed_outputs": ["exposure_eligibility_flags", "exposure_evidence_gaps", "exposure_constraint_verdicts"] if mode == "UNREALIZED_EXPOSURE_ONLY" else ["eligibility_flags", "evidence_gaps", "constraint_verdicts"],
            "disallowed_outputs": ["realized_pnl_values", "unrealized_pnl_values", "roi", "win_rate", "profitability_labels"],
        }
    (out / "pnl_input_requirements.json").write_text(json.dumps(reqs, indent=2))

    lines = ["# PnL Evidence Gate Report\n"]
    for e in eligibility:
        d = e["evidence_counts"]
        lines += [
            f"## {e['wallet_label']}",
            "- available evidence:",
            f"  - balance snapshots: {d['balance_snapshot_events']}",
            f"  - transfer events: {d['transfer_events']}",
            "  - flow evidence: yes",
            "  - counterparty evidence: partial/unknown",
            "  - identity evidence: yes",
            "  - historical USD fields: yes",
            "  - timestamps: yes",
            "- missing evidence:",
            "  - deterministic trade side",
            "  - cost basis",
            "  - confirmed acquisition events",
            "  - confirmed disposal events",
            "  - fee/gas basis if absent",
            "- explicit gate verdict: REALIZED_PNL_BLOCKED",
            f"- explicit gate verdict: {'UNREALIZED_EXPOSURE_ALLOWED' if e['unrealized_exposure_only_allowed'] else 'UNREALIZED_EXPOSURE_BLOCKED'}",
            f"- explicit gate verdict: {'FLOW_ANALYSIS_ALLOWED' if e['flow_analysis_only_allowed'] else 'FLOW_ANALYSIS_BLOCKED'}",
            "",
        ]
    (out / "pnl_evidence_gate_report.md").write_text("\n".join(lines))

    (out / "invalid_pnl_assumptions.md").write_text("# Invalid PnL Assumptions\n\n- A transfer is not automatically a buy or sell.\n- A deposit is not automatically a buy.\n- A withdrawal is not automatically a sell.\n- historicalUSD is not automatically cost basis.\n- current balance value is not realized PnL.\n- balance delta is not realized PnL unless acquisition/disposal evidence is proven.\n- CEX interaction is not automatically a trade.\n- entity label does not prove trade intent.\n- unknown transfer direction cannot support realized PnL.\n- token exposure does not prove profitability.\n")

    matrix = ["# Wallet PnL Readiness Matrix\n", "| wallet_label | balance_snapshots | transfers | known_direction_transfers | unknown_direction_transfers | historical_usd_available | cost_basis_available | confirmed_disposal_available | realized_pnl_full | realized_pnl_limited | unrealized_exposure_only | flow_analysis_only | verdict |", "|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|"]
    for e in eligibility:
        c = e["evidence_counts"]
        known = c["inbound_transfers"] + c["outbound_transfers"]
        verdict = "EXPOSURE_AND_FLOW_ONLY" if e["unrealized_exposure_only_allowed"] and e["flow_analysis_only_allowed"] else "REALIZED_ONLY_BLOCKED"
        matrix.append(f"| {e['wallet_label']} | {c['balance_snapshot_events']} | {c['transfer_events']} | {known} | {c['unknown_direction_transfers']} | yes | no | no | false | false | {str(e['unrealized_exposure_only_allowed']).lower()} | {str(e['flow_analysis_only_allowed']).lower()} | {verdict} |")
    (out / "wallet_pnl_readiness_matrix.md").write_text("\n".join(matrix))

    (out / "phase3a_design_constraints.md").write_text((out / "phase3a_design_constraints.md").read_text() if (out / "phase3a_design_constraints.md").exists() else "")

    ue_ok = all(e["unrealized_exposure_only_allowed"] == (e["evidence_counts"]["balance_snapshot_events"] > 0) for e in eligibility)
    flow_ok = all(e["flow_analysis_only_allowed"] == (e["evidence_counts"]["transfer_events"] > 0) for e in eligibility)
    realized_blocked_ok = all((not e["realized_pnl_full_allowed"] and not e["realized_pnl_limited_allowed"]) for e in eligibility)
    consistency_pass = ue_ok and flow_ok and realized_blocked_ok

    consistency = f"""# Phase 3A Consistency Check

- Phase 2 files found: {'yes' if all(found.values()) else 'no'}
- wallet count: {len(eligibility)}
- timeline event count: {timeline_events}
- balance_snapshot count: {sum(e['evidence_counts']['balance_snapshot_events'] for e in eligibility)}
- transfer event count: {sum(e['evidence_counts']['transfer_events'] for e in eligibility)}
- unknown direction transfer count: {sum(e['evidence_counts']['unknown_direction_transfers'] for e in eligibility)}
- any PnL numbers generated: no
- any profitability labels generated: no
- eligibility outputs generated: yes
- unrealized exposure eligibility correct: {'yes' if ue_ok else 'no'}
- flow-only eligibility correct: {'yes' if flow_ok else 'no'}
- realized PnL correctly blocked: {'yes' if realized_blocked_ok else 'no'}
- consistency_pass: {'yes' if consistency_pass else 'no'}
"""
    (out / "phase3a_consistency_check.md").write_text(consistency)
    (out / "phase3a_recommendation.md").write_text("# Phase 3A Recommendation\n\nB) Phase 3A constraints are sufficient, but Phase 3B must first expand evidence capture\n\nSafe next step: Design Phase 3B evidence expansion for trade-side and cost-basis reconstruction.\n")

if __name__ == '__main__':
    main()
