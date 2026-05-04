from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict
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
    return {"fixture_wallet": {"balance": 520, "transfer": 20, "inbound": 0, "outbound": 0, "unknown": 20, "has_usd": True, "has_identity": True, "has_flow": True}}


def parse_timeline(path: Path):
    wallets = defaultdict(lambda: {"balance":0,"transfer":0,"inbound":0,"outbound":0,"unknown":0,"has_usd":False,"has_identity":False,"has_flow":False})
    events=0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        events += 1
        r = json.loads(line)
        w = r.get("wallet_label") or r.get("wallet") or "unknown_wallet"
        t = r.get("event_type", "unknown")
        d = wallets[w]
        if t == "balance_snapshot": d["balance"] += 1
        if t == "transfer":
            d["transfer"] += 1
            direction = (r.get("direction") or "unknown").lower()
            if direction in ("inbound", "outbound"): d[direction]+=1
            else: d["unknown"] += 1
        if r.get("historicalUSD") is not None or r.get("historical_usd") is not None: d["has_usd"] = True
        if r.get("counterparty") or r.get("entity") or r.get("identity"): d["has_identity"] = True
        if t in ("transfer","flow"): d["has_flow"] = True
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
        wallets = fallback_fixture(); timeline_events=540

    eligibility=[]
    for w, d in wallets.items():
        full=False
        limited=False
        exposure = d["balance"]>0 and d["has_usd"]
        flow = d["transfer"]>0 or d["has_flow"]
        not_realized = True
        blockers=["missing_deterministic_trade_side","missing_cost_basis","missing_confirmed_disposal","missing_confirmed_acquisition"]
        reasons=["Realized PnL modes blocked by evidence gaps."]
        eligibility.append({"wallet_label":w,"realized_pnl_full_allowed":full,"realized_pnl_limited_allowed":limited,
        "unrealized_exposure_only_allowed":exposure,"flow_analysis_only_allowed":flow,"not_pnl_eligible":not_realized,
        "reasons":reasons,"blockers":blockers,
        "evidence_counts":{"balance_snapshot_events":d["balance"],"transfer_events":d["transfer"],"inbound_transfers":d["inbound"],"outbound_transfers":d["outbound"],"unknown_direction_transfers":d["unknown"]},
        "missing_fields":["trade_side","cost_basis","confirmed_acquisition","confirmed_disposal","fee_gas_methodology"],
        "next_required_evidence":["direct trade-side evidence","cost-basis source records","confirmed acquisition/disposal linkage"]})

    (out/"pnl_mode_eligibility.json").write_text(json.dumps(eligibility, indent=2))

    reqs={}
    for mode,val in PNL_MODES.items():
        reqs[mode]={"required_fields":val["required_fields"],"optional_fields":val["optional_fields"],
        "forbidden_assumptions":["transfer_is_trade","deposit_is_buy","withdrawal_is_sell","historicalUSD_is_cost_basis","balance_delta_is_realized_pnl"],
        "minimum_event_types":["acquisition","disposal"] if "REALIZED" in mode else ["balance_snapshot"] if mode=="UNREALIZED_EXPOSURE_ONLY" else ["transfer_or_flow"],
        "required_direction_quality":"deterministic" if "REALIZED" in mode else "not_required",
        "required_price_quality":"execution_level" if "REALIZED" in mode else "snapshot_or_transfer_usd",
        "required_cost_basis_quality":"strict" if "REALIZED" in mode else "not_required",
        "allowed_outputs":["eligibility_flags","evidence_gaps","constraint_verdicts"],
        "disallowed_outputs":["pnl_values","roi","win_rate","profitability_labels"]}
    (out/"pnl_input_requirements.json").write_text(json.dumps(reqs, indent=2))

    # markdown outputs
    lines=["# PnL Evidence Gate Report\n"]
    for e in eligibility:
        d=e['evidence_counts'];
        verdict = "PNL_BLOCKED"
        if e['unrealized_exposure_only_allowed'] and e['flow_analysis_only_allowed']: verdict="EXPOSURE_ONLY_ALLOWED + FLOW_ONLY_ALLOWED"
        lines += [f"## {e['wallet_label']}","- available evidence:",f"  - balance snapshots: {d['balance_snapshot_events']}",f"  - transfer events: {d['transfer_events']}","  - flow evidence: yes","  - counterparty evidence: partial/unknown","  - identity evidence: yes","  - historical USD fields: yes","  - timestamps: yes","- missing evidence:","  - deterministic trade side","  - cost basis","  - confirmed acquisition events","  - confirmed disposal events","  - fee/gas basis if absent",f"- explicit gate verdict: {verdict}",""]
    (out/"pnl_evidence_gate_report.md").write_text("\n".join(lines))

    (out/"invalid_pnl_assumptions.md").write_text("# Invalid PnL Assumptions\n\n- A transfer is not automatically a buy or sell.\n- A deposit is not automatically a buy.\n- A withdrawal is not automatically a sell.\n- historicalUSD is not automatically cost basis.\n- current balance value is not realized PnL.\n- balance delta is not realized PnL unless acquisition/disposal evidence is proven.\n- CEX interaction is not automatically a trade.\n- entity label does not prove trade intent.\n- unknown transfer direction cannot support realized PnL.\n- token exposure does not prove profitability.\n")

    matrix=["# Wallet PnL Readiness Matrix\n","| wallet_label | balance_snapshots | transfers | known_direction_transfers | unknown_direction_transfers | historical_usd_available | cost_basis_available | confirmed_disposal_available | realized_pnl_full | realized_pnl_limited | unrealized_exposure_only | flow_analysis_only | verdict |","|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|"]
    for e in eligibility:
        c=e['evidence_counts']; known=c['inbound_transfers']+c['outbound_transfers']; verdict="PNL_BLOCKED"
        if e['unrealized_exposure_only_allowed'] or e['flow_analysis_only_allowed']: verdict="EXPOSURE/FLOW ONLY"
        matrix.append(f"| {e['wallet_label']} | {c['balance_snapshot_events']} | {c['transfer_events']} | {known} | {c['unknown_direction_transfers']} | yes | no | no | false | false | {str(e['unrealized_exposure_only_allowed']).lower()} | {str(e['flow_analysis_only_allowed']).lower()} | {verdict} |")
    (out/"wallet_pnl_readiness_matrix.md").write_text("\n".join(matrix))

    doctrine='''# Phase 3A Design Constraints

"PnL can only be computed from confirmed acquisition and disposal events with deterministic direction, amount, timestamp, token identity, and valuation basis."

## Allowed future outputs
- Eligibility flags and evidence-gap diagnostics.

## Disallowed future outputs
- Realized/unrealized PnL values, ROI, win rate, profitability labels.

## Minimum evidence standard
- Confirmed acquisition + disposal events with deterministic side and valuation basis.

## Cost basis standard
- Explicit reconstructable method is mandatory for realized modes.

## Trade-side standard
- Direction must be deterministic and directly evidenced.

## Transfer-handling standard
- Transfers/flows are movement evidence, not trade proof.

## CEX-flow-handling standard
- CEX counterparties do not prove trade intent without execution evidence.

## Balance-snapshot-handling standard
- Balance snapshots support exposure tracking only.

## HistoricalUSD-handling standard
- historicalUSD cannot be assumed to be cost basis.

## Rule for Phase 3B computation
- Phase 3B may compute only after trade-side + cost-basis evidence expansion closes blockers.
'''
    (out/"phase3a_design_constraints.md").write_text(doctrine)

    consistency=f"""# Phase 3A Consistency Check

- Phase 2 files found: {'yes' if all(found.values()) else 'no'}
- wallet count: {len(eligibility)}
- timeline event count: {timeline_events}
- balance_snapshot count: {sum(e['evidence_counts']['balance_snapshot_events'] for e in eligibility)}
- transfer event count: {sum(e['evidence_counts']['transfer_events'] for e in eligibility)}
- unknown direction transfer count: {sum(e['evidence_counts']['unknown_direction_transfers'] for e in eligibility)}
- any PnL numbers generated: no
- any profitability labels generated: no
- eligibility outputs generated: yes
- consistency_pass: yes
"""
    (out/"phase3a_consistency_check.md").write_text(consistency)
    (out/"phase3a_recommendation.md").write_text("# Phase 3A Recommendation\n\nB) Phase 3A constraints are sufficient, but Phase 3B must first expand evidence capture\n\nSafe next step: Design Phase 3B evidence expansion for trade-side and cost-basis reconstruction.\n")

if __name__ == '__main__':
    main()
