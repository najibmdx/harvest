#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from parse_arkham_samples import parse_sample_file

DEFAULT_CONFIG = {
    "phase1_output_dir": "hunter_phase1/output",
    "offline_only": True,
    "allow_new_api_calls": False,
    "wallet_labels": [],
    "max_records_per_category": 10000,
}


def load_config(base_dir: Path) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    cfg_path = base_dir / "config" / "phase2_config.json"
    if cfg_path.exists():
        try:
            cfg.update(json.loads(cfg_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def redacted(addr: str | None) -> str | None:
    if not addr:
        return None
    if len(addr) < 12:
        return addr
    return f"{addr[:6]}...{addr[-4:]}"


def pick(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    output_dir = base / "output"
    recon_dir = output_dir / "reconstructed"
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(base)
    phase1 = Path(cfg["phase1_output_dir"])
    index_path = phase1 / "wallet_evidence_index.json"
    raw_dir = phase1 / "raw_samples"

    if not index_path.exists():
        raise SystemExit(f"missing required input: {index_path}")

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    wallets = index_data.get("wallets", index_data if isinstance(index_data, list) else [])

    timeline: List[Dict[str, Any]] = []
    exposure_summary = {}
    flow_summary = {}
    counterparty_summary = {}
    identity_summary = {}
    coverage_rows = []
    pnl_rows = []

    for w in wallets:
        wallet_label = w.get("wallet_label") or w.get("label") or w.get("wallet") or "unknown_wallet"
        if cfg["wallet_labels"] and wallet_label not in cfg["wallet_labels"]:
            continue

        addr = w.get("address") or w.get("wallet_address")
        addr_redacted = redacted(addr)
        categories = w.get("categories", {})
        if isinstance(categories, list):
            categories = {c: {} for c in categories}

        available = []
        failed = []
        parsed_by_cat = defaultdict(list)

        for cat in ["balances", "history", "flow", "counterparties", "intelligence", "intelligence_enriched", "transfers"]:
            cat_ref = categories.get(cat)
            sample_files = []
            if isinstance(cat_ref, dict):
                sample_files = cat_ref.get("sample_files", [])
            elif isinstance(cat_ref, list):
                sample_files = cat_ref
            elif isinstance(cat_ref, str):
                sample_files = [cat_ref]

            for sf in sample_files:
                p = raw_dir / sf
                parsed = parse_sample_file(p, max_records=cfg["max_records_per_category"])
                parsed_by_cat[cat].append(parsed)
                if parsed["meta"]["status"] == "ok":
                    available.append(cat)
                else:
                    failed.append(cat)

        available = sorted(set(available))
        failed = sorted(set(failed))

        tokens, chains = set(), set()
        usd_by_chain = defaultdict(float)
        quote_times = set()
        top_token_usd = defaultdict(float)
        missing_balance_fields = set()

        inbound = outbound = unknown = transfer_like = 0
        usd_flow_fields = set()
        missing_flow_fields = set()

        known_cp, entity_cp, cex_cp, unknown_cp = set(), set(), set(), 0
        cp_missing = set()

        entity_name = entity_id = entity_type = label_name = None
        identity_chains = set()

        for cat, parsed_list in parsed_by_cat.items():
            for parsed in parsed_list:
                if parsed["meta"]["status"] != "ok":
                    timeline.append({"wallet_label": wallet_label, "address_redacted": addr_redacted, "event_type": "parser_error", "source_category": cat, "timestamp": None, "chain": None, "tx_hash": None, "token_symbol": None, "token_id": None, "token_address": None, "amount": None, "usd_value": None, "direction": None, "counterparty": None, "counterparty_entity": None, "raw_source_file": parsed["file"], "evidence_quality": "unusable", "missing_fields": parsed["meta"].get("errors", ["unknown_parser_error"])})
                    continue

                for r in parsed["records"]:
                    chain = pick(r, ["chain", "network", "blockchain"])
                    if chain:
                        chains.add(str(chain))
                    timestamp = pick(r, ["timestamp", "time", "blockTimestamp", "quoteTime"])
                    token_symbol = pick(r, ["tokenSymbol", "symbol", "token"])
                    token_id = pick(r, ["tokenId", "assetId", "id"])
                    token_address = pick(r, ["tokenAddress", "contractAddress", "address"])
                    usd_value = pick(r, ["usdValue", "valueUsd", "usd", "balanceUsd"])
                    amount = pick(r, ["amount", "balance", "balanceExact", "quantity"])
                    tx_hash = pick(r, ["txHash", "transactionHash", "hash"])

                    if cat in {"intelligence", "intelligence_enriched", "history"}:
                        entity_name = entity_name or pick(r, ["entityName", "entity", "name"])
                        entity_id = entity_id or pick(r, ["entityId", "id"])
                        entity_type = entity_type or pick(r, ["entityType", "type"])
                        label_name = label_name or pick(r, ["label", "labelName"])
                        if chain:
                            identity_chains.add(str(chain))

                    direction = pick(r, ["direction", "flowDirection"])
                    from_addr = pick(r, ["fromAddress", "from", "sender"])
                    to_addr = pick(r, ["toAddress", "to", "recipient"])
                    if not direction and addr:
                        if from_addr and str(from_addr).lower() == addr.lower():
                            direction = "outbound"
                        elif to_addr and str(to_addr).lower() == addr.lower():
                            direction = "inbound"

                    counterparty = pick(r, ["counterparty", "counterpartyAddress"])
                    counterparty_entity = pick(r, ["counterpartyEntity", "counterpartyEntityName"])

                    if cat == "balances":
                        if token_symbol:
                            tokens.add(str(token_symbol))
                        for req in ["balance", "balanceExact"]:
                            if r.get(req) is None:
                                missing_balance_fields.add(req)
                        if usd_value is not None and chain:
                            try:
                                usd_by_chain[str(chain)] += float(usd_value)
                            except Exception:
                                pass
                        if token_symbol and usd_value is not None:
                            try:
                                top_token_usd[str(token_symbol)] += float(usd_value)
                            except Exception:
                                pass
                        qt = pick(r, ["quoteTime"])
                        if qt is not None:
                            quote_times.add(str(qt))

                    if cat in {"flow", "transfers"}:
                        transfer_like += 1
                        if direction == "inbound":
                            inbound += 1
                        elif direction == "outbound":
                            outbound += 1
                        else:
                            unknown += 1
                        if usd_value is not None:
                            usd_flow_fields.add("usd_value")
                        for req in ["amount", "timestamp"]:
                            if pick(r, [req, "time", "blockTimestamp"]) is None:
                                missing_flow_fields.add(req)

                    if cat == "counterparties":
                        if counterparty:
                            known_cp.add(str(counterparty))
                        else:
                            unknown_cp += 1
                            cp_missing.add("counterparty")
                        if counterparty_entity:
                            entity_cp.add(str(counterparty_entity))
                            if "cex" in str(counterparty_entity).lower() or "exchange" in str(counterparty_entity).lower():
                                cex_cp.add(str(counterparty_entity))

                    missing_fields = [
                        f for f, v in {
                            "timestamp": timestamp,
                            "chain": chain,
                            "token_symbol": token_symbol,
                            "amount": amount,
                        }.items() if v is None
                    ]
                    quality = "high" if len(missing_fields) <= 1 else "low"
                    if direction and quality == "low":
                        quality = "medium"

                    timeline.append({
                        "wallet_label": wallet_label,
                        "address_redacted": addr_redacted,
                        "event_type": cat,
                        "source_category": cat,
                        "timestamp": timestamp,
                        "chain": chain,
                        "tx_hash": tx_hash,
                        "token_symbol": token_symbol,
                        "token_id": token_id,
                        "token_address": token_address,
                        "amount": amount,
                        "usd_value": usd_value,
                        "direction": direction,
                        "counterparty": counterparty,
                        "counterparty_entity": counterparty_entity,
                        "raw_source_file": parsed["file"],
                        "evidence_quality": quality,
                        "missing_fields": missing_fields,
                    })

        exposure_summary[wallet_label] = {
            "tokens_observed": sorted(tokens),
            "chains_observed": sorted(chains),
            "total_usd_balance_by_chain": usd_by_chain,
            "top_token_exposures_by_usd": dict(sorted(top_token_usd.items(), key=lambda x: x[1], reverse=True)[:10]),
            "quote_time_coverage": sorted(quote_times),
            "missing_balance_fields": sorted(missing_balance_fields),
        }
        flow_summary[wallet_label] = {
            "flow_evidence_found": "flow" in available,
            "transfer_evidence_found": "transfers" in available,
            "number_of_transfer_like_records_parsed": transfer_like,
            "inbound_count": inbound,
            "outbound_count": outbound,
            "unknown_direction_count": unknown,
            "usd_flow_fields": sorted(usd_flow_fields),
            "missing_flow_fields": sorted(missing_flow_fields),
        }
        counterparty_summary[wallet_label] = {
            "counterparty_evidence_found": "counterparties" in available,
            "known_counterparties": sorted(known_cp),
            "arkham_entity_counterparties": sorted(entity_cp),
            "cex_counterparties": sorted(cex_cp),
            "unknown_counterparties": unknown_cp,
            "missing_fields": sorted(cp_missing),
        }
        identity_summary[wallet_label] = {
            "arkham_entity_evidence": bool(entity_name or entity_id),
            "arkham_label_evidence": bool(label_name),
            "entity_name": entity_name,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "cex_identification_evidence": bool(entity_type and "cex" in str(entity_type).lower()),
            "chains_with_identity_evidence": sorted(identity_chains),
            "missing_identity_evidence": [k for k, v in {"entity_name": entity_name, "entity_id": entity_id, "entity_type": entity_type, "label_name": label_name}.items() if not v],
        }

        wallet_events = [e for e in timeline if e["wallet_label"] == wallet_label]
        usable = "yes" if any(e["event_type"] == "transfers" for e in wallet_events) else "no"
        coverage_rows.append((wallet_label, available, failed, len(wallet_events), usable))

        has_transfers = flow_summary[wallet_label]["transfer_evidence_found"]
        has_amounts = transfer_like > 0 and "amount" not in flow_summary[wallet_label]["missing_flow_fields"]
        has_prices = len(exposure_summary[wallet_label]["top_token_exposures_by_usd"]) > 0
        has_timestamps = "timestamp" not in flow_summary[wallet_label]["missing_flow_fields"]
        has_direction = unknown < transfer_like if transfer_like else False
        has_cost_basis = False
        has_disposals = outbound > 0
        missing = []
        if not has_transfers:
            missing.append("transfers")
        if not has_amounts:
            missing.append("token amounts")
        if not has_prices:
            missing.append("token prices/usd valuation fields")
        if not has_timestamps:
            missing.append("timestamps")
        if not has_direction:
            missing.append("trade direction")
        if not has_cost_basis:
            missing.append("cost basis")
        if not has_disposals:
            missing.append("disposal/sell evidence")

        realized = "no" if missing else "partial"
        unrealized = "partial" if has_amounts and has_prices else "no"
        if not missing:
            verdict = "FEASIBLE_FOR_LIMITED_RECONSTRUCTION"
        elif len(missing) <= 2:
            verdict = "PARTIALLY_FEASIBLE_WITH_MORE_DATA"
        else:
            verdict = "NOT_FEASIBLE_YET"
        pnl_rows.append((wallet_label, realized, unrealized, has_transfers, has_amounts, has_prices, has_timestamps, has_direction, has_cost_basis, has_disposals, missing, verdict))

    (output_dir / "wallet_activity_timeline.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in timeline) + ("\n" if timeline else ""), encoding="utf-8")
    (output_dir / "wallet_asset_exposure_summary.json").write_text(json.dumps(exposure_summary, indent=2), encoding="utf-8")
    (output_dir / "wallet_flow_summary.json").write_text(json.dumps(flow_summary, indent=2), encoding="utf-8")
    (output_dir / "wallet_counterparty_summary.json").write_text(json.dumps(counterparty_summary, indent=2), encoding="utf-8")
    (output_dir / "wallet_identity_summary.json").write_text(json.dumps(identity_summary, indent=2), encoding="utf-8")

    coverage_md = ["# Reconstruction Coverage Report", ""]
    for label, avail, fail, cnt, usable in coverage_rows:
        missing = sorted(set(["balances", "history", "flow", "counterparties", "intelligence", "intelligence_enriched", "transfers"]) - set(avail))
        blockers = ["no transfer evidence"] if usable == "no" else []
        coverage_md += [f"## {label}", f"- Evidence categories available: {', '.join(avail) if avail else 'none'}", f"- Evidence categories missing: {', '.join(missing) if missing else 'none'}", f"- Successful parsed categories: {', '.join(avail) if avail else 'none'}", f"- Failed parsed categories: {', '.join(fail) if fail else 'none'}", f"- Number of timeline events produced: {cnt}", f"- Usable for Phase 3: {usable}", f"- Blockers: {', '.join(blockers) if blockers else 'none'}", ""]
    (output_dir / "reconstruction_coverage_report.md").write_text("\n".join(coverage_md), encoding="utf-8")

    pnl_md = ["# PnL Feasibility Report", "", "Phase 2 does not compute PnL. It only evaluates feasibility.", ""]
    for row in pnl_rows:
        label, realized, unrealized, t, a, p, ts, d, cb, disp, missing, verdict = row
        pnl_md += [f"## {label}", f"- Can realized PnL be computed now? {realized}", f"- Can unrealized exposure be estimated now? {unrealized}", "- Required evidence available:", f"  - transfers: {'yes' if t else 'no'}", f"  - token amounts: {'yes' if a else 'no'}", f"  - token prices: {'yes' if p else 'no'}", f"  - timestamps: {'yes' if ts else 'no'}", f"  - trade direction: {'yes' if d else 'no'}", f"  - cost basis: {'yes' if cb else 'no'}", f"  - disposal/sell evidence: {'yes' if disp else 'no'}", f"- Missing evidence: {', '.join(missing) if missing else 'none'}", f"- Verdict: {verdict}", ""]
    (output_dir / "pnl_feasibility_report.md").write_text("\n".join(pnl_md), encoding="utf-8")

    verdict = "C"
    if pnl_rows and any(r[-1] == "FEASIBLE_FOR_LIMITED_RECONSTRUCTION" for r in pnl_rows):
        verdict = "A"
    elif pnl_rows and any(r[-1] == "PARTIALLY_FEASIBLE_WITH_MORE_DATA" for r in pnl_rows):
        verdict = "B"
    rec_map = {
        "A": ("Phase 2 reconstruction is sufficient to proceed to limited PnL reconstruction design", "Proceed to Phase 3A: PnL Reconstruction Design"),
        "B": ("Phase 2 reconstruction is partially sufficient but needs more evidence coverage", "Expand Phase 1 evidence capture with additional endpoints/fixtures"),
        "C": ("Phase 2 reconstruction is insufficient", "Fix parsing/reconstruction coverage first"),
    }
    summary = f"Wallets processed: {len(coverage_rows)}. Timeline events: {len(timeline)}."
    blockers = sorted(set(x for row in pnl_rows for x in row[-2]))
    rec_md = [f"# {verdict}) {rec_map[verdict][0]}", "", f"- Evidence summary: {summary}", f"- Main blockers: {', '.join(blockers) if blockers else 'none'}", f"- Safe next step: {rec_map[verdict][1]}"]
    (output_dir / "phase2_recommendation.md").write_text("\n".join(rec_md), encoding="utf-8")


if __name__ == "__main__":
    main()
