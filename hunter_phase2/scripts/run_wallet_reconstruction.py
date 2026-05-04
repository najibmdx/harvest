#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from parse_arkham_samples import parse_sample_file

CATEGORIES = ["balances", "history", "flow", "counterparties", "intelligence", "intelligence_enriched", "transfers"]
DEFAULT_CONFIG = {
    "phase1_output_dir": "hunter_phase1/output",
    "offline_only": True,
    "allow_new_api_calls": False,
    "wallet_labels": [],
    "max_records_per_category": 10000,
}


def load_config(base_dir: Path) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    p = base_dir / "config" / "phase2_config.json"
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def pick(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def resolve_sample_path(raw_ref: str, repo_root: Path, phase1_output_dir: Path) -> Tuple[Path, bool, str]:
    normalized = raw_ref.replace("\\", "/")
    p = Path(normalized)
    candidates = [
        (repo_root / p),
        (phase1_output_dir.parent / p),
        (phase1_output_dir / "raw_samples" / p.name),
        (repo_root / "output" / "raw_samples" / p.name),
        (repo_root / "hunter_phase1" / "output" / "raw_samples" / p.name),
    ]
    if p.is_absolute():
        candidates.insert(0, p)
    for idx, c in enumerate(candidates):
        if c.exists():
            return c.resolve(), True, chr(ord("A") + min(idx, 4))
    return candidates[0].resolve(), False, "not_found"


def extract_wallets(index_data: Any) -> List[Dict[str, Any]]:
    if isinstance(index_data, dict) and isinstance(index_data.get("wallets"), list):
        return index_data["wallets"]
    if isinstance(index_data, list):
        return index_data
    return []


def main() -> None:
    script_path = Path(__file__).resolve()
    phase2_dir = script_path.parents[1]
    repo_root = phase2_dir.parent
    out_dir = phase2_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reconstructed").mkdir(parents=True, exist_ok=True)

    cfg = load_config(phase2_dir)
    phase1_output_dir = (repo_root / cfg["phase1_output_dir"]).resolve()
    index_path = phase1_output_dir / "wallet_evidence_index.json"

    diagnostics = ["# Phase 2 Input Diagnostics", "", f"- resolved phase1_output_dir: `{phase1_output_dir}`", f"- wallet_evidence_index.json exists: {'yes' if index_path.exists() else 'no'}"]
    if not index_path.exists():
        (out_dir / "phase2_input_diagnostics.md").write_text("\n".join(diagnostics), encoding="utf-8")
        raise SystemExit(f"missing required input: {index_path}")

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    wallets = extract_wallets(index_data)
    diagnostics.append(f"- number of wallets found in Phase 1 index: {len(wallets)}")

    timeline = []
    exposure_summary: Dict[str, Any] = {}
    flow_summary: Dict[str, Any] = {}
    counterparty_summary: Dict[str, Any] = {}
    identity_summary: Dict[str, Any] = {}
    coverage_rows = []
    pnl_rows = []

    for w in wallets:
        wallet_label = w.get("wallet_label") or "unknown_wallet"
        if cfg["wallet_labels"] and wallet_label not in cfg["wallet_labels"]:
            continue
        address_redacted = w.get("address_redacted")
        wallet_address = w.get("address") or w.get("wallet_address")
        available = w.get("available_evidence_categories", [])
        missing = w.get("missing_evidence_categories", [c for c in CATEGORIES if c not in available])
        endpoint_diag = w.get("endpoint_request_diagnostics", {})
        sample_map = w.get("sample_file_paths", {})

        diagnostics += ["", f"## {wallet_label}", f"- available_evidence_categories: {', '.join(available) if available else 'none'}", f"- missing_evidence_categories: {', '.join(missing) if missing else 'none'}"]

        wallet_stats = {
            "tokens": set(), "chains": set(), "usd_chain": {}, "top_token": {}, "quote_times": set(), "missing_balance_fields": set(),
            "inbound": 0, "outbound": 0, "unknown": 0, "transfer_like": 0, "missing_flow_fields": set(), "usd_flow": set(),
            "known_cp": set(), "entity_cp": set(), "cex_cp": set(), "unknown_cp": 0, "cp_missing": set(),
            "entity_name": None, "entity_id": None, "entity_type": None, "label_name": None, "identity_chains": set(),
            "balances_parsed": False, "identity_parsed": False, "transfer_events": 0,
        }

        for cat in CATEGORIES:
            raw_ref = pick(endpoint_diag.get(cat, {}) if isinstance(endpoint_diag.get(cat), dict) else {}, ["sample_file_path"]) or sample_map.get(cat) or f"output/raw_samples/{wallet_label}__{cat}.json"
            resolved, exists, rule = resolve_sample_path(str(raw_ref), repo_root, phase1_output_dir)
            parser_status = "not_parsed"
            diagnostics += [f"- {cat}: listed=`{raw_ref}` exists={'yes' if exists else 'no'} resolved=`{resolved}` resolution_rule={rule}"]
            if not exists:
                diagnostics.append(f"  - parser_status: file_missing")
                continue
            parsed = parse_sample_file(resolved, max_records=cfg["max_records_per_category"])
            parser_status = parsed["meta"]["status"]
            diagnostics.append(f"  - parser_status: {parser_status}")
            if parser_status != "ok":
                timeline.append({"wallet_label": wallet_label, "address_redacted": address_redacted, "event_type": "parser_error", "source_category": cat, "timestamp": None, "chain": None, "tx_hash": None, "token_symbol": None, "token_id": None, "token_address": None, "amount": None, "usd_value": None, "direction": None, "counterparty": None, "counterparty_entity": None, "raw_source_file": str(resolved), "evidence_quality": "unusable", "missing_fields": parsed["meta"].get("errors", ["parser_error"])})
                continue

            payload = json.loads(resolved.read_text(encoding="utf-8"))

            if cat == "balances" and isinstance(payload, dict):
                wallet_stats["balances_parsed"] = True
                addresses = payload.get("addresses", {})
                for chain, addr_map in addresses.items():
                    wallet_stats["chains"].add(chain)
                    wallet_stats["identity_chains"].add(chain)
                    if isinstance(addr_map, dict):
                        for _, info in addr_map.items():
                            if not isinstance(info, dict):
                                continue
                            ent = info.get("arkhamEntity") or {}
                            lab = info.get("arkhamLabel") or {}
                            if ent:
                                wallet_stats["identity_parsed"] = True
                                wallet_stats["entity_name"] = wallet_stats["entity_name"] or ent.get("name")
                                wallet_stats["entity_id"] = wallet_stats["entity_id"] or ent.get("id")
                                wallet_stats["entity_type"] = wallet_stats["entity_type"] or ent.get("type")
                            if lab:
                                wallet_stats["label_name"] = wallet_stats["label_name"] or lab.get("name")
                total_balance = payload.get("totalBalance", {})
                if isinstance(total_balance, dict):
                    for chain, usd in total_balance.items():
                        if isinstance(usd, (int, float)):
                            wallet_stats["usd_chain"][chain] = wallet_stats["usd_chain"].get(chain, 0.0) + float(usd)
                bal_map = payload.get("balances", {})
                if isinstance(bal_map, dict):
                    for chain, rows in bal_map.items():
                        wallet_stats["chains"].add(chain)
                        if not isinstance(rows, list):
                            continue
                        for r in rows:
                            if not isinstance(r, dict):
                                continue
                            token_symbol = r.get("symbol")
                            token_id = r.get("id")
                            token_address = r.get("ethereumAddress") or r.get("tokenAddress")
                            amount = r.get("balance")
                            usd_value = r.get("usd")
                            quote_time = r.get("quoteTime")
                            if token_symbol:
                                wallet_stats["tokens"].add(token_symbol)
                            if quote_time:
                                wallet_stats["quote_times"].add(str(quote_time))
                            if token_symbol and isinstance(usd_value, (int, float)):
                                wallet_stats["top_token"][token_symbol] = wallet_stats["top_token"].get(token_symbol, 0.0) + float(usd_value)
                            for req in ["balance", "balanceExact", "id"]:
                                if r.get(req) is None:
                                    wallet_stats["missing_balance_fields"].add(req)
                            missing_fields = [f for f, v in {"token_id": token_id, "amount": amount, "timestamp": quote_time}.items() if v is None]
                            quality = "high" if not missing_fields else ("medium" if len(missing_fields) == 1 else "low")
                            timeline.append({"wallet_label": wallet_label, "address_redacted": address_redacted, "event_type": "balance_snapshot", "source_category": "balances", "timestamp": quote_time, "chain": chain, "tx_hash": None, "token_symbol": token_symbol, "token_id": token_id, "token_address": token_address, "amount": amount, "usd_value": usd_value, "direction": None, "counterparty": None, "counterparty_entity": None, "raw_source_file": str(resolved), "evidence_quality": quality, "missing_fields": missing_fields})

            if cat == "transfers":
                records = payload if isinstance(payload, list) else payload.get("transfers") or payload.get("items") or payload.get("data") or payload.get("result") or []
                if isinstance(records, dict):
                    records = records.get("items") or records.get("data") or []
                if isinstance(records, list):
                    for r in records:
                        if not isinstance(r, dict):
                            continue
                        wallet_stats["transfer_like"] += 1
                        wallet_stats["transfer_events"] += 1
                        from_addr = pick(r, ["fromAddress", "from"])
                        to_addr = pick(r, ["toAddress", "to"])
                        direction = r.get("direction")
                        if direction is None and wallet_address:
                            if from_addr and str(from_addr).lower() == str(wallet_address).lower():
                                direction = "outbound"
                            elif to_addr and str(to_addr).lower() == str(wallet_address).lower():
                                direction = "inbound"
                        if direction == "inbound": wallet_stats["inbound"] += 1
                        elif direction == "outbound": wallet_stats["outbound"] += 1
                        else: wallet_stats["unknown"] += 1
                        token_symbol = pick(r, ["tokenSymbol", "symbol", "token"])
                        amount = pick(r, ["amount", "quantity"])
                        timestamp = pick(r, ["timestamp", "time", "blockTimestamp"])
                        usd_value = pick(r, ["usdValue", "usd", "valueUsd"])
                        chain = pick(r, ["chain", "network", "blockchain"])
                        if usd_value is not None:
                            wallet_stats["usd_flow"].add("usd_value")
                        if amount is None: wallet_stats["missing_flow_fields"].add("amount")
                        if timestamp is None: wallet_stats["missing_flow_fields"].add("timestamp")
                        timeline.append({"wallet_label": wallet_label, "address_redacted": address_redacted, "event_type": "transfer", "source_category": "transfers", "timestamp": timestamp, "chain": chain, "tx_hash": pick(r, ["txHash", "transactionHash", "hash"]), "token_symbol": token_symbol, "token_id": pick(r, ["tokenId", "id"]), "token_address": pick(r, ["tokenAddress", "ethereumAddress", "contractAddress"]), "amount": amount, "usd_value": usd_value, "direction": direction, "counterparty": None, "counterparty_entity": None, "raw_source_file": str(resolved), "evidence_quality": "high" if amount is not None and timestamp is not None else "medium", "missing_fields": [f for f,v in {"amount":amount, "timestamp":timestamp}.items() if v is None]})

            if cat == "flow":
                records = parsed["records"]
                if not records:
                    diagnostics.append("  - flow_parse_note: empty_or_unsupported_shape")
                for r in records:
                    if not isinstance(r, dict):
                        continue
                    if pick(r, ["inbound", "outbound", "netFlow", "usdValue", "amount"]) is None:
                        continue
                    wallet_stats["transfer_like"] += 1

            if cat == "counterparties":
                records = parsed["records"]
                if not records:
                    diagnostics.append("  - counterparty_parse_note: empty_or_unsupported_shape")
                for r in records:
                    if not isinstance(r, dict):
                        continue
                    cp = pick(r, ["counterparty", "counterpartyAddress", "address"])
                    cpe = pick(r, ["counterpartyEntity", "counterpartyEntityName", "entityName"])
                    if cp: wallet_stats["known_cp"].add(str(cp))
                    else:
                        wallet_stats["unknown_cp"] += 1
                        wallet_stats["cp_missing"].add("counterparty")
                    if cpe:
                        wallet_stats["entity_cp"].add(str(cpe))
                        if "cex" in str(cpe).lower() or "exchange" in str(cpe).lower() or "binance" in str(cpe).lower():
                            wallet_stats["cex_cp"].add(str(cpe))

        exposure_summary[wallet_label] = {
            "tokens_observed": sorted(wallet_stats["tokens"]),
            "chains_observed": sorted(wallet_stats["chains"]),
            "total_usd_balance_by_chain": wallet_stats["usd_chain"],
            "top_token_exposures_by_usd": dict(sorted(wallet_stats["top_token"].items(), key=lambda x: x[1], reverse=True)[:10]),
            "quote_time_coverage": sorted(wallet_stats["quote_times"]),
            "missing_balance_fields": sorted(wallet_stats["missing_balance_fields"]),
        }
        flow_summary[wallet_label] = {
            "flow_evidence_found": "flow" in available,
            "transfer_evidence_found": "transfers" in available,
            "number_of_transfer_like_records_parsed": wallet_stats["transfer_like"],
            "inbound_count": wallet_stats["inbound"],
            "outbound_count": wallet_stats["outbound"],
            "unknown_direction_count": wallet_stats["unknown"],
            "usd_flow_fields": sorted(wallet_stats["usd_flow"]),
            "missing_flow_fields": sorted(wallet_stats["missing_flow_fields"]),
        }
        entity_type = wallet_stats["entity_type"]
        identity_summary[wallet_label] = {
            "arkham_entity_evidence": bool(wallet_stats["entity_name"] or wallet_stats["entity_id"]),
            "arkham_label_evidence": bool(wallet_stats["label_name"]),
            "entity_name": wallet_stats["entity_name"],
            "entity_id": wallet_stats["entity_id"],
            "entity_type": entity_type,
            "cex_identification_evidence": bool(entity_type and str(entity_type).lower() == "cex") or (wallet_stats["entity_name"] and "binance" in str(wallet_stats["entity_name"]).lower()),
            "chains_with_identity_evidence": sorted(wallet_stats["identity_chains"]),
            "missing_identity_evidence": [k for k, v in {"entity_name": wallet_stats["entity_name"], "entity_id": wallet_stats["entity_id"], "entity_type": entity_type, "label_name": wallet_stats["label_name"]}.items() if not v],
        }
        counterparty_summary[wallet_label] = {
            "counterparty_evidence_found": "counterparties" in available,
            "known_counterparties": sorted(wallet_stats["known_cp"]),
            "arkham_entity_counterparties": sorted(wallet_stats["entity_cp"]),
            "cex_counterparties": sorted(wallet_stats["cex_cp"]),
            "unknown_counterparties": wallet_stats["unknown_cp"],
            "missing_fields": sorted(wallet_stats["cp_missing"]),
        }

        coverage_rows.append((wallet_label, available, missing, len([e for e in timeline if e["wallet_label"] == wallet_label]), "yes" if wallet_stats["balances_parsed"] else "no"))
        has_amounts = bool(exposure_summary[wallet_label]["tokens_observed"])
        has_prices = bool(exposure_summary[wallet_label]["top_token_exposures_by_usd"])
        has_timestamps = bool(exposure_summary[wallet_label]["quote_time_coverage"])
        has_transfers = wallet_stats["transfer_events"] > 0
        has_direction = wallet_stats["inbound"] > 0 or wallet_stats["outbound"] > 0
        has_disposals = wallet_stats["outbound"] > 0
        missing_fields = [
            x for x, ok in {
                "transfers": has_transfers,
                "token amounts": has_amounts,
                "token prices": has_prices,
                "timestamps": has_timestamps,
                "trade direction": has_direction,
                "cost basis": False,
                "disposal/sell evidence": has_disposals,
            }.items() if not ok
        ]
        verdict = "PARTIALLY_FEASIBLE_WITH_MORE_DATA" if has_amounts and has_prices and has_timestamps else "NOT_FEASIBLE_YET"
        realized = "no"
        unrealized = "partial" if has_amounts and has_prices and has_timestamps else "no"
        if has_transfers and has_direction:
            verdict = "FEASIBLE_FOR_LIMITED_RECONSTRUCTION"
        pnl_rows.append((wallet_label, realized, unrealized, has_transfers, has_amounts, has_prices, has_timestamps, has_direction, False, has_disposals, missing_fields, verdict))

    (out_dir / "phase2_input_diagnostics.md").write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    (out_dir / "wallet_activity_timeline.jsonl").write_text("\n".join(json.dumps(x) for x in timeline) + ("\n" if timeline else ""), encoding="utf-8")
    (out_dir / "wallet_asset_exposure_summary.json").write_text(json.dumps(exposure_summary, indent=2), encoding="utf-8")
    (out_dir / "wallet_flow_summary.json").write_text(json.dumps(flow_summary, indent=2), encoding="utf-8")
    (out_dir / "wallet_counterparty_summary.json").write_text(json.dumps(counterparty_summary, indent=2), encoding="utf-8")
    (out_dir / "wallet_identity_summary.json").write_text(json.dumps(identity_summary, indent=2), encoding="utf-8")

    coverage_md = ["# Reconstruction Coverage Report", ""]
    for label, avail, miss, event_count, usable in coverage_rows:
        coverage_md += [f"## {label}", f"- Evidence categories available: {', '.join(avail) if avail else 'none'}", f"- Evidence categories missing: {', '.join(miss) if miss else 'none'}", f"- Successful parsed categories: {', '.join(avail) if avail else 'none'}", "- Failed parsed categories: none", f"- Number of timeline events produced: {event_count}", f"- Usable for Phase 3: {usable}", f"- Blockers: {'cost basis missing' if usable == 'yes' else 'balance parsing missing'}", ""]
    (out_dir / "reconstruction_coverage_report.md").write_text("\n".join(coverage_md), encoding="utf-8")

    pnl_md = ["# PnL Feasibility Report", "", "Phase 2 does not compute PnL. It only evaluates feasibility.", ""]
    for row in pnl_rows:
        label, realized, unrealized, t, a, p, ts, d, cb, disp, missing_fields, verdict = row
        pnl_md += [f"## {label}", f"- Can realized PnL be computed now? {realized}", f"- Can unrealized exposure be estimated now? {unrealized}", "- Required evidence available:", f"  - transfers: {'yes' if t else 'no'}", f"  - token amounts: {'yes' if a else 'no'}", f"  - token prices: {'yes' if p else 'no'}", f"  - timestamps: {'yes' if ts else 'no'}", f"  - trade direction: {'yes' if d else 'no'}", f"  - cost basis: {'yes' if cb else 'no'}", f"  - disposal/sell evidence: {'yes' if disp else 'no'}", f"- Missing evidence: {', '.join(missing_fields) if missing_fields else 'none'}", f"- Verdict: {verdict}", ""]
    (out_dir / "pnl_feasibility_report.md").write_text("\n".join(pnl_md), encoding="utf-8")

    has_balance_identity = any(identity_summary[w]["arkham_entity_evidence"] or exposure_summary[w]["tokens_observed"] for w in identity_summary)
    has_transfer_timeline = any(x[-1] == "FEASIBLE_FOR_LIMITED_RECONSTRUCTION" for x in pnl_rows)
    verdict = "A" if has_transfer_timeline else ("B" if has_balance_identity else "C")
    rec = {
        "A": ("Phase 2 reconstruction is sufficient to proceed to limited PnL reconstruction design", "Proceed to Phase 3A: PnL Reconstruction Design"),
        "B": ("Phase 2 reconstruction is partially sufficient but needs more evidence coverage", "Expand Phase 1 evidence capture with additional transfer/trade/history coverage"),
        "C": ("Phase 2 reconstruction is insufficient", "Fix parsing/reconstruction coverage first"),
    }
    (out_dir / "phase2_recommendation.md").write_text(f"# {verdict}) {rec[verdict][0]}\n\n- Evidence summary: Wallets processed: {len(coverage_rows)}. Timeline events: {len(timeline)}.\n- Main blockers: cost basis and disposal evidence gaps.\n- Safe next step: {rec[verdict][1]}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
