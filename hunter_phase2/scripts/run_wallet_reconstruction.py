#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from parse_arkham_samples import parse_sample_file

CATEGORIES = ["balances", "history", "flow", "counterparties", "intelligence", "intelligence_enriched", "transfers"]
DEFAULT_CONFIG = {"phase1_output_dir": "hunter_phase1/output", "offline_only": True, "allow_new_api_calls": False, "wallet_labels": [], "max_records_per_category": 10000}


def load_config(base_dir: Path) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    p = base_dir / "config" / "phase2_config.json"
    if p.exists():
        try: cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception: pass
    return cfg


def pick(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if d.get(k) is not None: return d[k]
    return None


def resolve_sample_path(raw_ref: str, repo_root: Path, phase1_output_dir: Path) -> Tuple[Path, bool, str]:
    p = Path(raw_ref.replace("\\", "/"))
    candidates = [repo_root / p, phase1_output_dir.parent / p, phase1_output_dir / "raw_samples" / p.name, repo_root / "output" / "raw_samples" / p.name, repo_root / "hunter_phase1" / "output" / "raw_samples" / p.name]
    if p.is_absolute(): candidates.insert(0, p)
    for i, c in enumerate(candidates):
        if c.exists(): return c.resolve(), True, chr(ord("A") + min(i, 4))
    return candidates[0].resolve(), False, "not_found"


def resolve_project_paths(script_path: Path) -> Tuple[Path, Path]:
    cwd = Path.cwd().resolve()
    if (cwd / "hunter_phase2").exists(): return cwd, cwd / "hunter_phase2"
    phase2_dir = script_path.parents[1]
    if phase2_dir.name == "hunter_phase2": return phase2_dir.parent, phase2_dir
    return cwd, cwd / "hunter_phase2"


def resolve_phase1_output_dir(repo_root: Path, cfg_phase1: str) -> Tuple[Path, List[Path]]:
    attempted = []
    cp = Path(cfg_phase1)
    attempted.append(cp if cp.is_absolute() else (repo_root / cp).resolve())
    attempted.append((repo_root / "hunter_phase1" / "output").resolve())
    attempted.append((repo_root / "output").resolve())
    for c in attempted:
        if (c / "wallet_evidence_index.json").exists(): return c, attempted
    return attempted[0], attempted


def extract_wallets(index_data: Any) -> List[Dict[str, Any]]:
    if isinstance(index_data, dict) and isinstance(index_data.get("wallets"), list): return index_data["wallets"]
    if isinstance(index_data, list): return index_data
    return []


def find_balance_payload(parsed: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and ("balances" in payload or "addresses" in payload): return payload
    for rec in parsed.get("records", []):
        if isinstance(rec, dict) and ("balances" in rec or "addresses" in rec): return rec
    return {}


def main() -> None:
    script_path = Path(__file__).resolve()
    repo_root, phase2_dir = resolve_project_paths(script_path)
    out_dir = phase2_dir / "output"; out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "reconstructed").mkdir(parents=True, exist_ok=True)
    cfg = load_config(phase2_dir)
    phase1_output_dir, attempts = resolve_phase1_output_dir(repo_root, cfg["phase1_output_dir"])
    index_path = phase1_output_dir / "wallet_evidence_index.json"
    diagnostics = ["# Phase 2 Input Diagnostics", "", f"- script path: `{script_path}`", f"- current working directory: `{Path.cwd().resolve()}`", f"- resolved repo root: `{repo_root}`", f"- resolved phase1_output_dir: `{phase1_output_dir}`", f"- wallet_evidence_index.json exists: {'yes' if index_path.exists() else 'no'}"]
    if not index_path.exists():
        diagnostics.append("- attempted phase1_output_dir paths:")
        for p in attempts: diagnostics.append(f"  - `{p}`")
        (out_dir / "phase2_input_diagnostics.md").write_text("\n".join(diagnostics), encoding="utf-8")
        raise SystemExit("missing required input: " + str(index_path))

    wallets = extract_wallets(json.loads(index_path.read_text(encoding="utf-8")))
    diagnostics.append(f"- number of wallets found in Phase 1 index: {len(wallets)}")

    timeline=[]; exposure_summary={}; flow_summary={}; counterparty_summary={}; identity_summary={}; coverage_rows=[]; pnl_rows=[]
    for w in wallets:
        wl = w.get("wallet_label") or "unknown_wallet"
        if cfg["wallet_labels"] and wl not in cfg["wallet_labels"]: continue
        ar = w.get("address_redacted"); wa = w.get("address") or w.get("wallet_address")
        available = w.get("available_evidence_categories", []); missing = w.get("missing_evidence_categories", [c for c in CATEGORIES if c not in available])
        ed = w.get("endpoint_request_diagnostics", {}); sm = w.get("sample_file_paths", {})
        diagnostics += ["", f"## {wl}", f"- available_evidence_categories: {', '.join(available) if available else 'none'}", f"- missing_evidence_categories: {', '.join(missing) if missing else 'none'}"]
        s = {"tokens":set(),"chains":set(),"usd_chain":{},"top_token":{},"quote_times":set(),"missing_balance_fields":set(),"inbound":0,"outbound":0,"unknown":0,"transfer_like":0,"missing_flow_fields":set(),"usd_flow":set(),"known_cp":set(),"entity_cp":set(),"cex_cp":set(),"unknown_cp":0,"cp_missing":set(),"entity_name":None,"entity_id":None,"entity_type":None,"label_name":None,"identity_chains":set(),"balances_parsed":False,"transfer_events":0,"address_chain_count":0,"balance_chain_count":0,"balance_token_rows":0,"identity_records":0,"balance_snapshot_events":0}

        for cat in CATEGORIES:
            raw_ref = pick(ed.get(cat, {}) if isinstance(ed.get(cat), dict) else {}, ["sample_file_path"]) or sm.get(cat) or f"output/raw_samples/{wl}__{cat}.json"
            resolved, exists, rule = resolve_sample_path(str(raw_ref), repo_root, phase1_output_dir)
            diagnostics.append(f"- {cat}: listed=`{raw_ref}` exists={'yes' if exists else 'no'} resolved=`{resolved}` resolution_rule={rule}")
            if not exists: diagnostics.append("  - parser_status: file_missing"); continue
            parsed = parse_sample_file(resolved, max_records=cfg["max_records_per_category"])
            diagnostics.append(f"  - parser_status: {parsed['meta']['status']}")
            if parsed["meta"]["status"] != "ok": continue
            payload = find_balance_payload(parsed, json.loads(resolved.read_text(encoding="utf-8"))) if cat == "balances" else json.loads(resolved.read_text(encoding="utf-8"))

            if cat == "balances" and isinstance(payload, dict):
                s["balances_parsed"] = True
                addresses = payload.get("addresses", {})
                if isinstance(addresses, dict): s["address_chain_count"] = len(addresses)
                for chain, addrmap in (addresses.items() if isinstance(addresses, dict) else []):
                    s["chains"].add(chain); s["identity_chains"].add(chain)
                    if isinstance(addrmap, dict):
                        for _, info in addrmap.items():
                            if not isinstance(info, dict): continue
                            ent = info.get("arkhamEntity") or {}; lab = info.get("arkhamLabel") or {}
                            if ent:
                                s["identity_records"] += 1
                                s["entity_name"] = s["entity_name"] or ent.get("name"); s["entity_id"] = s["entity_id"] or ent.get("id"); s["entity_type"] = s["entity_type"] or ent.get("type")
                            if lab: s["label_name"] = s["label_name"] or lab.get("name")
                tb = payload.get("totalBalance", {})
                if isinstance(tb, dict):
                    for chain, usd in tb.items():
                        s["usd_chain"][chain] = usd
                bm = payload.get("balances", {})
                if isinstance(bm, dict): s["balance_chain_count"] = len(bm)
                for chain, rows in (bm.items() if isinstance(bm, dict) else []):
                    s["chains"].add(chain)
                    if not isinstance(rows, list): continue
                    for r in rows:
                        if not isinstance(r, dict): continue
                        s["balance_token_rows"] += 1
                        symbol, tid, taddr, amount, usd, qt = r.get("symbol"), r.get("id"), r.get("ethereumAddress"), r.get("balance"), r.get("usd"), r.get("quoteTime")
                        if symbol: s["tokens"].add(symbol)
                        if tid: s["tokens"].add(tid)
                        if qt: s["quote_times"].add(str(qt))
                        if symbol and isinstance(usd,(int,float)): s["top_token"][f"{chain}:{symbol}"] = float(usd)
                        for req in ["id","balance","balanceExact"]:
                            if r.get(req) is None: s["missing_balance_fields"].add(req)
                        miss = [f for f,v in {"token_id":tid,"amount":amount,"timestamp":qt}.items() if v is None]
                        timeline.append({"wallet_label":wl,"address_redacted":ar,"event_type":"balance_snapshot","source_category":"balances","timestamp":qt,"chain":chain,"tx_hash":None,"token_symbol":symbol,"token_id":tid,"token_address":taddr,"amount":amount,"usd_value":usd,"direction":None,"counterparty":None,"counterparty_entity":None,"raw_source_file":str(resolved),"evidence_quality":"high" if not miss else "medium","missing_fields":miss})
                        s["balance_snapshot_events"] += 1

            if cat == "transfers":
                recs = payload if isinstance(payload,list) else payload.get("transfers") or payload.get("items") or payload.get("data") or payload.get("result") or []
                if isinstance(recs, dict): recs = recs.get("items") or recs.get("data") or []
                for r in recs if isinstance(recs,list) else []:
                    if not isinstance(r,dict): continue
                    s["transfer_like"] += 1; s["transfer_events"] += 1
                    fa, ta = pick(r,["fromAddress","from"]), pick(r,["toAddress","to"]); direction = r.get("direction")
                    if direction is None and wa:
                        if fa and str(fa).lower()==str(wa).lower(): direction="outbound"
                        elif ta and str(ta).lower()==str(wa).lower(): direction="inbound"
                    if direction=="inbound": s["inbound"] += 1
                    elif direction=="outbound": s["outbound"] += 1
                    else: s["unknown"] += 1

        diagnostics.append(f"- balances extraction counts: number of chains in addresses={s['address_chain_count']}, number of chains in balances={s['balance_chain_count']}, number of token rows extracted={s['balance_token_rows']}, number of identity records extracted={s['identity_records']}, number of balance_snapshot events generated={s['balance_snapshot_events']}")
        exposure_summary[wl] = {"tokens_observed":sorted(s["tokens"]),"chains_observed":sorted(s["chains"]),"total_usd_balance_by_chain":s["usd_chain"],"top_token_exposures_by_usd":dict(sorted(s["top_token"].items(), key=lambda x:x[1], reverse=True)[:10]),"quote_time_coverage":sorted(s["quote_times"]),"missing_balance_fields":sorted(s["missing_balance_fields"])}
        identity_summary[wl] = {"arkham_entity_evidence":bool(s["entity_name"] or s["entity_id"]),"arkham_label_evidence":bool(s["label_name"]),"entity_name":s["entity_name"],"entity_id":s["entity_id"],"entity_type":s["entity_type"],"cex_identification_evidence":str(s["entity_type"] or "").lower()=="cex","chains_with_identity_evidence":sorted(s["identity_chains"]),"missing_identity_evidence":[k for k,v in {"entity_name":s["entity_name"],"entity_id":s["entity_id"],"entity_type":s["entity_type"],"label_name":s["label_name"]}.items() if not v]}
        flow_summary[wl] = {"flow_evidence_found":"flow" in available,"transfer_evidence_found":"transfers" in available,"number_of_transfer_like_records_parsed":s["transfer_like"],"inbound_count":s["inbound"],"outbound_count":s["outbound"],"unknown_direction_count":s["unknown"],"usd_flow_fields":sorted(s["usd_flow"]),"missing_flow_fields":sorted(s["missing_flow_fields"])}
        counterparty_summary[wl] = {"counterparty_evidence_found":"counterparties" in available,"known_counterparties":sorted(s["known_cp"]),"arkham_entity_counterparties":sorted(s["entity_cp"]),"cex_counterparties":sorted(s["cex_cp"]),"unknown_counterparties":s["unknown_cp"],"missing_fields":sorted(s["cp_missing"])}
        has_amounts = s["balance_token_rows"] > 0; has_prices = bool(exposure_summary[wl]["top_token_exposures_by_usd"]); has_timestamps = bool(exposure_summary[wl]["quote_time_coverage"])
        has_transfers = s["transfer_events"] > 0; has_direction = s["inbound"] > 0 or s["outbound"] > 0
        missing_fields = [x for x, ok in {"transfers":has_transfers,"token amounts":has_amounts,"token prices":has_prices,"timestamps":has_timestamps,"trade direction":has_direction,"cost basis":False,"disposal/sell evidence":False}.items() if not ok]
        verdict = "PARTIALLY_FEASIBLE_WITH_MORE_DATA" if has_amounts and has_prices and has_timestamps else "NOT_FEASIBLE_YET"
        if has_transfers and has_direction and has_timestamps: verdict = "FEASIBLE_FOR_LIMITED_RECONSTRUCTION"
        pnl_rows.append((wl,"no","partial" if has_amounts and has_prices and has_timestamps else "no",has_transfers,has_amounts,has_prices,has_timestamps,has_direction,False,False,missing_fields,verdict))
        coverage_rows.append((wl, available, missing, len([e for e in timeline if e["wallet_label"]==wl]), "yes" if s["balances_parsed"] else "no"))

    (out_dir / "phase2_input_diagnostics.md").write_text("\n".join(diagnostics)+"\n", encoding="utf-8")
    (out_dir / "wallet_activity_timeline.jsonl").write_text("\n".join(json.dumps(x) for x in timeline)+("\n" if timeline else ""), encoding="utf-8")
    (out_dir / "wallet_asset_exposure_summary.json").write_text(json.dumps(exposure_summary, indent=2), encoding="utf-8")
    (out_dir / "wallet_flow_summary.json").write_text(json.dumps(flow_summary, indent=2), encoding="utf-8")
    (out_dir / "wallet_counterparty_summary.json").write_text(json.dumps(counterparty_summary, indent=2), encoding="utf-8")
    (out_dir / "wallet_identity_summary.json").write_text(json.dumps(identity_summary, indent=2), encoding="utf-8")
    pnl_md=["# PnL Feasibility Report","", "Phase 2 does not compute PnL. It only evaluates feasibility.",""]
    for r in pnl_rows:
        l,real,unr,t,a,p,ts,d,cb,disp,miss,v=r
        pnl_md += [f"## {l}",f"- Can realized PnL be computed now? {real}",f"- Can unrealized exposure be estimated now? {unr}","- Required evidence available:",f"  - transfers: {'yes' if t else 'no'}",f"  - token amounts: {'yes' if a else 'no'}",f"  - token prices: {'yes' if p else 'no'}",f"  - timestamps: {'yes' if ts else 'no'}",f"  - trade direction: {'yes' if d else 'no'}",f"  - cost basis: {'yes' if cb else 'no'}",f"  - disposal/sell evidence: {'yes' if disp else 'no'}",f"- Missing evidence: {', '.join(miss) if miss else 'none'}",f"- Verdict: {v}",""]
    (out_dir / "pnl_feasibility_report.md").write_text("\n".join(pnl_md), encoding="utf-8")
    has_transfer = any(x[-1]=="FEASIBLE_FOR_LIMITED_RECONSTRUCTION" for x in pnl_rows)
    has_bal_snap = any(e.get("event_type")=="balance_snapshot" for e in timeline)
    verdict = "A" if has_transfer else ("B" if has_bal_snap else "C")
    next_step = "Proceed to Phase 3A: PnL Reconstruction Design" if verdict=="A" else ("Expand Phase 1 evidence capture with additional transfer/trade/history coverage" if verdict=="B" else "Fix parsing/reconstruction coverage first")
    title = {"A":"Phase 2 reconstruction is sufficient to proceed to limited PnL reconstruction design","B":"Phase 2 reconstruction is partially sufficient but needs more evidence coverage","C":"Phase 2 reconstruction is insufficient"}[verdict]
    (out_dir / "phase2_recommendation.md").write_text(f"# {verdict}) {title}\n\n- Evidence summary: Wallets processed: {len(coverage_rows)}. Timeline events: {len(timeline)}.\n- Main blockers: cost basis and disposal evidence gaps.\n- Safe next step: {next_step}\n", encoding="utf-8")

if __name__ == "__main__": main()
