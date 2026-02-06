#!/usr/bin/env python3
"""
panda_liveProof.py

Windows CMD quick run:
  python panda_liveProof.py --db masterwalletsdb.db --outdir exports_live_proofs --limit 200
  python panda_liveProof.py --db masterwalletsdb.db --mint <MINT_ADDRESS> --debug
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

STATE_ORDER = [
    "QUIET",
    "IGNITION",
    "COORDINATION_SPIKE",
    "EARLY_PHASE",
    "PERSISTENCE_CONFIRMED",
    "PARTICIPATION_EXPANSION",
    "PRESSURE_PEAKING",
    "EXHAUSTION_DETECTED",
    "DISSIPATION",
]
STATE_INDEX = {name: idx for idx, name in enumerate(STATE_ORDER)}

PROOF_TARGETS = {
    "TOKEN_PERSISTENCE_CONFIRMED": "PERSISTENCE_CONFIRMED",
    "TOKEN_PARTICIPATION_EXPANSION": "PARTICIPATION_EXPANSION",
    "TOKEN_PRESSURE_PEAKING": "PRESSURE_PEAKING",
    "TOKEN_EXHAUSTION_DETECTED": "EXHAUSTION_DETECTED",
    "TOKEN_DISSIPATION": "DISSIPATION",
}


class Phase3EngineError(RuntimeError):
    pass


class MintMappingError(RuntimeError):
    pass


@dataclass
class MappingPlan:
    mode: str
    mint_expr: str
    join_clause: str


@dataclass
class TraceRecord:
    mint: str
    episode_id: str
    proof: str
    transition_time: int
    from_state: str
    to_state: str
    supporting_events: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine PANDA LIVE proof traces with existing Phase 3 engine.")
    parser.add_argument("--db", default="masterwalletsdb.db")
    parser.add_argument("--outdir", default="exports_live_proofs")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--mint")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--explain", action="store_true")
    return parser.parse_args()


def table_columns(conn: sqlite3.Connection) -> Dict[str, set[str]]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]
    out: Dict[str, set[str]] = {}
    for t in tables:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({t})").fetchall()}
        out[t] = cols
    return out


def locate_phase3_engine(repo_root: Path) -> Tuple[Any, Path]:
    state_needles = ["QUIET", "IGNITION", "COORDINATION_SPIKE", "PERSISTENCE_CONFIRMED", "DISSIPATION"]
    candidates: List[Path] = []
    for pyf in sorted(repo_root.glob("*.py")):
        if pyf.name == Path(__file__).name:
            continue
        text = pyf.read_text(encoding="utf-8", errors="ignore")
        if all(needle in text for needle in state_needles):
            candidates.append(pyf)

    if not candidates:
        raise Phase3EngineError(
            "Phase 3 token state engine not found. Missing module/file containing state constants "
            "[QUIET..DISSIPATION]."
        )

    engine_file = candidates[0]
    spec = importlib.util.spec_from_file_location("panda_phase3_engine", str(engine_file))
    if not spec or not spec.loader:
        raise Phase3EngineError(f"Found candidate engine file but failed to load import spec: {engine_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, engine_file


def choose_mapping(cols: Dict[str, set[str]]) -> MappingPlan:
    if "whale_events" not in cols:
        raise MintMappingError("Missing required table whale_events.")

    whale_cols = cols["whale_events"]
    if "mint" in whale_cols:
        return MappingPlan("direct_whale_events_mint", "we.mint", "")

    if "flow_ref" not in whale_cols:
        raise MintMappingError("whale_events missing flow_ref; cannot deterministically map whale events to mint.")

    if "swaps" in cols:
        swap_cols = cols["swaps"]
        if "signature" in swap_cols:
            for mint_col in ["mint", "token_mint", "output_mint", "input_mint"]:
                if mint_col in swap_cols:
                    return MappingPlan(
                        f"whale_events_flow_ref_to_swaps_{mint_col}",
                        f"s.{mint_col}",
                        "JOIN swaps s ON s.signature = we.flow_ref",
                    )

    if "wallet_token_flow" in cols:
        wtf_cols = cols["wallet_token_flow"]
        if "signature" in wtf_cols and "mint" in wtf_cols:
            return MappingPlan(
                "whale_events_flow_ref_to_wallet_token_flow_mint",
                "wtf.mint",
                "JOIN wallet_token_flow wtf ON wtf.signature = we.flow_ref",
            )

    if "spl_transfers_v2" in cols:
        st_cols = cols["spl_transfers_v2"]
        if "signature" in st_cols:
            for mint_col in ["mint", "token_mint"]:
                if mint_col in st_cols:
                    return MappingPlan(
                        f"whale_events_flow_ref_to_spl_transfers_v2_{mint_col}",
                        f"st.{mint_col}",
                        "JOIN spl_transfers_v2 st ON st.signature = we.flow_ref",
                    )

    raise MintMappingError(
        "Cannot prove mint mapping path for whale_events. Required deterministic key is missing. "
        "Need at least one of: whale_events.mint OR joinable signature path from whale_events.flow_ref "
        "to swaps(wallet_token_flow/spl_transfers_v2) with a mint column."
    )


def get_target_mints(conn: sqlite3.Connection, plan: MappingPlan, limit: int, single_mint: Optional[str]) -> List[str]:
    if single_mint:
        return [single_mint]

    q = f"""
        SELECT DISTINCT {plan.mint_expr} AS mint
        FROM whale_events we
        {plan.join_clause}
        WHERE {plan.mint_expr} IS NOT NULL AND TRIM({plan.mint_expr}) <> ''
        ORDER BY mint
        LIMIT ?
    """
    rows = conn.execute(q, (limit,)).fetchall()
    return [r[0] for r in rows]


def load_events_for_mint(conn: sqlite3.Connection, plan: MappingPlan, mint: str) -> List[Dict[str, Any]]:
    q = f"""
        SELECT we.event_time,
               'WHALE_EVENT' AS kind,
               we.wallet,
               we.event_type,
               we.window,
               we.sol_amount_lamports,
               we.flow_ref
        FROM whale_events we
        {plan.join_clause}
        WHERE {plan.mint_expr} = ?
        ORDER BY we.event_time ASC, we.flow_ref ASC, we.event_type ASC
    """
    rows = conn.execute(q, (mint,)).fetchall()

    events = []
    for t, kind, wallet, event_type, window, amount, flow_ref in rows:
        side = "BUY" if "BUY" in (event_type or "") else "SELL" if "SELL" in (event_type or "") else "UNKNOWN"
        events.append(
            {
                "t": int(t) if t is not None else 0,
                "kind": kind,
                "wallet": wallet,
                "event_type": event_type,
                "window": window,
                "side": side,
                "amount_sol": (amount or 0) / 1_000_000_000,
                "flow_ref": flow_ref,
            }
        )
    return events


def run_engine(module: Any, mint: str, events: List[Dict[str, Any]], conn: sqlite3.Connection) -> Dict[str, Any]:
    for fn_name in ["replay_token_events", "run_phase3_replay", "process_token_events", "replay_events"]:
        fn = getattr(module, fn_name, None)
        if callable(fn):
            return fn(mint=mint, events=events, conn=conn)

    for cls_name in ["Phase3Engine", "PandaLiveEngine", "TokenStateEngine"]:
        cls = getattr(module, cls_name, None)
        if cls:
            inst = cls()
            for meth in ["replay", "run", "process"]:
                m = getattr(inst, meth, None)
                if callable(m):
                    return m(mint=mint, events=events, conn=conn)

    raise Phase3EngineError(
        f"Imported engine module '{module.__name__}' does not expose a supported replay callable. "
        "Expected one of functions [replay_token_events, run_phase3_replay, process_token_events, replay_events] "
        "or engine classes [Phase3Engine, PandaLiveEngine, TokenStateEngine]."
    )


def supporting_slice(events: Sequence[Dict[str, Any]], transition_time: int, max_items: int = 8) -> List[Dict[str, Any]]:
    around = [e for e in events if abs(int(e.get("t", 0)) - transition_time) <= 300]
    return around[:max_items]


def normalize_transitions(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    transitions = result.get("transitions") or result.get("state_transitions") or []
    norm = []
    for tr in transitions:
        norm.append(
            {
                "t": int(tr.get("t") or tr.get("time") or tr.get("transition_time") or 0),
                "from": str(tr.get("from") or tr.get("from_state") or ""),
                "to": str(tr.get("to") or tr.get("to_state") or ""),
                "episode_id": str(tr.get("episode_id") or tr.get("episode") or ""),
            }
        )
    norm.sort(key=lambda x: (x["t"], x["from"], x["to"], x["episode_id"]))
    return norm


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    try:
        engine_module, engine_path = locate_phase3_engine(Path.cwd())
    except Phase3EngineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    cols = table_columns(conn)
    try:
        mapping = choose_mapping(cols)
    except MintMappingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    mints = get_target_mints(conn, mapping, args.limit, args.mint)
    ensure_outdir(Path(args.outdir))

    summary_path = Path(args.outdir) / "proofs_summary.tsv"
    trace_path = Path(args.outdir) / "proofs_traces.jsonl"

    proof_hits_global = {k: 0 for k in PROOF_TARGETS}
    reverse_found_global = False

    rows_out: List[Tuple[Any, ...]] = []
    traces_out: List[TraceRecord] = []

    for mint in mints:
        events = load_events_for_mint(conn, mapping, mint)
        if not events:
            rows_out.append((mint, 0, 0, 0, 0, 0, 0, 0))
            continue

        try:
            result = run_engine(engine_module, mint, events, conn)
        except Phase3EngineError as exc:
            print(f"ERROR: {exc} (engine file: {engine_path})", file=sys.stderr)
            return 2

        transitions = normalize_transitions(result)
        flags = {proof: 0 for proof in PROOF_TARGETS}
        reverse = 0
        episodes = {tr["episode_id"] for tr in transitions if tr["episode_id"]}

        for tr in transitions:
            to_state = tr["to"].replace("TOKEN_", "")
            from_state = tr["from"].replace("TOKEN_", "")
            for proof, target in PROOF_TARGETS.items():
                if to_state == target and flags[proof] == 0:
                    flags[proof] = 1
                    traces_out.append(
                        TraceRecord(
                            mint=mint,
                            episode_id=tr["episode_id"],
                            proof=proof,
                            transition_time=tr["t"],
                            from_state=tr["from"],
                            to_state=tr["to"],
                            supporting_events=supporting_slice(events, tr["t"]),
                        )
                    )
            if from_state in STATE_INDEX and to_state in STATE_INDEX and STATE_INDEX[to_state] < STATE_INDEX[from_state]:
                reverse = 1
                reverse_found_global = True
                if not any(t.proof == "TOKEN_REVERSE_TRANSITION" and t.mint == mint for t in traces_out):
                    traces_out.append(
                        TraceRecord(
                            mint=mint,
                            episode_id=tr["episode_id"],
                            proof="TOKEN_REVERSE_TRANSITION",
                            transition_time=tr["t"],
                            from_state=tr["from"],
                            to_state=tr["to"],
                            supporting_events=supporting_slice(events, tr["t"]),
                        )
                    )

        for k, v in flags.items():
            proof_hits_global[k] += v

        rows_out.append(
            (
                mint,
                flags["TOKEN_PERSISTENCE_CONFIRMED"],
                flags["TOKEN_PARTICIPATION_EXPANSION"],
                flags["TOKEN_PRESSURE_PEAKING"],
                flags["TOKEN_EXHAUSTION_DETECTED"],
                flags["TOKEN_DISSIPATION"],
                reverse,
                len(episodes),
            )
        )

        if args.debug or args.explain:
            print(
                f"[DEBUG] mint={mint} whale_events={len(events)} transitions={len(transitions)} episodes={len(episodes)}",
                file=sys.stderr,
            )

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "mint\thas_persistence\thas_participation_expansion\thas_pressure_peaking\thas_exhaustion\thas_dissipation\thas_reverse\tepisodes_scanned\n"
        )
        for row in rows_out:
            f.write("\t".join(str(x) for x in row) + "\n")

    with trace_path.open("w", encoding="utf-8", newline="") as f:
        for tr in traces_out:
            f.write(
                json.dumps(
                    {
                        "mint": tr.mint,
                        "episode_id": tr.episode_id,
                        "proof": tr.proof,
                        "transition_time": tr.transition_time,
                        "from_state": tr.from_state,
                        "to_state": tr.to_state,
                        "supporting_events": tr.supporting_events,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    # Safety check: PASS in summary must have trace.
    trace_index = {(tr.mint, tr.proof) for tr in traces_out}
    for row in rows_out:
        mint = row[0]
        check = {
            "TOKEN_PERSISTENCE_CONFIRMED": row[1],
            "TOKEN_PARTICIPATION_EXPANSION": row[2],
            "TOKEN_PRESSURE_PEAKING": row[3],
            "TOKEN_EXHAUSTION_DETECTED": row[4],
            "TOKEN_DISSIPATION": row[5],
        }
        for proof, val in check.items():
            if val and (mint, proof) not in trace_index:
                print(f"ERROR: PASS without trace for mint={mint} proof={proof}", file=sys.stderr)
                return 4

    if not rows_out:
        print("No mints found for scan. Wrote empty outputs.")

    print(f"mints_scanned={len(rows_out)}")
    print(f"mints_with_TOKEN_PERSISTENCE_CONFIRMED={proof_hits_global['TOKEN_PERSISTENCE_CONFIRMED']}")
    print(f"mints_with_TOKEN_PARTICIPATION_EXPANSION={proof_hits_global['TOKEN_PARTICIPATION_EXPANSION']}")
    print(f"mints_with_TOKEN_PRESSURE_PEAKING={proof_hits_global['TOKEN_PRESSURE_PEAKING']}")
    print(f"mints_with_TOKEN_EXHAUSTION_DETECTED={proof_hits_global['TOKEN_EXHAUSTION_DETECTED']}")
    print(f"mints_with_TOKEN_DISSIPATION={proof_hits_global['TOKEN_DISSIPATION']}")
    print(f"reverse_transition_found={'YES' if reverse_found_global else 'NO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
