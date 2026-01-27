#!/usr/bin/env python3
"""
Solana Memecoin CA Scanner (Helius RPC)
- Prompts for token CA (mint)
- Prompts for scan minutes (default 60)
- Scans from token birth for N minutes OR stops early if "dead"
- Writes JSONL logs: <outdir>/<MINT>.jsonl
- Repeats until user types exit/quit/q
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests

# -------------------------
# Deterministic defaults
# -------------------------
DEFAULT_MINUTES = 60
DEAD_GAP_SECS_DEFAULT = 5 * 60         # "dead" = no txs for 5 minutes
SIG_PAGE_LIMIT_DEFAULT = 1000          # max per Solana RPC page
TX_FETCH_SLEEP_SECS = 0.12             # be polite to RPC


def helius_rpc_url() -> str:
    key = os.environ.get("HELIUS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing HELIUS_API_KEY environment variable.")
    return f"https://mainnet.helius-rpc.com/?api-key={key}"


def rpc_call(url: str, method: str, params: list, timeout: int = 40) -> Any:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")
    return data.get("result")


def get_signatures_page(
    url: str,
    address: str,
    before: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    cfg: Dict[str, Any] = {"limit": int(limit)}
    if before:
        cfg["before"] = before
    return rpc_call(url, "getSignaturesForAddress", [address, cfg])


def find_birth_ts_and_sig(url: str, mint: str, limit: int) -> Tuple[int, str, int]:
    """
    Birth = oldest signature retrievable for the mint address (by paging back).
    Returns (birth_ts, birth_sig, pages_scanned).
    """
    before = None
    oldest_ts: Optional[int] = None
    oldest_sig: Optional[str] = None
    pages = 0

    while True:
        rows = get_signatures_page(url, mint, before=before, limit=limit)
        pages += 1
        if not rows:
            break

        last = rows[-1]
        sig = last.get("signature")
        ts = last.get("blockTime")
        if sig and ts:
            oldest_sig = sig
            oldest_ts = int(ts)

        before = rows[-1].get("signature")

        if len(rows) < limit:
            break

        if pages > 500:
            raise RuntimeError("Too many pages while finding birth (safety stop).")

    if oldest_ts is None or oldest_sig is None:
        raise RuntimeError("Could not determine token birth (no signature with blockTime found).")

    return oldest_ts, oldest_sig, pages


def collect_signatures_in_window(
    url: str,
    mint: str,
    birth_ts: int,
    window_secs: int,
    dead_gap_secs: int,
    limit: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch signatures and filter to [birth_ts, birth_ts + window_secs],
    sorted ascending. Stop early if a gap >= dead_gap_secs is observed.
    Returns (filtered_signatures, pages_scanned).
    """
    end_ts = birth_ts + window_secs
    before = None
    all_rows: List[Dict[str, Any]] = []
    pages = 0

    while True:
        rows = get_signatures_page(url, mint, before=before, limit=limit)
        pages += 1
        if not rows:
            break

        all_rows.extend(rows)
        before = rows[-1].get("signature")

        last_ts = rows[-1].get("blockTime")
        if last_ts is not None and int(last_ts) <= birth_ts:
            break

        if len(rows) < limit:
            break

        if pages > 500:
            raise RuntimeError("Too many pages while collecting window (safety stop).")

    # Filter to time window
    filtered = []
    for r in all_rows:
        bt = r.get("blockTime")
        if bt is None:
            continue
        bt = int(bt)
        if birth_ts <= bt <= end_ts:
            filtered.append(r)

    filtered.sort(key=lambda x: (int(x.get("blockTime", 0)), x.get("signature", "")))

    # Early stop on "dead gap"
    if filtered:
        kept = [filtered[0]]
        for r in filtered[1:]:
            prev_ts = int(kept[-1]["blockTime"])
            cur_ts = int(r["blockTime"])
            if (cur_ts - prev_ts) >= dead_gap_secs:
                # dead detected: no txs for dead_gap_secs
                break
            kept.append(r)
        filtered = kept

    return filtered, pages


def get_transaction(url: str, signature: str) -> Optional[Dict[str, Any]]:
    params = [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
    try:
        return rpc_call(url, "getTransaction", params)
    except Exception:
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def normalize_mint(inp: str) -> str:
    return inp.strip()


def is_exit_cmd(s: str) -> bool:
    s = s.strip().lower()
    return s in {"q", "quit", "exit"}


def prompt_minutes(default_minutes: int) -> int:
    raw = input(f"Minutes to scan? (default {default_minutes}): ").strip()
    if raw == "":
        return default_minutes
    if is_exit_cmd(raw):
        raise KeyboardInterrupt()
    try:
        minutes = int(raw)
        if minutes <= 0:
            print("Minutes must be > 0. Using default.")
            return default_minutes
        return minutes
    except ValueError:
        print("Invalid number. Using default.")
        return default_minutes


def scan_one_mint(
    url: str,
    mint: str,
    outdir: str,
    minutes: int,
    dead_gap_secs: int,
    page_limit: int,
) -> None:
    window_secs = minutes * 60

    print(f"\n[scan] mint={mint} minutes={minutes} dead_gap_secs={dead_gap_secs}")
    birth_ts, birth_sig, birth_pages = find_birth_ts_and_sig(url, mint, limit=page_limit)
    print(f"[birth] ts={birth_ts} sig={birth_sig} (birth_pages={birth_pages})")

    sig_rows, sig_pages = collect_signatures_in_window(
        url=url,
        mint=mint,
        birth_ts=birth_ts,
        window_secs=window_secs,
        dead_gap_secs=dead_gap_secs,
        limit=page_limit,
    )
    print(f"[window] sigs_kept={len(sig_rows)} (sig_pages={sig_pages})")

    events: List[Dict[str, Any]] = []
    total = len(sig_rows)
    for i, r in enumerate(sig_rows, start=1):
        sig = r.get("signature")
        bt = r.get("blockTime")
        slot = r.get("slot")
        err = r.get("err")

        tx = get_transaction(url, sig) if sig else None

        events.append(
            {
                "mint": mint,
                "sig": sig,
                "blockTime": bt,
                "slot": slot,
                "err": err,
                "signature_row": {
                    "confirmationStatus": r.get("confirmationStatus"),
                    "memo": r.get("memo"),
                },
                "tx": tx,  # may be None if fetch failed
            }
        )

        # Progress
        if total > 0:
            if i == 1 or i == total or (i % 25 == 0):
                pct = (i / total) * 100.0
                print(f"[progress] {i}/{total} ({pct:.1f}%) sig={sig}")
        time.sleep(TX_FETCH_SLEEP_SECS)

    ensure_dir(outdir)
    out_path = os.path.join(outdir, f"{mint}.jsonl")
    write_jsonl(out_path, events)
    print(f"[done] wrote_lines={len(events)} -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive Solana CA scanner (birth->N minutes or dead) to JSONL.")
    ap.add_argument("--outdir", required=True, help="Output directory for JSONL logs")
    ap.add_argument("--default-minutes", type=int, default=DEFAULT_MINUTES, help="Default scan duration in minutes")
    ap.add_argument("--dead-gap-secs", type=int, default=DEAD_GAP_SECS_DEFAULT, help="Stop early if no txs for this many seconds")
    ap.add_argument("--page-limit", type=int, default=SIG_PAGE_LIMIT_DEFAULT, help="Signatures page limit per RPC call")
    args = ap.parse_args()

    url = helius_rpc_url()
    ensure_dir(args.outdir)

    print("=== Solana CA Scanner ===")
    print("Type 'exit' / 'quit' / 'q' at CA prompt to stop.\n")

    try:
        while True:
            raw = input("Token CA (mint) to scan: ").strip()
            if raw == "":
                continue
            if is_exit_cmd(raw):
                print("[exit] bye.")
                return 0

            mint = normalize_mint(raw)
            minutes = prompt_minutes(int(args.default_minutes))

            try:
                scan_one_mint(
                    url=url,
                    mint=mint,
                    outdir=args.outdir,
                    minutes=minutes,
                    dead_gap_secs=int(args.dead_gap_secs),
                    page_limit=int(args.page_limit),
                )
            except requests.HTTPError as e:
                print(f"[error] HTTPError: {e}")
            except Exception as e:
                print(f"[error] {type(e).__name__}: {e}")

            print("\n[next] ready for next CA (or type exit)\n")

    except KeyboardInterrupt:
        print("\n[exit] interrupted. bye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
