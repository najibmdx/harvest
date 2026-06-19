#!/usr/bin/env python3
"""
Solana Memecoin CA Scanner (Helius RPC) - Fast
- Prompts for token CA (mint)
- Prompts for scan minutes (default 60)
- Scans from token birth for N minutes OR stops early if "dead"
- Writes JSONL logs: <outdir>/<MINT>.jsonl
- Repeats until user types exit/quit/q
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

# -------------------------
# Deterministic defaults
# -------------------------
DEFAULT_MINUTES = 60
DEAD_GAP_SECS_DEFAULT = 5 * 60         # "dead" = no txs for 5 minutes
SIG_PAGE_LIMIT_DEFAULT = 1000          # max per Solana RPC page
DEFAULT_WORKERS = 12

# Retry/backoff settings
MAX_ATTEMPTS = 6
BASE_BACKOFF_SECS = 0.4
MAX_BACKOFF_SECS = 8.0
TX_TIMEOUT_SECS = 40


def helius_rpc_url() -> str:
    key = os.environ.get("HELIUS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing HELIUS_API_KEY environment variable.")
    return f"https://mainnet.helius-rpc.com/?api-key={key}"


def rpc_call(url: str, method: str, params: list, timeout: int = TX_TIMEOUT_SECS) -> Any:
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


def _sleep_with_jitter(secs: float) -> None:
    jitter = (time.time() % 1.0) * 0.1
    time.sleep(max(0.0, secs + jitter))


def get_transaction_with_retry(url: str, signature: str) -> Tuple[Optional[Dict[str, Any]], int, int]:
    params = [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getTransaction", "params": params}

    retry_429 = 0
    retry_503 = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(url, json=payload, timeout=TX_TIMEOUT_SECS)
            if resp.status_code in (429, 503):
                if resp.status_code == 429:
                    retry_429 += 1
                else:
                    retry_503 += 1
                if attempt >= MAX_ATTEMPTS:
                    return None, retry_429, retry_503
                backoff = min(MAX_BACKOFF_SECS, BASE_BACKOFF_SECS * (2 ** (attempt - 1)))
                _sleep_with_jitter(backoff)
                continue
            if 400 <= resp.status_code < 600:
                return None, retry_429, retry_503
            data = resp.json()
            if "error" in data:
                return None, retry_429, retry_503
            return data.get("result"), retry_429, retry_503
        except (requests.Timeout, requests.ConnectionError, requests.RequestException):
            if attempt >= MAX_ATTEMPTS:
                return None, retry_429, retry_503
            backoff = min(MAX_BACKOFF_SECS, BASE_BACKOFF_SECS * (2 ** (attempt - 1)))
            _sleep_with_jitter(backoff)

    return None, retry_429, retry_503


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


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_progress(
    completed: int,
    total: int,
    ok: int,
    null: int,
    r429: int,
    r503: int,
    start_ts: float,
) -> None:
    elapsed = time.time() - start_ts
    pct = (completed / total * 100.0) if total else 100.0
    rate = (completed / elapsed) if elapsed > 0 else 0.0
    eta = ((total - completed) / rate) if rate > 0 else 0.0
    line = (
        f"[tx] {completed}/{total} ({pct:.1f}%) "
        f"ok={ok} null={null} r429={r429} r503={r503} "
        f"elapsed={_format_duration(elapsed)} rate={rate:.1f} tx/s ETA={_format_duration(eta)}"
    )
    print(f"\r{line}", end="", flush=True)


def scan_one_mint(
    url: str,
    mint: str,
    outdir: str,
    minutes: int,
    dead_gap_secs: int,
    page_limit: int,
    workers: int,
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
    for r in sig_rows:
        events.append(
            {
                "mint": mint,
                "sig": r.get("signature"),
                "blockTime": r.get("blockTime"),
                "slot": r.get("slot"),
                "err": r.get("err"),
                "signature_row": {
                    "confirmationStatus": r.get("confirmationStatus"),
                    "memo": r.get("memo"),
                },
                "tx": None,
            }
        )

    total = len(sig_rows)
    completed = 0
    ok = 0
    null = 0
    r429 = 0
    r503 = 0
    last_update_completed = 0
    last_update_pct = 0.0
    start_ts = time.time()
    futures = {}
    interrupted = False

    def maybe_update_progress() -> None:
        nonlocal last_update_completed, last_update_pct
        if total == 0:
            return
        pct = (completed / total) * 100.0
        if completed == total or (completed - last_update_completed) >= 50 or (pct - last_update_pct) >= 1.0:
            _print_progress(completed, total, ok, null, r429, r503, start_ts)
            last_update_completed = completed
            last_update_pct = pct

    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            for idx, r in enumerate(sig_rows):
                sig = r.get("signature")
                if not sig:
                    continue
                future = executor.submit(get_transaction_with_retry, url, sig)
                futures[future] = idx
        except KeyboardInterrupt:
            interrupted = True

        try:
            for future in as_completed(futures):
                try:
                    tx, c429, c503 = future.result()
                except Exception:
                    tx, c429, c503 = None, 0, 0
                idx = futures[future]
                events[idx]["tx"] = tx
                completed += 1
                if tx is None:
                    null += 1
                else:
                    ok += 1
                r429 += c429
                r503 += c503
                maybe_update_progress()
        except KeyboardInterrupt:
            interrupted = True

        if interrupted:
            for future in futures:
                if not future.done():
                    future.cancel()

    if total > 0:
        _print_progress(completed, total, ok, null, r429, r503, start_ts)
        print()

    ensure_dir(outdir)
    out_path = os.path.join(outdir, f"{mint}.jsonl")
    write_jsonl(out_path, events)

    if interrupted:
        print(f"[exit] interrupted; wrote partial log -> {out_path}")
    else:
        print(f"[done] wrote_lines={len(events)} -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive Solana CA scanner (birth->N minutes or dead) to JSONL.")
    ap.add_argument("--outdir", required=True, help="Output directory for JSONL logs")
    ap.add_argument("--default-minutes", type=int, default=DEFAULT_MINUTES, help="Default scan duration in minutes")
    ap.add_argument("--dead-gap-secs", type=int, default=DEAD_GAP_SECS_DEFAULT, help="Stop early if no txs for this many seconds")
    ap.add_argument("--page-limit", type=int, default=SIG_PAGE_LIMIT_DEFAULT, help="Signatures page limit per RPC call")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel getTransaction workers")
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
                    workers=int(args.workers),
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
