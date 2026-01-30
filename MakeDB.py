#!/usr/bin/env python3
"""
ingest_kolscan_wallet_logs.py

Ingest wallet scan logs (*.jsonl / *.json) into a SQLite DB.

Designed for logs shaped like the sample you provided (fields like):
- scan_wallet
- blockTime, slot, transactionIndex, version
- transaction.signatures[0]
- pre_balance_SOL, post_balance_SOL, balance_delta_SOL
- spl_in_transfers / spl_out_transfers (optional)

Idempotent:
- Uses UNIQUE(scan_wallet, signature) so reruns won't duplicate rows.

Usage (Windows CMD example):
  python ingest_kolscan_wallet_logs.py --logs-dir "C:\path\to\logs" --db "C:\path\kolscan_wallets_7d.db"

Tip:
- Put the DB in the same folder as logs if you want it portable.
"""

import argparse
import glob
import hashlib
import json
import os
import sqlite3
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SOL_MINTS = {"SOL", "So11111111111111111111111111111111111111112"}


def iter_records(file_path: str) -> Iterable[Dict[str, Any]]:
    fp = file_path.lower()
    if fp.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    elif fp.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError:
                return
        if isinstance(obj, list):
            for x in obj:
                if isinstance(x, dict):
                    yield x
        elif isinstance(obj, dict):
            # common wrappers: {"data":[...]}
            data = obj.get("data")
            if isinstance(data, list):
                for x in data:
                    if isinstance(x, dict):
                        yield x
            else:
                yield obj


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Files table (optional state tracking / evidence)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS files (
      file_path TEXT PRIMARY KEY,
      file_name TEXT,
      file_size INTEGER,
      mtime INTEGER,
      sha256 TEXT,
      ingested_at INTEGER
    );
    """)

    # Wallets table (address + optional label derived from filename)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS wallets (
      wallet_address TEXT PRIMARY KEY,
      wallet_label TEXT
    );
    """)

    # Core tx table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tx (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      scan_wallet TEXT NOT NULL,
      signature TEXT NOT NULL,
      block_time INTEGER,
      slot INTEGER,
      tx_index INTEGER,
      version INTEGER,
      pre_balance_sol REAL,
      post_balance_sol REAL,
      balance_delta_sol REAL,
      spl_in_count INTEGER,
      spl_out_count INTEGER,
      err TEXT,
      source_file TEXT,
      raw_json TEXT NOT NULL,
      ingested_at INTEGER NOT NULL,
      UNIQUE(scan_wallet, signature)
    );
    """)

    # Optional SPL transfers table (if your logs contain per-tx transfer lists)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS spl_transfers (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      scan_wallet TEXT NOT NULL,
      signature TEXT NOT NULL,
      direction TEXT NOT NULL,           -- 'in' or 'out'
      mint TEXT,
      amount TEXT,
      from_addr TEXT,
      to_addr TEXT,
      raw_json TEXT NOT NULL,
      ingested_at INTEGER NOT NULL
    );
    """)

    # Swaps parsed deterministically from tx.raw_json (Helius enriched format)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS swaps (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      scan_wallet TEXT NOT NULL,
      signature TEXT NOT NULL,
      block_time INTEGER,
      dex TEXT,
      in_mint TEXT,
      in_amount_raw TEXT,
      out_mint TEXT,
      out_amount_raw TEXT,
      has_sol_leg INTEGER NOT NULL,
      sol_direction TEXT,
      sol_amount_lamports INTEGER,
      token_mint TEXT,
      token_amount_raw TEXT,
      UNIQUE(scan_wallet, signature, in_mint, out_mint, in_amount_raw, out_amount_raw)
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_wallet_time ON tx(scan_wallet, block_time);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_sig ON tx(signature);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_spl_sig ON spl_transfers(signature);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_swaps_wallet_time ON swaps(scan_wallet, block_time);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_swaps_sig ON swaps(signature);")

    conn.commit()


def is_sol_mint(mint: Optional[str]) -> bool:
    return isinstance(mint, str) and mint in SOL_MINTS


def coerce_amount_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val
    return None


def parse_int_amount(val: Any) -> Optional[int]:
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    if isinstance(val, str) and val.isdigit():
        return int(val)
    return None


def extract_amount_from_entry(entry: Any) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    if "amount" in entry:
        return coerce_amount_str(entry.get("amount"))
    token_amount = entry.get("tokenAmount")
    if isinstance(token_amount, dict):
        for key in ("amount", "uiAmountString", "uiAmount"):
            if key in token_amount:
                return coerce_amount_str(token_amount.get(key))
    elif token_amount is not None:
        return coerce_amount_str(token_amount)
    raw_token_amount = entry.get("rawTokenAmount")
    if isinstance(raw_token_amount, dict):
        for key in ("tokenAmount", "amount"):
            if key in raw_token_amount:
                return coerce_amount_str(raw_token_amount.get(key))
    elif raw_token_amount is not None:
        return coerce_amount_str(raw_token_amount)
    return None


def normalize_token_entries(entries: Any) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    if isinstance(entries, list):
        seq = entries
    elif isinstance(entries, dict):
        seq = [entries]
    else:
        seq = []
    for entry in seq:
        if not isinstance(entry, dict):
            continue
        mint = entry.get("mint") or entry.get("token") or entry.get("mintAddress")
        amount = extract_amount_from_entry(entry)
        if isinstance(mint, str) and mint and amount is not None:
            tokens.append((mint, amount))
    return tokens


def build_swap_row(
    scan_wallet: str,
    signature: str,
    block_time: Optional[int],
    dex: Optional[str],
    in_mint: Optional[str],
    in_amount_raw: Optional[str],
    out_mint: Optional[str],
    out_amount_raw: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not in_mint or not out_mint or in_amount_raw is None or out_amount_raw is None:
        return None

    in_amount_str = coerce_amount_str(in_amount_raw)
    out_amount_str = coerce_amount_str(out_amount_raw)
    if in_amount_str is None or out_amount_str is None:
        return None

    has_sol_leg = 1 if (is_sol_mint(in_mint) or is_sol_mint(out_mint)) else 0
    sol_direction = None
    sol_amount_lamports = None
    token_mint = None
    token_amount_raw = None

    if has_sol_leg:
        in_is_sol = is_sol_mint(in_mint)
        out_is_sol = is_sol_mint(out_mint)
        if in_is_sol and not out_is_sol:
            sol_direction = "buy"
            sol_amount_lamports = parse_int_amount(in_amount_str)
            token_mint = out_mint
            token_amount_raw = out_amount_str
        elif out_is_sol and not in_is_sol:
            sol_direction = "sell"
            sol_amount_lamports = parse_int_amount(out_amount_str)
            token_mint = in_mint
            token_amount_raw = in_amount_str
        else:
            return None

        if sol_amount_lamports is None:
            return None

    return {
        "scan_wallet": scan_wallet,
        "signature": signature,
        "block_time": block_time,
        "dex": dex,
        "in_mint": in_mint,
        "in_amount_raw": in_amount_str,
        "out_mint": out_mint,
        "out_amount_raw": out_amount_str,
        "has_sol_leg": has_sol_leg,
        "sol_direction": sol_direction,
        "sol_amount_lamports": sol_amount_lamports,
        "token_mint": token_mint,
        "token_amount_raw": token_amount_raw,
    }


def extract_swaps_from_event(
    rec: Dict[str, Any],
    scan_wallet: str,
    signature: str,
    block_time: Optional[int],
) -> List[Dict[str, Any]]:
    events = rec.get("events")
    if not isinstance(events, dict):
        return []
    swap = events.get("swap")
    if not isinstance(swap, dict):
        return []

    dex = swap.get("program") or swap.get("dex") or swap.get("source")

    token_inputs = normalize_token_entries(swap.get("tokenInputs") or swap.get("tokenIn"))
    token_outputs = normalize_token_entries(swap.get("tokenOutputs") or swap.get("tokenOut"))

    native_input = swap.get("nativeInput") if isinstance(swap.get("nativeInput"), dict) else {}
    native_output = swap.get("nativeOutput") if isinstance(swap.get("nativeOutput"), dict) else {}

    native_in_amount = coerce_amount_str(
        native_input.get("lamports") or native_input.get("amount") or native_input.get("nativeAmount")
    )
    native_out_amount = coerce_amount_str(
        native_output.get("lamports") or native_output.get("amount") or native_output.get("nativeAmount")
    )

    swaps: List[Dict[str, Any]] = []

    if native_in_amount and token_outputs and not token_inputs and len(token_outputs) == 1:
        out_mint, out_amount = token_outputs[0]
        row = build_swap_row(
            scan_wallet,
            signature,
            block_time,
            dex,
            "SOL",
            native_in_amount,
            out_mint,
            out_amount,
        )
        if row:
            swaps.append(row)
        return swaps

    if native_out_amount and token_inputs and not token_outputs and len(token_inputs) == 1:
        in_mint, in_amount = token_inputs[0]
        row = build_swap_row(
            scan_wallet,
            signature,
            block_time,
            dex,
            in_mint,
            in_amount,
            "SOL",
            native_out_amount,
        )
        if row:
            swaps.append(row)
        return swaps

    if not native_in_amount and not native_out_amount and len(token_inputs) == 1 and len(token_outputs) == 1:
        in_mint, in_amount = token_inputs[0]
        out_mint, out_amount = token_outputs[0]
        row = build_swap_row(
            scan_wallet,
            signature,
            block_time,
            dex,
            in_mint,
            in_amount,
            out_mint,
            out_amount,
        )
        if row:
            swaps.append(row)

    return swaps


def extract_swaps_from_transfers(
    rec: Dict[str, Any],
    scan_wallet: str,
    signature: str,
    block_time: Optional[int],
) -> List[Dict[str, Any]]:
    token_transfers = rec.get("tokenTransfers")
    native_transfers = rec.get("nativeTransfers")
    if not isinstance(token_transfers, list) and not isinstance(native_transfers, list):
        return []

    incoming: Dict[str, int] = {}
    outgoing: Dict[str, int] = {}

    def add_amount(bucket: Dict[str, int], mint: str, amount: int) -> None:
        bucket[mint] = bucket.get(mint, 0) + amount

    if isinstance(token_transfers, list):
        for t in token_transfers:
            if not isinstance(t, dict):
                continue
            mint = t.get("mint") or t.get("token") or t.get("mintAddress")
            if not isinstance(mint, str) or not mint:
                continue
            amount_raw = extract_amount_from_entry(t)
            amount_int = parse_int_amount(amount_raw)
            if amount_int is None:
                continue
            from_addr = t.get("fromUserAccount") or t.get("from") or t.get("source")
            to_addr = t.get("toUserAccount") or t.get("to") or t.get("destination")
            if from_addr == scan_wallet:
                add_amount(outgoing, mint, amount_int)
            elif to_addr == scan_wallet:
                add_amount(incoming, mint, amount_int)

    if isinstance(native_transfers, list):
        for t in native_transfers:
            if not isinstance(t, dict):
                continue
            amount_raw = t.get("lamports") or t.get("amount") or t.get("nativeAmount")
            amount_int = parse_int_amount(amount_raw)
            if amount_int is None:
                continue
            from_addr = t.get("fromUserAccount") or t.get("from") or t.get("source")
            to_addr = t.get("toUserAccount") or t.get("to") or t.get("destination")
            if from_addr == scan_wallet:
                add_amount(outgoing, "SOL", amount_int)
            elif to_addr == scan_wallet:
                add_amount(incoming, "SOL", amount_int)

    if len(incoming) != 1 or len(outgoing) != 1:
        return []

    in_mint, in_amount = next(iter(incoming.items()))
    out_mint, out_amount = next(iter(outgoing.items()))
    if in_mint == out_mint:
        return []

    row = build_swap_row(
        scan_wallet,
        signature,
        block_time,
        None,
        out_mint,
        str(out_amount),
        in_mint,
        str(in_amount),
    )
    return [row] if row else []


def extract_swaps_from_tx(
    rec: Dict[str, Any],
    scan_wallet: str,
    signature: str,
    block_time: Optional[int],
) -> List[Dict[str, Any]]:
    swaps = extract_swaps_from_event(rec, scan_wallet, signature, block_time)
    if swaps:
        return swaps
    return extract_swaps_from_transfers(rec, scan_wallet, signature, block_time)


def parse_decimal_amount(val: Any) -> Optional[Decimal]:
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val
    if isinstance(val, int):
        return Decimal(val)
    if isinstance(val, float):
        return Decimal(str(val))
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return None
        try:
            return Decimal(stripped)
        except InvalidOperation:
            return None
    return None


def extract_transfer_amount_raw(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("rawAmount", "raw_amount", "amount", "amount_raw"):
        if key in entry:
            return coerce_amount_str(entry.get(key))
    token_amount = entry.get("tokenAmount")
    if isinstance(token_amount, dict):
        for key in ("amount", "uiAmountString", "uiAmount"):
            if key in token_amount:
                return coerce_amount_str(token_amount.get(key))
    elif token_amount is not None:
        return coerce_amount_str(token_amount)
    for key in ("uiAmountString", "uiAmount"):
        if key in entry:
            return coerce_amount_str(entry.get(key))
    return None


def backfill_swaps_from_tx_raw_json(
    conn: sqlite3.Connection,
    *,
    batch_size: int = 5000,
    limit: Optional[int] = None,
) -> int:
    inserted = 0
    remaining = limit
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, scan_wallet, signature, block_time, raw_json, pre_balance_sol, post_balance_sol
        FROM tx
        WHERE raw_json IS NOT NULL AND raw_json != ''
        ORDER BY id
        """
    )
    while True:
        if remaining is not None and remaining <= 0:
            break
        fetch_size = batch_size
        if remaining is not None:
            fetch_size = min(fetch_size, remaining)
        rows = cur.fetchmany(fetch_size)
        if not rows:
            break
        for _, scan_wallet, signature, block_time, raw_json, pre_bal, post_bal in rows:
            try:
                rec = json.loads(raw_json)
            except json.JSONDecodeError:
                continue

            sol_delta_val = rec.get("balance_delta_SOL")
            if sol_delta_val is None and pre_bal is not None and post_bal is not None:
                sol_delta_val = Decimal(str(post_bal)) - Decimal(str(pre_bal))
            sol_delta = parse_decimal_amount(sol_delta_val)
            if sol_delta is None or sol_delta == 0:
                continue

            sol_amount_lamports = int(round(abs(sol_delta) * Decimal("1000000000")))
            sol_direction = "buy" if sol_delta < 0 else "sell"

            spl_in = rec.get("spl_in_transfers")
            spl_out = rec.get("spl_out_transfers")
            candidates: Dict[str, Tuple[Decimal, str]] = {}
            for transfers in (spl_in, spl_out):
                if not isinstance(transfers, list):
                    continue
                for entry in transfers:
                    if not isinstance(entry, dict):
                        continue
                    mint = entry.get("mint") or entry.get("token") or entry.get("mintAddress")
                    if not isinstance(mint, str) or not mint or is_sol_mint(mint):
                        continue
                    amount_raw = extract_transfer_amount_raw(entry)
                    amount_dec = parse_decimal_amount(amount_raw)
                    if amount_dec is None:
                        continue
                    abs_amount = abs(amount_dec)
                    if mint not in candidates or abs_amount > candidates[mint][0]:
                        candidates[mint] = (abs_amount, amount_raw)

            if not candidates:
                continue

            if len(candidates) == 1:
                token_mint, (_, token_amount_raw) = next(iter(candidates.items()))
            else:
                token_mint, (_, token_amount_raw) = max(
                    candidates.items(), key=lambda item: item[1][0]
                )

            if token_amount_raw is None:
                continue

            conn.execute(
                "DELETE FROM swaps WHERE scan_wallet = ? AND signature = ?;",
                (scan_wallet, signature),
            )
            before = conn.total_changes
            conn.execute(
                """
                INSERT OR IGNORE INTO swaps(
                  scan_wallet, signature, block_time, dex,
                  in_mint, in_amount_raw, out_mint, out_amount_raw,
                  has_sol_leg, sol_direction, sol_amount_lamports,
                  token_mint, token_amount_raw
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    scan_wallet,
                    signature,
                    block_time,
                    None,
                    None,
                    None,
                    None,
                    None,
                    1,
                    sol_direction,
                    sol_amount_lamports,
                    token_mint,
                    token_amount_raw,
                ),
            )
            if conn.total_changes > before:
                inserted += 1
        if remaining is not None:
            remaining -= len(rows)
    return inserted


def insert_swaps(conn: sqlite3.Connection, swaps: List[Dict[str, Any]]) -> int:
    inserted = 0
    for swap in swaps:
        before = conn.total_changes
        conn.execute(
            """
            INSERT OR IGNORE INTO swaps(
              scan_wallet, signature, block_time, dex,
              in_mint, in_amount_raw, out_mint, out_amount_raw,
              has_sol_leg, sol_direction, sol_amount_lamports,
              token_mint, token_amount_raw
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                swap["scan_wallet"],
                swap["signature"],
                swap["block_time"],
                swap["dex"],
                swap["in_mint"],
                swap["in_amount_raw"],
                swap["out_mint"],
                swap["out_amount_raw"],
                swap["has_sol_leg"],
                swap["sol_direction"],
                swap["sol_amount_lamports"],
                swap["token_mint"],
                swap["token_amount_raw"],
            ),
        )
        if conn.total_changes > before:
            inserted += 1
    return inserted


def backfill_swaps(conn: sqlite3.Connection) -> int:
    """
    Parse swaps from existing tx rows. Idempotent: uses INSERT OR IGNORE.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tx.scan_wallet, tx.signature, tx.block_time, tx.raw_json
        FROM tx
        LEFT JOIN swaps
          ON swaps.scan_wallet = tx.scan_wallet
         AND swaps.signature = tx.signature
        WHERE swaps.signature IS NULL
        """
    )
    inserted = 0
    for scan_wallet, signature, block_time, raw_json in cur.fetchall():
        try:
            rec = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        swaps = extract_swaps_from_tx(rec, scan_wallet, signature, block_time)
        inserted += insert_swaps(conn, swaps)
    return inserted


def safe_get_signature(rec: Dict[str, Any]) -> Optional[str]:
    tx = rec.get("transaction") or {}
    sigs = tx.get("signatures")
    if isinstance(sigs, list) and sigs:
        s = sigs[0]
        return s if isinstance(s, str) and s else None
    # fallback: some logs might use "signature" top-level
    s = rec.get("signature")
    return s if isinstance(s, str) and s else None


def safe_get_err(rec: Dict[str, Any]) -> Optional[str]:
    meta = rec.get("meta")
    if isinstance(meta, dict):
        err = meta.get("err")
        if err is None:
            return None
        # err can be dict/list/string; keep compact JSON
        try:
            return json.dumps(err, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            return str(err)
    return None


def normalize_transfer_obj(x: Any) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Attempt to extract common fields, but store full raw regardless.
    Since transfer shapes vary, we keep this permissive.
    """
    mint = amount = from_addr = to_addr = None
    if isinstance(x, dict):
        mint = x.get("mint") or x.get("token") or x.get("mintAddress")
        amount = x.get("amount") or x.get("uiAmountString") or x.get("uiAmount")
        from_addr = x.get("from") or x.get("fromUserAccount") or x.get("source")
        to_addr = x.get("to") or x.get("toUserAccount") or x.get("destination")
        # force to string where useful
        if amount is not None and not isinstance(amount, str):
            amount = str(amount)
    return (
        mint if isinstance(mint, str) else None,
        amount if isinstance(amount, str) else None,
        from_addr if isinstance(from_addr, str) else None,
        to_addr if isinstance(to_addr, str) else None,
    )


def upsert_wallet(conn: sqlite3.Connection, wallet_address: str, wallet_label: Optional[str]) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO wallets(wallet_address, wallet_label) VALUES(?, ?);",
        (wallet_address, wallet_label or None),
    )
    # if exists but label is NULL, fill it
    if wallet_label:
        conn.execute(
            "UPDATE wallets SET wallet_label = COALESCE(wallet_label, ?) WHERE wallet_address = ?;",
            (wallet_label, wallet_address),
        )


def ingest_file(
    conn: sqlite3.Connection,
    file_path: str,
    wallet_label: str,
    skip_if_unchanged: bool,
) -> Tuple[int, int, int]:
    """
    Returns (inserted_tx_rows, inserted_spl_rows, inserted_swap_rows)
    """
    p = Path(file_path)
    stat = p.stat()
    mtime = int(stat.st_mtime)
    fsize = int(stat.st_size)
    sha = sha256_file(file_path)

    cur = conn.cursor()
    cur.execute("SELECT sha256 FROM files WHERE file_path = ?;", (file_path,))
    row = cur.fetchone()
    if skip_if_unchanged and row and row[0] == sha:
        return (0, 0, 0)

    now = int(time.time())

    # record file state
    conn.execute(
        """
        INSERT INTO files(file_path, file_name, file_size, mtime, sha256, ingested_at)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(file_path) DO UPDATE SET
          file_size=excluded.file_size,
          mtime=excluded.mtime,
          sha256=excluded.sha256,
          ingested_at=excluded.ingested_at;
        """,
        (file_path, p.name, fsize, mtime, sha, now),
    )

    inserted_tx = 0
    inserted_spl = 0
    inserted_swaps = 0

    for rec in iter_records(file_path):
        scan_wallet = rec.get("scan_wallet")
        if not isinstance(scan_wallet, str) or not scan_wallet:
            # If your logs *always* have scan_wallet, great.
            # If not, you can fall back to filename label, but wallet address is better.
            continue

        upsert_wallet(conn, scan_wallet, wallet_label)

        sig = safe_get_signature(rec)
        if not sig:
            continue

        block_time = rec.get("blockTime")
        slot = rec.get("slot")
        tx_index = rec.get("transactionIndex")
        version = rec.get("version")

        pre_bal = rec.get("pre_balance_SOL")
        post_bal = rec.get("post_balance_SOL")
        delta_bal = rec.get("balance_delta_SOL")
        spl_in_count = rec.get("spl_in_count")
        spl_out_count = rec.get("spl_out_count")

        err = safe_get_err(rec)

        raw = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))

        # Insert tx row (ignore duplicates)
        before_changes = conn.total_changes
        conn.execute(
            """
            INSERT OR IGNORE INTO tx(
              scan_wallet, signature, block_time, slot, tx_index, version,
              pre_balance_sol, post_balance_sol, balance_delta_sol,
              spl_in_count, spl_out_count, err, source_file, raw_json, ingested_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                scan_wallet, sig,
                int(block_time) if isinstance(block_time, (int, float)) else None,
                int(slot) if isinstance(slot, (int, float)) else None,
                int(tx_index) if isinstance(tx_index, (int, float)) else None,
                int(version) if isinstance(version, (int, float)) else None,
                float(pre_bal) if isinstance(pre_bal, (int, float)) else None,
                float(post_bal) if isinstance(post_bal, (int, float)) else None,
                float(delta_bal) if isinstance(delta_bal, (int, float)) else None,
                int(spl_in_count) if isinstance(spl_in_count, (int, float)) else None,
                int(spl_out_count) if isinstance(spl_out_count, (int, float)) else None,
                err,
                file_path,
                raw,
                now,
            ),
        )
        if conn.total_changes > before_changes:
            inserted_tx += 1

            # Only ingest SPL transfers for rows we actually inserted (prevents doubling)
            for direction, k in (("in", "spl_in_transfers"), ("out", "spl_out_transfers")):
                arr = rec.get(k)
                if not isinstance(arr, list) or not arr:
                    continue
                for x in arr:
                    mint, amount, from_addr, to_addr = normalize_transfer_obj(x)
                    conn.execute(
                        """
                        INSERT INTO spl_transfers(
                          scan_wallet, signature, direction, mint, amount, from_addr, to_addr, raw_json, ingested_at
                        ) VALUES (?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            scan_wallet, sig, direction,
                            mint, amount, from_addr, to_addr,
                            json.dumps(x, ensure_ascii=False, separators=(",", ":")),
                            now,
                        ),
                    )
                    inserted_spl += 1

            swaps = extract_swaps_from_tx(
                rec,
                scan_wallet,
                sig,
                int(block_time) if isinstance(block_time, (int, float)) else None,
            )
            inserted_swaps += insert_swaps(conn, swaps)

    return (inserted_tx, inserted_spl, inserted_swaps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", required=True, help="Folder containing wallet scan logs (*.jsonl/*.json).")
    ap.add_argument("--db", required=True, help="SQLite DB file path to create/populate.")
    ap.add_argument("--glob", default="*.jsonl", help="Glob pattern (default: *.jsonl). Use *.jsonl;*.json if needed.")
    ap.add_argument("--skip-unchanged", action="store_true", help="Skip files whose sha256 hash is unchanged since last ingest.")
    ap.add_argument("--commit-every", type=int, default=2000, help="Commit every N inserted tx rows (default 2000).")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists() or not logs_dir.is_dir():
        print(f"ERROR: logs-dir not found or not a folder: {logs_dir}")
        return 2

    # support multiple globs separated by ;
    patterns = [x.strip() for x in args.glob.split(";") if x.strip()]
    files = []
    for pat in patterns:
        files.extend(glob.glob(str(logs_dir / pat)))
    files = sorted(set(files))

    if not files:
        print(f"ERROR: no files matched patterns {patterns} in {logs_dir}")
        return 2

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    ensure_schema(conn)

    total_tx = 0
    total_spl = 0
    total_swaps = 0
    committed_tx = 0

    start = time.time()
    for i, fp in enumerate(files, 1):
        label = Path(fp).name.split(".", 1)[0]  # e.g. "woopig" from "woopig.last168h.jsonl"
        try:
            ins_tx, ins_spl, ins_swaps = ingest_file(conn, fp, label, skip_if_unchanged=args.skip_unchanged)
            total_tx += ins_tx
            total_spl += ins_spl
            total_swaps += ins_swaps
            committed_tx += ins_tx

            # periodic commit
            if committed_tx >= args.commit_every:
                conn.commit()
                committed_tx = 0

            print(f"[{i}/{len(files)}] {Path(fp).name}  +tx={ins_tx} +spl={ins_spl} +swaps={ins_swaps}")
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception as e:
            print(f"[WARN] failed file={fp} err={e}")

    conn.commit()
    backfilled_from_raw = backfill_swaps_from_tx_raw_json(conn)
    if backfilled_from_raw:
        conn.commit()
        total_swaps += backfilled_from_raw
    backfilled_swaps = backfill_swaps(conn)
    if backfilled_swaps:
        conn.commit()
        total_swaps += backfilled_swaps
    elapsed = time.time() - start

    # quick stats
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM tx;")
    tx_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM wallets;")
    wallet_count = cur.fetchone()[0]
    cur.execute("SELECT MIN(block_time), MAX(block_time) FROM tx;")
    mnmx = cur.fetchone()
    conn.close()

    print("\nDONE")
    print(f"DB: {db_path.resolve()}")
    print(f"Wallets: {wallet_count}")
    print(f"Tx rows: {tx_count} (this run inserted {total_tx})")
    print(f"SPL rows: {total_spl}")
    print(
        "Swaps rows: "
        f"{total_swaps} (includes raw backfill {backfilled_from_raw}, "
        f"events backfill {backfilled_swaps})"
    )
    if mnmx and mnmx[0] and mnmx[1]:
        print(f"Time window (unix): {mnmx[0]} -> {mnmx[1]}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
