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
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


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

    cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_wallet_time ON tx(scan_wallet, block_time);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_sig ON tx(signature);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_spl_sig ON spl_transfers(signature);")

    conn.commit()


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


def ingest_file(conn: sqlite3.Connection, file_path: str, wallet_label: str, skip_if_unchanged: bool) -> Tuple[int, int]:
    """
    Returns (inserted_tx_rows, inserted_spl_rows)
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
        return (0, 0)

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

    return (inserted_tx, inserted_spl)


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
    committed_tx = 0

    start = time.time()
    for i, fp in enumerate(files, 1):
        label = Path(fp).name.split(".", 1)[0]  # e.g. "woopig" from "woopig.last168h.jsonl"
        try:
            ins_tx, ins_spl = ingest_file(conn, fp, label, skip_if_unchanged=args.skip_unchanged)
            total_tx += ins_tx
            total_spl += ins_spl
            committed_tx += ins_tx

            # periodic commit
            if committed_tx >= args.commit_every:
                conn.commit()
                committed_tx = 0

            print(f"[{i}/{len(files)}] {Path(fp).name}  +tx={ins_tx} +spl={ins_spl}")
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception as e:
            print(f"[WARN] failed file={fp} err={e}")

    conn.commit()
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
    if mnmx and mnmx[0] and mnmx[1]:
        print(f"Time window (unix): {mnmx[0]} -> {mnmx[1]}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
