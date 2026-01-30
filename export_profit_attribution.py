#!/usr/bin/env python3
"""
Export profit attribution TSVs using the same realized PnL logic as db_qa_shell.
"""

import argparse
import csv
import os
import sqlite3
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from db_qa_shell import build_time_clause, fetch_wallet_pnl_tokens, format_sol


def ensure_outdir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def fetch_wallets(conn: sqlite3.Connection) -> List[str]:
    sql = """
        SELECT DISTINCT scan_wallet
        FROM swaps
        WHERE has_sol_leg = 1
        ORDER BY scan_wallet ASC
    """
    cur = conn.execute(sql)
    return [row[0] for row in cur.fetchall() if row[0]]


def fetch_wallet_bounds(conn: sqlite3.Connection, wallet: str) -> Tuple[Optional[int], Optional[int]]:
    sql = """
        SELECT MIN(block_time) AS first_ts,
               MAX(block_time) AS last_ts
        FROM swaps
        WHERE scan_wallet = ?
          AND has_sol_leg = 1
    """
    row = conn.execute(sql, [wallet]).fetchone()
    return row[0], row[1]


def fetch_wallet_token_bounds(conn: sqlite3.Connection, wallet: str) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    sql = """
        SELECT token_mint,
               MIN(block_time) AS first_ts,
               MAX(block_time) AS last_ts
        FROM swaps
        WHERE scan_wallet = ?
          AND has_sol_leg = 1
          AND token_mint IS NOT NULL
        GROUP BY token_mint
    """
    cur = conn.execute(sql, [wallet])
    return {row[0]: (row[1], row[2]) for row in cur.fetchall() if row[0]}


def fetch_wallet_ambiguous(conn: sqlite3.Connection, wallet: str) -> int:
    sql = """
        SELECT COUNT(*)
        FROM swaps
        WHERE scan_wallet = ?
          AND has_sol_leg = 1
          AND token_mint IS NULL
    """
    row = conn.execute(sql, [wallet]).fetchone()
    return int(row[0] or 0)


def fetch_ecosystem_token_bounds(conn: sqlite3.Connection) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    sql = """
        SELECT token_mint,
               MIN(block_time) AS first_ts,
               MAX(block_time) AS last_ts
        FROM swaps
        WHERE has_sol_leg = 1
          AND token_mint IS NOT NULL
        GROUP BY token_mint
    """
    cur = conn.execute(sql)
    return {row[0]: (row[1], row[2]) for row in cur.fetchall() if row[0]}


def format_ts(value: Optional[int]) -> str:
    return "" if value is None else str(int(value))


def write_tsv(path: str, headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def summarize_wallet(
    wallet: str,
    token_rows: List[dict],
    ambiguous_total: int,
    first_ts: Optional[int],
    last_ts: Optional[int],
) -> List[str]:
    total_realized = sum((row["realized_pnl"] for row in token_rows), Decimal("0"))
    sorted_rows = sorted(token_rows, key=lambda r: (-r["realized_pnl"], r["token_mint"]))
    top1_token = sorted_rows[0]["token_mint"] if sorted_rows else ""
    top1_pnl = sorted_rows[0]["realized_pnl"] if sorted_rows else Decimal("0")
    top3_total = sum((row["realized_pnl"] for row in sorted_rows[:3]), Decimal("0"))

    if total_realized == 0:
        top1_share = Decimal("0")
        top3_share = Decimal("0")
    else:
        top1_share = top1_pnl / total_realized
        top3_share = top3_total / total_realized

    return [
        wallet,
        format_sol(total_realized),
        str(len(token_rows)),
        top1_token,
        format_sol(top1_pnl),
        format_sol(top1_share),
        format_sol(top3_share),
        str(ambiguous_total),
        format_ts(first_ts),
        format_ts(last_ts),
    ]


def build_export_rows(
    conn: sqlite3.Connection,
    wallets: Sequence[str],
) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:
    time_clause, time_params = build_time_clause(None, None)

    wallet_token_rows: List[dict] = []
    wallet_summary_rows: List[List[str]] = []
    qa_rows: List[List[str]] = []

    for wallet in wallets:
        pnl_rows = fetch_wallet_pnl_tokens(conn, wallet, time_clause, time_params)
        token_bounds = fetch_wallet_token_bounds(conn, wallet)
        wallet_first_ts, wallet_last_ts = fetch_wallet_bounds(conn, wallet)
        ambiguous_total = fetch_wallet_ambiguous(conn, wallet)

        for row in pnl_rows:
            token_mint = row["token_mint"]
            first_ts, last_ts = token_bounds.get(token_mint, (None, None))
            wallet_token_rows.append(
                {
                    "wallet_address": wallet,
                    "token_mint": token_mint,
                    "token_symbol": "",
                    "realized_pnl": row["realized_pnl"],
                    "trades": row["trade_count"],
                    "gross_in": row["sol_spent"],
                    "gross_out": row["sol_received"],
                    "turnover": row["sol_spent"] + row["sol_received"],
                    "ambiguous_txs": ambiguous_total,
                    "first_ts": first_ts,
                    "last_ts": last_ts,
                }
            )

        wallet_summary_rows.append(
            summarize_wallet(
                wallet,
                pnl_rows,
                ambiguous_total,
                wallet_first_ts,
                wallet_last_ts,
            )
        )

        if pnl_rows:
            top_row = max(pnl_rows, key=lambda r: (r["realized_pnl"], r["token_mint"]))
            qa_rows.append(
                [
                    wallet,
                    top_row["token_mint"],
                    format_sol(top_row["realized_pnl"]),
                    str(top_row["trade_count"]),
                    str(ambiguous_total),
                    "top_token",
                ]
            )
        else:
            qa_rows.append([wallet, "", "0", "0", str(ambiguous_total), "no_tokens"])

    wallet_token_rows.sort(
        key=lambda r: (r["wallet_address"], -r["realized_pnl"], r["token_mint"])
    )

    wallet_token_export = [
        [
            row["wallet_address"],
            row["token_mint"],
            row["token_symbol"],
            format_sol(row["realized_pnl"]),
            str(row["trades"]),
            format_sol(row["gross_in"]),
            format_sol(row["gross_out"]),
            format_sol(row["turnover"]),
            str(row["ambiguous_txs"]),
            format_ts(row["first_ts"]),
            format_ts(row["last_ts"]),
        ]
        for row in wallet_token_rows
    ]

    wallet_summary_rows.sort(key=lambda r: r[0])

    ecosystem_bounds = fetch_ecosystem_token_bounds(conn)
    ecosystem_map: Dict[str, dict] = {}
    seen_token_wallet = set()
    for row in wallet_token_rows:
        token_mint = row["token_mint"]
        wallet_address = row["wallet_address"]
        bucket = ecosystem_map.setdefault(
            token_mint,
            {
                "token_symbol": "",
                "wallets": set(),
                "realized_pnl": Decimal("0"),
                "trades": 0,
                "turnover": Decimal("0"),
                "ambiguous": 0,
            },
        )
        bucket["realized_pnl"] += row["realized_pnl"]
        bucket["trades"] += row["trades"]
        bucket["turnover"] += row["turnover"]
        token_wallet = (token_mint, wallet_address)
        if token_wallet not in seen_token_wallet:
            seen_token_wallet.add(token_wallet)
            bucket["wallets"].add(wallet_address)
            bucket["ambiguous"] += row["ambiguous_txs"]

    ecosystem_rows: List[List[str]] = []
    for token_mint, bucket in ecosystem_map.items():
        first_ts, last_ts = ecosystem_bounds.get(token_mint, (None, None))
        ecosystem_rows.append(
            [
                token_mint,
                bucket["token_symbol"],
                str(len(bucket["wallets"])),
                format_sol(bucket["realized_pnl"]),
                str(bucket["trades"]),
                format_sol(bucket["turnover"]),
                str(bucket["ambiguous"]),
                format_ts(first_ts),
                format_ts(last_ts),
            ]
        )

    ecosystem_rows.sort(key=lambda r: (-Decimal(r[3]), r[0]))
    qa_rows.sort(key=lambda r: (r[0], -Decimal(r[2]), r[1]))

    return wallet_token_export, wallet_summary_rows, ecosystem_rows, qa_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite DB file path.")
    ap.add_argument("--outdir", required=True, help="Output directory for TSVs.")
    ap.add_argument("--limit-wallets", type=int, help="Process only the first N wallets.")
    ap.add_argument("--wallet", help="Export just one wallet.")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"DB not found: {args.db}")

    ensure_outdir(args.outdir)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    wallets = fetch_wallets(conn)
    if args.wallet:
        wallets = [args.wallet]
    if args.limit_wallets is not None:
        wallets = wallets[: args.limit_wallets]

    wallet_token_rows, wallet_summary_rows, ecosystem_rows, qa_rows = build_export_rows(
        conn, wallets
    )

    write_tsv(
        os.path.join(args.outdir, "wallet_token_profit_rank.tsv"),
        [
            "wallet_address",
            "token_mint",
            "token_symbol",
            "realized_pnl_sol",
            "trades",
            "gross_in_sol",
            "gross_out_sol",
            "turnover_sol",
            "ambiguous_txs",
            "first_ts",
            "last_ts",
        ],
        wallet_token_rows,
    )
    write_tsv(
        os.path.join(args.outdir, "wallet_profit_summary.tsv"),
        [
            "wallet_address",
            "total_realized_pnl_sol",
            "tokens_traded",
            "top1_token_mint",
            "top1_pnl_sol",
            "top1_share",
            "top3_share",
            "ambiguous_txs_total",
            "first_ts",
            "last_ts",
        ],
        wallet_summary_rows,
    )
    write_tsv(
        os.path.join(args.outdir, "ecosystem_token_profit.tsv"),
        [
            "token_mint",
            "token_symbol",
            "wallets_traded",
            "total_realized_pnl_sol",
            "total_trades",
            "total_turnover_sol",
            "ambiguous_txs_total",
            "first_ts",
            "last_ts",
        ],
        ecosystem_rows,
    )
    write_tsv(
        os.path.join(args.outdir, "qa_profit_spotcheck.tsv"),
        [
            "wallet_address",
            "token_mint",
            "realized_pnl_sol",
            "trades",
            "ambiguous_txs",
            "note",
        ],
        qa_rows,
    )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
