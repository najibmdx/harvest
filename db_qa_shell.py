#!/usr/bin/env python3
"""
SQLite QA shell for Solana wallet scans.
"""

import argparse
import os
import sqlite3
import sys
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

SOL_LAMPORTS = Decimal("1000000000")


def normalize_alias(command: str) -> str:
    normalized = " ".join(command.strip().lower().split())
    normalized = normalized.replace("?", "")
    if normalized == "who made the most sol":
        return "top wallets 10"
    if normalized == "who made the most pnl":
        return "top pnl wallets 10"
    return command


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def build_time_clause(start_ts: Optional[int], end_ts: Optional[int]) -> Tuple[str, List[int]]:
    clauses = []
    params: List[int] = []
    if start_ts is not None:
        clauses.append("block_time >= ?")
        params.append(start_ts)
    if end_ts is not None:
        clauses.append("block_time <= ?")
        params.append(end_ts)
    if clauses:
        return " AND " + " AND ".join(clauses), params
    return "", params


def lamports_to_sol(lamports: int) -> Decimal:
    return Decimal(lamports) / SOL_LAMPORTS


def format_sol(value: Decimal) -> str:
    formatted = f"{value:.9f}"
    return formatted.rstrip("0").rstrip(".") if formatted else "0"


def handle_wallet_pnl_tokens(
    conn: sqlite3.Connection,
    wallet: str,
    limit: int,
    time_clause: str,
    time_params: Sequence[int],
) -> None:
    sql = f"""
        SELECT token_mint,
               SUM(CASE WHEN sol_direction = 'buy' THEN sol_amount_lamports ELSE 0 END) AS sol_spent_lamports,
               SUM(CASE WHEN sol_direction = 'sell' THEN sol_amount_lamports ELSE 0 END) AS sol_received_lamports,
               COUNT(*) AS trade_count
        FROM swaps
        WHERE scan_wallet = ?
          AND has_sol_leg = 1
          AND token_mint IS NOT NULL
          {time_clause}
        GROUP BY token_mint
    """
    cur = conn.execute(sql, [wallet, *time_params])
    rows = []
    for token_mint, sol_spent_lamports, sol_received_lamports, trade_count in cur.fetchall():
        sol_spent = lamports_to_sol(int(sol_spent_lamports or 0))
        sol_received = lamports_to_sol(int(sol_received_lamports or 0))
        realized_pnl = sol_received - sol_spent
        rows.append(
            {
                "token_mint": token_mint,
                "trade_count": int(trade_count or 0),
                "sol_spent": sol_spent,
                "sol_received": sol_received,
                "realized_pnl": realized_pnl,
            }
        )

    rows.sort(
        key=lambda r: (r["realized_pnl"], r["sol_received"]),
        reverse=True,
    )
    if limit:
        rows = rows[:limit]

    table_rows = [
        [
            r["token_mint"],
            str(r["trade_count"]),
            format_sol(r["sol_spent"]),
            format_sol(r["sol_received"]),
            format_sol(r["realized_pnl"]),
        ]
        for r in rows
    ]
    print_table(
        ["token_mint", "trade_count", "sol_spent", "sol_received", "realized_pnl_sol"],
        table_rows,
    )


def handle_top_pnl_wallets(
    conn: sqlite3.Connection,
    limit: int,
    time_clause: str,
    time_params: Sequence[int],
) -> None:
    sql = f"""
        SELECT s.scan_wallet,
               COALESCE(w.wallet_label, '') AS wallet_label,
               SUM(CASE WHEN s.sol_direction = 'buy' THEN s.sol_amount_lamports ELSE 0 END) AS sol_spent_lamports,
               SUM(CASE WHEN s.sol_direction = 'sell' THEN s.sol_amount_lamports ELSE 0 END) AS sol_received_lamports,
               COUNT(*) AS trade_count
        FROM swaps s
        LEFT JOIN wallets w ON w.wallet_address = s.scan_wallet
        WHERE s.has_sol_leg = 1
          {time_clause}
        GROUP BY s.scan_wallet
    """
    cur = conn.execute(sql, time_params)
    rows = []
    for wallet, label, sol_spent_lamports, sol_received_lamports, trade_count in cur.fetchall():
        sol_spent = lamports_to_sol(int(sol_spent_lamports or 0))
        sol_received = lamports_to_sol(int(sol_received_lamports or 0))
        realized_pnl = sol_received - sol_spent
        rows.append(
            {
                "wallet": wallet,
                "label": label,
                "sol_spent": sol_spent,
                "sol_received": sol_received,
                "trade_count": int(trade_count or 0),
                "realized_pnl": realized_pnl,
            }
        )

    rows.sort(
        key=lambda r: (r["realized_pnl"], r["sol_received"]),
        reverse=True,
    )
    if limit:
        rows = rows[:limit]

    table_rows = [
        [
            r["wallet"],
            r["label"],
            format_sol(r["realized_pnl"]),
            format_sol(r["sol_spent"]),
            format_sol(r["sol_received"]),
            str(r["trade_count"]),
        ]
        for r in rows
    ]
    print_table(
        ["wallet", "label", "realized_pnl_sol", "sol_spent", "sol_received", "trade_count"],
        table_rows,
    )


def handle_top_wallets(
    conn: sqlite3.Connection,
    limit: int,
    time_clause: str,
    time_params: Sequence[int],
) -> None:
    sql = f"""
        SELECT t.scan_wallet,
               COALESCE(w.wallet_label, '') AS wallet_label,
               SUM(COALESCE(t.balance_delta_sol, 0)) AS net_sol,
               COUNT(*) AS tx_count
        FROM tx t
        LEFT JOIN wallets w ON w.wallet_address = t.scan_wallet
        WHERE 1=1
          {time_clause}
        GROUP BY t.scan_wallet
        ORDER BY net_sol DESC
        LIMIT ?
    """
    cur = conn.execute(sql, [*time_params, limit])
    table_rows = [
        [
            wallet,
            label,
            str(net_sol if net_sol is not None else 0),
            str(tx_count),
        ]
        for wallet, label, net_sol, tx_count in cur.fetchall()
    ]
    print_table(["wallet", "label", "net_sol", "tx_count"], table_rows)


def print_help() -> None:
    print(
        """
Commands:
  wallet <WALLET> pnl tokens [N]    Realized PnL per token (SOL leg only).
  top pnl wallets [N]               Top wallets by realized PnL (SOL leg only).
  top wallets [N]                   Top wallets by net SOL balance delta.
  cls                               Clear screen (Windows compatible).
  help                              Show this help.
  quit / exit / q                   Exit.

Aliases:
  who made the most sol  -> top wallets 10
  who made the most pnl  -> top pnl wallets 10
""".strip()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite DB file path.")
    ap.add_argument("--start", type=int, help="Filter: block_time >= start (unix seconds).")
    ap.add_argument("--end", type=int, help="Filter: block_time <= end (unix seconds).")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}")
        return 2

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    time_clause, time_params = build_time_clause(args.start, args.end)

    print("DB QA Shell. Type 'help' for commands.")
    while True:
        try:
            raw = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw.strip():
            continue

        raw = normalize_alias(raw)
        parts = raw.strip().split()
        if not parts:
            continue

        cmd = parts[0].lower()
        if cmd in {"quit", "exit", "q"}:
            break
        if cmd == "help":
            print_help()
            continue
        if cmd == "cls":
            if os.name == "nt":
                os.system("cls")
            else:
                print("\n" * 120)
            continue

        if cmd == "wallet" and len(parts) >= 4 and parts[2].lower() == "pnl" and parts[3].lower() == "tokens":
            wallet = parts[1]
            limit = int(parts[4]) if len(parts) >= 5 else 20
            handle_wallet_pnl_tokens(conn, wallet, limit, time_clause, time_params)
            continue

        if cmd == "top" and len(parts) >= 2 and parts[1].lower() == "pnl":
            if len(parts) >= 3 and parts[2].lower() == "wallets":
                limit = int(parts[3]) if len(parts) >= 4 else 20
                handle_top_pnl_wallets(conn, limit, time_clause, time_params)
                continue

        if cmd == "top" and len(parts) >= 2 and parts[1].lower() == "wallets":
            limit = int(parts[2]) if len(parts) >= 3 else 20
            handle_top_wallets(conn, limit, time_clause, time_params)
            continue

        print("Unknown command. Type 'help' for usage.")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
