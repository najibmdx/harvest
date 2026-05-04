#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from typing import Any


def build_tx_hash_inventory(transfers_by_wallet: dict[str, list[dict[str, Any]]], source_file: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for wallet_label, events in sorted(transfers_by_wallet.items()):
        hashes = [
            (ev.get("tx_hash") or ev.get("txHash") or ev.get("transactionHash") or "").strip()
            for ev in events
        ]
        present = [h for h in hashes if h]
        counts = Counter(present)
        duplicates = sum(c - 1 for c in counts.values() if c > 1)
        out.append({
            "wallet_label": wallet_label,
            "total_transfer_events": len(events),
            "unique_tx_hash_count": len(counts),
            "tx_hashes": sorted(counts.keys()),
            "missing_tx_hash_count": len([h for h in hashes if not h]),
            "duplicate_tx_hash_count": duplicates,
            "source_file": source_file,
        })
    return out
