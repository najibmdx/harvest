#!/usr/bin/env python3
from __future__ import annotations

from typing import Any


def classify_transfer_event(event: dict[str, Any]) -> str:
    cp = " ".join(str(event.get(k, "")) for k in ["counterparty", "entity", "fromEntity", "toEntity"]).lower()
    if any(x in cp for x in ["exchange", "cex", "binance", "coinbase", "kraken", "okx", "bybit"]):
        return "CEX_FLOW_ONLY"
    return "TRANSFER_ONLY"


def tx_has_trade_side(detail: dict[str, Any]) -> bool:
    txt = str(detail).lower()
    required_markers = ["sold", "bought", "amount", "token"]
    return all(m in txt for m in required_markers)


def tx_has_fee_gas(detail: dict[str, Any]) -> tuple[bool, list[str]]:
    fields = []
    for f in ["gas", "gasPrice", "gasUsed", "fee", "networkFee", "txFee"]:
        if f in detail:
            fields.append(f)
    if not fields:
        txt = str(detail).lower()
        for f in ["gas", "fee"]:
            if f in txt:
                fields.append(f"string:{f}")
    return bool(fields), fields


def cost_basis_candidate_possible(direction: str, detail: dict[str, Any]) -> bool:
    if direction not in {"inbound", "outbound"}:
        return False
    txt = str(detail).lower()
    required = ["token", "amount", "timestamp", "tx"]
    has_value = ("price" in txt) or ("value" in txt) or ("historicalusd" in txt)
    return all(x in txt for x in required) and has_value
