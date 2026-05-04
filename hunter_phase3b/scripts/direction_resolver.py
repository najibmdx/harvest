#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WalletDirectionSummary:
    wallet_label: str
    wallet_address_available: bool
    transfer_events_seen: int
    inbound_count: int
    outbound_count: int
    unknown_direction_count: int
    direction_resolution_method: str
    unresolved_reason: str
    resolved_transfer_events: list[dict[str, Any]]


def _is_evm_address(value: str) -> bool:
    return isinstance(value, str) and value.lower().startswith("0x") and len(value) == 42


def _normalize_for_compare(value: Any, is_evm: bool) -> str:
    if not isinstance(value, str):
        return ""
    v = value.strip()
    return v.lower() if is_evm else v


def resolve_transfer_directions(
    transfers_by_wallet: dict[str, list[dict[str, Any]]],
    wallet_addresses: dict[str, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for wallet_label, events in sorted(transfers_by_wallet.items()):
        wallet_address = wallet_addresses.get(wallet_label, "")
        address_available = bool(wallet_address)
        evm_mode = _is_evm_address(wallet_address)
        normalized_wallet = _normalize_for_compare(wallet_address, evm_mode)

        inbound = outbound = unknown = 0
        resolved_events: list[dict[str, Any]] = []

        for ev in events:
            from_addr = ev.get("fromAddress") or ev.get("from") or ""
            to_addr = ev.get("toAddress") or ev.get("to") or ""
            norm_from = _normalize_for_compare(from_addr, evm_mode)
            norm_to = _normalize_for_compare(to_addr, evm_mode)

            direction = "unknown"
            reason = "wallet_address_unavailable"
            if address_available:
                reason = "no_address_match"
                if normalized_wallet and norm_from == normalized_wallet:
                    direction = "outbound"
                    reason = "matched_from_address"
                elif normalized_wallet and norm_to == normalized_wallet:
                    direction = "inbound"
                    reason = "matched_to_address"

            if direction == "inbound":
                inbound += 1
            elif direction == "outbound":
                outbound += 1
            else:
                unknown += 1

            resolved_events.append({
                "timestamp": ev.get("timestamp"),
                "txHash": ev.get("txHash"),
                "fromAddress": from_addr,
                "toAddress": to_addr,
                "direction": direction,
                "resolution_reason": reason,
            })

        unresolved_reason = "none"
        if not address_available:
            unresolved_reason = "wallet_full_address_missing"
        elif unknown > 0:
            unresolved_reason = "address_present_but_transfer_parties_do_not_match_wallet"

        summary = WalletDirectionSummary(
            wallet_label=wallet_label,
            wallet_address_available=address_available,
            transfer_events_seen=len(events),
            inbound_count=inbound,
            outbound_count=outbound,
            unknown_direction_count=unknown,
            direction_resolution_method=(
                "evm_case_insensitive_address_match" if evm_mode else "exact_string_address_match"
            ),
            unresolved_reason=unresolved_reason,
            resolved_transfer_events=resolved_events,
        )
        results.append(summary.__dict__)
    return results
