"""Deterministic Phase 3A PnL eligibility rule definitions (no computation)."""

from __future__ import annotations

PNL_MODES = {
    "REALIZED_PNL_FULL": {
        "required_fields": [
            "confirmed_acquisition_events",
            "confirmed_disposal_events",
            "deterministic_wallet_direction",
            "token_amount",
            "timestamp",
            "execution_price_or_historical_usd_at_execution",
            "cost_basis_method",
            "chain_token_identity",
            "transaction_hash",
            "no_unresolved_transfer_as_trade_ambiguity",
        ],
        "optional_fields": ["fee_gas_data"],
    },
    "REALIZED_PNL_LIMITED": {
        "required_fields": [
            "confirmed_acquisition_events",
            "confirmed_disposal_events",
            "deterministic_wallet_direction",
            "reconstructable_cost_basis",
            "no_transfer_only_trade_inference",
        ],
        "optional_fields": ["fee_gas_data_explicitly_excluded"],
    },
    "UNREALIZED_EXPOSURE_ONLY": {
        "required_fields": [
            "current_balance_snapshot",
            "token_amount",
            "token_price_or_usd_value",
            "quote_time",
            "chain_token_identity",
        ],
        "optional_fields": [],
    },
    "FLOW_ANALYSIS_ONLY": {
        "required_fields": ["transfer_or_flow_records", "timestamp", "amount_or_usd_value"],
        "optional_fields": [],
    },
    "NOT_PNL_ELIGIBLE": {
        "required_fields": [
            "no_deterministic_trade_side",
            "no_cost_basis",
            "no_confirmed_disposal_evidence",
            "balances_transfers_flows_only",
        ],
        "optional_fields": [],
    },
}

FORBIDDEN_ASSUMPTIONS = [
    "transfer_is_trade",
    "deposit_is_buy",
    "withdrawal_is_sell",
    "historical_usd_is_cost_basis",
    "balance_snapshot_is_trade",
]
