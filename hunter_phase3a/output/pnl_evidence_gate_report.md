# PnL Evidence Gate Report

## binance_public_evm_fixture
- available evidence:
  - balance snapshots: 520
  - transfer events: 20
  - flow evidence: yes
  - counterparty evidence: partial/unknown
  - identity evidence: yes
  - historical USD fields: yes
  - timestamps: yes
- missing evidence:
  - deterministic trade side
  - cost basis
  - confirmed acquisition events
  - confirmed disposal events
  - fee/gas basis if absent
- explicit gate verdict: REALIZED_PNL_BLOCKED
- explicit gate verdict: UNREALIZED_EXPOSURE_ALLOWED
- explicit gate verdict: FLOW_ANALYSIS_ALLOWED
