# PnL Evidence Gate Report

## fixture_wallet
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
- explicit gate verdict: EXPOSURE_ONLY_ALLOWED + FLOW_ONLY_ALLOWED
