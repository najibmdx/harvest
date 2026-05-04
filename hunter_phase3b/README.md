# Hunter Phase 3B — Evidence Expansion for Trade-Side and Cost-Basis Reconstruction

## What Phase 3B does
- Loads Phase 2 transfer and snapshot evidence.
- Resolves transfer direction only with deterministic wallet-address matching.
- Inventories transfer transaction hashes.
- Optionally captures read-only Arkham `/tx/{hash}` details for explicit Phase 2 hashes.
- Produces conservative evidence classification reports for trade-side, cost-basis, linkage, and fee/gas methodology.

## What Phase 3B does NOT do
- Does **not** compute realized/unrealized PnL.
- Does **not** compute ROI or win rate.
- Does **not** label profitability.
- Does **not** rank wallets.
- Does **not** call external price APIs.

> **Warning:** Phase 3B expands evidence only. It does not compute PnL.

## Required inputs
- `hunter_phase2/output/wallet_activity_timeline.jsonl`
- `hunter_phase2/output/wallet_flow_summary.json`
- `hunter_phase2/output/wallet_identity_summary.json`
- `hunter_phase2/output/phase2_consistency_check.md`
- `hunter_phase3a/output/pnl_mode_eligibility.json`
- `hunter_phase3a/output/pnl_input_requirements.json`
- `hunter_phase3a/output/phase3a_design_constraints.md`
- optional wallet addresses from:
  - `config/wallet_inputs.json`
  - `hunter_phase1/config/wallet_inputs.json`

## Optional Arkham capture
- Enabled only when `allow_api_calls=true` and `offline_only=false` in config.
- Uses existing shared Arkham client (`hunter_phase1/scripts/arkham_client.py`) when available.
- Calls only `/tx/{hash}` and only for hashes present in Phase 2 transfer events.

## How to run
```bash
python hunter_phase3b/scripts/run_evidence_expansion.py
```

## Outputs
- `hunter_phase3b/output/transfer_direction_resolution.json`
- `hunter_phase3b/output/transaction_hash_inventory.json`
- `hunter_phase3b/output/tx_detail_request_plan.json`
- `hunter_phase3b/output/tx_detail_capture_index.json`
- `hunter_phase3b/output/raw_tx_details/`
- `hunter_phase3b/output/trade_side_evidence_report.md`
- `hunter_phase3b/output/cost_basis_evidence_report.md`
- `hunter_phase3b/output/acquisition_disposal_linkage_report.md`
- `hunter_phase3b/output/fee_gas_methodology_report.md`
- `hunter_phase3b/output/phase3b_evidence_expansion_report.md`
- `hunter_phase3b/output/phase3b_consistency_check.md`
- `hunter_phase3b/output/phase3b_recommendation.md`
