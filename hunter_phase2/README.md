# Hunter Phase 2: Wallet Activity Reconstruction + PnL Feasibility Review

## What Phase 2 does
- Runs **offline-first** reconstruction from Phase 1 captured Arkham evidence.
- Parses raw sample payloads for balances, flow, counterparties, intelligence, and transfers.
- Produces structured wallet activity timeline evidence (`jsonl`) and wallet-level summary artifacts.
- Produces a PnL feasibility assessment report without calculating any PnL values.

## What Phase 2 does NOT do
- No trading logic, alerts, dashboarding, ML, ranking, or profitability claims.
- No realized/unrealized PnL computation.
- No inferred cost basis, missing prices, or missing timestamps.
- No new Arkham API calls in default mode (`offline_only=true`, `allow_new_api_calls=false`).

> **Warning:** Phase 2 does not compute PnL. It only determines whether PnL reconstruction is feasible.

## Required Phase 1 inputs
- `hunter_phase1/output/wallet_evidence_index.json`
- `hunter_phase1/output/raw_samples/`

## How to run
1. (Optional) copy `hunter_phase2/config/phase2_config.example.json` to `hunter_phase2/config/phase2_config.json` and edit values.
2. Run:

```bash
python3 hunter_phase2/scripts/run_wallet_reconstruction.py
```

## Output files
- `hunter_phase2/output/wallet_activity_timeline.jsonl`: one event record per line.
- `hunter_phase2/output/wallet_asset_exposure_summary.json`: wallet exposure/balance evidence summary.
- `hunter_phase2/output/wallet_flow_summary.json`: transfer/flow availability and directionality summary.
- `hunter_phase2/output/wallet_counterparty_summary.json`: counterparty and Arkham entity evidence summary.
- `hunter_phase2/output/wallet_identity_summary.json`: identity/entity/label evidence summary.
- `hunter_phase2/output/reconstruction_coverage_report.md`: category-level parsing coverage and blockers.
- `hunter_phase2/output/pnl_feasibility_report.md`: per-wallet feasibility verdicts for later PnL design.
- `hunter_phase2/output/phase2_recommendation.md`: final Phase 2 recommendation (A/B/C only).
- `hunter_phase2/output/reconstructed/`: reserved per-wallet reconstructed artifacts directory.
