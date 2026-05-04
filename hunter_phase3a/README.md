# Hunter Phase 3A — PnL Reconstruction Design Constraints

## What Phase 3A does
- Defines strict evidence gates for future PnL computation modes.
- Audits available Phase 2 evidence and missing fields.
- Produces machine-readable eligibility and requirements artifacts.

## What Phase 3A does NOT do
- Does **not** compute realized or unrealized PnL.
- Does **not** compute ROI/win-rate.
- Does **not** label wallets profitable/unprofitable.
- Does **not** call APIs.

> **Warning:** Phase 3A does not compute PnL. It only defines the constraints required before PnL computation is allowed.

## Required Phase 2 inputs
- `hunter_phase2/output/wallet_activity_timeline.jsonl`
- `hunter_phase2/output/wallet_asset_exposure_summary.json`
- `hunter_phase2/output/wallet_flow_summary.json`
- `hunter_phase2/output/wallet_identity_summary.json`
- `hunter_phase2/output/pnl_feasibility_report.md`
- `hunter_phase2/output/phase2_consistency_check.md`
- `hunter_phase2/output/phase2_recommendation.md`
- `hunter_phase2/output/raw_sample_shape_diagnostics.json`

## How to run
```bash
python3 hunter_phase3a/scripts/run_pnl_design_constraints.py
```

## Output files
- `output/pnl_evidence_gate_report.md`: per-wallet evidence and gate verdict.
- `output/pnl_input_requirements.json`: formal requirements by PnL mode.
- `output/pnl_mode_eligibility.json`: per-wallet mode eligibility flags and blockers.
- `output/invalid_pnl_assumptions.md`: forbidden inference list.
- `output/wallet_pnl_readiness_matrix.md`: tabular readiness matrix.
- `output/phase3a_design_constraints.md`: formal doctrine.
- `output/phase3a_consistency_check.md`: policy/consistency checks.
- `output/phase3a_recommendation.md`: single recommendation verdict.
