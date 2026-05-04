# Hunter Phase 1 — Wallet Extraction + Evidence Capture

This module performs **read-only Arkham evidence capture** for known wallet inputs.

## Scope
- Calls only approved Phase 0B endpoint families.
- Captures raw Arkham responses and writes sanitized files.
- Produces machine-readable and markdown evidence summary outputs.

## Non-goals
- No trading logic.
- No PnL computation.
- No wallet ranking.
- No alerts/dashboard/ML.

## Required input
Create `hunter_phase1/config/wallet_inputs.json` from `wallet_inputs.example.json`.

## Environment variables
- `ARKHAM_API_KEY`
- `ARKHAM_API_BASE_URL`
- `ARKHAM_AUTH_MODE` (defaults to `api-key`; this mode is required)

## Run
```bash
python3 hunter_phase1/scripts/run_wallet_evidence_capture.py
```

## Outputs
- `hunter_phase1/output/raw_samples/*.json`
- `hunter_phase1/output/wallet_evidence_index.json`
- `hunter_phase1/output/endpoint_success_matrix.md`
- `hunter_phase1/output/missing_evidence_report.md`
- `hunter_phase1/output/phase1_recommendation.md`

## Notes
- API keys are never printed or persisted.
- Addresses are redacted in indexed metadata and saved envelopes.
