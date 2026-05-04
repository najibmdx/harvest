# Hunter Phase 0 — Arkham API Capability Auditor

## What this audit does
- Audits **only** discoverable Arkham **read-only GET** API capabilities based on provided docs/spec.
- Validates environment setup without exposing secrets.
- Attempts safe test calls only for endpoints discovered from docs/spec.
- Saves sanitized sample responses and generated capability artifacts.
- Produces explicit UNKNOWN states when required inputs are missing.

## What this audit does NOT do
- Does **not** build Hunter Phase 1.
- Does **not** include trading logic, strategy, ranking, alerts, UI, ML, or database design.
- Does **not** invent endpoints, params, or schemas.
- Does **not** hardcode keys or store secrets.

## Install dependencies
```bash
cd hunter_api_audit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure environment variables
1. Copy `.env.example` to `.env`.
2. Fill values from your environment:
   - `ARKHAM_API_KEY`
   - `ARKHAM_API_BASE_URL`
   - `ARKHAM_DOCS_URL` (optional but strongly recommended for endpoint discovery)

## Run the audit
```bash
cd hunter_api_audit
python scripts/run_audit.py
```

## Output location
All artifacts are written under:
- `hunter_api_audit/output/`
- sanitized response samples: `hunter_api_audit/output/sample_responses/`

## Warning
This repository component contains **no trading logic** and is strictly limited to API capability auditing.
