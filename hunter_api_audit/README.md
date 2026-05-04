# Hunter Phase 0 — Arkham API Capability Auditor

## What this audit does
- Audits **only** discoverable Arkham **read-only GET** API capabilities based on provided docs/spec.
- Validates environment setup without exposing secrets.
- Attempts safe test calls only for endpoints discovered from docs/spec.
- Handles HTML docs pages by saving a sanitized snapshot and extracting only clear API-like endpoint paths/spec-link candidates without executing JavaScript.
- Supports manual safe endpoint seeding via `config/endpoints_seed.json` (enabled `GET` endpoints only).
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
   - `ARKHAM_OPENAPI_FILE` (optional local OpenAPI/spec JSON file path)

## Run the audit
```bash
cd hunter_api_audit
python scripts/run_audit.py
```

## Optional manual endpoint seed
- Copy `config/endpoints_seed.example.json` to `config/endpoints_seed.json`.
- Add only verified read-only endpoints.
- Set `"enabled": true` only for endpoints you want tested.
- The auditor will never invent params and will skip non-GET endpoints.

## OpenAPI local fallback (for blocked remote docs)
- If `ARKHAM_DOCS_URL` is blocked or returns HTTP 403, manually download the Arkham OpenAPI/spec JSON and store it locally.
- Set:
  - `ARKHAM_OPENAPI_FILE=config/openapi.json`
- The auditor will prioritize local OpenAPI loading before remote docs fetch.

## Output location
All artifacts are written under:
- `hunter_api_audit/output/`
- sanitized response samples: `hunter_api_audit/output/sample_responses/`
- docs HTML snapshot (if docs URL returns HTML): `hunter_api_audit/output/docs_snapshot.html`

## Warning
This repository component contains **no trading logic** and is strictly limited to API capability auditing.
