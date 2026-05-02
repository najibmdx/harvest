# Padre KOLs Manager Exporter (Local-Only)

This tool exports KOL display names + **full wallet addresses** from the authenticated Padre tracker UI (`KOLs Manager`) using your own local Chrome session.

## Security and scope
- Local-only extraction from your active authenticated browser session.
- No bypassing auth/CAPTCHA/paywalls.
- No external APIs.
- No server component.
- No storage of passwords/cookies/private keys/seed phrases/tokens.

## Prerequisites
- Node.js 18+
- Google Chrome

## Install
```bash
npm install
```

## Launch Chrome with remote debugging
Close all Chrome windows first, then run one of these:

### Windows
```powershell
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%USERPROFILE%\padre-debug-profile"
```

### macOS
```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222 --user-data-dir="$HOME/padre-debug-profile"
```

Then, in that Chrome window:
1. Log into Padre manually.
2. Ensure you can access `https://trade.padre.gg/tracker`.

## Run exporter
```bash
npm run export:padre-kols
```

## Output files
- `output/padre_kols.csv`
- `output/padre_kols.json`
- `output/padre_kols.sqlite`
- `output/debug_network_candidates.json`
- `output/debug_dom_snapshot.html`
- `output/debug_dom_rows.json`
- `output/debug_run_log.txt`

## Extraction order
1. Connect to existing Chrome via CDP (`http://127.0.0.1:9222`).
2. Open tracker and activate `KOLs Manager`.
3. Observe fetch/xhr JSON responses for 15 seconds.
4. Recursively score candidate wallet+name fields.
5. Build DOM row evidence from visible KOL rows and candidate display names.
6. Pair DOM names to full wallets using clipboard-first, fragment match second (hybrid mode).
7. Validate Solana-like full addresses and reject obvious truncation.
8. Reject numeric/internal IDs as display names and fail if all names are numeric/empty.
9. Deduplicate by wallet; preserve alternate names in JSON `names`.
5. If network confidence is insufficient, fallback to DOM + copy-button clipboard flow.
6. Validate Solana-like full addresses and reject obvious truncation.
7. Deduplicate by wallet; preserve alternate names in JSON `names`.

## Failure behavior
If full wallet addresses cannot be recovered, the script exits with a clear error and does not silently emit partial/truncated data.
