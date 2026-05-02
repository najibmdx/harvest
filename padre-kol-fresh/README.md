# padre-kol-fresh

Local-only Node.js scraper that connects to your already logged-in Chrome debug session and exports KOL owner display names with full Solana wallet addresses from Padre Tracker -> KOLs Manager.

## Install

```bash
npm install
```

## Run

```bash
npm run export:padre-kols
```

## Launch Chrome (Windows)

```powershell
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1 --user-data-dir="%USERPROFILE%\padre-debug-profile" --no-first-run --no-default-browser-check --disable-extensions
```

## Manual steps

1. Close Chrome
2. Launch debug Chrome
3. Open Padre tracker
4. Log in manually
5. Open KOLs Manager
6. Run scraper

## Output files

- `output/padre_kols.csv`
- `output/padre_kols.json`
- `output/debug_run_log.txt`
- `output/debug_dom_snapshot.html`
- `output/debug_dom_rows.json`
