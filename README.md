# padre-kol-clean

Local-only Node.js exporter for Padre `KOLs Manager` data from your own authenticated Chrome session.

## Install
```bash
npm install
```

## Launch Chrome with remote debugging (Windows)
```powershell
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1 --user-data-dir="%USERPROFILE%\padre-debug-profile" --no-first-run --no-default-browser-check --disable-extensions
```

## Required before running
1. Manually log in to Padre in that Chrome window.
2. Open: `https://trade.padre.gg/tracker`
3. Open the **KOLs Manager** section.

## Run
```bash
npm run export:padre-kols
```

## Output
- `output/padre_kols.csv`
- `output/padre_kols.json`
- `output/debug_network_candidates.json`
- `output/debug_dom_snapshot.html`
- `output/debug_dom_rows.json`
- `output/debug_run_log.txt`
