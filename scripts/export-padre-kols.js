#!/usr/bin/env node
/* eslint-disable no-console */
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const OUTPUT_DIR = path.resolve(process.cwd(), 'output');
const LOG_PATH = path.join(OUTPUT_DIR, 'debug_run_log.txt');
const DOM_SNAPSHOT_PATH = path.join(OUTPUT_DIR, 'debug_dom_snapshot.html');
const DOM_ROWS_PATH = path.join(OUTPUT_DIR, 'debug_dom_rows.json');
const NETWORK_CANDIDATES_PATH = path.join(OUTPUT_DIR, 'debug_network_candidates.json');
const CSV_PATH = path.join(OUTPUT_DIR, 'padre_kols.csv');
const JSON_PATH = path.join(OUTPUT_DIR, 'padre_kols.json');
const SQLITE_PATH = path.join(OUTPUT_DIR, 'padre_kols.sqlite');
const CDP_URL = 'http://127.0.0.1:9222';

const WALLET_KEYS = ['wallet', 'walletAddress', 'address', 'owner', 'publicKey', 'pubkey', 'account', 'trader', 'kol'];
const NAME_KEYS = ['name', 'displayName', 'username', 'handle', 'label', 'nickname', 'ownerName'];

const BASE58_RE = /[1-9A-HJ-NP-Za-km-z]{32,44}/g;

const nowIso = () => new Date().toISOString();
const ensureOutput = () => fs.mkdirSync(OUTPUT_DIR, { recursive: true });
function ensureOutput() {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

function nowIso() {
  return new Date().toISOString();
}

function createLogger() {
  const lines = [];
  const log = (msg) => {
    const line = `[${nowIso()}] ${msg}`;
    lines.push(line);
    console.log(line);
  };
  const flush = () => fs.writeFileSync(LOG_PATH, `${lines.join('\n')}\n`, 'utf8');
  return { log, flush };
}

function isLikelySolanaAddress(v) {
  if (typeof v !== 'string') return false;
  const s = v.trim();
  return s.length >= 32 && s.length <= 44 && !/[\s.…]/.test(s) && /^[1-9A-HJ-NP-Za-km-z]+$/.test(s);
}

function isNumericOnly(s) {
  return typeof s === 'string' && /^\d+$/.test(s.trim());
}

function isLikelyInternalIdName(name) {
  if (!name || typeof name !== 'string') return true;
  const s = name.trim();
  if (!s) return true;
  if (isNumericOnly(s) && s.length >= 6) return true;
  return false;
  if (s.length < 32 || s.length > 44) return false;
  if (/\.{2,}|…/.test(s)) return false;
  if (/\s/.test(s)) return false;
  if (!/^[1-9A-HJ-NP-Za-km-z]+$/.test(s)) return false;
  if (s.length <= 8) return false;
  return true;
}

function isLikelyTruncated(v) {
  if (typeof v !== 'string') return true;
  const s = v.trim();
  return s.length <= 8 || /\.{2,}|…/.test(s);
}

function normalizeName(v) {
  return typeof v === 'string' ? v : (v == null ? '' : String(v));
}

function recursiveWalk(value, visitor, pathStack = []) {
  visitor(value, pathStack);
  if (Array.isArray(value)) value.forEach((item, i) => recursiveWalk(item, visitor, pathStack.concat(String(i))));
  else if (value && typeof value === 'object') Object.entries(value).forEach(([k, v]) => recursiveWalk(v, visitor, pathStack.concat(k)));
}

function discoverFromJson(payload, meta) {
  const out = [];
  recursiveWalk(payload, (node, pth) => {
    if (!node || typeof node !== 'object' || Array.isArray(node)) return;
    const entries = Object.entries(node);
    const walletHits = entries.filter(([k, v]) => WALLET_KEYS.map(x => x.toLowerCase()).includes(k.toLowerCase()) && typeof v === 'string');
    const nameHits = entries.filter(([k, v]) => NAME_KEYS.map(x => x.toLowerCase()).includes(k.toLowerCase()) && typeof v === 'string');

    for (const [walletKey, walletValue] of walletHits) {
      const wallet = walletValue.trim();
      out.push({
        wallet_address: wallet,
        name: nameHits[0] ? nameHits[0][1] : '',
        source: 'network',
        extracted_at: nowIso(),
        meta: {
          url: meta.url,
          status: meta.status,
          path: pth.join('.'),
          wallet_key: walletKey,
          name_key: nameHits[0] ? nameHits[0][0] : null,
          full_wallet: isLikelySolanaAddress(wallet)
        }
      });
    }
  });
  return out;
}

function fragmentMatchesWallet(fragment, wallet) {
  if (!fragment || !wallet) return false;
  const f = fragment.replace(/\s+/g, '').replace(/…/g, '').replace(/\./g, '');
  if (f.length < 4) return false;
  return wallet.startsWith(f) || wallet.includes(f);
}

function cleanCandidateName(raw, rowText) {
  if (!raw || typeof raw !== 'string') return '';
  let name = raw.replace(/\s+/g, ' ').trim();
  if (!name) return '';
  if (/^copy$/i.test(name) || /^copied!?$/i.test(name)) return '';
  if (isLikelyInternalIdName(name)) return '';

  const row = (rowText || '').replace(/\s+/g, ' ');
  if (row) {
    const walletTokens = row.match(BASE58_RE) || [];
    walletTokens.forEach((w) => { if (name.includes(w)) name = name.replace(w, '').trim(); });
  }

  if (isLikelyInternalIdName(name)) return '';
  return name;
}

function consolidate(rows) {
  if (Array.isArray(value)) {
    value.forEach((item, idx) => recursiveWalk(item, visitor, pathStack.concat(String(idx))));
  } else if (value && typeof value === 'object') {
    Object.entries(value).forEach(([k, v]) => recursiveWalk(v, visitor, pathStack.concat(k)));
  }
}

function discoverFromJson(payload, meta) {
  const candidates = [];
  recursiveWalk(payload, (node, pth) => {
    if (!node || typeof node !== 'object' || Array.isArray(node)) return;

    const entries = Object.entries(node);
    const walletHits = [];
    const nameHits = [];

    for (const [k, v] of entries) {
      const lowerK = k.toLowerCase();
      if (WALLET_KEYS.map((x) => x.toLowerCase()).includes(lowerK) && typeof v === 'string') {
        walletHits.push({ key: k, value: v });
      }
      if (NAME_KEYS.map((x) => x.toLowerCase()).includes(lowerK) && typeof v === 'string') {
        nameHits.push({ key: k, value: v });
      }
    }

    if (walletHits.length > 0) {
      for (const w of walletHits) {
        const wallet = String(w.value || '').trim();
        const full = isLikelySolanaAddress(wallet);
        const nameFromNode = nameHits[0] ? normalizeName(nameHits[0].value) : '';
        let inferredName = nameFromNode;

        if (!inferredName) {
          for (const [k, v] of entries) {
            if (typeof v === 'string' && v.trim() && !isLikelySolanaAddress(v) && !isLikelyTruncated(v)) {
              inferredName = v;
              break;
            }
          }
        }

        const score = (full ? 4 : 0) + (nameFromNode ? 2 : 0) + (WALLET_KEYS.includes(w.key) ? 1 : 0);
        candidates.push({
          wallet_address: wallet,
          name: inferredName,
          source: 'network',
          extracted_at: nowIso(),
          meta: {
            url: meta.url,
            status: meta.status,
            path: pth.join('.'),
            wallet_key: w.key,
            name_key: nameHits[0] ? nameHits[0].key : null,
            score,
            full_wallet: full
          }
        });
      }
    }
  });
  return candidates;
}

function consolidateRows(rows) {
  const byWallet = new Map();
  for (const r of rows) {
    if (!isLikelySolanaAddress(r.wallet_address)) continue;
    const wallet = r.wallet_address.trim();
    const name = r.name || '';
    if (!byWallet.has(wallet)) {
      byWallet.set(wallet, { name, names: name ? [name] : [], wallet_address: wallet, source: r.source, extracted_at: r.extracted_at });
    } else if (name && !byWallet.get(wallet).names.includes(name)) {
      byWallet.get(wallet).names.push(name);
    }
  }
  return [...byWallet.values()].map((r) => ({ ...r, name: r.name || (r.names[0] || '') }));
}

function toCsv(rows) {
  const esc = (v) => /[",\n]/.test(String(v ?? '')) ? `"${String(v ?? '').replace(/"/g, '""')}"` : String(v ?? '');
  return `name,wallet_address,source,extracted_at\n${rows.map((r) => [esc(r.name), esc(r.wallet_address), esc(r.source), esc(r.extracted_at)].join(',')).join('\n')}\n`;
    const name = normalizeName(r.name);
    if (!byWallet.has(wallet)) {
      byWallet.set(wallet, {
        name,
        names: name ? [name] : [],
        wallet_address: wallet,
        source: r.source,
        extracted_at: r.extracted_at
      });
    } else {
      const existing = byWallet.get(wallet);
      if (name && !existing.names.includes(name)) existing.names.push(name);
    }
  }
  return [...byWallet.values()].map((row) => ({
    name: row.name || (row.names[0] || ''),
    names: row.names,
    wallet_address: row.wallet_address,
    source: row.source,
    extracted_at: row.extracted_at
  }));
}

function toCsv(rows) {
  const header = 'name,wallet_address,source,extracted_at';
  const esc = (v) => {
    const s = v == null ? '' : String(v);
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };
  return `${header}\n${rows.map((r) => [esc(r.name), esc(r.wallet_address), esc(r.source), esc(r.extracted_at)].join(',')).join('\n')}\n`;
}

function writeSqliteWithPython(rows, log) {
  const payloadPath = path.join(OUTPUT_DIR, '.tmp_rows.json');
  fs.writeFileSync(payloadPath, JSON.stringify(rows), 'utf8');
  const py = [
    'import json, sqlite3',
    `rows=json.load(open(r"${payloadPath}","r",encoding="utf-8"))`,
    `conn=sqlite3.connect(r"${SQLITE_PATH}")`,
    'cur=conn.cursor()',
    'cur.execute("DROP TABLE IF EXISTS padre_kols")',
    'cur.execute("CREATE TABLE padre_kols (name TEXT, names TEXT, wallet_address TEXT PRIMARY KEY, source TEXT, extracted_at TEXT)")',
    'for r in rows:',
    '  cur.execute("INSERT OR REPLACE INTO padre_kols (name,names,wallet_address,source,extracted_at) VALUES (?,?,?,?,?)", (r.get("name",""), json.dumps(r.get("names",[]), ensure_ascii=False), r["wallet_address"], r.get("source",""), r.get("extracted_at","")))',
    'conn.commit(); conn.close()'
  ].join(';');
  const { execSync } = require('child_process');
  execSync(`python3 -c '${py.replace(/'/g, "'\\''")}'`, { stdio: 'pipe' });
  fs.unlinkSync(payloadPath);
  log('SQLite export completed.');
}

async function run() {
  ensureOutput();
  const { log, flush } = createLogger();
  let browser;

  try {
    log('Starting export flow.');
    browser = await chromium.connectOverCDP(CDP_URL);
    const context = browser.contexts()[0];
    if (!context) throw new Error('No browser context found. Launch Chrome with remote debugging and log in first.');
    log(`Connecting to Chrome over CDP at ${CDP_URL}.`);
    browser = await chromium.connectOverCDP(CDP_URL);

    const context = browser.contexts()[0];
    if (!context) throw new Error('No browser context found. Ensure Chrome is launched with remote debugging and has at least one profile context.');

    let page = context.pages().find((p) => p.url().includes('trade.padre.gg')) || context.pages()[0];
    if (!page) page = await context.newPage();

    const networkCandidates = [];
    page.on('response', async (resp) => {
      try {
        if (!['fetch', 'xhr'].includes(resp.request().resourceType())) return;
        const ct = (resp.headers()['content-type'] || '').toLowerCase();
        if (!ct.includes('json') && !ct.includes('javascript')) return;
        const text = await resp.text();
        let json;
        try { json = JSON.parse(text); } catch { return; }
        const found = discoverFromJson(json, { url: resp.url(), status: resp.status() });
        if (found.length) networkCandidates.push(...found);
      } catch {}
    });

    await page.goto('https://trade.padre.gg/tracker', { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForTimeout(4000);
    const tab = page.locator('text=KOLs Manager').first();
    await tab.waitFor({ timeout: 30000 });
    await tab.click({ timeout: 10000 });

    log('Capturing network for 15 seconds.');
    await page.waitForTimeout(15000);

    const domHtml = await page.content();
    fs.writeFileSync(DOM_SNAPSHOT_PATH, domHtml, 'utf8');
    fs.writeFileSync(NETWORK_CANDIDATES_PATH, JSON.stringify(networkCandidates, null, 2), 'utf8');

    const networkWallets = consolidate(networkCandidates.filter((r) => r.meta && r.meta.full_wallet));
    const networkWalletSet = new Set(networkWallets.map((r) => r.wallet_address));
    if (!networkWallets.length) throw new Error('No valid full wallets found in network capture.');
    log(`Valid network wallets: ${networkWallets.length}`);

    const domRows = await page.evaluate(() => {
      const base58 = /[1-9A-HJ-NP-Za-km-z]{4,44}/g;
      const rows = [];
      const rowNodes = Array.from(document.querySelectorAll('tr, [role="row"], li, article, section > div, div'));

      for (const node of rowNodes) {
        const text = (node.innerText || node.textContent || '').replace(/\s+/g, ' ').trim();
        if (!text || text.length < 5) continue;
        if (!/copy|wallet|kol|manager/i.test(text)) continue;

        const buttons = Array.from(node.querySelectorAll('button, [role="button"], [aria-label], [title]'));
        const copyButton = buttons.find((b) => /copy/i.test((b.getAttribute('aria-label') || '') + ' ' + (b.getAttribute('title') || '') + ' ' + (b.textContent || '')));
        if (!copyButton) continue;

        const nameEl = node.querySelector('[data-testid*="name"], [class*="name"], a, strong, h1, h2, h3, h4, h5, span');
        const nameText = (nameEl ? nameEl.textContent : node.textContent || '').replace(/\s+/g, ' ').trim();
        const fragments = text.match(base58) || [];
        const visibleWalletFragment = fragments.sort((a, b) => b.length - a.length)[0] || '';

        const marker = `padre-copy-${rows.length}`;
        copyButton.setAttribute('data-padre-copy-marker', marker);

        rows.push({ row_index: rows.length, raw_text: text, candidate_name: nameText, visible_wallet_fragment: visibleWalletFragment, copy_marker: marker });
      }
      return rows;
    });

    log(`DOM rows discovered: ${domRows.length}`);

    const debugRows = [];
    const hybridRows = [];

    for (const row of domRows) {
      let clipboardWallet = '';
      let matchedNetworkWallet = '';
      let matchMethod = 'none';

      try {
        const btn = page.locator(`[data-padre-copy-marker="${row.copy_marker}"]`).first();
        if (await btn.count()) {
          await btn.click({ timeout: 2000 });
          await page.waitForTimeout(200);
          clipboardWallet = await page.evaluate(async () => {
            try { return (await navigator.clipboard.readText() || '').trim(); } catch { return ''; }
          });
          if (isLikelySolanaAddress(clipboardWallet)) {
            matchedNetworkWallet = clipboardWallet;
            matchMethod = 'clipboard';
          }
        }
      } catch {}

      if (!matchedNetworkWallet && row.visible_wallet_fragment) {
        const matched = [...networkWalletSet].find((w) => fragmentMatchesWallet(row.visible_wallet_fragment, w));
        if (matched) {
          matchedNetworkWallet = matched;
          matchMethod = 'fragment';
        }
      }

      const cleanedName = cleanCandidateName(row.candidate_name, row.raw_text);
      debugRows.push({
        row_index: row.row_index,
        raw_text: row.raw_text,
        candidate_name: cleanedName,
        visible_wallet_fragment: row.visible_wallet_fragment,
        clipboard_wallet: clipboardWallet,
        matched_network_wallet: matchedNetworkWallet,
        match_method: matchMethod
      });

      if (matchedNetworkWallet) {
        hybridRows.push({
          name: cleanedName,
          wallet_address: matchedNetworkWallet,
          source: matchMethod === 'clipboard' ? 'clipboard' : 'hybrid',
          extracted_at: nowIso()
        });
      } else {
        log(`Unpaired row ${row.row_index}: could not match wallet. Name candidate='${cleanedName}'`);
      }
    }

    fs.writeFileSync(DOM_ROWS_PATH, JSON.stringify(debugRows, null, 2), 'utf8');

    const hybridByWallet = new Map(hybridRows.map((r) => [r.wallet_address, r]));
    const finalRows = networkWallets.map((nw) => {
      const h = hybridByWallet.get(nw.wallet_address);
      return {
        name: h ? h.name : '',
        wallet_address: nw.wallet_address,
        source: h ? h.source : 'network',
        extracted_at: nowIso(),
        names: h && h.name ? [h.name] : []
      };
    });

    const nonNumericNames = finalRows.filter((r) => r.name && !isLikelyInternalIdName(r.name));
    if (finalRows.length && nonNumericNames.length === 0) {
      throw new Error('Name extraction failed: all extracted names are empty or numeric/internal IDs.');
    }

    fs.writeFileSync(CSV_PATH, toCsv(finalRows), 'utf8');
    fs.writeFileSync(JSON_PATH, JSON.stringify(finalRows, null, 2), 'utf8');

    try {
      if (!fs.existsSync(SQLITE_PATH)) {
        fs.writeFileSync(SQLITE_PATH, '');
      }
      log('SQLite export skipped (optional, no Python dependency).');
    } catch (e) {
      log(`SQLite optional export skipped due to error: ${e.message}`);
    }

    log(`Export completed. Wallets: ${finalRows.length}. Named rows: ${nonNumericNames.length}.`);
    flush();
  } catch (err) {
    log(`ERROR: ${err.message}`);
    flush();
    const responseHandler = async (resp) => {
      try {
        const req = resp.request();
        const type = req.resourceType();
        if (!['fetch', 'xhr'].includes(type)) return;

        const ct = (resp.headers()['content-type'] || '').toLowerCase();
        if (!ct.includes('application/json') && !ct.includes('text/json') && !ct.includes('javascript')) return;

        const txt = await resp.text();
        let parsed;
        try { parsed = JSON.parse(txt); } catch { return; }
        const found = discoverFromJson(parsed, { url: resp.url(), status: resp.status() });
        if (found.length) networkCandidates.push(...found);
      } catch {
        // best effort
      }
    };

    page.on('response', responseHandler);

    log('Navigating to tracker page.');
    await page.goto('https://trade.padre.gg/tracker', { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForTimeout(4000);

    log('Locating KOLs Manager section/tab.');
    const kolLocator = page.locator('text=KOLs Manager').first();
    await kolLocator.waitFor({ timeout: 30000 });
    await kolLocator.click({ timeout: 10000 });

    log('Capturing Fetch/XHR JSON responses for 15 seconds.');
    await page.waitForTimeout(15000);

    fs.writeFileSync(NETWORK_CANDIDATES_PATH, JSON.stringify(networkCandidates, null, 2), 'utf8');
    log(`Saved network candidates: ${networkCandidates.length}`);

    let finalRows = [];
    const confident = consolidateRows(
      networkCandidates
        .filter((r) => r.meta && r.meta.full_wallet)
        .sort((a, b) => (b.meta.score || 0) - (a.meta.score || 0))
    );

    if (confident.length > 0) {
      finalRows = confident.map((r) => ({ ...r, source: 'network' }));
      log(`Selected extraction mode: network. Valid wallets: ${finalRows.length}`);
    } else {
      log('Network extraction did not find confident full wallet/name pairs. Falling back to DOM+clipboard.');
      const html = await page.content();
      fs.writeFileSync(DOM_SNAPSHOT_PATH, html, 'utf8');
      log('Saved DOM snapshot.');

      const domRows = await page.evaluate(async () => {
        function findWalletText(el) {
          const texts = [el.innerText || '', el.textContent || ''];
          return texts.find((t) => /[1-9A-HJ-NP-Za-km-z]{32,44}/.test(t)) || '';
        }

        const candidates = [];
        const blocks = Array.from(document.querySelectorAll('tr, [role="row"], li, div'));
        for (const block of blocks) {
          const txt = (block.textContent || '').trim();
          if (!txt || txt.length < 4) continue;
          if (!/kol|manager|wallet|address/i.test(document.body.innerText || '')) break;

          const btn = block.querySelector('button, [role="button"]');
          if (!btn) continue;
          const walletVisible = findWalletText(block);
          candidates.push({
            name: (block.querySelector('[data-testid*="name"], .name, .username, a, span') || block).textContent?.trim() || '',
            walletVisible,
            hasButton: !!btn
          });
        }
        return candidates;
      });

      log(`DOM candidate rows found: ${domRows.length}`);

      const copiedRows = [];
      const copyButtons = page.locator('button:has-text("Copy"), [aria-label*="copy" i], [title*="copy" i], button >> svg');
      const copyCount = await copyButtons.count();
      log(`Detected potential copy buttons: ${copyCount}`);

      for (let i = 0; i < copyCount; i++) {
        try {
          await copyButtons.nth(i).click({ timeout: 2000 });
          await page.waitForTimeout(200);
          const clip = await page.evaluate(async () => {
            try { return await navigator.clipboard.readText(); } catch { return ''; }
          });
          if (isLikelySolanaAddress(clip)) {
            const name = domRows[i] ? domRows[i].name : '';
            copiedRows.push({ name, wallet_address: clip.trim(), source: 'clipboard', extracted_at: nowIso() });
          }
        } catch {
          // continue
        }
      }

      finalRows = consolidateRows(copiedRows);
      if (!finalRows.length) {
        throw new Error('Failed to extract full wallet addresses from network and DOM/clipboard.');
      }
      log(`Selected extraction mode: clipboard. Valid wallets: ${finalRows.length}`);
    }

    const csv = toCsv(finalRows);
    fs.writeFileSync(CSV_PATH, csv, 'utf8');
    fs.writeFileSync(JSON_PATH, JSON.stringify(finalRows, null, 2), 'utf8');
    writeSqliteWithPython(finalRows, log);

    if (!fs.existsSync(DOM_SNAPSHOT_PATH)) {
      fs.writeFileSync(DOM_SNAPSHOT_PATH, await page.content(), 'utf8');
    }

    log(`Export completed. Rows exported: ${finalRows.length}`);
    flush();
  } catch (err) {
    try {
      fs.writeFileSync(LOG_PATH, `${fs.existsSync(LOG_PATH) ? fs.readFileSync(LOG_PATH, 'utf8') : ''}[${nowIso()}] ERROR: ${err.message}\n`, 'utf8');
    } catch {}
    console.error(`ERROR: ${err.message}`);
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
  }
}

run();
