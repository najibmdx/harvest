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
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
  }
}

run();
