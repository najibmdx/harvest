#!/usr/bin/env node
/* eslint-disable no-console */
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer-core');

const OUTPUT_DIR = path.resolve(process.cwd(), 'output');
const PATHS = {
  runLog: path.join(OUTPUT_DIR, 'debug_run_log.txt'),
  networkCandidates: path.join(OUTPUT_DIR, 'debug_network_candidates.json'),
  domSnapshot: path.join(OUTPUT_DIR, 'debug_dom_snapshot.html'),
  domRows: path.join(OUTPUT_DIR, 'debug_dom_rows.json'),
  csv: path.join(OUTPUT_DIR, 'padre_kols.csv'),
  json: path.join(OUTPUT_DIR, 'padre_kols.json')
};

const FORBIDDEN_NAME_TOKENS = new Set(['copy', 'copied', 'wallet', 'address', 'open', 'view', 'edit', 'delete']);
const BASE58_FULL_RE = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;
const BASE58_FRAGMENT_RE = /[1-9A-HJ-NP-Za-km-z]{6,}/g;

function nowIso() { return new Date().toISOString(); }
function ensureOutput() { fs.mkdirSync(OUTPUT_DIR, { recursive: true }); }
function writeJson(file, data) { fs.writeFileSync(file, JSON.stringify(data, null, 2), 'utf8'); }

function isBase58Like(text) {
  return typeof text === 'string' && /^[1-9A-HJ-NP-Za-km-z]+$/.test(text.trim());
}

function isFullWallet(text) {
  if (typeof text !== 'string') return false;
  const v = text.trim();
  return BASE58_FULL_RE.test(v);
}

function isNumericName(v) {
  return typeof v === 'string' && /^\d+$/.test(v.trim());
}

function isValidDisplayName(v) {
  if (typeof v !== 'string') return false;
  const name = v.trim();
  if (!name) return false;
  if (isNumericName(name)) return false;
  if (isFullWallet(name)) return false;
  if (isBase58Like(name) && name.length >= 32) return false;
  if (FORBIDDEN_NAME_TOKENS.has(name.toLowerCase())) return false;
  return true;
}

function extractWalletsRecursively(value, outputSet) {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (isFullWallet(trimmed)) outputSet.add(trimmed);
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) extractWalletsRecursively(item, outputSet);
    return;
  }
  if (value && typeof value === 'object') {
    for (const val of Object.values(value)) extractWalletsRecursively(val, outputSet);
  }
}

function toCsv(rows) {
  const esc = (v) => {
    const s = String(v ?? '');
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const header = 'name,wallet_address,source,extracted_at';
  const body = rows.map((r) => [esc(r.name), esc(r.wallet_address), esc(r.source), esc(r.extracted_at)].join(','));
  return `${header}\n${body.join('\n')}\n`;
}

async function getWsEndpoint() {
  const res = await fetch('http://127.0.0.1:9222/json/version');
  if (!res.ok) throw new Error(`Failed to fetch CDP version endpoint: HTTP ${res.status}`);
  const data = await res.json();
  if (!data.webSocketDebuggerUrl) throw new Error('webSocketDebuggerUrl missing from /json/version');
  return data.webSocketDebuggerUrl;
}

async function openOrReuseTrackerPage(browser, log) {
  const pages = await browser.pages();
  const existing = pages.find((p) => (p.url() || '').includes('trade.padre.gg/tracker'));
  const page = existing || await browser.newPage();
  if (!existing) {
    await page.goto('https://trade.padre.gg/tracker', { waitUntil: 'domcontentloaded', timeout: 120000 });
  }
  await page.bringToFront();
  await page.waitForTimeout(2000);
  log('tracker page opened');
  return page;
}

async function openKolsManager(page, log) {
  const buttonHandles = await page.$$('button, [role="button"], a, div, span');
  for (const handle of buttonHandles) {
    const text = await page.evaluate((el) => (el.textContent || '').trim(), handle);
    if (/^kols manager$/i.test(text) || /kols manager/i.test(text)) {
      await handle.click({ delay: 20 });
      await page.waitForTimeout(1500);
      log('KOLs Manager found');
      return;
    }
  }
  throw new Error('Could not locate "KOLs Manager" in visible UI. Open it manually and rerun.');
}

async function main() {
  ensureOutput();
  const logs = [];
  const log = (m) => {
    const line = `[${nowIso()}] ${m}`;
    logs.push(line);
    console.log(line);
  };

  let browser;
  const networkWallets = new Set();
  const networkEvidence = [];

  try {
    const ws = await getWsEndpoint();
    browser = await puppeteer.connect({ browserWSEndpoint: ws, defaultViewport: null, protocolTimeout: 120000 });
    log('connected to Chrome');

    const page = await openOrReuseTrackerPage(browser, log);

    page.on('response', async (response) => {
      try {
        const headers = response.headers();
        const ct = headers['content-type'] || headers['Content-Type'] || '';
        if (!ct.includes('application/json')) return;
        const data = await response.json();
        const walletsBefore = networkWallets.size;
        extractWalletsRecursively(data, networkWallets);
        const walletsAfter = networkWallets.size;
        if (walletsAfter > walletsBefore) {
          networkEvidence.push({
            url: response.url(),
            status: response.status(),
            newly_found_count: walletsAfter - walletsBefore,
            timestamp: nowIso()
          });
        }
      } catch (_) {}
    });

    await openKolsManager(page, log);
    await page.waitForTimeout(15000);

    const networkPayload = {
      extracted_at: nowIso(),
      wallet_count: networkWallets.size,
      wallets: Array.from(networkWallets),
      evidence: networkEvidence
    };
    writeJson(PATHS.networkCandidates, networkPayload);
    log(`network candidates found: ${networkWallets.size}`);

    const html = await page.content();
    fs.writeFileSync(PATHS.domSnapshot, html, 'utf8');

    const domRows = await page.evaluate(() => {
      const rowSelectors = ['tr', '[role="row"]', '.row', '.table-row', '[class*="row"]'];
      const rows = [];
      const seen = new Set();
      for (const sel of rowSelectors) {
        const nodes = document.querySelectorAll(sel);
        for (const el of nodes) {
          const rawText = (el.textContent || '').replace(/\s+/g, ' ').trim();
          if (!rawText || rawText.length < 6) continue;
          if (seen.has(rawText)) continue;
          seen.add(rawText);

          const tokens = rawText.split(' ').filter(Boolean);
          const candidateName = tokens.find((t) => t && !/^\d+$/.test(t)) || '';
          const fragments = rawText.match(/[1-9A-HJ-NP-Za-km-z]{6,}/g) || [];
          const visibleWalletFragment = fragments.find((f) => f.length >= 6 && f.length < 32) || '';

          const clickable = Array.from(el.querySelectorAll('button,[role="button"],svg,[class*="copy"], [aria-label*="copy" i]'));
          const copyTargetIndex = clickable.findIndex((node) => {
            const label = `${node.getAttribute('aria-label') || ''} ${node.textContent || ''}`.toLowerCase();
            return label.includes('copy') || label.includes('wallet');
          });

          rows.push({ raw_text: rawText, candidate_name: candidateName, visible_wallet_fragment: visibleWalletFragment, copy_target_index: copyTargetIndex });
        }
      }
      return rows;
    });

    const mappedRows = [];
    const exported = [];
    const wallets = Array.from(networkWallets);

    for (let i = 0; i < domRows.length; i += 1) {
      const r = domRows[i];
      const row = {
        row_index: i,
        raw_text: r.raw_text,
        candidate_name: isValidDisplayName(r.candidate_name) ? r.candidate_name : '',
        visible_wallet_fragment: r.visible_wallet_fragment || '',
        clipboard_wallet: '',
        matched_network_wallet: '',
        match_method: 'none'
      };

      if (row.candidate_name && row.visible_wallet_fragment && row.visible_wallet_fragment.length >= 6) {
        const match = wallets.find((w) => w.startsWith(row.visible_wallet_fragment) || w.includes(row.visible_wallet_fragment));
        if (match) {
          row.matched_network_wallet = match;
          row.match_method = 'fragment';
        }
      }

      if (!row.matched_network_wallet && row.candidate_name) {
        const copied = await page.evaluate(async (rowText) => {
          const all = Array.from(document.querySelectorAll('tr,[role="row"],.row,.table-row,[class*="row"]'));
          const rowEl = all.find((el) => ((el.textContent || '').replace(/\s+/g, ' ').trim() === rowText));
          if (!rowEl) return '';
          const candidates = Array.from(rowEl.querySelectorAll('button,[role="button"],[aria-label*="copy" i],[class*="copy" i]'));
          const target = candidates.find((el) => {
            const label = `${el.getAttribute('aria-label') || ''} ${el.textContent || ''}`.toLowerCase();
            return label.includes('copy') || label.includes('wallet');
          });
          if (!target) return '';
          target.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
          await new Promise((resolve) => setTimeout(resolve, 300));
          if (!navigator.clipboard || !navigator.clipboard.readText) return '';
          try {
            return (await navigator.clipboard.readText()).trim();
          } catch {
            return '';
          }
        }, r.raw_text);

        if (isFullWallet(copied)) {
          row.clipboard_wallet = copied;
          row.match_method = 'clipboard';
        }
      }

      if (row.match_method === 'fragment') {
        exported.push({ name: row.candidate_name, wallet_address: row.matched_network_wallet, source: 'network+dom', extracted_at: nowIso() });
      } else if (row.match_method === 'clipboard') {
        exported.push({ name: row.candidate_name, wallet_address: row.clipboard_wallet, source: 'clipboard', extracted_at: nowIso() });
      }

      mappedRows.push(row);
    }

    writeJson(PATHS.domRows, mappedRows);
    log(`DOM rows found: ${mappedRows.length}`);

    const dedup = [];
    const seenWallet = new Set();
    for (const item of exported) {
      if (!isValidDisplayName(item.name)) continue;
      if (!isFullWallet(item.wallet_address)) continue;
      if (seenWallet.has(item.wallet_address)) continue;
      seenWallet.add(item.wallet_address);
      dedup.push(item);
    }

    log(`rows paired: ${dedup.length}`);

    if (dedup.length === 0) {
      throw new Error('No confident name-wallet pairs extracted. Check debug files and ensure KOLs Manager rows are visible.');
    }

    if (dedup.every((x) => !x.name || isNumericName(x.name))) {
      throw new Error('Extracted names are empty or numeric IDs only. Failing to prevent invalid export.');
    }

    fs.writeFileSync(PATHS.csv, toCsv(dedup), 'utf8');
    writeJson(PATHS.json, dedup);
    log('CSV written');
    log('JSON written');
  } catch (err) {
    log(`failures: ${err.message}`);
    throw err;
  } finally {
    fs.writeFileSync(PATHS.runLog, `${logs.join('\n')}\n`, 'utf8');
    if (browser) await browser.disconnect();
  }
}

main().catch(() => {
  process.exitCode = 1;
});
