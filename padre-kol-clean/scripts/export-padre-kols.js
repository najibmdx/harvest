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

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const BASE58_FULL_RE = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;
const FORBIDDEN_NAME_TOKENS = new Set(['copy address', 'copy', 'copied', 'wallet', 'address', 'open', 'view', 'edit', 'delete']);

function nowIso() { return new Date().toISOString(); }
function ensureOutput() { fs.mkdirSync(OUTPUT_DIR, { recursive: true }); }
function writeJson(file, data) { fs.writeFileSync(file, JSON.stringify(data, null, 2), 'utf8'); }

function isFullWallet(text) { return typeof text === 'string' && BASE58_FULL_RE.test(text.trim()); }
function isNumericName(v) { return typeof v === 'string' && /^\d+$/.test(v.trim()); }
function looksLikeId(v) { return typeof v === 'string' && /^\d{6,}$/.test(v.trim()); }

function isValidDisplayName(v) {
  if (typeof v !== 'string') return false;
  const name = v.trim();
  if (!name || isNumericName(name) || looksLikeId(name) || isFullWallet(name)) return false;
  if (FORBIDDEN_NAME_TOKENS.has(name.toLowerCase())) return false;
  if (/^[1-9A-HJ-NP-Za-km-z]{6,31}$/.test(name)) return false;
  return true;
}

function extractWalletsRecursively(value, outputSet) {
  if (typeof value === 'string') { if (isFullWallet(value.trim())) outputSet.add(value.trim()); return; }
  if (Array.isArray(value)) { for (const item of value) extractWalletsRecursively(item, outputSet); return; }
  if (value && typeof value === 'object') for (const v of Object.values(value)) extractWalletsRecursively(v, outputSet);
}

function toCsv(rows) {
  const esc = (v) => { const s = String(v ?? ''); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return `name,wallet_address,source,extracted_at\n${rows.map((r) => [esc(r.name), esc(r.wallet_address), esc(r.source), esc(r.extracted_at)].join(',')).join('\n')}\n`;
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
  if (!existing) await page.goto('https://trade.padre.gg/tracker', { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.bringToFront();
  await sleep(2000);
  log('tracker page opened');
  return page;
}

async function openKolsManager(page, log) {
  const handles = await page.$$('button, [role="button"], a, div, span');
  for (const h of handles) {
    const text = await page.evaluate((el) => (el.textContent || '').trim(), h);
    if (/kols manager/i.test(text)) {
      await h.click({ delay: 20 });
      await sleep(1500);
      log('KOLs Manager found');
      return;
    }
  }
  throw new Error('Could not locate "KOLs Manager" in visible UI. Open it manually and rerun.');
}

async function scrapeVisibleDomRows(page) {
  return page.evaluate(() => {
    const walletPrefixes = ['button-wallet-track-select-', 'button-kol-settings-'];
    const rows = Array.from(document.querySelectorAll('div[data-zebra]'));

    const pickName = (row, rawText, wallet) => {
      const spanTexts = Array.from(row.querySelectorAll('span,div,p')).map((n) => (n.textContent || '').replace(/\s+/g, ' ').trim()).filter(Boolean);
      const candidates = [...spanTexts, ...(rawText ? rawText.split(/\s{2,}|\n/).map((t) => t.trim()).filter(Boolean) : [])];
      for (const c of candidates) {
        if (!c) continue;
        if (/^copy address$/i.test(c)) continue;
        if (/^\d+$/.test(c)) continue;
        if (/^[1-9A-HJ-NP-Za-km-z]{6,31}$/.test(c)) continue;
        if (wallet && c.includes(wallet)) continue;
        if (c.length < 2) continue;
        return c;
      }
      return '';
    };

    return rows.map((row, idx) => {
      const rawText = (row.textContent || '').replace(/\s+/g, ' ').trim();
      const buttons = Array.from(row.querySelectorAll('[id]'));
      let wallet = '';
      for (const b of buttons) {
        const id = b.id || '';
        for (const prefix of walletPrefixes) {
          if (id.startsWith(prefix)) {
            wallet = id.slice(prefix.length).trim();
            if (wallet) break;
          }
        }
        if (wallet) break;
      }
      const fragment = (rawText.match(/[1-9A-HJ-NP-Za-km-z]{6,31}/g) || [])[0] || '';
      const candidateName = pickName(row, rawText, wallet);
      return { row_index: idx, raw_text: rawText, candidate_name: candidateName, wallet_from_button_id: wallet, visible_wallet_fragment: fragment, match_method: 'dom-button-id' };
    });
  });
}

async function main() {
  ensureOutput();
  const logs = [];
  const log = (m) => { const line = `[${nowIso()}] ${m}`; logs.push(line); console.log(line); };
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
        const ct = (response.headers()['content-type'] || '').toLowerCase();
        if (!ct.includes('application/json')) return;
        const data = await response.json();
        const before = networkWallets.size;
        extractWalletsRecursively(data, networkWallets);
        if (networkWallets.size > before) networkEvidence.push({ url: response.url(), status: response.status(), timestamp: nowIso() });
      } catch (_) {}
    });

    await openKolsManager(page, log);
    await sleep(2000);

    const mergedRows = [];
    const byWallet = new Map();
    let staleCount = 0;
    let prevScrollTop = -1;

    for (let i = 0; i < 120; i += 1) {
      const batch = await scrapeVisibleDomRows(page);
      let newWallets = 0;

      for (const row of batch) {
        const wallet = (row.wallet_from_button_id || '').trim();
        const name = (row.candidate_name || '').trim();
        if (!isFullWallet(wallet)) continue;
        if (!isValidDisplayName(name)) continue;
        if (!byWallet.has(wallet)) {
          byWallet.set(wallet, { ...row, row_index: mergedRows.length });
          mergedRows.push(byWallet.get(wallet));
          newWallets += 1;
        }
      }

      staleCount = newWallets === 0 ? staleCount + 1 : 0;

      const scrollResult = await page.evaluate(() => {
        const candidates = Array.from(document.querySelectorAll('div')).filter((el) => el.scrollHeight > el.clientHeight + 100);
        if (!candidates.length) return { found: false, scrollTop: 0, changed: false };
        candidates.sort((a, b) => b.clientHeight - a.clientHeight);
        const scroller = candidates[0];
        const before = scroller.scrollTop;
        scroller.scrollTop = Math.min(scroller.scrollTop + Math.max(300, Math.floor(scroller.clientHeight * 0.8)), scroller.scrollHeight);
        return { found: true, scrollTop: scroller.scrollTop, changed: scroller.scrollTop !== before };
      });

      if (!scrollResult.found || (!scrollResult.changed && scrollResult.scrollTop === prevScrollTop) || staleCount >= 6) break;
      prevScrollTop = scrollResult.scrollTop;
      await sleep(250);
    }

    fs.writeFileSync(PATHS.domSnapshot, await page.content(), 'utf8');
    writeJson(PATHS.networkCandidates, { extracted_at: nowIso(), wallet_count: networkWallets.size, wallets: Array.from(networkWallets), evidence: networkEvidence });
    log(`network candidates found: ${networkWallets.size}`);

    writeJson(PATHS.domRows, mergedRows);
    log(`DOM rows found: ${mergedRows.length}`);

    const exported = mergedRows.map((r) => ({
      name: r.candidate_name,
      wallet_address: r.wallet_from_button_id,
      source: 'dom-button-id',
      extracted_at: nowIso()
    }));

    log(`rows paired: ${exported.length}`);
    if (exported.length === 0) throw new Error('No confident name-wallet pairs extracted from virtualized DOM rows.');

    fs.writeFileSync(PATHS.csv, toCsv(exported), 'utf8');
    writeJson(PATHS.json, exported);
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

main().catch(() => { process.exitCode = 1; });
