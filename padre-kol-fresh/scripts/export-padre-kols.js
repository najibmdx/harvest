#!/usr/bin/env node

const fs = require('fs/promises');
const path = require('path');
const puppeteer = require('puppeteer-core');

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const OUT_DIR = path.join(__dirname, '..', 'output');
const FILES = {
  log: path.join(OUT_DIR, 'debug_run_log.txt'),
  snapshot: path.join(OUT_DIR, 'debug_dom_snapshot.html'),
  rows: path.join(OUT_DIR, 'debug_dom_rows.json'),
  csv: path.join(OUT_DIR, 'padre_kols.csv'),
  json: path.join(OUT_DIR, 'padre_kols.json')
};

async function writeOutputs({ runLogLines, snapshotHtml, rowDebug, accepted }) {
  const extractedAt = new Date().toISOString();
  const jsonRows = accepted.map((r) => ({
    name: r.name,
    wallet_address: r.wallet,
    source: 'button-id',
    extracted_at: extractedAt
  }));

  const csvHeader = 'name,wallet_address,source,extracted_at';
  const csvRows = jsonRows.map((r) => [r.name, r.wallet_address, r.source, r.extracted_at]
    .map((v) => `"${String(v).replace(/"/g, '""')}"`).join(','));

  await fs.writeFile(FILES.log, `${runLogLines.join('\n')}\n`, 'utf8');
  await fs.writeFile(FILES.snapshot, snapshotHtml || '', 'utf8');
  await fs.writeFile(FILES.rows, `${JSON.stringify(rowDebug, null, 2)}\n`, 'utf8');
  await fs.writeFile(FILES.csv, `${csvHeader}\n${csvRows.join('\n')}${csvRows.length ? '\n' : ''}`, 'utf8');
  await fs.writeFile(FILES.json, `${JSON.stringify(jsonRows, null, 2)}\n`, 'utf8');
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });

  const runLog = [];
  let snapshotHtml = '';
  const allDebugRows = [];
  const acceptedByWallet = new Map();
  let scrollAttempts = 0;
  let failureReason = null;
  let browser;

  try {
    const versionRes = await fetch('http://127.0.0.1:9222/json/version');
    if (!versionRes.ok) throw new Error(`Failed to fetch /json/version: HTTP ${versionRes.status}`);
    const version = await versionRes.json();
    const ws = version.webSocketDebuggerUrl;
    if (!ws) throw new Error('webSocketDebuggerUrl not found from Chrome DevTools endpoint.');

    browser = await puppeteer.connect({
      browserWSEndpoint: ws,
      defaultViewport: null,
      protocolTimeout: 120000
    });
    runLog.push('connected to Chrome: true');

    const pages = await browser.pages();
    let page = pages.find((p) => p.url().includes('trade.padre.gg/tracker'));
    if (!page) {
      page = await browser.newPage();
      await page.goto('https://trade.padre.gg/tracker', { waitUntil: 'domcontentloaded' });
    }

    const activeUrl = page.url();
    runLog.push(`active page URL: ${activeUrl}`);

    await sleep(1200);
    snapshotHtml = await page.evaluate(() => document.documentElement.outerHTML);
    const pageStats = await page.evaluate(() => ({
      hasKolsManager: document.body && document.body.innerText.includes('KOLs Manager'),
      zebraCount: document.querySelectorAll('[data-zebra]').length,
      walletTrackCount: document.querySelectorAll('[id^="button-wallet-track-select-"]').length,
      kolSettingsCount: document.querySelectorAll('[id^="button-kol-settings-"]').length
    }));

    runLog.push(`document.body.innerText includes "KOLs Manager": ${pageStats.hasKolsManager}`);
    runLog.push(`count [data-zebra]: ${pageStats.zebraCount}`);
    runLog.push(`count [id^="button-wallet-track-select-"]: ${pageStats.walletTrackCount}`);
    runLog.push(`count [id^="button-kol-settings-"]: ${pageStats.kolSettingsCount}`);

    if (!pageStats.hasKolsManager) {
      throw new Error('KOLs Manager not visible. Please log in manually and open KOLs Manager.');
    }

    const maxScrollAttempts = 200;
    let staleRounds = 0;

    while (scrollAttempts < maxScrollAttempts && staleRounds < 5) {
      const pass = await page.evaluate(() => {
        const isVisibleText = (el) => {
          if (!el) return false;
          const style = window.getComputedStyle(el);
          if (style.display === 'none' || style.visibility === 'hidden') return false;
          const rect = el.getBoundingClientRect();
          return rect.width > 0 && rect.height > 0;
        };

        const isLikelyWallet = (v) => /^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(v) && !/[.…]/.test(v);
        const rowFromEl = (el) => {
          const zebra = el.closest('[data-zebra]');
          if (zebra) return zebra;
          let cur = el;
          for (let i = 0; i < 8 && cur && cur.parentElement; i += 1) {
            cur = cur.parentElement;
            const txt = (cur.innerText || '').trim();
            if (txt.length >= 2) return cur;
          }
          return el.parentElement || el;
        };

        const forbidden = new Set(['Copy address', 'KOLs Manager', 'Track', 'Settings', 'Copy', 'Manage']);
        const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
        const chooseName = (row, wallet) => {
          const first = clean(row.querySelector('img[src*="/kols/"]')?.closest('div')?.parentElement?.querySelector('span')?.textContent);
          const good = (t) => {
            if (!t) return false;
            if (forbidden.has(t)) return false;
            if (/^\d+$/.test(t)) return false;
            if (t.length < 2 && !/^@?[A-Za-z0-9_]$/.test(t)) return false;
            if (isLikelyWallet(t)) return false;
            if (wallet && (wallet.includes(t) || t.includes(wallet))) return false;
            if (/address|wallet|copy|manager|search/i.test(t)) return false;
            return true;
          };
          if (good(first)) return first;

          const candidates = Array.from(row.querySelectorAll('span,div'))
            .filter((el) => isVisibleText(el))
            .map((el) => clean(el.textContent))
            .filter(Boolean);

          for (const c of candidates) {
            if (good(c)) return c;
          }
          return '';
        };

        const buttons = Array.from(document.querySelectorAll('[id^="button-wallet-track-select-"],[id^="button-kol-settings-"]'));
        const rows = [];

        buttons.forEach((el, idx) => {
          const id = el.id || '';
          const wallet = id.replace('button-wallet-track-select-', '').replace('button-kol-settings-', '').trim();
          const row = rowFromEl(el);
          const rawText = clean(row?.innerText || '');
          let reject = null;

          if (!isLikelyWallet(wallet)) reject = 'invalid_wallet';
          const candidate = chooseName(row, wallet);
          if (!reject && !candidate) reject = 'invalid_name';

          rows.push({
            row_index: idx,
            raw_text: rawText,
            candidate_name: candidate,
            wallet_from_button_id: wallet,
            accepted: !reject,
            reject_reason: reject,
            match_method: 'button-id'
          });
        });

        const accepted = rows
          .filter((r) => r.accepted)
          .map((r) => ({ wallet: r.wallet_from_button_id, name: r.candidate_name }));

        const findScrollable = () => {
          for (const b of buttons) {
            let p = b.parentElement;
            let hops = 0;
            while (p && hops < 12) {
              if (p.scrollHeight > p.clientHeight + 40) return p;
              p = p.parentElement;
              hops += 1;
            }
          }
          const zebras = Array.from(document.querySelectorAll('[data-zebra]'));
          for (const z of zebras) {
            let p = z.parentElement;
            let hops = 0;
            while (p && hops < 12) {
              if (p.scrollHeight > p.clientHeight + 40) return p;
              p = p.parentElement;
              hops += 1;
            }
          }
          return null;
        };

        const scroller = findScrollable();
        if (scroller) {
          scroller.scrollTop += Math.floor(scroller.clientHeight * 0.8);
        } else {
          window.scrollBy(0, Math.floor(window.innerHeight * 0.8));
        }

        return { rows, accepted };
      });

      allDebugRows.push(...pass.rows);
      const before = acceptedByWallet.size;
      for (const item of pass.accepted) {
        if (!acceptedByWallet.has(item.wallet)) acceptedByWallet.set(item.wallet, item.name);
      }
      const grew = acceptedByWallet.size > before;
      staleRounds = grew ? 0 : staleRounds + 1;
      scrollAttempts += 1;
      await sleep(700);
    }

    if ((pageStats.walletTrackCount + pageStats.kolSettingsCount) === 0) {
      throw new Error('Wallet button IDs not found. Make sure Padre KOLs Manager rows are visible.');
    }

    if (acceptedByWallet.size === 0 && (pageStats.walletTrackCount + pageStats.kolSettingsCount) > 0) {
      throw new Error('Wallet button IDs found but extraction failed. Check debug_dom_rows.json.');
    }
  } catch (err) {
    failureReason = err && err.message ? err.message : String(err);
  } finally {
    runLog.push(`scroll attempts: ${scrollAttempts}`);
    runLog.push(`total unique wallets extracted: ${acceptedByWallet.size}`);
    if (failureReason) runLog.push(`failure reason: ${failureReason}`);

    const accepted = Array.from(acceptedByWallet.entries()).map(([wallet, name]) => ({ wallet, name }));
    await writeOutputs({ runLogLines: runLog, snapshotHtml, rowDebug: allDebugRows, accepted });

    if (browser) await browser.disconnect();
  }

  if (failureReason) {
    console.error(failureReason);
    process.exit(1);
  }

  console.log(`Done. Extracted ${acceptedByWallet.size} wallets.`);
}

main();
