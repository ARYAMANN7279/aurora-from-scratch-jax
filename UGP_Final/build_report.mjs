// Reproducible Markdown -> HTML -> PDF build. Requires pandoc, Chrome, Node with WebSocket.
// Usage: node build_report.mjs [output.pdf]   (UGP_Final_Report.md -> UGP_Final_Report.pdf)
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const work = path.join(root, '.build');
const profile = path.join(work, 'chrome-profile');
const html = path.join(work, 'report.html');
const output = path.resolve(process.argv[2] || path.join(root, 'UGP_Final_Report.pdf'));
const chrome = process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
await fs.mkdir(profile, { recursive: true });
execFileSync('pandoc', ['UGP_Final_Report.md', '-f', 'gfm+tex_math_dollars', '-t', 'html5', '--standalone', '--mathml',
  '--metadata', 'pagetitle=UGP Final Report', '--embed-resources', '--resource-path', root,
  '--css', 'report.css', '-o', html], { cwd: root, stdio: 'inherit' });
const activePort = path.join(profile, 'DevToolsActivePort');
await fs.rm(activePort, { force: true });
const browser = spawn(chrome, ['--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' });
let ws;
try {
  let port, endpoint;
  for (let i = 0; i < 200; i++) {
    try {
      [port, endpoint] = (await fs.readFile(activePort, 'utf8')).trim().split('\n');
      if (port && endpoint) break;
    } catch {}
    if (browser.exitCode !== null) throw new Error(`Chrome exited ${browser.exitCode}`);
    await new Promise(r => setTimeout(r, 100));
  }
  if (!port) throw new Error('Chrome debugger did not become ready');
  ws = new WebSocket(`ws://127.0.0.1:${port}${endpoint}`);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  let counter = 0;
  const pending = new Map();
  ws.onmessage = event => {
    const m = JSON.parse(event.data);
    if (pending.has(m.id)) {
      const p = pending.get(m.id);
      clearTimeout(p.timeout);
      pending.delete(m.id);
      m.error ? p.reject(new Error(JSON.stringify(m.error))) : p.resolve(m.result);
    }
  };
  const call = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++counter;
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`${method} timed out`)); }, 60000);
    pending.set(id, { resolve, reject, timeout });
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
  const { targetId } = await call('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await call('Target.attachToTarget', { targetId, flatten: true });
  await call('Page.enable', {}, sessionId);
  await call('Page.navigate', { url: pathToFileURL(html).href }, sessionId);
  let ready = false;
  for (let i = 0; i < 200; i++) {
    const r = await call('Runtime.evaluate', { expression: 'document.readyState === "complete"', returnByValue: true }, sessionId);
    if (r.result.value) { ready = true; break; }
    await new Promise(r => setTimeout(r, 100));
  }
  if (!ready) throw new Error('Report page did not finish loading');
  await call('Runtime.evaluate', { expression: 'document.fonts.ready.then(() => true)', awaitPromise: true }, sessionId);
  const check = await call('Runtime.evaluate', {
    expression: 'Array.from(document.images).filter(i => !i.complete || !i.naturalWidth).map(i => i.getAttribute("src").slice(0, 100))',
    returnByValue: true,
  }, sessionId);
  if (check.result.value.length) throw new Error(`Missing report images: ${JSON.stringify(check.result.value)}`);
  const pdf = await call('Page.printToPDF', { printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font:9px Georgia;width:100%;text-align:center;color:#555"><span class="pageNumber"></span></div>',
    paperWidth: 8.2677, paperHeight: 11.6929, marginTop: .67, marginBottom: .75,
    marginLeft: .63, marginRight: .63,
  }, sessionId);
  await fs.writeFile(output, Buffer.from(pdf.data, 'base64'));
  console.log(`Wrote ${output}`);
} finally {
  ws?.close();
  browser.kill('SIGTERM');
}
