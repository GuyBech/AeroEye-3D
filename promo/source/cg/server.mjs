// Local-only frame sink: serves the renderer page, stores what it posts, then quits.
//   node server.mjs "<job>" [port]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const job = process.argv[2];
const port = Number(process.argv[3] || 8173);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.frag': 'text/plain' };
let chrome;

const read = req => new Promise(r => { const c = []; req.on('data', d => c.push(d)); req.on('end', () => r(Buffer.concat(c))); });

const srv = http.createServer(async (req, res) => {
  const u = new URL(req.url, 'http://x');
  if (req.method === 'GET') {
    const f = path.join(HERE, u.pathname === '/' ? 'index.html' : u.pathname.slice(1));
    if (!fs.existsSync(f)) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'content-type': types[path.extname(f)] || 'text/plain', 'cache-control': 'no-store' });
    return res.end(fs.readFileSync(f));
  }
  const body = await read(req);
  const p = u.pathname.split('/');
  if (p[1] === 'log') console.log(body.toString());
  else if (p[1] === 'f') { fs.mkdirSync(path.join(HERE, 'frames', p[2]), { recursive: true }); fs.writeFileSync(path.join(HERE, 'frames', p[2], p[3] + '.jpg'), body); }
  else if (p[1] === 'm') { fs.mkdirSync(path.join(HERE, 'frames', p[2]), { recursive: true }); fs.writeFileSync(path.join(HERE, 'frames', p[2], 'meta.json'), body); }
  else if (p[1] === 'done') { res.end('ok'); chrome?.kill(); srv.close(); setTimeout(() => process.exit(0), 300); return; }
  res.end('ok');
});

srv.listen(port, '127.0.0.1', () => {
  const url = `http://127.0.0.1:${port}/?job=${encodeURIComponent(job)}`;
  chrome = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', [
    '--headless=new', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--enable-unsafe-swiftshader=false', '--force_high_performance_gpu',
    '--disable-gpu-watchdog', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
    '--no-first-run', '--no-default-browser-check', `--user-data-dir=${path.join(HERE, '.profile')}`, '--window-size=1920,1080', url,
  ], { stdio: 'ignore' });
  chrome.on('exit', c => console.log('browser exit', c));
});
setTimeout(() => { console.log('TIMEOUT'); chrome?.kill(); process.exit(2); }, Number(process.env.LIMIT_MS || 900000));
