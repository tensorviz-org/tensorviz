// Use Codex's supported API so comments, credentials and unrelated settings survive.
const { spawn } = require('node:child_process');
const { createInterface } = require('node:readline');

async function withCodexConfig(action, { command = 'codex', env = process.env, cwd = process.cwd() } = {}) {
  const child = spawn(command, ['app-server', '--stdio'], { env, cwd, stdio: ['pipe', 'pipe', 'ignore'] });
  const lines = createInterface({ input: child.stdout });
  const pending = new Map();
  const failure = () => new Error('Codex could not read or save its configuration. Check the official CLI and retry; existing connections are preserved.');
  let next = 0, stopped = false;
  const rejectAll = () => { stopped = true; for (const item of pending.values()) { clearTimeout(item.timer); item.reject(failure()); } pending.clear(); };
  child.on('error', rejectAll);
  child.on('exit', rejectAll);
  child.stdin.on('error', rejectAll);
  lines.on('line', line => {
    let message;
    try { message = JSON.parse(line); } catch { rejectAll(); return; }
    const item = pending.get(message.id);
    if (!item) return;
    pending.delete(message.id); clearTimeout(item.timer);
    if (message.error) item.reject(failure());
    else item.resolve(message.result);
  });
  const request = (method, params) => new Promise((resolve, reject) => {
    if (stopped) { reject(failure()); return; }
    const id = ++next;
    const timer = setTimeout(() => { pending.delete(id); reject(failure()); }, 15000);
    pending.set(id, { resolve, reject, timer });
    child.stdin.write(JSON.stringify({ id, method, params }) + '\n');
  });
  try {
    await request('initialize', { clientInfo: { name: 'tensorviz-installer', version: '1' } });
    child.stdin.write(JSON.stringify({ method: 'initialized' }) + '\n');
    return await action(request);
  } finally { rejectAll(); lines.close(); child.stdin.end(); child.kill(); }
}

module.exports = { withCodexConfig };
