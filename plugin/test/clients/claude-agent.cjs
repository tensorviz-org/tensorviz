// Opt-in ordinary model conversation using the machine's authorized Claude login.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { createInterface } = require('node:readline');

test('Claude completes an ordinary model conversation with the packaged plugin', { timeout: 300000 }, async () => {
  assert.equal(process.env.TENSORVIZ_TEST_CLAUDE_AGENT, '1', 'Explicitly opt in to this authenticated model test.');
  const python = process.env.TENSORVIZ_TEST_PYTHON;
  assert.ok(python && path.isAbsolute(python), 'Choose an existing compatible interpreter.');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-claude-agent-')));
  const project = path.join(root, 'project'), plugin = path.join(root, 'plugin');
  const pending = new Map(), calls = [];
  let child, lines, nextId = 0, result, approvals = 0;
  try {
    await fs.mkdir(project);
    await fs.cp(process.env.TENSORVIZ_TEST_PLUGIN ?? path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
    await fs.writeFile(path.join(project, 'model.py'), 'from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x):\n  return self.fc(x)\n');
    await fs.writeFile(path.join(project, 'tensorviz.toml'), 'version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = { args = [], kwargs = {} }\ninputs = { args = [{kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {} }\n');
    child = spawn(process.env.TENSORVIZ_TEST_CLAUDE || 'claude', ['-p', '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
      '--no-session-persistence', '--setting-sources', 'user', '--plugin-dir', plugin,
      '--allowedTools', 'Read,Skill,mcp__plugin_tensorviz_tensorviz__*', '--disallowedTools', 'Bash,Write,Edit,Agent',
      '--append-system-prompt', 'Work only on the synthetic project and the supplied plugin. Do not install packages, edit files, read credentials, access unrelated projects or delegate work.'],
    { env: { ...process.env, TENSORVIZ_DATA_DIR: path.join(root, 'state') }, cwd: project, detached: true, stdio: ['pipe', 'pipe', 'pipe'] });
    child.stderr.on('data', () => {});
    lines = createInterface({ input: child.stdout });
    lines.on('line', line => {
      let message; try { message = JSON.parse(line); } catch { return; }
      if (message.type === 'control_response' && pending.has(message.response.request_id)) {
        const request = pending.get(message.response.request_id);
        clearTimeout(request.timer); pending.delete(message.response.request_id);
        message.response.subtype === 'error' ? request.reject(new Error(message.response.error)) : request.resolve(message.response.response);
      } else if (message.type === 'control_request') {
        const request = message.request;
        if (request.subtype === 'elicitation' && request.mode === 'form' && request.message.includes(project)
          && request.message.includes(python) && /inspect environment|capture/.test(request.message)) {
          approvals++;
          child.stdin.write(JSON.stringify({ type: 'control_response', response: { subtype: 'success', request_id: message.request_id,
            response: { action: 'accept', content: { confirm: true } } } }) + '\n');
        } else {
          console.log('Unexpected approval kind:', request.subtype);
          child.stdin.write(JSON.stringify({ type: 'control_response', response: { subtype: 'error', request_id: message.request_id,
            error: 'Only the exact synthetic capture is authorized by this test.' } }) + '\n');
        }
      } else if (message.type === 'assistant') {
        for (const item of message.message?.content ?? []) if (item.type === 'tool_use') {
          calls.push(item.name); console.log('Claude tool:', item.name);
        }
      } else if (message.type === 'result') result = message;
    });
    const request = data => new Promise((resolve, reject) => {
      const id = String(++nextId), timer = setTimeout(() => { pending.delete(id); reject(new Error(`${data.subtype} timed out`)); }, 30000);
      pending.set(id, { resolve, reject, timer });
      child.stdin.write(JSON.stringify({ type: 'control_request', request_id: id, request: data }) + '\n');
    });
    await request({ subtype: 'initialize' });
    const prompt = `Use tensorViz to explain the tensor flow through Tiny in ${project}/model.py and verify its measured input and output shapes using tensorviz.toml. Use the existing interpreter ${python}. Do not edit files or install packages. A browser is not attached to this model test; report measured evidence without claiming visual display.`;
    child.stdin.write(JSON.stringify({ type: 'user', message: { role: 'user', content: prompt }, parent_tool_use_id: null, session_id: '' }) + '\n');
    const deadline = Date.now() + 240000;
    while (!result && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 200));
    assert.ok(result, 'Claude did not finish the model conversation.');
    assert.equal(result.is_error, false, result.result);
    assert.ok(calls.some(name => name.endsWith('__open_graph')), 'The model must request a capture.');
    assert.ok(calls.some(name => name.endsWith('__get_graph')), 'The model must inspect measured evidence.');
    assert.match(result.result, /2\s*[,×x]\s*4/);
    assert.match(result.result, /2\s*[,×x]\s*3/);
    assert.equal(approvals, 1, 'The fixture must use exactly one native TensorViz permission.');
    console.log('Ordinary Claude conversation captured Tiny and reported measured [2, 4] -> [2, 3].');
  } finally {
    for (const request of pending.values()) clearTimeout(request.timer);
    lines?.close();
    if (child?.pid) try { process.kill(-child.pid, 'SIGTERM'); } catch {}
    await fs.rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }
});
