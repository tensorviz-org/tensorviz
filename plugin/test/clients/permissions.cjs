const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { configureCodexPermissions } = require('../../../plugins/tensorviz/scripts/codex-permissions.cjs');

test('Codex saves only the installed TensorViz server policy through its supported API', { timeout: 45000 }, async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-client-permissions-'));
  const file = path.join(root, 'config.toml');
  const config = '# preserve my choices\napproval_policy = "on-request"\nsandbox_mode = "workspace-write"\n[plugins."another@market"]\nenabled = false\n[plugins."tensorviz@tensorviz-standalone"]\nenabled = true\n[plugins."tensorviz@tensorviz-standalone".mcp_servers.tensorviz.tools.inspect_graph]\napproval_mode = "prompt"\n';
  try {
    await fs.writeFile(file, config);
    const result = await configureCodexPermissions({ command: process.env.TENSORVIZ_TEST_CODEX || 'codex', env: { ...process.env, CODEX_HOME: root } });
    assert.equal(result.status, 'ok');
    const after = await fs.readFile(file, 'utf8');
    assert.match(after, /# preserve my choices/);
    assert.match(after, /approval_policy = "on-request"/);
    assert.match(after, /sandbox_mode = "workspace-write"/);
    assert.match(after, /\[plugins\."another@market"\]\nenabled = false/);
    assert.match(after, /\[plugins\."tensorviz@tensorviz-standalone"\.mcp_servers\.tensorviz\]\ndefault_tools_approval_mode = "approve"/);
    assert.match(after, /tools\.inspect_graph\]\napproval_mode = "prompt"/);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
