import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import { writeFileSync } from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { prepareAgentConfig, applyAgentConfig, readAgentConfig, agentConfigFile, setUpMcpAgent } from '../../mcpAgentSetup';
import * as vscode from '../vscodeStub';

suite('approved user-level agent setup', () => {
  const url = 'http://127.0.0.1:8765/mcp';
  let directory: string;
  setup(async () => { vscode.reset(); directory = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-agent-config-')); });
  teardown(async () => { await fs.rm(directory, { recursive: true, force: true }); vscode.reset(); });
  test('Codex append preserves comments, unrelated servers and quoted TOML keys', async () => {
    const before = '# retain me\nmodel="test"\n[mcp_servers."another.server"]\nurl="https://example.com"\n';
    const plan = await prepareAgentConfig('Codex', '/unused', before, url);
    assert.ok(plan.after.startsWith(before)); assert.match(plan.after, /\[mcp_servers.tensorviz\]/);
    assert.equal((await prepareAgentConfig('Codex', '/unused', plan.after, url)).after, plan.after);
    await assert.rejects(prepareAgentConfig('Codex', '/unused', '[mcp_servers."tensorviz"]\ncommand="other"', url), /already exists/);
  });
  test('Claude user scope preserves project settings and explicitly uses HTTP', async () => {
    const before = '{"projects":{"/repo":{"mcpServers":{"a":{"url":"https://example.com"}}}},"other":42}';
    const plan = await prepareAgentConfig('Claude Code', '/unused', before, url);
    const parsed = JSON.parse(plan.after);
    assert.deepEqual(parsed.projects, JSON.parse(before).projects); assert.equal(parsed.other, 42);
    assert.deepEqual(parsed.mcpServers.tensorviz, { type: 'http', url });
    assert.equal(agentConfigFile('Claude Code', '/temporary', {}), '/temporary/.claude.json');
  });
  test('Cursor comments survive; malformed or conflicting configuration is refused', async () => {
    const plan = await prepareAgentConfig('Cursor', '/unused', '{\n// keep\n"other": 3,\n}', url);
    assert.match(plan.after, /\/\/ keep/); assert.match(plan.after, /"other": 3/);
    await assert.rejects(prepareAgentConfig('Cursor', '/unused', '{oops', url));
    await assert.rejects(prepareAgentConfig('Cursor', '/unused', '{"mcpServers":{"tensorviz":{"url":"https://elsewhere"}}}', url), /already exists/);
    await assert.rejects(prepareAgentConfig('Codex', '/unused', 'broken="', url));
  });
  test('writes exactly the reviewed change and refuses intervening edits or symlinks', async () => {
    const file = path.join(directory, 'mcp.json');
    const plan = await prepareAgentConfig('Cursor', file, null, url);
    await applyAgentConfig(plan); assert.equal(await readAgentConfig(file), plan.after);
    await assert.rejects(applyAgentConfig(plan), /changed during review/);
    const link = path.join(directory, 'link.json'); await fs.symlink(file, link);
    await assert.rejects(readAgentConfig(link), /non-regular/);
  });
  test('a changed local Codex port updates only the server URL and preserves comments and other settings', async () => {
    const old = 'http://127.0.0.1:8849/mcp';
    for (const before of [
      `# \"${old}\"\n[mcp_servers.\"tensorviz\"]\nurl = '${old}' # keep this\nenabled = true\n[mcp_servers.other]\nurl = \"${old}\"\n`,
      `[mcp_servers]\ntensorviz = { url = \"${old}\", startup_timeout_sec = 20 }\n`,
    ]) {
      const plan = await prepareAgentConfig('Codex', '/unused', before, url);
      assert.equal(plan.previousUrl, old);
      const { parse } = await import('smol-toml');
      const expected = parse(before) as any; expected.mcp_servers.tensorviz.url = url;
      assert.deepEqual(parse(plan.after), expected);
      assert.ok(plan.after.includes('# keep this') || plan.after.includes('startup_timeout_sec = 20'));
      if (before.startsWith('#')) assert.ok(plan.after.startsWith(`# \"${old}\"`));
    }
  });
  for (const agent of ['Claude Code', 'Cursor'] as const) test(`${agent} updates a local port without replacing server options`, async () => {
    const entry = { url: 'http://127.0.0.1:8849/mcp', type: 'http', headers: { 'X-Test': 'fixture' } };
    const plan = await prepareAgentConfig(agent, '/unused', JSON.stringify({ mcpServers: { tensorviz: entry }, keep: 42 }), url);
    assert.deepEqual(JSON.parse(plan.after), { mcpServers: { tensorviz: { ...entry, url } }, keep: 42 });
    assert.equal(plan.previousUrl, entry.url);
    await assert.rejects(prepareAgentConfig(agent, '/unused', JSON.stringify({ mcpServers: { tensorviz: { ...entry, enabled: false } } }), url), /disabled/);
  });
  test('a local port update needs explicit approval and leaves the old entry on dismissal', async () => {
    const file = path.join(directory, 'config.toml');
    const before = '[mcp_servers.tensorviz]\nurl="http://127.0.0.1:8849/mcp"\n';
    await fs.writeFile(file, before);
    vscode.state.quickPickAnswers = ['Codex'];
    let guided = false;
    vscode.setInfoResponder((_message, items) => { assert.deepEqual(items, ['Update TensorViz']); return undefined; });
    await setUpMcpAgent(url, () => { guided = true; }, () => file);
    assert.equal(await readAgentConfig(file), before); assert.equal(guided, false);
    vscode.state.quickPickAnswers = ['Codex'];
    vscode.setInfoResponder(() => 'Update TensorViz');
    await setUpMcpAgent(url, () => { guided = true; }, () => file);
    assert.ok((await readAgentConfig(file))!.includes(url)); assert.equal(guided, true);
  });
  test('dismissing setup consent writes no configuration', async () => {
    vscode.state.quickPickAnswers = ['Codex'];
    let asked = false;
    vscode.setInfoResponder(message => { asked = message.startsWith('Add TensorViz'); return undefined; });
    const file = path.join(directory, 'config.toml');
    let guideShown = false;
    await setUpMcpAgent(url, () => { guideShown = true; }, () => file);
    assert.equal(await readAgentConfig(file), null);
    assert.equal(asked, true);
    assert.equal(guideShown, false);
    assert.deepEqual(vscode.state.errorMessages, []);
  });
  test('explicit approval installs the reviewed entry in the selected user file', async () => {
    vscode.state.quickPickAnswers = ['Claude Code'];
    vscode.setInfoResponder(message => message.startsWith('Add TensorViz') ? 'Add TensorViz' : undefined);
    const file = path.join(directory, '.claude.json');
    let guideShown = false;
    await setUpMcpAgent(url, () => { guideShown = true; }, () => file);
    assert.deepEqual(JSON.parse((await readAgentConfig(file))!).mcpServers.tensorviz, { type: 'http', url });
    assert.equal(guideShown, true, 'approved setup explains how to continue with the selected model');
    assert.deepEqual(vscode.state.errorMessages, []);
  });
  for (const agent of ['Codex', 'Claude Code']) {
    test(`${agent} setup only offers the VS Code MCP connection`, async () => {
      vscode.state.quickPickAnswers = [agent];
      vscode.setInfoResponder((_message, items) => {
        assert.deepEqual(items, ['Add TensorViz'], 'extension setup must not install a second plugin');
        return 'Add TensorViz';
      });
      const file = path.join(directory, 'settings');
      let guidedAgent: unknown;
      await setUpMcpAgent(url, selected => { guidedAgent = selected; }, () => file);
      assert.ok((await readAgentConfig(file))!.includes(url));
      assert.equal(guidedAgent, agent, 'the configured connection should skip duplicate manual URL setup');
    });
  }
  test('an existing matching connection opens guidance without rewriting settings or asking again', async () => {
    vscode.state.quickPickAnswers = ['Codex'];
    const file = path.join(directory, 'config.toml');
    const before = `[mcp_servers.tensorviz]\nurl="${url}"\n# retained\n`;
    await fs.writeFile(file, before);
    let consentAsked = false, guidedAgent: unknown;
    vscode.setInfoResponder(message => { if (message.startsWith('Add TensorViz')) consentAsked = true; return undefined; });
    await setUpMcpAgent(url, selected => { guidedAgent = selected; }, () => file);
    assert.equal(await fs.readFile(file, 'utf8'), before);
    assert.equal(consentAsked, false);
    assert.equal(guidedAgent, 'Codex');
  });
  test('settings changed during MCP consent are preserved and prevent success guidance', async () => {
    vscode.state.quickPickAnswers = ['Codex'];
    const file = path.join(directory, 'config.toml');
    vscode.setInfoResponder(() => { writeFileSync(file, '# user edit'); return 'Add TensorViz'; });
    let guided = false;
    await setUpMcpAgent(url, () => { guided = true; }, () => file);
    assert.equal(guided, false);
    assert.equal(await readAgentConfig(file), '# user edit');
    assert.match(vscode.state.errorMessages[0], /changed during review/);
  });
});
