const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mkdtemp, cp, rm, readFile, writeFile } = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');

test('a copied plugin discovers tools with no checkout, VS Code, Python, or PATH', async () => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), 'tensorviz-standalone-'));
  const plugin = path.join(temporary, 'plugin with spaces');
  await cp(path.resolve(__dirname, '../../../plugins/tensorviz'), plugin, { recursive: true });
  const { pluginVersion } = await import('../../scripts/release-version.mjs');
  const manifest = JSON.parse(await readFile(path.join(plugin, 'plugin.json'), 'utf8'));
  await writeFile(path.join(plugin, 'plugin.json'), JSON.stringify({ ...manifest, version: pluginVersion() }));
  const transport = new StdioClientTransport({ command: process.execPath,
    args: [path.join(plugin, 'runtime/host.cjs')], cwd: temporary, env: { PATH: '' }, stderr: 'pipe' });
  let diagnostics = '';
  transport.stderr?.on('data', chunk => { diagnostics += chunk; });
  const client = new Client({ name: 'standalone-discovery-test', version: '1' });
  try {
    await client.connect(transport);
    assert.equal(client.getServerVersion().title, 'tensorViz');
    const { tools } = await client.listTools();
    assert.deepEqual(tools.map(tool => tool.name).sort(), ['list_models', 'get_graph', 'inspect_nodes', 'get_selection',
      'focus_view', 'open_graph_view', 'open_graph', 'get_execution_environment', 'list_captures', 'inspect_capture',
      'get_model_context', 'find_computation', 'query_computation', 'analyze_computation_change',
      'get_graph_setup', 'suggest_graph_inputs', 'configure_graph_setup', 'configure_graph_environment',
      'validate_model', 'get_run', 'inspect_run', 'cancel_run', 'compare_runs', 'open_graph_diff',
      'open_graph_panel', 'graph_panel_read', 'graph_panel_exchange', 'detach_graph_panel', 'close_graph_panel'].sort());
    assert.equal(tools.find(tool => tool.name === 'list_models').annotations.readOnlyHint, true);
    for (const name of ['get_pending_permission', 'present_permission', 'get_machine_info',
      'read_project_file', 'request_project_command', 'get_project_command', 'submit_node_explanation']) {
      assert.equal(tools.some(tool => tool.name === name), false, `${name} belongs only to the private Canvas agent endpoint`);
    }
    const environment = tools.find(tool => tool.name === 'configure_graph_environment');
    assert.equal(environment.annotations.readOnlyHint, false);
    assert.equal(environment.annotations.destructiveHint, true);
    assert.equal(environment.annotations.openWorldHint, true);
    const unopenedEnvironment = await client.callTool({ name: 'configure_graph_environment', arguments: {
      openingId: 'unopened', sourceDigest: 'a'.repeat(64), action: 'select_environment', candidateId: 'unselected',
    } });
    assert.equal(unopenedEnvironment.structuredContent.error.code, 'NO_MODEL', 'environment configuration requires an existing opening and must not provision during discovery');
    for (const [name, query] of [
      ['get_model_context', { revision: 'a'.repeat(64) }],
      ['find_computation', { text: 'Linear' }],
      ['query_computation', { anchors: [{ kind: 'value', id: 'missing' }] }],
      ['analyze_computation_change', { changes: [{ valueId: 'missing', shape: [2, 4] }] }],
    ]) {
      const annotations = tools.find(tool => tool.name === name).annotations;
      assert.equal(annotations.readOnlyHint, true);
      assert.equal(annotations.destructiveHint, false);
      assert.equal(annotations.openWorldHint, false);
      const result = await client.callTool({ name, arguments: {
        modelId: 'unopened', captureId: 'missing', sourceId: 'missing', ...query,
      } });
      assert.equal(result.structuredContent.error.code, 'NO_MODEL', 'analysis must not open or run a model');
    }
    const result = await client.callTool({ name: 'list_models', arguments: {} });
    assert.equal(result.structuredContent.host.kind, 'standalone');
    assert.equal(result.structuredContent.workspaceSelectionRequired, true);
    assert.equal(result.structuredContent.capabilities.nativeEditor, false);
    assert.equal(result.structuredContent.capabilities.analytics, false);
    const opener = tools.find(tool => tool.name === 'open_graph_panel');
    assert.deepEqual(opener._meta['openai/ui'].entrypoints, [{ type: 'thread' }]);
    assert.equal(opener._meta['openai/ui'].preferredModelDisplayMode, 'fullscreen');
    const resource = await client.readResource({ uri: opener._meta.ui.resourceUri });
    assert.equal(resource.contents[0].mimeType, 'text/html;profile=mcp-app');
    assert.match(resource.contents[0].text, /<title>Model graph<\/title>/);
    assert.ok(Buffer.byteLength(resource.contents[0].text) <= 5 * 1024 * 1024, 'T3 Code must be able to snapshot the HTML resource');
    assert.ok(Buffer.byteLength(JSON.stringify(resource)) < 10 * 1024 * 1024, 'UI resources must fit the default MCP stdio limit');
    assert.deepEqual(resource.contents[0]._meta.ui.csp, { connectDomains: [], resourceDomains: [] });
    assert.deepEqual(tools.find(tool => tool.name === 'graph_panel_exchange')._meta.ui.visibility, ['app']);
    assert.equal(tools.find(tool => tool.name === 'graph_panel_exchange').annotations.readOnlyHint, false);
    assert.deepEqual(tools.find(tool => tool.name === 'graph_panel_read')._meta.ui.visibility, ['app']);
    assert.equal(tools.find(tool => tool.name === 'graph_panel_read').annotations.readOnlyHint, true);
    const opened = await client.callTool({ name: 'open_graph_panel', arguments: {} });
    assert.equal(opened.structuredContent.displayed, false);
    const panelId = opened._meta.panelId;
    const empty = await client.callTool({ name: 'graph_panel_read', arguments: { panelId, after: 0 } });
    assert.equal(empty._meta.panel.waitingForModel, true, 'opening the panel must not start Python');
    assert.deepEqual(empty.content, [], 'Canvas data must stay out of model tool content');
    await client.callTool({ name: 'close_graph_panel', arguments: { panelId } });
    const closed = await client.callTool({ name: 'graph_panel_exchange', arguments: { panelId, after: 0 } });
    assert.equal(closed.structuredContent.error.code, 'PANEL_CLOSED');
    assert.equal(diagnostics, '');
  } finally {
    await client.close();
    await rm(temporary, { recursive: true, force: true });
  }
});
