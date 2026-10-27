// Called only by the explicit installer after the user accepts plugin permissions.
const { withCodexConfig } = require('./codex-config.cjs');

async function configureCodexPermissions(options) {
  return withCodexConfig(async request => {
    const result = await request('config/value/write', {
      keyPath: 'plugins."tensorviz@tensorviz-standalone".mcp_servers.tensorviz.default_tools_approval_mode',
      value: 'approve', mergeStrategy: 'replace',
    });
    if (!['ok', 'okOverridden'].includes(result?.status)) throw new Error('Codex did not confirm TensorViz tool permission. Rerun the installer to finish setup.');
    return { status: result.status, scope: 'tensorviz@tensorviz-standalone/tensorviz',
      ...(result.status === 'okOverridden' ? { detail: 'A higher-priority client or organization policy still applies.' } : {}) };
  }, options);
}

module.exports = { configureCodexPermissions };
