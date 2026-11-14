/** Poll only readiness; never retry an explicit server/discovery failure. */
export async function waitForInstalledMcp(fetchStatus, {timeoutMs = 30000, intervalMs = 250,
  now = Date.now, sleep = ms => new Promise(resolve => setTimeout(resolve, ms)), onStatus = () => {}} = {}) {
  const deadline = now() + timeoutMs;
  const serverName = 'tensorviz';
  while (now() < deadline) {
    const status = await fetchStatus({serverName, remainingMs: deadline - now()});
    await onStatus(status);
    if (now() >= deadline) throw new Error('Installed MCP readiness exceeded the 30-second deadline');
    if (!Array.isArray(status?.data) || status.data.length !== 1) throw new Error('Expected exactly one installed MCP server');
    const server = status.data[0];
    if (server.name !== serverName) throw new Error('Installed MCP server name changed');
    if (server.toolsError || ['failed', 'cancelled', 'disabled', 'authenticationRequired'].includes(server.runtimeStatus))
      throw new Error(`Installed MCP startup failed: ${server.toolsError || server.runtimeStatus}`);
    if (Object.values(server.tools ?? {}).some(tool => tool.name === 'open_graph')) return server;
    if (![null, undefined, 'notStarted', 'starting'].includes(server.runtimeStatus))
      throw new Error(`Installed MCP has no expected tools in terminal state: ${server.runtimeStatus}`);
    await sleep(Math.min(intervalMs, Math.max(0, deadline - now())));
  }
  throw new Error('Installed MCP tools were not ready within 30 seconds');
}
