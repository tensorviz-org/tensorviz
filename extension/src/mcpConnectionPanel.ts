import { randomBytes } from 'node:crypto';
import * as path from 'node:path';
import * as vscode from 'vscode';
import type { LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import { graphMcpHelpRequest, type GraphMcpHelpContext } from '@tensorviz/contracts/graphHelpRequest';
import type { AgentKind } from './mcpAgentSetup';
import type { McpStatus } from './mcpLifecycle';

const VIEW_TYPE = 'tensorviz.mcpConnection';

let panel: vscode.WebviewPanel | undefined;
let currentUrl = '';
let currentPrompt = '';
let currentTargetPrompt = '';
let currentHelp: GraphMcpHelpContext | undefined;
let currentHelpPrompt = '';
let currentLocation: 'local' | 'ssh' | 'desktop' = 'local';
let currentAgent: AgentKind | undefined;
let currentStatus: McpStatus | undefined;

/** Writing agent settings does not prove that an agent connected. */
export function updateMcpConnectionStatus(status: McpStatus): void {
  currentStatus = status;
  if (currentLocation !== 'local' || !panel) return;
  const changedPort = status.url !== undefined && status.url !== currentUrl;
  const unavailable = status.state === 'Off' || changedPort;
  const connected = !unavailable && status.state === 'Connected';
  void panel.webview.postMessage({ type: 'connection', repair: unavailable,
    title: changedPort ? 'The MCP address changed' : unavailable ? 'Graph tools are unavailable'
      : connected ? 'An agent is connected' : 'Waiting for your agent',
    detail: unavailable ? 'Run setup again to restore the VS Code MCP connection.'
      : connected ? 'TensorViz has received an agent connection. If your current chat still lacks tools, reconnect MCP in that agent or start a new chat.'
      : currentAgent ? 'The VS Code MCP connection is configured. Start a new agent chat with VS Code open. If tools are still missing, reconnect MCP in your agent or restart it. No URL setup is needed.'
        : 'TensorViz is ready to receive an agent connection. Set up your coding agent below, or reconnect the existing MCP connection in your agent.',
  });
}

export function mcpSetupPrompt(target?: LocalBrowserTargetIdentity): string {
  return target
    ? `Help configure ${target.cls} in ${target.file} in workspace ${target.workspaceFolder} using TensorViz MCP. Use open_graph with file ${JSON.stringify(path.join(target.workspaceFolder, target.file))} and cls ${JSON.stringify(target.cls)}, and verify the returned target. Reuse saved setup when available. Explain any missing values and proposed example inputs; ask before choosing unresolved values that cannot be established from source. Continue the returned request until capture succeeds or report a concrete blocker. Save the explicit profile in tensorviz.toml and explain what the captured example covers.`
    : 'Open my PyTorch model in TensorViz using MCP. Identify the intended model file and class, then use open_graph. Reuse saved setup when available. Explain any missing values and proposed example inputs; ask before choosing unresolved values that cannot be established from source. Continue the returned request until capture succeeds or report a concrete blocker. Save the explicit profile in tensorviz.toml and explain what the captured example covers.';
}

function html(url: string, target: LocalBrowserTargetIdentity | undefined, location: typeof currentLocation, agent?: AgentKind,
  help?: GraphMcpHelpContext): string {
  const nonce = randomBytes(16).toString('hex');
  const data = JSON.stringify({ url, prompt: mcpSetupPrompt(), targetPrompt: target ? mcpSetupPrompt(target) : '',
    configured: !!agent,
    helpPrompt: help ? graphMcpHelpRequest(help) : '',
    target: target ? `${target.cls} · ${target.file}\n${target.workspaceFolder}` : '' }).replace(/</g, '\\u003c');
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}';">
<title>tensorViz MCP</title>
<style nonce="${nonce}">
  body { margin: 0; padding: 28px 32px; max-width: 720px;
         font-family: var(--vscode-font-family, sans-serif);
         color: var(--vscode-editor-foreground, #ccc);
         background: var(--vscode-editor-background, #1e1e1e); }
  h1 { font-size: 15px; font-weight: 600; margin: 0 0 8px; }
  h2 { font-size: 13px; font-weight: 600; margin: 24px 0 8px; }
  p, li { font-size: 13px; opacity: .85; margin: 0 0 16px; }
  .row { display: flex; gap: 8px; align-items: stretch; flex-wrap: wrap; }
  code, pre { white-space: pre-wrap; overflow-wrap: anywhere; }
  .row code { flex: 1; min-width: 0; padding: 10px 12px; word-break: break-all; user-select: all;
         background: var(--vscode-textCodeBlock-background, #2b2b2b);
         border: 1px solid var(--vscode-panel-border, #3c3c3c); border-radius: 4px;
         font-family: var(--vscode-editor-font-family, monospace); font-size: 13px; }
  button { padding: 8px 12px; border: 0; border-radius: 4px; cursor: pointer;
           background: var(--vscode-button-background, #0e639c);
           color: var(--vscode-button-foreground, #fff); font-size: 13px; }
  button:hover { background: var(--vscode-button-hoverBackground, #1177bb); }
  button:focus-visible { outline: 2px solid var(--vscode-focusBorder); outline-offset: 2px; }
  #target, #prompt, #target-prompt, #help-prompt { font: 12px/1.5 var(--vscode-editor-font-family, monospace); color: var(--vscode-descriptionForeground); }
  details { margin: 20px 0; }
  summary { cursor: pointer; font-size: 13px; }
  .secondary { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); }
  .secondary:hover { background: var(--vscode-button-secondaryHoverBackground); }
  #status { min-height: 1.5em; margin-top: 8px; }
  .warn { margin: 18px 0 0; font-size: 12px; opacity: .7; }
</style>
</head>
<body>
${agent ? `<h1>VS Code connection configured for ${agent}</h1>
<p>You do not need to add another server or copy a URL.
Keep VS Code open, start a new ${agent} chat, and ask it to open your model in TensorViz.</p>
<h2 id="connection-title">Waiting for your agent</h2>
<p id="connection-detail" role="status" aria-live="polite">Agent settings are configured. Waiting for an agent to connect.</p>
<button id="setup-agent" hidden>Repair connection</button>` : `<h1>${help ? 'Get help through MCP' : 'Connect your coding agent'}</h1>
<p>Use your coding agent to open PyTorch models, prepare missing configuration and
example inputs, and inspect captured graphs. MCP connects that agent to TensorViz;
the connection works as you switch models and projects in this VS Code host.</p>
<p>You can also configure the Canvas form or edit
<code>tensorviz.toml</code> yourself; an agent is optional.</p>`}
${help && !agent && location === 'local' ? `<h2 id="connection-title">Checking agent connection</h2>
<p id="connection-detail" role="status" aria-live="polite">A configured URL alone does not mean an agent is connected.</p>` : ''}
${help ? `<h2>${help.opening.phase === 'needs_setup' && help.opening.environment ? 'Investigate this environment' : 'Investigate this failure'}</h2>
<p>Copy this request into your coding agent to use MCP directly. It identifies the model, selected environment and originating opening. No request is sent automatically.</p>
<button id="copy-help-prompt">Copy MCP investigation request</button>
<details><summary>Investigation request details</summary><pre id="help-prompt"></pre></details>` : ''}
${location === 'local' && !agent ? `<h2>Use the VS Code graph from your agent</h2>
<p>This connection uses the running VS Code extension. The tensorViz plugin installs separately in your coding agent and works with VS Code closed.
If you use the plugin, do not add this connection.</p>
<button id="setup-agent">Set up coding agent</button>
<p>Already connected? Start a new agent chat; do not add the URL again.</p>` : ''}
${agent ? '<details><summary>Connection details for other agents</summary>' : '<h2>Connect your agent</h2>'}
<p>${location === 'ssh'
    ? 'This URL runs on the SSH host. Use it for agents running on that server. For an agent running on your Mac or other desktop, explicitly forward a separate endpoint below.'
    : location === 'desktop'
      ? 'This URL is forwarded to this desktop. Add it in your desktop agent settings. Model files, Python and tools still run on the SSH host. This endpoint belongs to this VS Code window; after closing it or reconnecting SSH, forward again and update your agent URL.'
      : 'This URL runs on your local VS Code host.'}</p>
${location === 'ssh' ? '<button id="forward" class="secondary">Forward MCP to This Desktop</button><p>Forwarding permits processes on this desktop to use MCP as well as processes on the SSH host.</p>' : ''}
<ol>
<li>If you have not configured a connection yet, add a server in your agent’s MCP settings using this Streamable HTTP URL. No header or credential is needed.</li>
</ol>
<div class="row"><code id="url"></code><button id="copy">Copy URL</button></div>
<p>In VS Code Copilot, run <strong>MCP: Add Server</strong>, choose HTTP,
paste this URL and select your <strong>user profile</strong> to use it across projects.</p>
${agent ? '</details>' : '<p>Creating this URL does not connect an agent. Finish adding it in your agent’s settings, then check that the agent can access TensorViz’s tools.</p>'}
<h2>Try your model</h2>
<p>Tell your agent which model you want to open. You can copy this request and add the model file or class:</p>
<pre id="prompt"></pre>
<button id="copy-prompt" class="secondary">Copy setup request</button>
${target ? `<details>
<summary>Use the model open in Canvas</summary>
<pre id="target"></pre>
<p>This model was selected when the guide opened. Reopen the guide if you want to update it.</p>
<pre id="target-prompt"></pre>
<button id="copy-target-prompt" class="secondary">Copy request for this model</button>
</details>` : ''}
<p id="status" role="status" aria-live="polite"></p>
<h2>Continue without an agent</h2>
<p>Use the Canvas form or a valid TOML profile to capture the same measured graph.
Your selected Python environment, dependencies and any required factory or checkpoint
still need to be ready. The saved profile works after the agent disconnects.</p>
<button id="manual" class="secondary">Return to Canvas</button>
<p class="warn">The endpoint is reachable by processes on ${location === 'local' ? 'this computer' : location === 'ssh' ? 'the SSH host' : 'the SSH host and this desktop'} and validation runs Python.
Run <strong>Disconnect coding agents (MCP)</strong> when you are done.</p>
<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  const data = ${data};
  for (const key of ['url', 'prompt']) document.getElementById(key).textContent = data[key];
  if (data.helpPrompt) {
    document.getElementById('help-prompt').textContent = data.helpPrompt;
    document.getElementById('copy-help-prompt').addEventListener('click', () => vscode.postMessage({ type: 'copy_help_prompt' }));
  }
  if (data.target) {
    document.getElementById('target').textContent = data.target;
    document.getElementById('target-prompt').textContent = data.targetPrompt;
    document.getElementById('copy-target-prompt').addEventListener('click', () => vscode.postMessage({ type: 'copy_target_prompt' }));
  }
  document.getElementById('copy').addEventListener('click', () => vscode.postMessage({ type: 'copy' }));
  document.getElementById('copy-prompt').addEventListener('click', () => vscode.postMessage({ type: 'copy_prompt' }));
  document.getElementById('manual').addEventListener('click', () => vscode.postMessage({ type: 'manual' }));
  ${location === 'local' ? "document.getElementById('setup-agent').addEventListener('click', () => vscode.postMessage({ type: 'setup_agent' }));" : ''}
  ${location === 'ssh' ? "document.getElementById('forward').addEventListener('click', () => vscode.postMessage({ type: 'forward' }));" : ''}
  window.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'status') document.getElementById('status').textContent = event.data.text;
    if (event.data && event.data.type === 'connection' && document.getElementById('connection-title')) {
      document.getElementById('connection-title').textContent = event.data.title;
      document.getElementById('connection-detail').textContent = event.data.detail;
      if (data.configured) document.getElementById('setup-agent').hidden = !event.data.repair;
    }
  });
  vscode.postMessage({ type: 'ready' });
</script>
</body>
</html>`;
}

/** Show the endpoint URL with a Copy button in its own editor window. */
export function showMcpConnection(url: string, context: vscode.ExtensionContext, target?: LocalBrowserTargetIdentity,
  location: typeof currentLocation = 'local', configuredAgent?: AgentKind, help?: GraphMcpHelpContext): void {
  currentUrl = url;
  currentLocation = location;
  currentAgent = location === 'local' ? configuredAgent : undefined;
  currentPrompt = mcpSetupPrompt();
  currentTargetPrompt = target ? mcpSetupPrompt(target) : '';
  currentHelp = help ? JSON.parse(JSON.stringify(help)) : undefined;
  currentHelpPrompt = currentHelp ? graphMcpHelpRequest(currentHelp) : '';
  if (panel) {
    panel.webview.html = html(url, target, location, currentAgent, currentHelp);
    panel.reveal();
    return;
  }
  const created = vscode.window.createWebviewPanel(
    VIEW_TYPE, 'tensorViz MCP', vscode.ViewColumn.Active, { enableScripts: true },
  );
  panel = created;
  created.webview.html = html(url, target, location, currentAgent, currentHelp);
  const listener = created.webview.onDidReceiveMessage(async (message: unknown) => {
    const type = (message as { type?: unknown })?.type;
    if (type === 'ready') {
      if (currentStatus) updateMcpConnectionStatus(currentStatus);
      return;
    }
    if (type === 'setup_agent' && currentLocation === 'local') {
      await vscode.commands.executeCommand('tensorviz.connectMcp', currentHelp?.opening, true);
      return;
    }
    if (type === 'forward' && currentLocation === 'ssh') {
      await vscode.commands.executeCommand('tensorviz.forwardMcp', currentHelp?.opening);
      return;
    }
    if (type === 'manual') {
      await vscode.commands.executeCommand('tensorviz.openCanvas');
      created.dispose();
      return;
    }
    if (type !== 'copy' && type !== 'copy_prompt' && type !== 'copy_target_prompt' && type !== 'copy_help_prompt') return;
    if (type === 'copy_target_prompt' && !currentTargetPrompt) return;
    if (type === 'copy_help_prompt' && !currentHelpPrompt) return;
    try {
      await vscode.env.clipboard.writeText(type === 'copy' ? currentUrl : type === 'copy_target_prompt' ? currentTargetPrompt
        : type === 'copy_help_prompt' ? currentHelpPrompt : currentPrompt);
      await created.webview.postMessage({ type: 'status', text: type === 'copy' ? 'URL copied.'
        : type === 'copy_help_prompt' ? 'Investigation request copied.' : 'Setup request copied.' });
    } catch {
      await created.webview.postMessage({ type: 'status', text: 'Could not copy. Select and copy the text above.' });
    }
  });
  context.subscriptions.push(created);
  created.onDidDispose(() => {
    listener.dispose();
    if (panel === created) panel = undefined;
  });
}

/** Close the window, e.g. after the endpoint is disconnected. */
export function closeMcpConnection(): void {
  panel?.dispose();
}
