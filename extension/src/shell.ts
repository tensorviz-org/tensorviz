import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';

import { devServerUrl, withLaunchToken } from './config';
import { AppMessage, HostFile, PROTOCOL_VERSION } from '@tensorviz/contracts/protocol';
import { canvasHomeParts, type CanvasHomeState } from './canvasHome';
import { SHARE_GRAPH_FILE } from './commandIds';

export const VIEW_TYPE = 'tensorviz.canvas';

/** Where a `saveFile` / `saveFiles` message lands, relative to the workspace. */
const EXPORT_DIR = 'tensorviz-exports';

function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function nonce(): string {
  let text = '';
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += alphabet.charAt(Math.floor(Math.random() * alphabet.length));
  }
  return text;
}

const SHELL_STYLE = `
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden;
               background: var(--vscode-editor-background, #1e1e1e);
               color: var(--vscode-editor-foreground, #ccc);
               font-family: var(--vscode-font-family, sans-serif); }
  iframe { display: block; border: 0; width: 100%; height: 100%; }
  .err { box-sizing: border-box; height: 100%; overflow: auto; padding: 28px 24px;
    display: flex; flex-direction: column; gap: 32px; background: var(--vscode-sideBar-background); }
  .err-brand { display: flex; gap: 10px; align-items: center; font-size: 15px; font-weight: 600; }
  .err-brand svg { color: var(--vscode-textLink-foreground); }
  .err-copy { width: 100%; max-width: 460px; margin: auto; padding: 24px 0; }
  .err-mark { color: var(--vscode-textLink-foreground); margin-bottom: 24px; }
  .err h1 { font-size: 24px; line-height: 1.25; font-weight: 600; letter-spacing: -.5px;
    margin: 0 0 16px; overflow-wrap: anywhere; }
  .err pre { white-space: pre-wrap; overflow-wrap: anywhere; font-family: var(--vscode-font-family, sans-serif);
    font-size: 13px; line-height: 1.65; color: var(--vscode-descriptionForeground); margin: 0; }
  .err pre + pre { margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--vscode-widget-border); }
  @media (max-width: 260px) { .err { padding: 20px 16px; } .err h1 { font-size: 21px; } }
`;

export interface ShellTarget {
  /** The URL the iframe loads, token included. */
  appUrl: string;
  /** The origin half of that URL — the only thing frame-src may name. */
  appOrigin: string;
  /** The loopback port to map through to the extension host (Remote-SSH, Codespaces). */
  port: number;
  /** An asExternalUri URL already addresses the desktop tunnel; do not remap it. */
  forwarded?: boolean;
  /** Where saved files go. */
  workspaceDir: string;
}

/** The shell document.
 *
 * It is deliberately tiny: an iframe and a relay. The app inside it is the
 * production React build served by the backend over http://, because tldraw 5
 * treats a `vscode-webview:` origin as unlicensed production and unmounts the
 * editor five seconds in, while any http: origin is exempt (task.md §2.3, D2).
 * That is the whole reason this is not a first-party webview.
 *
 * The CSP is what decides whether anything renders: VS Code injects a strict
 * default that does not permit framing a local origin, and the symptom of
 * getting it wrong is a blank panel whose refusal appears only in the webview
 * devtools console.
 *
 * The `allow` attribute delegates clipboard permission into the cross-origin
 * child frame. The app prefers the host clipboard service, but the fallback path
 * (and any code that reaches navigator.clipboard directly) needs it.
 */
export function shellHtml(target: ShellTarget, homeState?: CanvasHomeState, contentVisible = false): string {
  const scriptNonce = nonce();
  const home = homeState ? canvasHomeParts(homeState) : undefined;
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; frame-src ${target.appOrigin}; script-src 'nonce-${scriptNonce}'; style-src 'unsafe-inline';">
<title>tensorViz</title>
<style>${SHELL_STYLE}${home?.style ?? ''}</style>
</head>
<body>
${home ? '<div id="canvas-home" class="canvas-home"' + (contentVisible ? ' hidden' : '') + '>' + home.body + '</div>' : ''}
<iframe id="app" src="${target.appUrl}"
        ${home && !contentVisible ? 'hidden aria-hidden="true"' : ''}
        allow="clipboard-read; clipboard-write"
        title="tensorViz canvas"></iframe>
<script nonce="${scriptNonce}">
  // Three hops, two of which this file owns: app -> shell -> extension host.
  // The shell answers 'ready' itself so the app learns the shell's origin
  // without a round trip through the host, then relays everything both ways.
  const vscodeApi = acquireVsCodeApi();
  const frame = document.getElementById('app');
  const APP_ORIGIN = ${JSON.stringify(target.appOrigin)};
  const WORKSPACE_ROOT = ${JSON.stringify(target.workspaceDir)};
  ${home ? 'const vscode = vscodeApi;' + home.script : ''}
  function canvasTheme() {
    const kind = document.body.getAttribute('data-vscode-theme-kind');
    return kind === 'vscode-light' || kind === 'vscode-high-contrast-light' ? 'light' : 'dark';
  }
  new MutationObserver(function () {
    if (frame && frame.contentWindow) {
      frame.contentWindow.postMessage({ type: 'canvasTheme', theme: canvasTheme() }, APP_ORIGIN);
    }
  }).observe(document.body, { attributes: true, attributeFilter: ['data-vscode-theme-kind'] });
  window.addEventListener('message', function (event) {
    if (frame && event.source === frame.contentWindow && event.origin === APP_ORIGIN) {
      if (event.data && event.data.type === 'ready') {
        frame.contentWindow.postMessage(
          { type: 'hostReady', protocolVersion: ${PROTOCOL_VERSION}, theme: canvasTheme() },
          APP_ORIGIN
        );
      }
      if (event.data && event.data.type === 'workspaceRoot') {
        // Answered here, not relayed. The shell already knows the folder, and a
        // round trip through the extension host would make the palette wait on
        // a process for something the shell can say immediately.
        frame.contentWindow.postMessage(
          { type: 'workspaceRoot', root: WORKSPACE_ROOT },
          APP_ORIGIN
        );
        return;
      }
      vscodeApi.postMessage(event.data);
      return;
    }
    if (event.data?.type === 'canvasHomeState' || event.data?.type === 'canvasContent') return;
    if (frame && frame.contentWindow) {
      const data = event.data && event.data.type === 'canvasConnected'
        ? { ...event.data, theme: canvasTheme() } : event.data;
      frame.contentWindow.postMessage(data, APP_ORIGIN);
    }
  });
</script>
</body>
</html>`;
}

/** Shown when the backend dies after the panel already exists. Needs no scripts
 *  and no command URIs — the retry affordance lives on the notification.
 *  An empty retry command omits restart advice for in-flight preparation. */
export function errorHtml(message: string, detail: string, retryCommand = 'tensorViz: Open Canvas'): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';">
<title>tensorViz</title>
<style>${SHELL_STYLE}</style>
</head>
<body>
<div class="err">
  <div class="err-brand"><svg aria-hidden="true" width="22" height="22" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.6">
    <path d="M8 7h16M8 7v18m0-9h16m0-9v18M8 25h16" />
    <circle cx="8" cy="7" r="3.2" /><circle cx="24" cy="7" r="3.2" /><circle cx="8" cy="25" r="3.2" />
    <circle cx="24" cy="25" r="3.2" /><circle cx="24" cy="16" r="3.2" />
  </svg>tensorViz</div>
  <section class="err-copy">
  <svg class="err-mark" aria-hidden="true" width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="7" y="7" width="18" height="18" rx="5" /><rect x="39" y="39" width="18" height="18" rx="5" />
    <path d="M25 16h10v9M39 48H29v-9M35 31h-6" stroke-linecap="round" />
  </svg>
  <h1>${escapeHtml(message)}</h1>
  <pre>${escapeHtml(detail)}</pre>
  ${retryCommand ? `<pre>Run "${escapeHtml(retryCommand)}" again to restart the backend.</pre>` : ''}
  </section>
</div>
</body>
</html>`;
}

/** A file name the host is willing to write: one path segment, nothing else. */
function safeSegment(name: string): string | undefined {
  if (!name || name === '.' || name === '..') {
    return undefined;
  }
  if (name.includes('/') || name.includes('\\') || path.isAbsolute(name)) {
    return undefined;
  }
  return name;
}

/** Shared webview services; no panel ownership or controller lifetime. */
export class CanvasBridge {
  private readonly disposables: vscode.Disposable[] = [];

  /** Last reported mount state, so the two-second heartbeat is logged on change only. */
  private canvasMounted: boolean | undefined;

  constructor(
    readonly webview: vscode.Webview,
    private readonly target: ShellTarget,
    private readonly channel: vscode.OutputChannel,
  ) {
    this.disposables.push(webview.onDidReceiveMessage((message: AppMessage) => void this.handle(message)));
  }

  /** Webview options that must match on a fresh panel and a restored one.
   *
   * The port mapping below is the difference between a working canvas and a
   * blank one under Remote-SSH or Codespaces, where the backend's loopback is the
   * remote box's, not the browser's. It is a no-op on local desktop. */
  static options(port: number, forwarded = false): vscode.WebviewOptions & vscode.WebviewPanelOptions {
    return {
      // Mandatory even though this document is "just an iframe": the flag governs
      // script execution for the whole frame tree, so false would load the React
      // app and never run it — a white panel with no error.
      enableScripts: true,
      // Without this, switching editor tabs tears down the iframe and reloads the
      // app, losing in-memory tldraw and unsaved graph state.
      retainContextWhenHidden: true,
      // The HTML is fully inline and loads nothing from disk, so grant nothing.
      localResourceRoots: [],
      portMapping: forwarded ? [] : [{ webviewPort: port, extensionHostPort: port }],
    };
  }

  /** Messages the app has sent, newest last. The Electron tests read this rather
   *  than racing the relay; nothing in the product consumes it. */
  readonly received: AppMessage[] = [];
  private listener: ((message: AppMessage) => boolean | void) | undefined;

  onMessage(listener: (message: AppMessage) => boolean | void): void {
    this.listener = listener;
  }

  /** Deliver a message as if the app had posted it. Used by the Electron tests,
   *  which cannot reach into the cross-origin iframe to make the app send one. */
  handleForTest(message: AppMessage): Promise<void> {
    return this.handle(message);
  }

  private async handle(message: AppMessage): Promise<void> {
    if (!message || typeof message.type !== 'string') {
      return;
    }
    if (this.listener?.(message) === true) return;
    this.received.push(message);

    switch (message.type) {
      case 'ready':
        if (message.protocolVersion !== PROTOCOL_VERSION) {
          // Refuse loudly: a half-understood protocol is worse than none.
          void vscode.window.showErrorMessage(
            `The tensorViz app speaks bridge protocol ${message.protocolVersion}; this extension speaks ${PROTOCOL_VERSION}. Rebuild the app (node extension/scripts/build-webview.mjs).`,
          );
          return;
        }
        this.channel.appendLine(`[app] ready (protocol ${message.protocolVersion})`);
        return;
      case 'canvasState':
        // Only on CHANGE. The app reports this every two seconds as a heartbeat, and
        // logging each tick buried the log it shares with the backend, the trace
        // worker and the projection: one session held 94 identical lines against 389
        // that said something. What this exists to catch is the transition — a canvas
        // that was mounted and is not any more — and that still reaches the log.
        if (message.mounted !== this.canvasMounted) {
          this.canvasMounted = message.mounted;
          this.channel.appendLine(`[app] canvas ${message.mounted ? 'mounted' : 'NOT MOUNTED'}`);
        }
        return;
      case 'openExternal':
        await vscode.env.openExternal(vscode.Uri.parse(message.url));
        return;
      case 'clipboard': {
        if (message.requestId !== undefined && (typeof message.requestId !== 'string' || !message.requestId
          || message.requestId.length > 128 || typeof message.text !== 'string' || message.text.length > 64000)) return;
        try {
          await vscode.env.clipboard.writeText(message.text);
          if (message.requestId) await this.webview.postMessage({ type: 'clipboardWritten', requestId: message.requestId });
        } catch {
          if (message.requestId) await this.webview.postMessage({ type: 'clipboardWritten', requestId: message.requestId, error: true });
        }
        return;
      }
      case 'readClipboard': {
        if (typeof message.requestId !== 'string' || !message.requestId || message.requestId.length > 128) return;
        try {
          const text = (await vscode.env.clipboard.readText()).slice(0, 16000);
          await this.webview.postMessage({ type: 'clipboardText', requestId: message.requestId, text });
        } catch {
          await this.webview.postMessage({ type: 'clipboardText', requestId: message.requestId, error: true });
        }
        return;
      }
      case 'showCommandPalette':
        await vscode.commands.executeCommand('workbench.action.showCommands');
        return;
      case 'openAsGraph':
        await vscode.commands.executeCommand('tensorviz.openAsGraph');
        return;
      case 'toast':
        this.toast(message.level, message.message);
        return;
      case 'saveFile':
        await this.writeExport(undefined, [message.file]);
        return;
      case 'saveFiles':
        await this.writeExport(message.dir, message.files);
        return;
      case 'shareGraph':
        // Add to tensorViz: websiteCommands.ts signs in if needed and uploads.
        await vscode.commands.executeCommand(SHARE_GRAPH_FILE, message.file);
        return;
      default:
        this.channel.appendLine(`[app] ignored message: ${message.type}`);
    }
  }

  private toast(level: 'info' | 'warn' | 'error', text: string): void {
    if (level === 'error') {
      void vscode.window.showErrorMessage(text);
    } else if (level === 'warn') {
      void vscode.window.showWarningMessage(text);
    } else {
      void vscode.window.showInformationMessage(text);
    }
  }

  /** Write an export where the user chooses, then offer to open or reveal it.
   *
   * `a[download]` does nothing inside a webview, so a download in IDE mode is
   * bytes over the bridge. One file asks for a path first. A set of files still
   * lands in tensorviz-exports. Nothing is opened until the user picks Open or Reveal. */
  private async writeExport(dir: string | undefined, files: HostFile[]): Promise<void> {
    const segment = dir === undefined ? undefined : safeSegment(dir);
    if (dir !== undefined && segment === undefined) {
      void vscode.window.showErrorMessage(`tensorViz refused to write an export to "${dir}".`);
      return;
    }
    const root = segment
      ? path.join(this.target.workspaceDir, EXPORT_DIR, segment)
      : path.join(this.target.workspaceDir, EXPORT_DIR);

    const accepted: { name: string; bytes: string; proposed: string }[] = [];
    for (const file of files) {
      const name = safeSegment(file.name);
      if (!name) {
        void vscode.window.showErrorMessage(`tensorViz refused to write a file named "${file.name}".`);
        continue;
      }
      accepted.push({ name, bytes: file.bytes, proposed: path.join(root, name) });
    }
    if (accepted.length === 0) return;
    const targets: string[] = [];
    if (accepted.length === 1) {
      const only = accepted[0];
      const picked = await vscode.window.showSaveDialog({
        defaultUri: vscode.Uri.file(only.proposed),
        saveLabel: 'Save',
        filters: only.name.endsWith('.svg') ? { SVG: ['svg'] } : undefined,
      });
      if (!picked) return;
      fs.mkdirSync(path.dirname(picked.fsPath), { recursive: true });
      fs.writeFileSync(picked.fsPath, Buffer.from(only.bytes, 'base64'));
      targets.push(picked.fsPath);
      this.channel.appendLine(`[app] wrote ${picked.fsPath}`);
    } else {
      fs.mkdirSync(root, { recursive: true });
      for (const file of accepted) {
        fs.writeFileSync(file.proposed, Buffer.from(file.bytes, 'base64'));
        targets.push(file.proposed);
        this.channel.appendLine(`[app] wrote ${file.proposed}`);
      }
    }
    const primary = targets.find((item) => item.endsWith('.py')) ?? targets[0];
    await this.confirmExport(primary);
  }

  private async confirmExport(file: string): Promise<void> {
    const pick = await vscode.window.showInformationMessage(`Saved ${path.basename(file)}.`, 'Open', 'Reveal');
    if (pick === 'Open') {
      const document = await vscode.workspace.openTextDocument(file);
      await vscode.window.showTextDocument(document, { viewColumn: vscode.ViewColumn.Beside, preview: false });
    } else if (pick === 'Reveal') {
      await vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(file));
    }
  }

  setError(message: string, detail: string): void {
    this.webview.html = errorHtml(message, detail);
  }

  dispose(): void {
    this.disposables.forEach(d => d.dispose());
    this.listener = undefined;
  }
}

/** Editor-panel owner for the read-only run comparison. */
export class CanvasShell extends CanvasBridge {
  private constructor(readonly panel: vscode.WebviewPanel, target: ShellTarget, channel: vscode.OutputChannel) {
    super(panel.webview, target, channel);
  }

  static attach(
    panel: vscode.WebviewPanel,
    target: ShellTarget,
    channel: vscode.OutputChannel,
    onDispose: () => void,
  ): CanvasShell {
    const shell = new CanvasShell(panel, target, channel);
    panel.webview.options = CanvasShell.options(target.port, target.forwarded);
    panel.webview.html = shellHtml(target);
    panel.onDidDispose(() => {
      shell.detach();
      onDispose();
    });
    return shell;
  }

  private detach(): void { super.dispose(); }
  reveal(): void { this.panel.reveal(this.panel.viewColumn); }
  override dispose(): void { this.panel.dispose(); }
}

/** The iframe URL and the origin the CSP will name.
 *
 * `tensorviz.devServerUrl` keeps an iframe-to-a-dev-server path available while
 * the IDE routes are still being built: when it is set the value is used
 * verbatim and the CSP origin is derived from it, so no port literal ever
 * appears in this source. */
export function shellTarget(
  port: number,
  token: string,
  workspaceDir: string,
): ShellTarget {
  const configured = devServerUrl(
    vscode.workspace.getConfiguration('tensorviz').get<string>('devServerUrl', ''),
    process.env,
  );
  // The dev server gets the token too: LocalTokenMiddleware checks the header on every /api
  // request, and the app only learns the token from its own URL (ide/main.tsx).
  const appUrl = configured
    ? withLaunchToken(configured, token)
    : `http://127.0.0.1:${port}/?token=${encodeURIComponent(token)}`;
  return { appUrl, appOrigin: new URL(appUrl).origin, port, workspaceDir };
}
