/**
 * The window APIs that do not work the same way inside an editor, behind one seam.
 *
 * Every call site changes by its callee token only — `window.prompt(` becomes
 * `hostServices.prompt(`, `navigator.clipboard.writeText(` becomes
 * `hostServices.copyText(` — so the source-text jest suites that slice
 * ModelBuilder.tsx / BuilderPageV2.tsx / App.tsx by literal keep their anchors
 * (task.md §8.1), and so web mode keeps the behaviour it has: with no bridge,
 * each function is the browser call it replaced, argument for argument.
 *
 * What actually differs in the IDE is downloads. `a[download]` is inert inside a
 * webview (task.md §3.3), so a saved file is posted to the extension host, which
 * writes it into the workspace and opens it. The modal family (prompt / confirm /
 * alert) stays synchronous on purpose: every call site is written as
 * `if (!hostServices.confirm(…)) return;`, and a host round-trip would have to
 * be async, which is a control-flow rewrite of a dozen call sites rather than a
 * shim. They therefore still use the browser modals inside the iframe.
 */
import { HostFile } from './protocol.generated';
import { bridgeActive, postToHost } from './bridge';

export function prompt(message: string, defaultValue?: string): string | null {
  return window.prompt(message, defaultValue);
}

export function confirm(message: string): boolean {
  return window.confirm(message);
}

export function alert(message: string): void {
  window.alert(message);
}

/** Third argument kept so `window.open(url, '_blank', 'noopener,noreferrer')`
 *  changes by its callee token alone; the host has no use for either. */
export function openExternal(url: string, target?: string, features?: string): void {
  if (bridgeActive()) {
    postToHost({ type: 'openExternal', url });
    return;
  }
  window.open(url, target, features);
}

export async function copyText(text: string): Promise<void> {
  if (bridgeActive()) {
    postToHost({ type: 'clipboard', text });
    return;
  }
  await navigator.clipboard?.writeText(text);
}

export function reload(): void {
  window.location.reload();
}

/** base64 without a data: prefix, which is what the protocol carries. */
function toBase64(bytes: ArrayBuffer): string {
  const view = new Uint8Array(bytes);
  let binary = '';
  for (let i = 0; i < view.length; i += 1) {
    binary += String.fromCharCode(view[i]);
  }
  return window.btoa(binary);
}

/** The pre-IDE download path, unchanged: an object URL, an anchor, a click. */
function downloadInBrowser(name: string, payload: Blob): void {
  const url = URL.createObjectURL(payload);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

async function toHostFile(name: string, payload: Blob): Promise<HostFile> {
  return { name, bytes: toBase64(await payload.arrayBuffer()) };
}

/** Add to tensorViz: hand the host a graph file to upload to the member's
 *  account. Only the native Canvas has a host; elsewhere this returns false. */
export function shareGraph(name: string, payload: Blob): boolean {
  if (!bridgeActive()) return false;
  void toHostFile(name, payload).then((file) => postToHost({ type: 'shareGraph', file }));
  return true;
}

/** Hand the user a file. In the IDE it lands in the workspace instead of the
 *  browser's download directory, because a webview has no download directory. */
export function saveFile(name: string, payload: Blob): void {
  if (!bridgeActive()) {
    downloadInBrowser(name, payload);
    return;
  }
  void toHostFile(name, payload).then((file) => postToHost({ type: 'saveFile', file }));
}
