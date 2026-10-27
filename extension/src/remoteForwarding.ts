import * as vscode from 'vscode';
import type { ShellTarget } from './shell';
import { unsupportedRemote, loopbackOrigin } from './remoteSupport';

/** Re-resolve on each explicit open: VS Code may recreate a tunnel on a new port.
 * Only the credential-free origin enters the forwarding API. */
export async function clientOrigin(origin: string): Promise<string> {
  const unsupported = unsupportedRemote(vscode.env.remoteName);
  if (unsupported) throw new Error(unsupported);
  if (!vscode.env.remoteName) return origin;
  const internal = loopbackOrigin(origin);
  const forwarded = await vscode.env.asExternalUri(vscode.Uri.parse(internal));
  return loopbackOrigin(forwarded.toString());
}

export async function forwardShellTarget(target: ShellTarget): Promise<ShellTarget> {
  if (!vscode.env.remoteName) return target;
  const origin = await clientOrigin(target.appOrigin);
  const url = new URL(target.appUrl);
  return { ...target, appOrigin: origin,
    appUrl: `${origin}${url.pathname}${url.search}${url.hash}`, forwarded: true };
}
