/** Sign in to tensorViz, Sign out of tensorViz, the website's sign-in callback, and Add to
 * tensorViz from the Canvas, which reaches the host as the internal SHARE_GRAPH_FILE command
 * with the graph file Export graph would save. websiteAccount.ts and websiteShare.ts do the
 * work; this module is their VS Code surface. Tokens never reach the log.
 */
import * as vscode from 'vscode';
import type { HostFile } from '@tensorviz/contracts/protocol';
import { SEND_FEEDBACK, SHARE_GRAPH_FILE, SIGN_IN_TO_WEBSITE, SIGN_OUT_OF_WEBSITE } from './commandIds';
import { editorCallback, WebsiteAccount, WebsiteError, type WebsiteMember } from './websiteAccount';
import { graphUrl, shareGraph, shareableGraph } from './websiteShare';
import { feedbackReport, operatingSystem, sendFeedback, type FeedbackInput } from '@tensorviz/graph-host/feedback';

const who = (member: WebsiteMember): string =>
  member.name && member.email ? `${member.name} (${member.email})` : member.name ?? member.email ?? 'your Google account';

export function registerWebsiteCommands(context: vscode.ExtensionContext, channel: vscode.OutputChannel): void {
  const account = new WebsiteAccount({
    secrets: context.secrets,
    callback: async () => editorCallback(vscode.env.uriScheme, context.extension.id),
    openExternal: async url => vscode.env.openExternal(vscode.Uri.parse(url)),
  });

  const report = async (what: string, error: unknown): Promise<boolean> => {
    if (error instanceof WebsiteError && error.code === 'cancelled') return false;
    const detail = error instanceof Error ? error.message : String(error);
    channel.appendLine(`[website] ${what} ${detail}`);
    // A modal keeps the whole sentence visible. A toast clips this message
    // (`Could not add the graph to tensorViz.` plus the reason) at the edge.
    const message = detail.startsWith(what) ? detail : `${what} ${detail}`;
    const retry = error instanceof WebsiteError && ['expired', 'signed-out'].includes(error.code);
    if (retry) return await vscode.window.showErrorMessage(message, { modal: true }, 'Retry') === 'Retry';
    await vscode.window.showErrorMessage(message, { modal: true });
    return false;
  };

  const signIn = (title: string) => {
    if (vscode.env.uiKind === vscode.UIKind.Web) {
      throw new WebsiteError('setup', 'Signing in to tensorViz needs VS Code on your desktop.');
    }
    return vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title, cancellable: true },
      (_progress, token) => account.signIn(token));
  };

  const signOut = async (): Promise<void> => {
    const had = await account.signOut();
    void vscode.window.showInformationMessage(had ? 'Signed out of tensorViz.' : 'You are not signed in to tensorViz.');
  };

  let adding = false;
  const share = async (file: HostFile | undefined): Promise<void> => {
    if (typeof file?.bytes !== 'string') return;
    if (adding) {
      void vscode.window.showInformationMessage('tensorViz is still adding your last graph.');
      return;
    }
    adding = true;
    let shared: { title: string; id: string } | undefined;
    let retry = false;
    try {
      const graph = shareableGraph(Buffer.from(file.bytes, 'base64'));
      if (!await account.session()) await signIn(`Sign in to tensorViz in your browser to add ${graph.title}`);
      const id = await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: `Adding ${graph.title} to tensorViz` },
        () => shareGraph(account, graph));
      channel.appendLine(`[website] shared graph ${id}`);
      shared = { title: graph.title, id };
    } catch (error) {
      retry = await report('Could not add the graph to tensorViz.', error);
    } finally {
      adding = false;
    }
    if (retry) return share(file);
    if (!shared) return;
    const config = account.config;
    await vscode.env.openExternal(vscode.Uri.parse(graphUrl(config, shared.id, true)));
    const choice = await vscode.window.showInformationMessage(
      `${shared.title} is on tensorViz, visible only to you. Choose who can see it on the page that opened.`, 'Copy link');
    if (choice === 'Copy link') {
      await vscode.env.clipboard.writeText(graphUrl(config, shared.id));
      vscode.window.setStatusBarMessage('Copied the graph’s tensorViz link', 4000);
    }
  };

  /** The Canvas's feedback button: the member's words, as them when signed in. Never signs in. */
  const feedback = async (input: FeedbackInput): Promise<{ status: 'sent' | 'failed'; detail: string }> => {
    try {
      const report = feedbackReport(input, 'canvas', { product: 'extension', version: String(context.extension.packageJSON.version ?? ''),
        os: operatingSystem(), editor: `${vscode.env.appName} ${vscode.version}` },
      (vscode.workspace.workspaceFolders ?? []).map(folder => folder.uri.fsPath));
      const id = await sendFeedback(account, report);
      channel.appendLine(`[website] sent feedback ${id}`);
      return { status: 'sent', detail: 'Thanks. Your feedback reached the tensorViz team.' };
    } catch (error) {
      const detail = error instanceof WebsiteError ? error.message : 'Could not send your feedback.';
      channel.appendLine(`[website] Could not send feedback. ${detail}`);
      return { status: 'failed', detail };
    }
  };

  const signInCommand = async (): Promise<void> => {
    try {
      const member = await account.member();
      if (member) {
        const choice = await vscode.window.showInformationMessage(`Signed in to tensorViz as ${who(member)}.`, 'Switch account', 'Sign out');
        if (choice === 'Sign out') return await signOut();
        if (choice !== 'Switch account') return;
      }
      const session = await signIn('Sign in to tensorViz in your browser');
      void vscode.window.showInformationMessage(`Signed in to tensorViz as ${who(session)}.`);
    } catch (error) {
      if (await report('Could not sign in to tensorViz.', error)) return signInCommand();
    }
  };

  context.subscriptions.push(
    vscode.window.registerUriHandler({
      handleUri(uri: vscode.Uri) {
        if (account.handleUri(uri.path, uri.query)) return;
        channel.appendLine(`[website] ignored a ${uri.path} link`);
        if (uri.path === '/auth' || uri.path === 'auth') {
          void vscode.window.showWarningMessage('This tensorViz sign-in has expired or was started in another window. Run "Sign in to tensorViz" again.');
        }
      },
    }),
    // Another window signed in or out: read the stored sign-in again on next use.
    context.secrets.onDidChange(event => { if (account.owns(event.key)) account.forget(); }),
    vscode.commands.registerCommand(SIGN_IN_TO_WEBSITE, signInCommand),
    vscode.commands.registerCommand(SIGN_OUT_OF_WEBSITE, async () => {
      try { await signOut(); } catch (error) { await report('Could not sign out of tensorViz.', error); }
    }),
    vscode.commands.registerCommand(SHARE_GRAPH_FILE, share),
    vscode.commands.registerCommand(SEND_FEEDBACK, feedback),
  );
}
