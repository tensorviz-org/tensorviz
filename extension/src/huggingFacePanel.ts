import { randomBytes } from 'node:crypto';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { architectureFor, downloadHubWeights, exampleInputs, HubAccessError, hubModelFiles,
  inspectHubWeights, MAX_DOWNLOAD, parseHubLink, type HubWeights } from './huggingFace';
import { unsupportedRemote } from './remoteSupport';

const TOKEN_KEY = 'tensorviz.huggingFace.readToken';

export function huggingFaceHtml(script: string, style: string, cspSource: string): string {
  const nonce = randomBytes(16).toString('hex');
  return `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${cspSource}; script-src 'nonce-${nonce}';">
<title>tensorViz · Hugging Face weights</title><link rel="stylesheet" href="${style}"></head><body>
<main>
  <header><div class="eyebrow">tensorViz / Hugging Face</div><h1>Inside the weights</h1>
    <p>Explore tensor shapes, then run an example to see how tensors flow through the model.</p></header>
  <form id="inspect-form"><label for="url">Model or safetensors link</label><div class="url-row">
    <input id="url" type="url" required placeholder="https://huggingface.co/owner/model/blob/main/model.safetensors">
    <button id="inspect" type="submit">Explore weights</button></div></form>
  <p id="status" role="status" aria-live="polite">Only the file header is read when you explore weights.</p>
  <section id="results" hidden>
    <div class="model-heading"><div><h2 id="model-name"></h2><div id="revision" class="muted mono"></div></div><span class="badge">Weights · not executed</span></div>
    <div id="stats" class="stats"></div>
    <section aria-label="Weight storage by group"><h3>Where the weights live</h3><p class="muted">Stored tensor bytes by top-level group. These are weight groups, not forward connections.</p><div id="groups"></div></section>
    <details class="run" id="run-setup"><summary>Run example inputs <span class="muted">→ capture the forward graph</span></summary><div><p id="run-description"></p></div>
      <div id="input-form"><p class="muted">Synthetic examples, editable before running. Integer inputs use token IDs; float inputs use model-ready features.</p>
        <div class="input-head"><span>Argument</span><span>Shape</span><span>Type</span><span>Fill</span><span></span></div>
        <div id="inputs"></div><button id="add-input" class="secondary" type="button">+ Add input</button>
        <p id="run-note" class="muted"></p><button id="run" type="button">Download &amp; run example</button>
      </div>
    </details>
    <section><div class="tensor-heading"><h3>Tensor shapes</h3><input id="search" type="search" aria-label="Search tensors" placeholder="Find a layer, tensor or shape…"></div>
      <p id="match-count" class="muted" aria-live="polite"></p>
      <div class="tensor-columns"><span>Name</span><span>Shape</span><span>Type</span><span>Bytes</span></div>
      <div id="tree"></div><button id="more" class="secondary" type="button" hidden>Show more matches</button>
    </section>
  </section>
</main><script nonce="${nonce}" src="${script}"></script></body></html>`;
}

/** The host owns network access, credentials and writes; the webview only selects inputs. */
export function openHuggingFace(context: vscode.ExtensionContext, initialUrl?: unknown): vscode.WebviewPanel {
  // Source lives in webview/, not media/. media/ is the gitignored React build, so assets
  // placed there never reach a fresh clone or the packaged extension.
  const assets = vscode.Uri.joinPath(context.extensionUri, 'webview');
  const panel = vscode.window.createWebviewPanel('tensorviz.huggingFace', 'Hugging Face weights', vscode.ViewColumn.Active,
    { enableScripts: true, localResourceRoots: [assets] });
  const media = (name: string) => panel.webview.asWebviewUri(vscode.Uri.joinPath(assets, name)).toString();
  panel.webview.html = huggingFaceHtml(media('hugging-face.js'), media('hugging-face.css'), panel.webview.cspSource);
  let weights: HubWeights | undefined;
  let active: AbortController | undefined;
  let disposed = false;
  const send = (value: unknown) => disposed ? Promise.resolve(false) : panel.webview.postMessage(value);

  async function withAccess<T>(operation: (token?: string) => Promise<T>): Promise<T> {
    const token = await context.secrets.get(TOKEN_KEY);
    try { return await operation(token); } catch (error) {
      if (!(error instanceof HubAccessError)) throw error;
      const supplied = await vscode.window.showInputBox({ title: 'Hugging Face access', password: true, ignoreFocusOut: true,
        prompt: 'Accept the model’s access conditions on Hugging Face, then enter a read token. It is stored in VS Code Secret Storage.' });
      if (!supplied?.trim()) throw error;
      const result = await operation(supplied.trim());
      await context.secrets.store(TOKEN_KEY, supplied.trim());
      return result;
    }
  }

  const listener = panel.webview.onDidReceiveMessage(async (message: unknown) => {
    const data = message as { type?: string; url?: unknown; inputs?: unknown } | null;
    if (data?.type === 'ready') {
      await send({ type: 'initial', url: typeof initialUrl === 'string' ? initialUrl : '' });
      return;
    }
    if (active || (data?.type !== 'inspect' && data?.type !== 'run')) return;
    const controller = new AbortController();
    active = controller;
    const timer = setTimeout(() => controller.abort(), data.type === 'inspect' ? 60000 : 10 * 60000);
    await send({ type: 'busy', busy: true });
    try {
      if (data.type === 'inspect') {
        weights = undefined;
        if (typeof data.url !== 'string' || data.url.length > 4096) throw new Error('Paste a Hugging Face model URL.');
        const link = parseHubLink(data.url);
        await send({ type: 'status', text: 'Reading the weight header…' });
        weights = await withAccess(token => inspectHubWeights(link, controller.signal, token));
        await send({ type: 'weights', weights, architecture: architectureFor(weights), inputs: exampleInputs(weights.config), maxDownload: MAX_DOWNLOAD });
      } else {
        if (!weights) throw new Error('Explore a safetensors file first.');
        const unsupported = unsupportedRemote(vscode.env.remoteName);
        if (unsupported) throw new Error(unsupported);
        if (!vscode.workspace.isTrusted) throw new Error('Trust this workspace before downloading and running a model.');
        const folders = vscode.workspace.workspaceFolders ?? [];
        const folder = folders.length === 1 ? folders[0] : await vscode.window.showWorkspaceFolderPick({ placeHolder: 'Import the model into this workspace' });
        if (!folder) throw new Error('Open a workspace folder to save and run this model.');
        const files = hubModelFiles(weights, data.inputs);
        if (weights.fileBytes > MAX_DOWNLOAD) throw new Error('This importer supports downloads up to 2 GiB. Use a local checkpoint recipe for this model.');
        const selected = weights;
        const stem = selected.repo.split('/')[1].replace(/[^A-Za-z0-9_]/g, '_').slice(0, 70);
        const directory = await fs.mkdtemp(path.join(folder.uri.fsPath, `hf_${stem}_`));
        let saved = false;
        try {
          await fs.mkdir(path.join(directory, 'weights'));
          await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: `Importing ${selected.repo}`, cancellable: true },
            async (progress, cancellation) => {
              const subscription = cancellation.onCancellationRequested(() => controller.abort());
              let reported = -1;
              try {
                await withAccess(async token => {
                  const destination = path.join(directory, 'weights', 'model.safetensors');
                  await downloadHubWeights(selected, destination, controller.signal, token, bytes => {
                    const percent = Math.floor(100 * bytes / selected.fileBytes);
                    if (percent !== reported) { progress.report({ message: `${percent}% downloaded` }); reported = percent; }
                  });
                });
              } finally { subscription.dispose(); }
            });
          controller.signal.throwIfAborted();
          // Keep the pinned source beside the editable recipe. No credentials enter these files.
          await fs.writeFile(path.join(directory, 'weights', 'config.json'), JSON.stringify(selected.config, null, 2) + '\n', { flag: 'wx' });
          await fs.writeFile(path.join(directory, 'huggingface.json'), JSON.stringify({ repo: selected.repo, revision: selected.commit,
            file: selected.file, fileBytes: selected.fileBytes }, null, 2) + '\n', { flag: 'wx' });
          await fs.writeFile(path.join(directory, 'tensorviz.toml'), files.profile, { flag: 'wx' });
          await fs.writeFile(path.join(directory, 'model.py'), files.model, { flag: 'wx' });
          saved = true;
          const uri = vscode.Uri.joinPath(folder.uri, path.basename(directory), 'model.py');
          await send({ type: 'status', text: 'Model and example saved. Opening Canvas with the selected Python environment…' });
          await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: uri.toString(), cls: 'HuggingFaceModel' });
          await send({ type: 'status', text: `Saved in ${path.basename(directory)}. Canvas shows environment setup and the execution result. Reopen model.py with Open as Graph to use these downloaded weights again.` });
        } finally {
          if (!saved) await fs.rm(directory, { recursive: true, force: true });
        }
      }
    } catch (error) {
      await send({ type: 'status', error: true, text: controller.signal.aborted ? 'Import cancelled or timed out. Try again when ready.'
        : error instanceof Error ? error.message : 'The Hugging Face import could not complete.' });
    } finally {
      clearTimeout(timer);
      active = undefined;
      await send({ type: 'busy', busy: false });
    }
  });
  panel.onDidDispose(() => { disposed = true; active?.abort(); listener.dispose(); });
  context.subscriptions.push(panel);
  return panel;
}
