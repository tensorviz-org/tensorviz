// Disposable installed-host driver for browser QA; never use a normal VS Code profile.
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');
const root = process.env.TV_QA_ROOT;
const write = (name, value) => fs.writeFileSync(path.join(root, 'control', name), JSON.stringify(value, null, 2), {mode: 0o600});
// The public command opens a project unless a class is named. QA deliberately
// opens one class, using fixture declarations rather than the retired chooser.
const openModel = async (doc, line = 1, cls) => {
  const classes = [...doc.getText().matchAll(/^class\s+(\w+)\s*\(/gm)];
  const chosen = cls || classes.filter(match => doc.positionAt(match.index).line < line).at(-1)?.[1] || classes[0]?.[1];
  if (!chosen) throw new Error('QA fixture has no model class');
  await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: doc.uri.toString(), cls: chosen });
  await vscode.commands.executeCommand('tensorviz.openCanvasInBrowser');
};
/** The embedded canvas's projection, reduced to what a QA assertion can compare:
 *  the toolbar's "N layers" is `nodes.length` off this very document, and the
 *  dependency list is the set of files an edit to which must redraw it. */
const embeddedSummary = (drawn) => {
  if (!drawn) return null;
  const nodes = (drawn.document && drawn.document.semantic && drawn.document.semantic.nodes) || [];
  return {
    file: drawn.file, cls: drawn.cls, revision: drawn.revision, traced: drawn.traced, greyed: drawn.greyed,
    nodeCount: nodes.length,
    nodeIds: nodes.map(node => node.node_id).sort(),
    dependencies: (drawn.dependencies || []).map(entry => entry.file),
    diagnostics: (drawn.diagnostics || []).map(entry => entry.code),
    truncated: drawn.truncated ?? null,
  };
};

exports.run = async function () {
  await require(path.join(__dirname, 'check-installed-workspace.cjs')).run();
  const ext = vscode.extensions.getExtension('tensorviz.tensorviz');
  if (!ext) throw new Error('Installed tensorViz extension missing');
  const api = (await ext.activate()).__test;
  const uri = vscode.Uri.file(path.join(root, 'fixture/complex_models/network.py'));
  const document = await vscode.workspace.openTextDocument(uri);
  const editor = await vscode.window.showTextDocument(document);
  editor.selection = new vscode.Selection(10, 0, 10, 0);
  await openModel(document, 11, 'Network');
  write('ready.json', {extensionPath: ext.extensionPath, fixture: uri.fsPath});
  let busy = false;
  await new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      if (busy) return;
      busy = true;
      try {
        const session = api.localBrowserSession();
        if (session) write('page.json', session);
        write('state.json', {target: api.localBrowserTarget(), origin: api.backendOrigin(), viewPresence: api.localBrowserViewPresence(), editor: vscode.window.activeTextEditor ? {file: vscode.window.activeTextEditor.document.uri.fsPath, line: vscode.window.activeTextEditor.selection.active.line + 1, dirty: vscode.window.activeTextEditor.document.isDirty} : null,
          // Polled, not command-driven: the embedded canvas redraws by itself when a file the
          // graph was built from is edited, and a driver can only see that by watching.
          embedded: embeddedSummary(api.projection())});
        const commandPath = path.join(root, 'control/command.json');
        if (!fs.existsSync(commandPath)) return;
        const command = JSON.parse(fs.readFileSync(commandPath, 'utf8'));
        fs.unlinkSync(commandPath);
        if (command.type === 'finish') { clearInterval(timer); resolve(); }
        else if (command.type === 'connect-mcp') {
          await vscode.commands.executeCommand('tensorviz.connectMcp');
          const connection = api.mcpConnection();
          if (!connection) throw new Error('MCP endpoint did not start');
          // Private disposable control data, never an evidence artifact or workspace file.
          write('mcp.json', { id: command.id, ...connection });
        } else if (command.type === 'disconnect-mcp') {
          await vscode.commands.executeCommand('tensorviz.disconnectMcp');
          fs.rmSync(path.join(root, 'control/mcp.json'), { force: true });
          write('mcp-disconnected.json', { id: command.id });
        } else if (command.type === 'select-model') {
          const file = path.resolve(root, 'fixture', command.file);
          if (!file.startsWith(path.join(root, 'fixture') + path.sep)) throw new Error('QA target must be inside the disposable fixture');
          const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(file));
          const editor = await vscode.window.showTextDocument(doc);
          editor.selection = new vscode.Selection(command.line - 1, 0, command.line - 1, 0);
          await openModel(doc, command.line, command.cls);
          write('selected-model.json', { id: command.id });
        } else if (command.type === 'inspect') {
          // A native diff can replace the source preview tab; still inspect its real buffer.
          await vscode.workspace.openTextDocument(uri);
          const files = {};
          for (const doc of vscode.workspace.textDocuments) {
            if (doc.uri.fsPath.startsWith(path.join(root, 'fixture'))) files[path.relative(path.join(root, 'fixture'), doc.uri.fsPath)] = { text: doc.getText(), version: doc.version, dirty: doc.isDirty };
          }
          const reviews = [];
          for (const group of vscode.window.tabGroups.all) for (const tab of group.tabs) {
            if (tab.input instanceof vscode.TabInputTextDiff && tab.input.modified.scheme.startsWith('tensorviz-review-')) {
              const before = await vscode.workspace.openTextDocument(tab.input.original);
              const after = await vscode.workspace.openTextDocument(tab.input.modified);
              reviews.push({ title: tab.label, before: before.getText(), after: after.getText() });
            }
          }
          write('source-state.json', { id: command.id, files, reviews });
        } else if (command.type === 'revert-clean') {
          // QA cleanup only: never discard text that differs from fixture bytes.
          const dirty = vscode.workspace.textDocuments.filter(doc => doc.isDirty && doc.uri.fsPath.startsWith(path.join(root, 'fixture') + path.sep));
          for (const doc of dirty) {
            if (doc.getText() !== fs.readFileSync(doc.uri.fsPath, 'utf8')) throw new Error('QA cleanup refuses changed fixture text');
          }
          for (const doc of dirty) {
            await vscode.window.showTextDocument(doc);
            await vscode.commands.executeCommand('workbench.action.files.revert');
          }
          write('clean-state.json', { id: command.id, reverted: dirty.length });
        } else if (command.type === 'undo') {
          const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(root, 'fixture', command.file)));
          await vscode.window.showTextDocument(doc);
          await vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
          const version = doc.version;
          await vscode.commands.executeCommand('undo');
          // The command can return before the renderer's document change arrives.
          const deadline = Date.now() + 5000;
          while (doc.version === version && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 50));
          if (doc.version === version) throw new Error('Native Undo did not change the QA buffer');
          write('undo-result.json', { id: command.id, text: doc.getText(), dirty: doc.isDirty });
        } else if (command.type === 'apply' || command.type === 'discard') {
          await vscode.commands.executeCommand(command.type === 'apply' ? 'tensorviz.applySourceProposal' : 'tensorviz.discardSourceProposal');
          write('decision-result.json', { id: command.id });
        } else if (command.type === 'open') {
          // Show the file first when one is named. `openAsGraph` with no argument
          // resolves its target from the ACTIVE editor, and a driver that has been
          // clicking around a browser cannot assume one is still focused — the
          // command would refuse with "Open a Python file" and look like the bug
          // under test.
          if (command.file) {
            const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(root, 'fixture', command.file)));
            await vscode.window.showTextDocument(doc);
          }
          const active = vscode.window.activeTextEditor?.document;
          if (!active) throw new Error('QA open needs an active fixture');
          await openModel(active, command.line || 1, command.cls);
          write('open-result.json', { id: command.id, file: command.file || null });
        }
        else if (command.type === 'edit') {
          const uri = vscode.Uri.file(path.join(root, 'fixture', command.file));
          const doc = await vscode.workspace.openTextDocument(uri);
          const original = doc.getText();
          if (!original.includes(command.before)) throw new Error('QA edit source mismatch');
          const edit = new vscode.WorkspaceEdit();
          edit.replace(uri, new vscode.Range(doc.positionAt(0), doc.positionAt(original.length)), original.replace(command.before, command.after));
          await vscode.workspace.applyEdit(edit);
          write('edit-result.json', {id: command.id, dirty: doc.isDirty});
        } else if (command.type === 'trace') {
          // What a user does to get MEASURED shapes: opt this workspace into tracing and save.
          // Global scope, not Workspace: the setting file must land in the disposable profile,
          // never as a .vscode/ write into the fixture the purity check hashed.
          await vscode.workspace.getConfiguration('tensorviz').update('traceOnSave', true, vscode.ConfigurationTarget.Global);
          const uri = vscode.Uri.file(path.join(root, 'fixture', command.file));
          const doc = await vscode.workspace.openTextDocument(uri);
          await vscode.window.showTextDocument(doc);
          const before = api.localBrowserTarget();
          // A clean document's save() is a no-op that fires no save event, so there would be
          // nothing for traceOnSave to hear. Dirty it first, the way a user does, with a
          // trailing comment that moves no existing line.
          if (!doc.isDirty) {
            const edit = new vscode.WorkspaceEdit();
            edit.insert(uri, doc.lineAt(doc.lineCount - 1).range.end, '\n# tensorviz QA: edited to trigger a trace on save\n');
            if (!await vscode.workspace.applyEdit(edit)) throw new Error('QA trace could not dirty the buffer');
          }
          if (!await doc.save()) throw new Error('QA trace could not save the buffer');
          // The trace is save-coalesced before it runs, so give the save time to ARM one
          // (`tracePending` is false both before and after), then wait for it to conclude.
          await new Promise(r => setTimeout(r, 4000));
          const deadline = Date.now() + (command.timeoutMs || 240000);
          while (api.localBrowserTracePending() && Date.now() < deadline) {
            await new Promise(r => setTimeout(r, 250));
          }
          write('trace-result.json', {id: command.id, file: command.file, before, after: api.localBrowserTarget(), pending: api.localBrowserTracePending()});
        } else if (command.type === 'embedded-graph') {
          // The OTHER graph surface — the in-editor webview canvas — so a driver can tell a
          // behaviour the local browser page lacks from one the extension lacks entirely.
          await api.openEmbeddedGraph({uri: vscode.Uri.file(path.join(root, 'fixture', command.file)).toString(), cls: command.cls});
          // The canvas toolbar's "N layers" is `nodes.length` off exactly this document, so
          // reporting it is how a driver compares the two surfaces by number instead of by
          // screenshot. Read as soon as the open resolves: that is the static tier, before any
          // save-coalesced trace replaces it — which is the tier the surfaces are compared on.
          write('embedded-result.json', {id: command.id, file: command.file, cls: command.cls,
            ...embeddedSummary(api.projection())});
        } else if (command.type === 'inlay-hints') {
          // The provider as the EDITOR sees it, not as the extension hopes: this is the same
          // call the inlay-hint renderer makes, so an unregistered provider reads as [].
          const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(root, 'fixture', command.file)));
          await vscode.window.showTextDocument(doc);
          const hints = await vscode.commands.executeCommand('vscode.executeInlayHintProvider', doc.uri,
            new vscode.Range(0, 0, doc.lineCount, 0)) || [];
          write('inlay-result.json', {id: command.id, file: command.file, lineCount: doc.lineCount,
            hints: hints.map(hint => ({
              line: hint.position.line + 1,
              label: typeof hint.label === 'string' ? hint.label : hint.label.map(part => part.value).join(''),
              source: doc.lineAt(hint.position.line).text,
            }))});
        } else if (command.type === 'scan') {
          const token = api.launchToken();
          const leaks = [];
          function scan(dir) {
            for (const entry of fs.readdirSync(dir, {withFileTypes:true})) {
              const p = path.join(dir, entry.name);
              if (entry.isDirectory()) scan(p);
              else if (token && fs.readFileSync(p).includes(Buffer.from(token))) leaks.push(path.relative(root, p));
            }
          }
          scan(path.join(root, 'evidence'));
          scan(path.join(root, 'profile/logs'));
          write('secret-scan.json', {launchTokenPresent: !!token, leaks});
        } else throw new Error('Unknown QA control command');
      } catch (error) { clearInterval(timer); reject(error); }
      finally { busy = false; }
    }, 500);
  });
};
