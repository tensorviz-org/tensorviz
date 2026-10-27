// Disposable host driver for where the graph opens.
//
// The unit tests pin the routing decision; they cannot show that a real VS Code
// actually renders the graph page in an editor panel. This drives the real
// extension in a real window and reports the webview tabs the editor ended up
// holding, for all three openers:
//
//   - `tensorviz.openAsGraphInEditor`  with the setting on `browser`
//   - `tensorviz.openAsGraphInBrowser` with the setting on `editor`
//   - `tensorviz.openAsGraph`          honouring the setting
//
// The first two are the interesting ones: each is run with the SETTING POINTING THE
// OTHER WAY, so a command that quietly fell back to `configuredGraphSurface()` shows
// up as a panel that never appeared, or a link that did.
//
// Run it through qa-host.mjs, which builds the disposable root and workspace:
//   node scripts/qa-host.mjs --root /tmp/tv-surface --driver scripts/qa-graph-surface.cjs
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');

const root = process.env.TV_QA_ROOT;
const write = (name, value) =>
  fs.writeFileSync(path.join(root, 'control', name), JSON.stringify(value, null, 2), { mode: 0o600 });

/** Every open tab that is a webview, with the view type VS Code gave it.
 *
 * VS Code namespaces a panel's view type on the tab input (`mainThreadWebview-…`),
 * so this reports the raw string rather than matching an assumed shape. */
function webviewTabs() {
  const tabs = [];
  for (const group of vscode.window.tabGroups.all) {
    for (const tab of group.tabs) {
      if (tab.input instanceof vscode.TabInputWebview) {
        tabs.push({ label: tab.label, viewType: tab.input.viewType, group: group.viewColumn });
      }
    }
  }
  return tabs;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

exports.run = async function () {
  const ext = vscode.extensions.getExtension('tensorviz.tensorviz');
  if (!ext) throw new Error('tensorViz extension missing');
  const api = (await ext.activate()).__test;

  const linkFile = path.join(root, 'control/opened-links.txt');
  const links = () =>
    fs.existsSync(linkFile) ? fs.readFileSync(linkFile, 'utf8').split('\n').filter(Boolean) : [];

  // Global, not workspace: writing a workspace setting would put a .vscode/ file
  // into the fixture, and the purity check exists to prove activation writes nothing.
  const setSurface = async (value) => {
    await vscode.workspace
      .getConfiguration('tensorviz')
      .update('graphSurface', value, vscode.ConfigurationTarget.Global);
    return vscode.workspace.getConfiguration('tensorviz').get('graphSurface');
  };

  /** Every lens the providers put on this file, title and command, in order. */
  const lensesFor = async (uri) => {
    const found = await vscode.commands.executeCommand('vscode.executeCodeLensProvider', uri, 50);
    return (found ?? []).map((lens) => ({
      line: lens.range.start.line,
      title: lens.command && lens.command.title,
      command: lens.command && lens.command.command,
    }));
  };

  const waitFor = async (predicate) => {
    for (let i = 0; i < 60; i += 1) {
      if (predicate()) return true;
      await sleep(500);
    }
    return false;
  };

  const uri = vscode.Uri.file(path.join(root, 'fixture/complex_models/network.py'));
  const document = await vscode.workspace.openTextDocument(uri);
  const editor = await vscode.window.showTextDocument(document);
  editor.selection = new vscode.Selection(10, 0, 10, 0);

  const registered = (await vscode.commands.getCommands(true))
    .filter((id) => id.startsWith('tensorviz.openAsGraph'))
    .sort();

  const report = { registered, legs: [] };

  // Leg 1 — forced editor, while the setting says browser.
  const browserConfigured = await setSurface('browser');
  let tabsBefore = webviewTabs();
  let linksBefore = links();
  await vscode.commands.executeCommand('tensorviz.openAsGraphInEditor');
  await waitFor(() => webviewTabs().length > tabsBefore.length);
  report.legs.push({
    leg: 'openAsGraphInEditor',
    graphSurface: browserConfigured,
    tabsBefore,
    tabsAfter: webviewTabs(),
    linksAdded: links().slice(linksBefore.length),
  });

  // The lens pair, with the setting still on browser: the alternative must be the editor.
  report.lensesWithBrowserDefault = await lensesFor(uri);

  // Leg 2 — forced browser, while the setting says editor.
  const editorConfigured = await setSurface('editor');
  report.lensesWithEditorDefault = await lensesFor(uri);
  tabsBefore = webviewTabs();
  linksBefore = links();
  await vscode.window.showTextDocument(document);
  await vscode.commands.executeCommand('tensorviz.openAsGraphInBrowser');
  await waitFor(() => links().length > linksBefore.length);
  report.legs.push({
    leg: 'openAsGraphInBrowser',
    graphSurface: editorConfigured,
    tabsBefore,
    tabsAfter: webviewTabs(),
    linksAdded: links().slice(linksBefore.length),
  });

  // Leg 3 — the default command, still honouring the setting (now `editor`).
  tabsBefore = webviewTabs();
  linksBefore = links();
  await vscode.window.showTextDocument(document);
  await vscode.commands.executeCommand('tensorviz.openAsGraph');
  await sleep(3000);
  report.legs.push({
    leg: 'openAsGraph',
    graphSurface: vscode.workspace.getConfiguration('tensorviz').get('graphSurface'),
    tabsBefore,
    tabsAfter: webviewTabs(),
    linksAdded: links().slice(linksBefore.length),
  });

  report.session = api.localBrowserSession() ?? null;
  write('surface.json', report);

  // Stay up so the window can be looked at, and close on command.
  await new Promise((resolve) => {
    const timer = setInterval(() => {
      const commandPath = path.join(root, 'control/command.json');
      if (!fs.existsSync(commandPath)) return;
      fs.unlinkSync(commandPath);
      clearInterval(timer);
      resolve();
    }, 500);
  });
};
