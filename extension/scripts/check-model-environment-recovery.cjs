/** Installed acceptance in qa-host.mjs's disposable profile and fixture. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
let vscode = require('vscode');

exports.run = async () => {
  const root = process.env.TV_QA_ROOT;
  assert.ok(root);
  const workspace = path.join(root, 'fixture');
  assert.equal(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath, workspace);
  // qa-host rebuilds root at launch, so the disposable target is its sibling.
  const python = path.join(`${root}-project-env`, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  assert.ok(fs.existsSync(python), 'Create a disposable Python 3.10 <qa-root>-project-env first');
  const inventory = () => JSON.parse(execFileSync(python, ['-c',
    'import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()}))'], { encoding: 'utf8' }));
  const before = inventory();
  assert.match(before.torch, /^2\.14\.1/);
  assert.ok(!before.pip && !before.pydantic && !before.fastapi);
  const extension = vscode.extensions.getExtension('tensorviz.tensorviz-vscode');
  assert.ok(extension);
  // Recent VS Code versions give an external test driver a separate API object.
  vscode = require('node:module').createRequire(path.join(extension.extensionPath, 'package.json'))('vscode');
  const api = (await extension.activate()).__test;
  await vscode.workspace.getConfiguration('tensorviz').update('pythonPath', python, vscode.ConfigurationTarget.Workspace);
  const originals = {};
  for (const name of ['showQuickPick', 'showInputBox', 'showWarningMessage', 'showInformationMessage', 'showErrorMessage']) originals[name] = vscode.window[name];
  let chosen = python, installs = 0;
  const failures = [];
  const write = (name, value) => fs.writeFileSync(path.join(root, 'evidence', name), JSON.stringify(value, null, 2));
  const settled = async () => {
    const deadline = Date.now() + 60000;
    while (Date.now() < deadline) {
      const state = api.graphOpeningState();
      if (state && ['ready', 'failed', 'needs_setup', 'cancelled'].includes(state.phase)) return state;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error(`Opening did not settle: ${JSON.stringify(api.graphOpeningState())}`);
  };
  const open = async file => {
    const document = await vscode.workspace.openTextDocument(path.join(workspace, file));
    await vscode.window.showTextDocument(document);
    await vscode.commands.executeCommand('tensorviz.openAsGraph', { uri: document.uri.toString(), cls: 'Net' });
    const state = await settled();
    write('latest-opening.json', state);
    return state;
  };
  try {
    vscode.window.showQuickPick = async (items, options) => {
      assert.equal(options.title, 'Choose the Python environment for tensorViz');
      const item = items.find(item => item.choice?.interpreter === chosen);
      return item ?? items.find(item => item.action === 'path');
    };
    vscode.window.showInputBox = async () => chosen;
    vscode.window.showWarningMessage = async (_message, ...args) => args.includes('Review missing libraries') ? 'Review missing libraries' : undefined;
    vscode.window.showInformationMessage = async (_message, ...args) => {
      if (!args.includes('Install missing libraries')) return undefined;
      const detail = args.find(value => typeof value === 'object')?.detail;
      assert.ok(detail.includes(python) && detail.includes('pydantic'));
      assert.ok(!/^torch /m.test(detail));
      installs++;
      return 'Install missing libraries';
    };
    vscode.window.showErrorMessage = async message => { failures.push(message); return undefined; };
    await vscode.commands.executeCommand('tensorviz.selectEnvironment');
    console.log('Installed environment selection and repair finished.');
    assert.deepEqual(failures, []);
    assert.equal(installs, 1);
    const after = inventory();
    for (const [name, version] of Object.entries(before)) assert.equal(after[name], version, `${name} changed`);
    for (const absent of ['pip', 'fastapi', 'uvicorn', 'libcst', 'websockets']) assert.ok(!after[absent], `${absent} must not be added`);
    assert.ok(after.pydantic);
    const service = api.launchToken();
    assert.ok(service);
    fs.writeFileSync(path.join(workspace, 'feedback_model.py'), `from torch import nn
class Net(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        return self.layer(x).relu()
`);
    fs.writeFileSync(path.join(workspace, 'existing_pipeline.py'), `from torch import nn
from feedback_model import Net
class Pipeline:
    def __init__(self):
        self.decoder = Net(nn.Linear(8, 4))
def build_component():
    return Pipeline().decoder
`);
    fs.writeFileSync(path.join(workspace, 'tensorviz.toml'), `version = 1
[profiles.feedback]
kind = "model"
model = "feedback_model:Net"
mode = "eval"
device = "cpu"
grad_enabled = false
factory = {file="existing_pipeline.py", callable="build_component", kwargs={}}
[profiles.feedback.inputs]
args = [{kind="tensor",shape=[2,8],dtype="float32",requires_grad=false,generator={kind="full",value=1}}]
kwargs = {}
`);
    let state = await open('feedback_model.py');
    assert.equal(state.phase, 'ready', JSON.stringify(state));
    assert.equal(state.hasCapture, true);
    assert.match(state.environment.candidates.find(item => item.id === state.environment.selectedId).python, /^3\.10\./);
    assert.equal(state.environment.candidates.find(item => item.id === state.environment.selectedId).interpreter, python);
    write('python310-capture.json', state);
    const alternate = path.join(path.dirname(python), process.platform === 'win32' ? 'python-switch.exe' : 'python-switch');
    fs.copyFileSync(python, alternate);
    fs.chmodSync(alternate, 0o755);
    chosen = alternate;
    await vscode.workspace.getConfiguration('tensorviz').update('pythonPath', alternate, vscode.ConfigurationTarget.Workspace);
    await vscode.commands.executeCommand('tensorviz.selectEnvironment');
    state = await settled();
    assert.equal(state.phase, 'ready', JSON.stringify(state));
    assert.equal(state.environment.candidates.find(item => item.id === state.environment.selectedId).interpreter, alternate);
    assert.equal(api.launchToken(), service, 'The model switch must preserve the service');
    write('environment-switch.json', { passed: true, servicePreserved: true, opening: state });
    fs.writeFileSync(path.join(workspace, 'broken_model.py'), 'import tensorviz_feedback_missing_library\n' + fs.readFileSync(path.join(workspace, 'feedback_model.py'), 'utf8'));
    state = await open('broken_model.py');
    assert.equal(state.phase, 'needs_setup', JSON.stringify(state));
    assert.equal(state.environment.reason, 'missing_libraries');
    assert.match(state.detail, /tensorviz_feedback_missing_library/);
    assert.equal(state.environment.candidates.find(item => item.id === state.environment.selectedId).interpreter, alternate);
    assert.deepEqual(failures, []);
    write('environment-recovery.json', { passed: true, installs, before, after, opening: state });
    console.log('Installed Python 3.10 capture, pipless repair, package preservation, component factory, environment switch and diagnostics passed.');
  } finally { Object.assign(vscode.window, originals); }
  await vscode.commands.executeCommand('tensorviz.openCanvasInBrowser');
  // Keep this disposable window available for native visual acceptance.
  fs.writeFileSync(path.join(root, 'control', 'ready.json'), JSON.stringify({ ready: true }));
  await new Promise((resolve, reject) => {
    let busy = false;
    const timer = setInterval(async () => {
      const file = path.join(root, 'control', 'command.json');
      if (busy || !fs.existsSync(file)) return;
      busy = true;
      try {
        const command = JSON.parse(fs.readFileSync(file, 'utf8')); fs.unlinkSync(file);
        if (command.type === 'finish') { clearInterval(timer); resolve(); return; }
        if (command.type === 'show-model') await open('feedback_model.py');
        if (command.type === 'show-error') await open('broken_model.py');
        write('latest-opening.json', api.graphOpeningState());
      } catch (error) { clearInterval(timer); reject(error); }
      finally { busy = false; }
    }, 250);
  });
};
