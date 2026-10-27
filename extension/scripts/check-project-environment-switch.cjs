/** Installed regression using only the disposable QA environment. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');
exports.run = async function () {
  const root = process.env.TV_QA_ROOT;
  const python = process.env.TV_QA_PYTHON;
  assert.ok(root && python && python.startsWith(path.join(root, 'project-env') + path.sep), 'Refusing a non-disposable Python');
  const alternate = path.join(path.dirname(python), `python-switch-${process.pid}${process.platform === 'win32' ? '.exe' : ''}`);
  assert.ok(!fs.existsSync(alternate));
  fs.copyFileSync(python, alternate, fs.constants.COPYFILE_EXCL);
  fs.chmodSync(alternate, 0o755);
  const parked = python + '.qa-parked';
  assert.ok(!fs.existsSync(parked));
  const config = vscode.workspace.getConfiguration('tensorviz');
  const configured = config.get('pythonPath');
  const originals = {};
  for (const name of ['showQuickPick', 'showWarningMessage', 'showInformationMessage', 'showErrorMessage']) originals[name] = vscode.window[name];
  const failures = [], notifications = [];
  let initial = true;
  let cancel = true;
  let picks = 0;
  const extension = vscode.extensions.getExtension('tensorviz.tensorviz');
  assert.ok(extension);
  const api = (await extension.activate()).__test;
  try {
    vscode.window.showQuickPick = async (items, options) => {
      assert.equal(options?.title, 'Choose the Python environment for tensorViz');
      picks++;
      if (cancel && !initial) return undefined;
      const picked = items.find(item => item.choice?.interpreter === (initial ? python : alternate));
      assert.ok(picked, 'replacement Python must be offered');
      return picked;
    };
    vscode.window.showWarningMessage = async (_message, ...args) => args.includes('Choose environment') ? 'Choose environment' : undefined;
    vscode.window.showInformationMessage = async message => { notifications.push(message); return undefined; };
    vscode.window.showErrorMessage = async message => { failures.push(message); return undefined; };
    await vscode.commands.executeCommand('tensorviz.selectEnvironment');
    assert.deepEqual(failures, []);
    initial = false;
    picks = 0;
    await require('./check-installed-workspace.cjs').run();
    const before = api.launchToken();
    assert.ok(before);
    await vscode.commands.executeCommand('tensorviz.selectEnvironment');
    assert.equal(picks, 1);
    assert.ok(api.launchToken() === before, 'cancelling a switch must retain the running backend');
    cancel = false;
    await config.update('pythonPath', alternate, vscode.ConfigurationTarget.Global);
    fs.renameSync(python, parked);
    await vscode.commands.executeCommand('tensorviz.checkEnvironment');
    assert.deepEqual(failures, []);
    assert.equal(picks, 2, 'failed compatibility must allow another environment');
    assert.ok(api.launchToken());
    assert.ok(api.launchToken() !== before, 'the backend must restart when Check Environment selects a different Python');
    assert.ok(notifications.some(message => message.includes('Open as Graph to reconnect')));
    await require('./check-installed-workspace.cjs').run();
    assert.ok(new URL(api.localBrowserSession().pageUrl).origin === api.backendOrigin(), 'the browser session must belong to the replacement backend');
    assert.ok(!fs.existsSync(path.join(api.globalStorageDir(), 'runtime')));
    fs.writeFileSync(path.join(root, 'evidence/environment-switch.json'), JSON.stringify({passed:true,cancelPreservedBackend:true,recoveryRestartedBackend:true}));
    console.log('Installed environment switch cancellation, recovery restart and graph reconnect: passed');
  } finally {
    if (fs.existsSync(parked)) fs.renameSync(parked, python);
    await config.update('pythonPath', configured, vscode.ConfigurationTarget.Global);
    Object.assign(vscode.window, originals);
    // Keep the selected disposable alias available for a subsequent reuse smoke test.
  }
};
