/** Installed-host regression. TV_QA_ROOT and TV_QA_PYTHON must be disposable.
 * The Python starts with Torch + backend dependencies except libcst. This test
 * approves a real additive install only in that temporary environment. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const vscode = require('vscode');
exports.run = async function () {
  const root = process.env.TV_QA_ROOT;
  const python = process.env.TV_QA_PYTHON;
  assert.ok(root && python && python.startsWith(path.join(root, 'project-env') + path.sep), 'Refusing a non-disposable Python');
  const extension = vscode.extensions.getExtension('tensorviz.tensorviz-vscode');
  assert.ok(extension);
  const api = (await extension.activate()).__test;
  const inventory = () => JSON.parse(execFileSync(python, ['-c',
    'import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions() if d.metadata["Name"]}))'],
    { encoding: 'utf8', env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' } }));
  const before = inventory();
  assert.ok(before.torch);
  assert.ok(!before.libcst, 'Start this regression with libcst absent');
  let mode = 'cancel-picker';
  let picks = 0, reviews = 0, installs = 0;
  const originals = {};
  const failures = [];
  for (const name of ['showQuickPick', 'showWarningMessage', 'showInformationMessage', 'showErrorMessage']) originals[name] = vscode.window[name];
  vscode.window.showQuickPick = async (items, options) => {
    assert.equal(options?.title, 'Choose the Python environment for tensorViz');
    picks++;
    return mode === 'cancel-picker' ? undefined : items.find(item => item.choice?.interpreter === python);
  };
  vscode.window.showWarningMessage = async (_message, ...args) => {
    if (args.includes('Review missing libraries')) { reviews++; return 'Review missing libraries'; }
    return undefined;
  };
  vscode.window.showInformationMessage = async (message, ...args) => {
    assert.ok(!message.includes('needs to install its own Python runtime'), 'Must not ask to download managed Python');
    if (args.includes('Install missing libraries')) {
      const detail = args.find(arg => arg && typeof arg === 'object')?.detail ?? '';
      assert.ok(detail.includes(python) && detail.includes('libcst'));
      assert.ok(!/^torch /m.test(detail), 'Torch must never appear in the install plan');
      installs++;
      return mode === 'install' ? 'Install missing libraries' : undefined;
    }
    return undefined;
  };
  vscode.window.showErrorMessage = async message => { failures.push(message); return undefined; };
  try {
    await vscode.commands.executeCommand('tensorviz.checkEnvironment');
    assert.equal(picks, 1);
    assert.deepEqual(inventory(), before);
    assert.ok(!fs.existsSync(path.join(api.globalStorageDir(), 'runtime')));
    mode = 'cancel-install';
    await vscode.commands.executeCommand('tensorviz.checkEnvironment');
    assert.equal(reviews, 1); assert.equal(installs, 1);
    assert.deepEqual(inventory(), before, 'Install dismissal must change no package');
    mode = 'install';
    await vscode.commands.executeCommand('tensorviz.checkEnvironment');
    assert.deepEqual(failures, []);
    assert.equal(installs, 2);
    const after = inventory();
    for (const [name, version] of Object.entries(before)) assert.equal(after[name], version, `${name} changed`);
    assert.ok(after.libcst);
    const pickedCount = picks;
    await vscode.commands.executeCommand('tensorviz.checkEnvironment');
    assert.equal(picks, pickedCount, 'Compatible remembered environment should be reused');
    assert.equal(installs, 2, 'No repeat install prompt');
    assert.ok(!fs.existsSync(path.join(api.globalStorageDir(), 'runtime')), 'Managed runtime must remain absent');
    fs.writeFileSync(path.join(root, 'evidence/project-environment.json'), JSON.stringify({
      passed: true, picks, reviews, installs, preservedPackages: Object.keys(before).length,
      torch: after.torch, added: Object.keys(after).filter(name => !(name in before)),
    }, null, 2));
    console.log('Installed project-environment selection, cancellation, additive install, and reuse: passed');
  } finally {
    Object.assign(vscode.window, originals);
  }
  await require('./qa-local-browser.cjs').run();
};
