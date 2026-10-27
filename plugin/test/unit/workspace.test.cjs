const { test } = require('node:test');
const assert = require('node:assert/strict');
const { mkdtemp, mkdir, writeFile, symlink, rm } = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { workspaceRoot, savedFile, sourceIO } = require('../../out/host/workspace.cjs');
const { Environments } = require('../../out/host/environment.cjs');
const { ProfileChoices } = require('../../out/host/profileChoices.cjs');

test('explicit workspace identity isolates worktrees and rejects files escaping through symlinks', async () => {
  const root = await workspaceRoot(await mkdtemp(path.join(os.tmpdir(), 'tensorviz-workspaces-')));
  try {
    const first = path.join(root, 'first'), second = path.join(root, 'second');
    await mkdir(first); await mkdir(second);
    await writeFile(path.join(first, 'model.py'), 'first'); await writeFile(path.join(second, 'model.py'), 'second');
    assert.notEqual(await workspaceRoot(first), await workspaceRoot(second));
    await assert.rejects(workspaceRoot(), { code: 'WORKSPACE_REQUIRED' });
    assert.equal((await savedFile(first, 'model.py')).text, 'first');
    await assert.rejects(savedFile(first, '../second/model.py'), { code: 'OUTSIDE_WORKSPACE' });
    await symlink(path.join(second, 'model.py'), path.join(first, 'alias.py'));
    await assert.rejects(savedFile(first, 'alias.py'), { code: 'OUTSIDE_WORKSPACE' });
    await mkdir(path.join(first, '.venv')); await writeFile(path.join(first, '.venv', 'ignored.py'), 'ignored');
    assert.deepEqual(await sourceIO(first).list(), ['model.py']);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('invalid explicit or saved Python never silently selects shell or another project environment', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'tensorviz-environments-'));
  try {
    const environments = new Environments('/not-used', path.join(root, 'state'));
    assert.equal((await environments.resolve(root)).source, 'selection_required');
    const candidate = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts' : 'bin');
    await mkdir(candidate, { recursive: true });
    await writeFile(path.join(candidate, process.platform === 'win32' ? 'python.exe' : 'python'), 'not executed by discovery');
    assert.equal((await environments.resolve(root)).source, 'selection_required', 'a unique project environment is still a suggestion');
    const selected = path.join(candidate, process.platform === 'win32' ? 'python.exe' : 'python');
    assert.ok((await environments.resolve(root)).candidates.includes(selected));
    await environments.remember(root, selected, true);
    assert.equal((await environments.resolve(root)).source, 'managed');
    const missing = await environments.resolve(root, path.join(root, 'missing'));
    assert.equal(missing.source, 'selection_required');
    assert.equal(missing.selected.interpreter, path.join(root, 'missing'));
    await environments.remember(root, path.join(root, 'missing'));
    assert.equal((await environments.resolve(root)).selected.interpreter, path.join(root, 'missing'));
    const other = path.join(root, 'other'); await mkdir(other);
    assert.equal((await environments.resolve(other)).source, 'selection_required');
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('saved profile choices isolate exact workspaces and model classes', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'tensorviz-profile-choices-'));
  try {
    const choices = new ProfileChoices(root);
    await choices.remember('/first/worktree', 'model.py', 'Net', 'large');
    const restored = new ProfileChoices(root);
    assert.equal(await restored.read('/first/worktree', 'model.py', 'Net'), 'large');
    assert.equal(await restored.read('/second/worktree', 'model.py', 'Net'), undefined);
    assert.equal(await restored.read('/first/worktree', 'model.py', 'Child'), undefined);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('past explicit environments are offered without borrowing managed project packages', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'tensorviz-environment-choices-'));
  try {
    const environments = new Environments('/not-used', path.join(root, 'state'));
    const bin = path.join(root, 'python'); await writeFile(bin, 'discovery never executes Python');
    const privateBin = path.join(root, 'private-python'); await writeFile(privateBin, 'isolated packages');
    await environments.remember('/previous', bin);
    await environments.remember('/duplicate', bin);
    await environments.remember('/private', privateBin, true);
    await environments.remember('/removed', path.join(root, 'missing'));
    const fresh = path.join(root, 'fresh'); await mkdir(fresh);
    const choices = await environments.resolve(fresh);
    assert.equal(choices.source, 'selection_required');
    assert.ok(choices.candidates.includes(bin));
    assert.ok(!choices.candidates.includes(privateBin));
    assert.equal(choices.candidates.filter(value => value === bin).length, 1);
    assert.match(choices.detail, /Compatibility will be checked/);
    assert.equal((await environments.resolve(fresh, bin)).source, 'explicit');
    await environments.remember(fresh, bin);
    assert.equal((await environments.resolve(fresh)).source, 'saved');
  } finally { await rm(root, { recursive: true, force: true }); }
});
