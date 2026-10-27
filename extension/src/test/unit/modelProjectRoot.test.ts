import * as assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { modelProjectRoot } from '../../modelProjectRoot';

suite('model project root', () => {
  let workspace: string;
  setup(async () => { workspace = await mkdtemp(path.join(tmpdir(), 'tensorviz-project-')); });
  teardown(async () => { await rm(workspace, { recursive: true, force: true }); });

  for (const [project, file] of [
    ['graph_ui', 'residual.py'], ['mcp_diagnostics', 'demo_models.py'],
    ['tensorviz-clean', 'models/mlp_tiny.py'], ['tensorviz-trust-test', 'models/mlp_tiny.py'],
    ['tensorviz-demo', 'custom/net.py'], ['tensorviz-sweep', 'models/resid.py'],
    ['tensorviz-test', 'models/simple.py'], ['tensorviz-research', 'models/attention.py'],
  ]) test(`the shipped ${project} models keep their root when opened from the repository`, () => {
    const repo = path.resolve(__dirname, '../../../..');
    const root = path.join(repo, 'examples', project);
    const model = path.join(root, file);
    assert.equal(modelProjectRoot(model, repo), root);
    assert.equal(modelProjectRoot(model, root), root);
  });

  for (const marker of ['tensorviz.toml', 'pyproject.toml', 'setup.py', 'setup.cfg', '.git']) {
    test(`uses the nearest ${marker} inside the workspace`, async () => {
      const project = path.join(workspace, 'examples', 'research');
      const file = path.join(project, 'models', 'net.py');
      await mkdir(path.dirname(file), { recursive: true });
      await writeFile(path.join(workspace, 'tensorviz.toml'), 'version = 1');
      await writeFile(path.join(project, marker), '');
      assert.equal(modelProjectRoot(file, workspace), project);
      assert.equal(modelProjectRoot(file, project), project, 'opening the subfolder must use the same identity');
    });
  }

  test('recognizes a nested Git checkout directory', async () => {
    const project = path.join(workspace, 'research');
    await mkdir(path.join(project, '.git'), { recursive: true });
    assert.equal(modelProjectRoot(path.join(project, 'model.py'), workspace), project);
  });

  test('does not bypass an invalid nested configuration at an outer root', async () => {
    const project = path.join(workspace, 'research');
    const config = path.join(project, 'tensorviz.toml');
    await mkdir(config, { recursive: true });
    assert.equal(modelProjectRoot(path.join(project, 'model.py'), workspace), project);
    await rm(config, { recursive: true });
    await symlink(path.join(project, 'missing.toml'), config);
    assert.equal(modelProjectRoot(path.join(project, 'model.py'), workspace), project);
  });

  for (const name of ['tensorviz-clean', 'my model', '123model']) {
    test(`keeps the non-module directory ${name} out of the Python import path`, async () => {
      const project = path.join(workspace, 'examples', name);
      await mkdir(path.join(project, 'models'), { recursive: true });
      assert.equal(modelProjectRoot(path.join(project, 'models', 'net.py'), workspace), project);
    });
  }

  test('preserves the workspace for unmarked Python packages, including a nonexistent live file', async () => {
    const file = path.join(workspace, 'package', 'models', 'net.py');
    assert.equal(modelProjectRoot(file, workspace), workspace);
    assert.equal(modelProjectRoot(file), path.dirname(file));
  });

  test('does not consult metadata outside the owning folder or accept a prefix sibling', async () => {
    const nested = path.join(workspace, 'project', 'models');
    await mkdir(nested, { recursive: true });
    await writeFile(path.join(workspace, 'project', 'tensorviz.toml'), '');
    assert.equal(modelProjectRoot(path.join(nested, 'net.py'), nested), nested);
    assert.throws(() => modelProjectRoot(path.join(`${nested}_other`, 'net.py'), nested), /outside/);
  });
});
