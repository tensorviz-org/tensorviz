import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as vscode from 'vscode';

import { OPEN_AS_GRAPH } from '../../commandIds';

// Run in a fresh host: TENSORVIZ_TEST_ONLY=coldDiscovery npm run test:electron
suite('coldDiscovery', () => {
  test('imported bases acquire CodeLenses without opening Canvas, then survive save and reopen', async function () {
    this.timeout(120_000);
    assert.ok(process.env.TENSORVIZ_TEST_PYTHON, 'TENSORVIZ_TEST_PYTHON is required');
    const extension = vscode.extensions.getExtension('tensorviz.tensorviz');
    assert.ok(extension);
    const api = (await extension.activate()).__test;
    assert.equal(api.backendOrigin(), undefined, 'a warm backend would mask the startup bug');
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorviz-cold-discovery-'));
    const uri = vscode.Uri.file(path.join(root, 'model.py'));
    const config = vscode.workspace.getConfiguration('tensorviz');
    const previous = config.inspect('repoPath')?.globalValue;
    const bases = [
      ['transformers', 'PreTrainedModel'], ['diffusers', 'ModelMixin'],
      ['lightning.pytorch', 'LightningModule'], ['peft', 'PeftModel'],
      ['timm.models.vision_transformer', 'VisionTransformer'],
      ['torchvision.models', 'ResNet'], ['torch_geometric.nn', 'MessagePassing'],
      ['mmengine.model', 'BaseModel'],
    ];
    const source = [
      'raise RuntimeError("Discovery must never execute model source")',
      ...bases.map(([module, base], i) => `from ${module} import ${base} as Base${i}`),
      'class SmartTurnMixin:\n    pass',
      ...bases.map((_, i) => `class Model${i}(SmartTurnMixin, Base${i}):\n    pass`),
      'class Child(Model0):\n    pass',
      'class Settings(UnknownBase):\n    pass',
    ].join('\n') + '\n';
    const expectedNames = [...bases.map((_, i) => `Model${i}`), 'Child'].sort();
    const expectLenses = async (offset: number) => {
      const deadline = Date.now() + 90_000;
      for (;;) {
        const lenses = (await vscode.commands.executeCommand<vscode.CodeLens[]>(
          'vscode.executeCodeLensProvider', uri,
        ) ?? []).filter((lens) => lens.command?.command === OPEN_AS_GRAPH);
        const found = lenses.map((lens) => lens.command!.arguments![0].cls).sort();
        if (JSON.stringify(found) === JSON.stringify(expectedNames) && lenses.every((lens) => {
          const name = lens.command!.arguments![0].cls;
          return lens.range.start.line === source.split('\n').findIndex((line) => line.startsWith(`class ${name}(`)) + offset;
        })) return;
        assert.ok(Date.now() < deadline, `Expected imported model lenses; got ${JSON.stringify(found)}`);
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    };
    try {
      fs.writeFileSync(uri.fsPath, source);
      await config.update('repoPath', root, vscode.ConfigurationTarget.Global);
      const document = await vscode.workspace.openTextDocument(uri);
      const editor = await vscode.window.showTextDocument(document);
      await expectLenses(0);
      assert.ok(api.backendOrigin(), 'opening the imported subclass must start discovery');
      assert.equal(api.localBrowserTarget(), undefined, 'discovery must not open a graph');
      assert.ok(await editor.edit((edit) => edit.insert(new vscode.Position(0, 0), '# saved edit\n')));
      assert.ok(await document.save());
      await expectLenses(1);
      await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
      await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(uri));
      await expectLenses(1);
    } finally {
      await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
      await config.update('repoPath', previous, vscode.ConfigurationTarget.Global);
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});
