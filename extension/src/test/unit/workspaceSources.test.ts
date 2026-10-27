import * as assert from 'node:assert/strict';

import {
  SOURCE_SET_LIMITS,
  SourceIO,
  collectSourceSet,
  sha256Text,
} from '../../workspaceSources';

function sourceIO(options: {
  listed: string[];
  disk?: Record<string, string>;
  open?: Record<string, { text: string; version: number }>;
  fail?: Set<string>;
  reads?: string[];
}): SourceIO {
  return {
    async list() {
      return options.listed;
    },
    async read(relativePath) {
      options.reads?.push(relativePath);
      if (options.fail?.has(relativePath) || options.disk?.[relativePath] === undefined) {
        throw new Error(`gone: ${relativePath}`);
      }
      return options.disk[relativePath];
    },
    openDocument(relativePath) {
      const entry = options.open?.[relativePath];
      return entry
        ? { getText: () => entry.text, version: entry.version }
        : undefined;
    },
  };
}

suite('workspaceSources.collectSourceSet', () => {
  test('a large repository admits the selected model inheritance chain before unrelated alphabetic files', async () => {
    const unrelated = Array.from({ length: SOURCE_SET_LIMITS.maxSources + 20 }, (_, i) => `aaa_tests/test_${String(i).padStart(3, '0')}.py`);
    const disk = Object.fromEntries(unrelated.map(file => [file, 'irrelevant = True\n']));
    Object.assign(disk, {
      'models/kimi/model.py': 'from models.common.decoder import Decoder\nclass KimiModel(Decoder):\n    pass\n',
      'models/common/decoder.py': 'from protocols.model import BaseModel\nclass Decoder(BaseModel):\n    pass\n',
      'protocols/model.py': 'from .module import Module\nclass BaseModel(Module):\n    pass\n',
      'protocols/module.py': 'from torch import nn\nclass Module(nn.Module):\n    pass\n',
    });
    const result = await collectSourceSet('models/kimi/model.py', sourceIO({ listed: Object.keys(disk).reverse(), disk }));
    assert.deepEqual(result.entries.slice(0, 4).map(entry => entry.file),
      ['models/kimi/model.py', 'models/common/decoder.py', 'protocols/model.py', 'protocols/module.py']);
    assert.equal(result.entries.length, SOURCE_SET_LIMITS.maxSources); assert.equal(result.truncated, true);
    assert.ok(!result.entries.some(entry => entry.file.startsWith('torch/')), 'external imports cannot authorize filesystem reads');
  });

  test('local import priority follows packages and relative modules, terminates cycles and uses live buffers', async () => {
    const reads: string[] = [];
    const disk = { 'pkg/model.py': 'from . import block\nimport pkg.exports as exported\nfrom ...escape import Missing\n',
      'pkg/block.py': 'from .model import Net\n', 'pkg/exports/__init__.py': 'from ..base import Base\n',
      'pkg/base.py': 'saved = True\n', 'aaa.py': 'unrelated = True\n' };
    const result = await collectSourceSet('pkg/model.py', sourceIO({ listed: Object.keys(disk), disk, reads,
      open: { 'pkg/base.py': { text: 'live = True\n', version: 7 } } }));
    assert.ok(result.entries.findIndex(entry => entry.file === 'pkg/base.py') < result.entries.findIndex(entry => entry.file === 'aaa.py'));
    assert.equal(result.entries.find(entry => entry.file === 'pkg/base.py')?.source, 'live = True\n');
    assert.equal(result.entries.find(entry => entry.file === 'pkg/base.py')?.document_version, 7);
    assert.equal(new Set(reads).size, reads.length); assert.equal(result.truncated, false);
    assert.ok(reads.every(file => Object.hasOwn(disk, file)), 'only inventoried, normalized Python files can be read');
  });

  test('documentation and commented imports cannot take priority over the actual model base', async () => {
    const disk = { 'model.py': '"""\nfrom aaa import Decoy\n"""\n# from aaa import Decoy\nfrom protocols.base import Base\n',
      'aaa.py': 'class Decoy: pass\n', 'protocols/base.py': 'class Base: pass\n' };
    const result = await collectSourceSet('model.py', sourceIO({ listed: Object.keys(disk), disk }));
    assert.deepEqual(result.entries.map(entry => entry.file), ['model.py', 'protocols/base.py', 'aaa.py']);
  });

  test('is root-first and deterministic, prefers dirty buffers, and rejects unsafe candidates', async () => {
    const reads: string[] = [];
    const result = await collectSourceSet(
      'models/net.py',
      sourceIO({
        listed: [
          'models/z.py',
          'models/net.py',
          'models/a.py',
          'models/z.py',
          '../escape.py',
          'models/readme.txt',
        ],
        disk: {
          'models/a.py': 'A = 1\n',
          'models/z.py': 'saved = True\n',
        },
        open: {
          'models/net.py': { text: 'class Net:\n    pass\n', version: 4 },
          'models/z.py': { text: 'dirty = True\n', version: 7 },
        },
        reads,
      }),
    );

    assert.deepEqual(result.entries.map((entry) => entry.file), [
      'models/net.py',
      'models/a.py',
      'models/z.py',
    ]);
    assert.equal(result.truncated, true, 'unsafe/non-Python candidates are disclosed as omitted');
    assert.deepEqual(reads, ['models/a.py'], 'open buffers never fall through to disk');
    assert.deepEqual(result.entries[0], {
      file: 'models/net.py',
      source: 'class Net:\n    pass\n',
      digest: sha256Text('class Net:\n    pass\n'),
      document_version: 4,
    });
    assert.equal(result.entries[2].source, 'dirty = True\n');
    assert.equal(result.entries[2].document_version, 7);
  });

  test('discloses a vanished dependency but keeps the projection; a missing root fails loud', async () => {
    const result = await collectSourceSet(
      'root.py',
      sourceIO({
        listed: ['root.py', 'alive.py', 'gone.py'],
        disk: { 'root.py': 'root = 1\n', 'alive.py': 'alive = 1\n' },
        fail: new Set(['gone.py']),
      }),
    );
    assert.deepEqual(result.entries.map((entry) => entry.file), ['root.py', 'alive.py']);
    assert.equal(result.truncated, true);

    await assert.rejects(
      collectSourceSet(
        'root.py',
        sourceIO({ listed: [], disk: {}, fail: new Set(['root.py']) }),
      ),
      /gone: root\.py/,
    );
  });

  test('enforces per-file, total-byte, and file-count ceilings without hiding truncation', async () => {
    const tooLarge = 'x'.repeat(SOURCE_SET_LIMITS.maxSourceBytes + 1);
    const perFile = await collectSourceSet(
      'root.py',
      sourceIO({
        listed: ['root.py', 'large.py'],
        disk: { 'root.py': 'root\n', 'large.py': tooLarge },
      }),
    );
    assert.deepEqual(perFile.entries.map((entry) => entry.file), ['root.py']);
    assert.equal(perFile.truncated, true);

    const chunk = 'x'.repeat(900_000);
    const totalDisk: Record<string, string> = { 'root.py': 'root\n' };
    const totalPaths = Array.from({ length: 10 }, (_, index) => `part_${index}.py`);
    totalPaths.forEach((file) => { totalDisk[file] = chunk; });
    const total = await collectSourceSet(
      'root.py',
      sourceIO({ listed: ['root.py', ...totalPaths], disk: totalDisk }),
    );
    assert.equal(total.entries.length, 9, 'root plus eight chunks fit below the total ceiling');
    assert.equal(total.truncated, true);

    const countPaths = Array.from({ length: SOURCE_SET_LIMITS.maxSources + 20 },
      (_, index) => `pkg/f_${String(index).padStart(3, '0')}.py`);
    const countDisk = Object.fromEntries([
      ['root.py', 'root\n'],
      ...countPaths.map((file) => [file, 'x\n']),
    ]);
    const count = await collectSourceSet(
      'root.py',
      sourceIO({ listed: ['root.py', ...countPaths], disk: countDisk }),
    );
    assert.equal(count.entries.length, SOURCE_SET_LIMITS.maxSources);
    assert.equal(count.truncated, true);
  });

  test('refuses an unsafe root before reading anything', async () => {
    await assert.rejects(
      collectSourceSet('../root.py', sourceIO({ listed: [], disk: {} })),
      /invalid workspace-relative Python root/,
    );
  });
});
