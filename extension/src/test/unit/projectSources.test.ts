import * as assert from 'node:assert/strict';

import {
  PROJECT_LIMITS,
  ProjectSourceIO,
  bodiesFor,
  collectProjectSources,
  hasSkippedSegment,
  isIgnored,
  parseIgnore,
  sha256Text,
} from '../../projectSources';

function io(options: {
  listed: string[];
  disk?: Record<string, string>;
  open?: Record<string, string>;
  ignore?: string;
  fail?: Set<string>;
}): ProjectSourceIO {
  return {
    async list() {
      return options.listed;
    },
    async read(relativePath) {
      if (options.fail?.has(relativePath) || options.disk?.[relativePath] === undefined) {
        throw new Error(`gone: ${relativePath}`);
      }
      return options.disk[relativePath];
    },
    openDocument(relativePath) {
      const text = options.open?.[relativePath];
      return text === undefined
        ? undefined
        : ({ getText: () => text, version: 1 } as never);
    },
    async readIgnore() {
      return options.ignore;
    },
  };
}

suite('projectSources — .gitignore', () => {
  test('a plain directory pattern excludes everything beneath it', () => {
    const rules = parseIgnore('wandb/\n');
    assert.equal(isIgnored('wandb/run-1/files/x.py', rules), true);
    assert.equal(isIgnored('src/wandb_helper.py', rules), false);
  });

  test('a glob matches at any depth unless anchored', () => {
    const rules = parseIgnore('*.generated.py\n');
    assert.equal(isIgnored('a/b/thing.generated.py', rules), true);
    assert.equal(isIgnored('a/b/thing.py', rules), false);
  });

  test('a leading slash anchors to the root', () => {
    const rules = parseIgnore('/build\n');
    assert.equal(isIgnored('build/x.py', rules), true);
    assert.equal(isIgnored('pkg/build/x.py', rules), false);
  });

  test('the last matching rule wins, so a negation re-includes', () => {
    const rules = parseIgnore('data/\n!data/keep.py\n');
    assert.equal(isIgnored('data/big.py', rules), true);
    assert.equal(
      isIgnored('data/keep.py', rules),
      false,
      'a later !rule must re-include, as git does',
    );
  });

  test('an unreadable pattern is skipped rather than approximated', () => {
    // A wrong exclusion hides a file the user can see, which is worse than a file
    // we failed to exclude.
    assert.deepEqual(parseIgnore('*.[ab]\n'), []);
  });

  test('comments and blank lines are not rules', () => {
    assert.deepEqual(parseIgnore('# comment\n\n   \n'), []);
  });
});

suite('projectSources — always-skipped directories', () => {
  test('caches, virtualenvs and vcs metadata never enter the manifest', () => {
    for (const rel of [
      '.git/config.py',
      'a/__pycache__/x.py',
      '.venv/lib/x.py',
      'node_modules/x/y.py',
      '.tensorviz/state.py',
    ]) {
      assert.equal(hasSkippedSegment(rel), true, `${rel} should be skipped`);
    }
    assert.equal(hasSkippedSegment('src/venv_utils.py'), false);
  });
});

suite('projectSources — collection', () => {
  test('digests exactly the text it will send', async () => {
    const set = await collectProjectSources(
      io({ listed: ['a.py'], disk: { 'a.py': 'X = 1\n' } }),
    );
    assert.deepEqual(set.manifest, [{ file: 'a.py', digest: sha256Text('X = 1\n') }]);
    assert.equal(set.texts.get('a.py'), 'X = 1\n');
  });

  test('an open dirty buffer wins over disk', async () => {
    const set = await collectProjectSources(
      io({ listed: ['a.py'], disk: { 'a.py': 'ON_DISK = 1\n' }, open: { 'a.py': 'EDITED = 1\n' } }),
    );
    assert.equal(set.texts.get('a.py'), 'EDITED = 1\n');
    assert.equal(set.manifest[0].digest, sha256Text('EDITED = 1\n'));
  });

  test('ignored files are absent from the manifest', async () => {
    const set = await collectProjectSources(
      io({
        listed: ['train.py', 'wandb/run/x.py'],
        disk: { 'train.py': 'A = 1\n', 'wandb/run/x.py': 'B = 1\n' },
        ignore: 'wandb/\n',
      }),
    );
    assert.deepEqual(
      set.manifest.map((entry) => entry.file),
      ['train.py'],
    );
  });

  test('a file that vanishes between listing and reading is skipped, not fatal', async () => {
    const set = await collectProjectSources(
      io({ listed: ['a.py', 'gone.py'], disk: { 'a.py': 'X = 1\n' }, fail: new Set(['gone.py']) }),
    );
    assert.deepEqual(
      set.manifest.map((entry) => entry.file),
      ['a.py'],
    );
  });

  test('an oversized file is excluded and admitted as truncation', async () => {
    const big = 'x'.repeat(PROJECT_LIMITS.maxBodyBytes + 1);
    const set = await collectProjectSources(
      io({ listed: ['big.py', 'a.py'], disk: { 'big.py': big, 'a.py': 'X = 1\n' } }),
    );
    assert.deepEqual(
      set.manifest.map((entry) => entry.file),
      ['a.py'],
      'a body we could never send must not be promised in the manifest',
    );
    assert.equal(set.truncated, true);
  });

  test('the manifest holds far more than the multi-file source cap', async () => {
    // The whole reason this module exists: 256 was the old ceiling.
    const listed = Array.from({ length: 900 }, (_unused, index) => `pkg/mod_${index}.py`);
    const disk = Object.fromEntries(listed.map((rel) => [rel, 'X = 1\n']));
    const set = await collectProjectSources(io({ listed, disk }));
    assert.equal(set.manifest.length, 900);
    assert.equal(set.truncated, false);
  });
});

suite('projectSources — bodies handshake', () => {
  const set = {
    manifest: [],
    texts: new Map([
      ['a.py', 'A = 1\n'],
      ['b.py', 'B = 1\n'],
    ]),
    truncated: false,
    discovered: 2,
  };

  test('sends bodies only for the paths the backend asked for', () => {
    const { bodies, deferred } = bodiesFor(set, ['b.py']);
    assert.deepEqual(bodies, [{ file: 'b.py', source: 'B = 1\n' }]);
    assert.deepEqual(deferred, []);
  });

  test('asking for nothing sends nothing', () => {
    assert.deepEqual(bodiesFor(set, []).bodies, []);
  });

  test('a request beyond the body budget defers rather than failing', () => {
    const many = new Map<string, string>();
    const wanted: string[] = [];
    for (let index = 0; index <= PROJECT_LIMITS.maxBodies + 10; index += 1) {
      const rel = `m${index}.py`;
      many.set(rel, 'X = 1\n');
      wanted.push(rel);
    }
    const { bodies, deferred } = bodiesFor(
      { manifest: [], texts: many, truncated: false, discovered: many.size },
      wanted,
    );
    assert.equal(bodies.length, PROJECT_LIMITS.maxBodies);
    assert.equal(deferred.length, 11, 'the rest wait for the next round');
  });

  test('a path the backend asks for that we no longer have is skipped', () => {
    assert.deepEqual(bodiesFor(set, ['ghost.py']).bodies, []);
  });
});
