import * as assert from 'node:assert/strict';

import { ProjectSourceIO } from '../../projectSources';
import { indexOrder, indexWorkspaceFolder, projectIdentityFor } from '../../projectIndex';
import { sha256 } from '../../sourceChanges';

function io(disk: Record<string, string>): ProjectSourceIO {
  return {
    async list() {
      return Object.keys(disk);
    },
    async read(relativePath) {
      const text = disk[relativePath];
      if (text === undefined) {
        throw new Error(`gone: ${relativePath}`);
      }
      return text;
    },
    openDocument() {
      return undefined;
    },
    async readIgnore() {
      return undefined;
    },
  };
}

/** A fake backend that records what it was sent and replays scripted answers. */
function backend(script: Array<Record<string, unknown>>) {
  const sent: Array<{ manifest: number; bodies: string[] }> = [];
  let call = 0;
  const post = async (_url: string, _token: string, body: unknown) => {
    const payload = body as { manifest: unknown[]; bodies: Array<{ file: string }> };
    sent.push({
      manifest: payload.manifest.length,
      bodies: payload.bodies.map((entry) => entry.file),
    });
    const answer = script[Math.min(call, script.length - 1)];
    call += 1;
    return {
      ok: true,
      status: 200,
      json: async () => answer,
    } as unknown as Response;
  };
  return { post, sent };
}

const OPTIONS = {
  origin: () => 'http://127.0.0.1:8765',
  token: () => 'launch-token',
};

const DISK = { 'train.py': 'X = 1\n', 'models/net.py': 'Y = 2\n' };

suite('projectIndex — the manifest/bodies handshake', () => {
  test('sends a manifest and NO bodies first', async () => {
    // The point of the handshake: on an unchanged workspace the backend already
    // holds every digest, so the common case ships no source at all.
    const fake = backend([{ status: 'ok', index: {}, digest: 'd', indexed: 2 }]);
    await indexWorkspaceFolder('/w', { ...OPTIONS, sourceIO: () => io(DISK), post: fake.post });

    assert.equal(fake.sent.length, 1);
    assert.equal(fake.sent[0].manifest, 2);
    assert.deepEqual(fake.sent[0].bodies, []);
  });

  test('answers need_bodies with exactly the paths asked for', async () => {
    const fake = backend([
      { status: 'need_bodies', missing: ['train.py'] },
      { status: 'ok', index: {}, digest: 'd', indexed: 2 },
    ]);
    const result = await indexWorkspaceFolder('/w', {
      ...OPTIONS,
      sourceIO: () => io(DISK),
      post: fake.post,
    });

    assert.deepEqual(fake.sent[1].bodies, ['train.py'], 'only the missing file is sent');
    assert.equal(result?.indexed, 2);
    assert.deepEqual(result?.sources, new Map(Object.entries(DISK).map(([file, text]) => [file, sha256(text)])),
      'review evidence must describe the source actually admitted, including cached bodies');
  });

  test('settles over several rounds when the first batch does not fit', async () => {
    const fake = backend([
      { status: 'need_bodies', missing: ['train.py'] },
      { status: 'need_bodies', missing: ['models/net.py'] },
      { status: 'ok', index: {}, digest: 'd', indexed: 2 },
    ]);
    const result = await indexWorkspaceFolder('/w', {
      ...OPTIONS,
      sourceIO: () => io(DISK),
      post: fake.post,
    });

    assert.equal(fake.sent.length, 3);
    assert.ok(result, 'a multi-round index must still succeed');
  });

  test('gives up loudly when the backend asks twice for the same bodies', async () => {
    // A silent retry loop here would look exactly like a hang.
    const logged: string[] = [];
    const fake = backend([{ status: 'need_bodies', missing: ['train.py'] }]);
    const result = await indexWorkspaceFolder('/w', {
      ...OPTIONS,
      sourceIO: () => io(DISK),
      post: fake.post,
      log: (message) => logged.push(message),
    });

    assert.equal(result, undefined);
    assert.ok(logged.some((line) => line.includes('same bodies')));
  });

  test('gives up when asked for a path it no longer holds', async () => {
    const logged: string[] = [];
    const fake = backend([{ status: 'need_bodies', missing: ['ghost.py'] }]);
    const result = await indexWorkspaceFolder('/w', {
      ...OPTIONS,
      sourceIO: () => io(DISK),
      post: fake.post,
      log: (message) => logged.push(message),
    });

    assert.equal(result, undefined);
    assert.ok(logged.some((line) => line.includes('no longer hold')));
  });

  test('a refusal is reported, not retried', async () => {
    const logged: string[] = [];
    const result = await indexWorkspaceFolder('/w', {
      ...OPTIONS,
      sourceIO: () => io(DISK),
      post: async () => ({ ok: false, status: 400, json: async () => ({}) }) as unknown as Response,
      log: (message) => logged.push(message),
    });

    assert.equal(result, undefined);
    assert.ok(logged.some((line) => line.includes('refused')));
  });
});

suite('projectIndex — degenerate inputs', () => {
  test('a folder with no Python is an empty project, not a failure', async () => {
    const result = await indexWorkspaceFolder('/w', {
      ...OPTIONS,
      sourceIO: () => io({}),
      post: async () => {
        throw new Error('must not call the backend for an empty folder');
      },
    });
    assert.equal(result?.indexed, 0);
    assert.equal(result?.truncated, false);
  });

  test('no backend yet means no index, not a crash', async () => {
    const result = await indexWorkspaceFolder('/w', {
      origin: () => undefined,
      token: () => undefined,
      sourceIO: () => io(DISK),
    });
    assert.equal(result, undefined);
  });

  test('discovery count is reported for the honest banner', async () => {
    const fake = backend([{ status: 'ok', index: {}, digest: 'd', indexed: 2 }]);
    const result = await indexWorkspaceFolder('/w', {
      ...OPTIONS,
      sourceIO: () => io(DISK),
      post: fake.post,
    });
    assert.equal(result?.discovered, 2);
  });
});

suite('projectIndex — multi-root ordering', () => {
  test('the folder holding the active editor is indexed first', () => {
    // A three-root workspace must not pay 3x before showing the root the user is
    // actually looking at.
    const order = indexOrder(['/a', '/b', '/c'], '/b/train.py');
    assert.deepEqual(order, ['/b', '/a', '/c']);
  });

  test('a nested folder beats its parent', () => {
    const order = indexOrder(['/repo', '/repo/packages/alpha'], '/repo/packages/alpha/train.py');
    assert.equal(order[0], '/repo/packages/alpha');
  });

  test('a file outside every folder leaves the order alone', () => {
    assert.deepEqual(indexOrder(['/a', '/b'], '/elsewhere/x.py'), ['/a', '/b']);
  });

  test('no active editor leaves the order alone', () => {
    assert.deepEqual(indexOrder(['/a', '/b'], undefined), ['/a', '/b']);
  });
});

suite('projectIndex — identity', () => {
  test('carries the derived root and the focused experiment', () => {
    const identity = projectIdentityFor('/w', 'packages/alpha', 'exp:1', 'a'.repeat(64), 3);
    assert.equal(identity.projectRoot, 'packages/alpha');
    assert.equal(identity.experimentId, 'exp:1');
    assert.equal(identity.indexRevision, 3);
  });

  test('an unfocused project is a real state', () => {
    assert.equal(projectIdentityFor('/w', '', '', 'a'.repeat(64), 0).experimentId, '');
  });
});
