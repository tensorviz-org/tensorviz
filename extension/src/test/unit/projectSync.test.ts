import * as assert from 'node:assert/strict';

import { IndexedProject, ProjectSync } from '../../projectSync';

const DIGEST_A = 'a'.repeat(64);
const DIGEST_B = 'b'.repeat(64);

function project(experiments: Array<[string, string, string?]>): IndexedProject {
  return {
    experiments: experiments.map(([id, label, root]) => ({
      experiment_id: id,
      label,
      entrypoint: `${label}.py`,
      root: root ?? '',
      closure: [`${label}.py`],
    })),
    unreached: [],
    roots: [''],
    truncated: false,
  };
}

function harness(options: { index?: IndexedProject; digest?: string } = {}) {
  const published: any[] = [];
  const logged: string[] = [];
  const documents: string[] = [];
  let index = options.index ?? project([['exp:1', 'train'], ['exp:2', 'sweep']]);
  let digest = options.digest ?? DIGEST_A;

  const sync = new ProjectSync({
    async index() {
      return { index, digest, indexed: 1, discovered: 1, truncated: index.truncated };
    },
    async document(_folder, experimentId, expanded) {
      documents.push(`${experimentId}:${expanded.join(',')}`);
      return { document: { experimentId }, spanmap: {}, expanded };
    },
    publish: (message) => published.push(message),
    log: (message) => logged.push(message),
  });

  return {
    sync,
    published,
    logged,
    documents,
    setIndex(next: IndexedProject, nextDigest: string) {
      index = next;
      digest = nextDigest;
    },
  };
}

suite('projectSync — opening', () => {
  test('an index without host source evidence cannot authorize a declaration', async () => {
    const h = harness();
    await h.sync.open('/w');
    assert.equal(h.sync.declarationContext(h.sync.identity()!), undefined);
  });

  test('publishes an index and then a document for the first experiment', async () => {
    const h = harness();
    await h.sync.open('/w');

    assert.deepEqual(
      h.published.map((m) => m.type),
      ['projectIndex', 'projectDocument'],
      'the index must arrive before the document drawn from it',
    );
    assert.equal(h.published[1].project.experimentId, 'exp:1');
  });

  test('the identity carries the focused experiment and its derived root', async () => {
    const h = harness({ index: project([['exp:1', 'train', 'packages/alpha']]) });
    await h.sync.open('/w');

    const identity = h.sync.identity();
    assert.equal(identity?.projectRoot, 'packages/alpha');
    assert.equal(identity?.experimentId, 'exp:1');
  });

  test('a workspace with no experiments publishes an index and no document', async () => {
    const h = harness({ index: project([]) });
    await h.sync.open('/w');

    assert.deepEqual(h.published.map((m) => m.type), ['projectIndex']);
  });
});

suite('projectSync — declaration authority', () => {
  test('re-index failure and a late index after invalidation cannot restore old write authority', async () => {
    let result: any = { index: project([['exp:1', 'train']]), digest: DIGEST_A, indexed: 1, discovered: 1,
      truncated: false, sources: new Map([['train.py', DIGEST_B]]) };
    let release!: (value: any) => void;
    const sync = new ProjectSync({ index: async () => result, document: async () => ({ document: {}, spanmap: {}, expanded: [] }), publish() {} });
    await sync.open('/w');
    const identity = sync.identity()!;
    assert.equal(sync.declarationContext(identity)?.entrypoint, 'train.py');
    result = new Promise(resolve => { release = resolve; });
    const pending = sync.open('/w');
    assert.equal(sync.declarationContext(identity), undefined, 'old source authority is revoked before indexing completes');
    release(undefined); await assert.rejects(pending, /index unavailable/);
    assert.equal(sync.declarationContext(identity), undefined, 'failed indexing cannot reuse old authority');
    result = new Promise(resolve => { release = resolve; });
    const late = sync.open('/w');
    sync.invalidateDeclarations();
    release({ index: project([['exp:1', 'train']]), digest: DIGEST_A, indexed: 1, discovered: 1,
      truncated: false, sources: new Map([['train.py', DIGEST_B]]) });
    await late;
    assert.equal(sync.declarationContext(identity), undefined, 'retarget/dispose must win over late success');
  });
});

suite('projectSync — re-indexing', () => {
  test('keeps the focused experiment so typing does not move the view', async () => {
    const h = harness();
    await h.sync.open('/w');
    await h.sync.focus(h.sync.identity()!, 'exp:2');

    h.setIndex(project([['exp:1', 'train'], ['exp:2', 'sweep']]), DIGEST_B);
    await h.sync.open('/w');

    assert.equal(h.sync.state().focused, 'exp:2');
  });

  test('falls back when the focused experiment is gone', async () => {
    const h = harness();
    await h.sync.open('/w');
    await h.sync.focus(h.sync.identity()!, 'exp:2');

    h.setIndex(project([['exp:1', 'train']]), DIGEST_B);
    await h.sync.open('/w');

    assert.equal(h.sync.state().focused, 'exp:1');
  });

  test('the revision advances so two indexes are ordered', async () => {
    const h = harness();
    await h.sync.open('/w');
    const first = h.sync.state().revision;
    h.setIndex(project([['exp:1', 'train']]), DIGEST_B);
    await h.sync.open('/w');

    assert.ok(h.sync.state().revision > first);
  });
});

suite('projectSync — refusing a stale page', () => {
  test('a focus request against an old index is refused, not served', async () => {
    // The page would otherwise steer the canvas at something the index no longer
    // describes — the same hazard a stale revealNode carries.
    const h = harness();
    await h.sync.open('/w');
    const old = h.sync.identity()!;

    h.setIndex(project([['exp:1', 'train'], ['exp:2', 'sweep']]), DIGEST_B);
    await h.sync.open('/w');

    const accepted = await h.sync.focus(old, 'exp:2');
    assert.equal(accepted, false);
    assert.ok(h.logged.some((line) => line.includes('stale index')));
  });

  test('an expand request against an old index is refused', async () => {
    const h = harness();
    await h.sync.open('/w');
    const old = h.sync.identity()!;
    h.setIndex(project([['exp:1', 'train']]), DIGEST_B);
    await h.sync.open('/w');

    assert.equal(await h.sync.expand(old, 'net', true), false);
  });

  test('focusing an experiment this index does not have is refused', async () => {
    const h = harness();
    await h.sync.open('/w');

    assert.equal(await h.sync.focus(h.sync.identity()!, 'exp:ghost'), false);
    assert.ok(h.logged.some((line) => line.includes('does not have')));
  });

  test('clearing the focus is allowed — showing no experiment is a real state', async () => {
    const h = harness();
    await h.sync.open('/w');

    assert.equal(await h.sync.focus(h.sync.identity()!, ''), true);
    assert.deepEqual(h.published.at(-1).document.nodes, [], 'clearing focus must clear the drawn experiment too');
  });
});

suite('projectSync — expansion', () => {
  test('opening a frame re-publishes the document with it expanded', async () => {
    const h = harness();
    await h.sync.open('/w');
    await h.sync.expand(h.sync.identity()!, 'net', true);

    assert.equal(h.documents.at(-1), 'exp:1:net');
  });

  test('closing a frame removes it', async () => {
    const h = harness();
    await h.sync.open('/w');
    await h.sync.expand(h.sync.identity()!, 'net', true);
    await h.sync.expand(h.sync.identity()!, 'net', false);

    assert.equal(h.documents.at(-1), 'exp:1:');
  });

  test('expansion is kept per experiment, so switching away does not lose it', async () => {
    // Reopening every frame after a glance at another experiment would make the
    // rail hostile to use.
    const h = harness();
    await h.sync.open('/w');
    await h.sync.expand(h.sync.identity()!, 'net', true);
    await h.sync.focus(h.sync.identity()!, 'exp:2');
    await h.sync.focus(h.sync.identity()!, 'exp:1');

    assert.equal(h.documents.at(-1), 'exp:1:net');
  });

  test('expansion survives a re-index', async () => {
    const h = harness();
    await h.sync.open('/w');
    await h.sync.expand(h.sync.identity()!, 'net', true);

    h.setIndex(project([['exp:1', 'train'], ['exp:2', 'sweep']]), DIGEST_B);
    await h.sync.open('/w');

    assert.equal(h.documents.at(-1), 'exp:1:net');
  });

  // A webview that reloads comes back with no memory and no way to ask, and
  // postMessage reports success for a message the page never saw. The class plane
  // already replays its projection when the app says `ready`; without the same on
  // this plane a reloaded panel drew a document with no index behind it — and both
  // the experiment rail and the status bar are gated on having one.
  test('republish says everything again without indexing again', async () => {
    const h = harness();
    await h.sync.open('/w');
    await h.sync.expand(h.sync.identity()!, 'net', true);
    const before = h.published.length;
    const documentsBefore = h.documents.length;

    await h.sync.republish();

    const replayed = h.published.slice(before);
    assert.deepEqual(replayed.map((m) => m.type), ['projectIndex', 'projectDocument']);
    // The same identity, not a new revision: nothing about the project changed, and
    // bumping it would make every page holding the old one refuse its next request.
    assert.deepEqual(replayed[0].project, h.sync.identity());
    // And the expansion the user opened is still open in what comes back.
    assert.equal(h.documents.at(-1), 'exp:1:net');
    assert.equal(h.documents.length, documentsBefore, 'reload replays the indexed document without reading changed buffers');
  });

  test('republish before anything is open says nothing at all', async () => {
    // Every window that never opened a project takes this path on every reload.
    const h = harness();
    await h.sync.republish();
    assert.deepEqual(h.published, []);
  });

  test('a replayed index carries the truncation the first one admitted', async () => {
    // Held rather than passed through: a replay that reported a partial project as
    // whole would be the one moment the rail's honest banner disappears.
    const h = harness({ index: { ...project([['exp:1', 'train']]), truncated: true } });
    await h.sync.open('/w');
    const before = h.published.length;

    await h.sync.republish();

    assert.equal(h.published.slice(before)[0].truncated, true);
  });
});


suite('projectSync — publication freshness', () => {
  test('late focus responses cannot overwrite a newer focus or be replayed under its identity', async () => {
    const published: Array<{ type: string; project: { experimentId: string } }> = [];
    const pending = new Map<string, (value: { document: object; spanmap: object; expanded: string[] }) => void>();
    let defer = false;
    const sync = new ProjectSync({
      index: async () => ({ index: project([['one', 'one'], ['two', 'two'], ['three', 'three']]),
        digest: DIGEST_A, indexed: 3, discovered: 3, truncated: false }),
      document: async (_folder, id, expanded) => defer ? new Promise(resolve => pending.set(id, resolve)) : { document: {}, spanmap: {}, expanded },
      publish: value => published.push(value),
    });
    await sync.open('/w'); defer = true;
    const second = sync.focus(sync.identity()!, 'two');
    const third = sync.focus(sync.identity()!, 'three');
    const before = published.length;
    await sync.republish();
    assert.deepEqual(published.slice(before).map(item => item.type), ['projectIndex'], 'old document is not replayed under the pending focus');
    pending.get('three')!({ document: {}, spanmap: {}, expanded: [] }); await third;
    pending.get('two')!({ document: {}, spanmap: {}, expanded: [] }); await second;
    assert.equal(published.at(-1)?.project.experimentId, 'three');
  });

  test('a document from changed source cannot claim the indexed digest', async () => {
    const published: string[] = [];
    const sync = new ProjectSync({
      index: async () => ({ index: project([['one', 'one']]), digest: DIGEST_A, indexed: 1, discovered: 1, truncated: false }),
      document: async () => ({ document: {}, spanmap: {}, expanded: [], digest: DIGEST_B }),
      publish: value => published.push(value.type),
    });
    await assert.rejects(sync.open('/w'), /sources changed/);
    assert.deepEqual(published, ['projectIndex']);
  });
});
