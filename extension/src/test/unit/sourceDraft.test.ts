import * as assert from 'node:assert/strict';
import { SourceDrafts, SourceDraftInput } from '../../sourceDraft';
import { sha256 } from '../../sourceChanges';
import type { GraphReviewState } from '@tensorviz/contracts/protocol';

const tick = () => new Promise<void>(resolve => setImmediate(resolve));
const before = 'class Net(Base): pass\n', after = 'class Net(Base):\n    fc2 = Linear(4, 4)\n';
const input = (id = 'proposal-1'): SourceDraftInput => ({ id,
  target: { workspaceFolder: '/project', file: 'model.py', cls: 'Net', sourceDigest: sha256(before), documentVersion: 1, backendRevision: 1 },
  artifacts: [{ path: 'model.py', before, after, before_digest: sha256(before), after_digest: sha256(after), edits: [] }],
  sources: { entries: [{ file: 'model.py', source: before, digest: sha256(before) }, { file: 'base.py', source: '# frozen', digest: sha256('# frozen') }], truncated: false },
});

suite('pending source draft authority', () => {
  let drafts: SourceDrafts, messages: any[], requests: any[], changed: () => void, current: boolean, release: (() => void) | undefined, failed: boolean;
  const latest = () => messages.at(-1)?.state as GraphReviewState | null;
  setup(() => {
    messages = []; requests = []; current = true; failed = false; release = undefined;
    drafts = new SourceDrafts({ endpoint: () => ({ origin: 'http://localhost:9999', token: 'private' }), publish: message => messages.push(message),
      changed: listener => { changed = listener; return { dispose() {} }; },
      fetchImpl: (async (url, options) => {
        const body = JSON.parse(String(options?.body)); requests.push({ url, body });
        if (release) await new Promise<void>(resolve => { const previous = release!; release = () => { previous(); resolve(); }; });
        if (failed) return new Response('', { status: 422 });
        return new Response(JSON.stringify({ cls: 'Net', source_digest: body.root.source_digest, graph_id: body.root.source_digest,
          revision: 0, traced: false, document: { semantic: { nodes: [] } }, spanmap: {}, diagnostics: [], dependencies: [], truncated: false }));
      }) as typeof fetch });
  });
  teardown(() => drafts.dispose());

  test('canvas decisions reach only the current native review once, with stale Accept refused', async () => {
    const decisions: boolean[] = [];
    const handle = drafts.start(input(), new AbortController().signal, () => current);
    handle.show(async () => {}, value => decisions.push(value)); await tick();
    drafts.handle({ action: 'acceptDraft', reviewId: 'old-proposal' } as any);
    assert.equal(decisions.length, 0);
    drafts.handle({ action: 'acceptDraft', reviewId: 'proposal-1' } as any);
    drafts.handle({ action: 'acceptDraft', reviewId: 'proposal-1' } as any);
    assert.deepEqual([...decisions], [true]);
    handle.dispose();
    const stale = drafts.start(input('proposal-2'), new AbortController().signal, () => current);
    stale.show(async () => {}, value => decisions.push(value)); await tick();
    current = false; changed();
    drafts.handle({ action: 'acceptDraft', reviewId: 'proposal-2' } as any);
    assert.deepEqual([...decisions], [true]);
    drafts.handle({ action: 'discardDraft', reviewId: 'proposal-2' } as any);
    assert.deepEqual(decisions, [true, false]);
  });

  test('failed accept republishes a fresh draft state to clear the pending decision', async () => {
    const handle = drafts.start(input(), new AbortController().signal, () => current);
    handle.show(async () => {}, () => {}); await tick();
    const previous = latest();
    drafts.handle({ action: 'acceptDraft', reviewId: 'proposal-1' } as any);
    handle.notifyDecisionFailed?.();
    assert.notStrictEqual(latest(), previous);
    assert.deepEqual(latest(), previous);
    assert.equal(drafts.active, true);
  });

  test('projects frozen reviewed bytes alongside the diff, never live edits or a model run', async () => {
    release = () => {};
    const handle = drafts.start(input(), new AbortController().signal, () => current);
    let revealed = 0;
    handle.show(async () => { revealed++; }, () => {});
    assert.equal(latest(), undefined, 'native review does not wait for the projector');
    release!(); await tick();
    assert.equal(latest()?.proposal?.id, 'proposal-1');
    assert.equal(latest()?.before.sourceDigest, sha256(before));
    assert.equal(latest()?.after.sourceDigest, sha256(after));
    assert.deepEqual(requests.map(request => request.body.sources[0].source), [before, after]);
    assert.ok(requests.every(request => request.url.endsWith('/api/ide/multifile/project') && request.body.review_only === true && !request.body.trace_result));
    assert.ok(requests.every(request => request.body.sources[1].source === '# frozen'));
    drafts.handle({ action: 'reveal', reviewId: 'proposal-1' } as any); assert.equal(revealed, 1);
    for (const action of ['close', 'refresh', 'selectModel', 'open']) drafts.handle({ action, reviewId: 'proposal-1' } as any);
    assert.ok(drafts.active, 'only the native proposal decision ends the draft');
    current = false; changed(); assert.equal(latest()?.stale, true);
    const staleMessages = messages.length;
    changed(); changed(); assert.equal(messages.length, staleMessages, 'stale typing must not repeatedly replace/re-layout the whole draft');
    handle.dispose(); assert.equal(latest(), null);
  });

  test('finished or cancelled proposals cannot return after slow projection replies', async () => {
    release = () => {};
    const abort = new AbortController();
    const old = drafts.start(input(), abort.signal, () => current); old.show(async () => {}, () => {});
    abort.abort();
    const finishOld = release!; release = undefined;
    const next = drafts.start(input('proposal-2'), new AbortController().signal, () => current); next.show(async () => {}, () => {});
    await tick(); assert.equal(latest()?.id, 'proposal-2');
    finishOld(); await tick(); old.dispose();
    assert.equal(latest()?.id, 'proposal-2', 'old completion/cleanup cannot overwrite a newer draft');
  });

  test('a multi-file draft projects every exact replacement and exposes the complete change list', async () => {
    const batch = input();
    batch.artifacts.push({ path: 'base.py', before: '# frozen', after: 'WIDTH = 16\n',
      before_digest: sha256('# frozen'), after_digest: sha256('WIDTH = 16\n') });
    const handle = drafts.start(batch, new AbortController().signal, () => true);
    handle.show(async () => {}, () => {}); await tick();
    assert.deepEqual(requests[0].body.sources.map((item: any) => item.source), [before, '# frozen']);
    assert.deepEqual(requests[1].body.sources.map((item: any) => item.source), [after, 'WIDTH = 16\n']);
    assert.deepEqual(latest()?.changedFiles, ['model.py', 'base.py']);
    assert.equal(latest()?.proposal?.id, batch.id);
    handle.dispose();
  });

  test('projection failures are explicit and a prepared graph appears only after the native diff opens', async () => {
    failed = true;
    const handle = drafts.start(input(), new AbortController().signal, () => current);
    await tick(); assert.equal(messages.length, 0);
    handle.show(async () => {}, () => {});
    assert.equal(latest()?.proposal?.unavailable, true);
    handle.dispose(); assert.equal(latest(), null);
  });

  test('typed drafts publish their revision boundary with the same native decisions', async () => {
    const decisions: boolean[] = [];
    const handle = drafts.start({ ...input(), canRevise: false }, new AbortController().signal, () => true);
    handle.show(async () => {}, accepted => decisions.push(accepted)); await tick();
    assert.equal(latest()?.proposal?.canRevise, false);
    drafts.handle({ action: 'discardDraft', reviewId: 'proposal-1' } as any);
    assert.deepEqual(decisions, [false]);
  });
});
