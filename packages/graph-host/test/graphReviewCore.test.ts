import * as assert from 'node:assert/strict';
import { projectReviewSnapshot } from '@tensorviz/graph-host/graphReviewCore';
import type { IdeMultifileProjectionResponse, IdeMultifileSpan } from '@tensorviz/contracts/protocol';

type Fact = { span: IdeMultifileSpan; constructor: IdeMultifileSpan | null; parent?: string; children?: string[] };
const digest = 'a'.repeat(64);
const at = (start: [number, number], end: [number, number]) => ({ file: 'model.py', start, end, kind: 'decl' }) as IdeMultifileSpan;
// A real review_only reply, trimmed. Review discovery sends a nested project's
// files project-relative, so the backend names model.py, not données/model.py.
const fc = '@declaration:["model.py","Net","fc"]', seq = '@declaration:["model.py","Net","seq"]';
const reply = (): IdeMultifileProjectionResponse => ({ protocol_version: 1, cls: 'Net', source_digest: digest, projection_digest: 'p',
  graph_id: 'g', revision: 1, traced: false, project_id: 'p', dependencies: [{ file: 'model.py', digest }], diagnostics: [], truncated: false,
  spanmap: { fc: at([5, 8], [5, 33]), seq: at([6, 8], [6, 60]), 'linear@Net#0': { ...at([6, 33], [6, 48]), kind: 'call' },
    [fc]: at([5, 8], [5, 33]), [seq]: at([6, 8], [6, 60]), [`${seq}[0]`]: at([6, 33], [6, 48]) },
  document: { semantic: { metadata: { reviewDeclarations: { complete: true, entries: {
    [fc]: { label: 'Net.fc', digest: 'f', ambiguous: false, span: at([5, 8], [5, 33]), constructor: at([5, 18], [5, 33]) },
    [seq]: { label: 'Net.seq', digest: 's', ambiguous: false, span: at([6, 8], [6, 60]), constructor: at([6, 19], [6, 60]), children: [`${seq}[0]`] },
    [`${seq}[0]`]: { label: 'Net.seq[0]', parent: seq, digest: 'l', contextDigest: 'c', ambiguous: false, span: at([6, 33], [6, 48]), constructor: at([6, 33], [6, 48]) },
  } } } } } });
const sources = { entries: [{ file: 'model.py', source: '', digest }], truncated: false };
const facts = (document: unknown) => (document as { semantic: { metadata: { reviewDeclarations: { entries: Record<string, Fact> } } } })
  .semantic.metadata.reviewDeclarations.entries;
const location = (span: unknown) => { const { file, start, end } = span as IdeMultifileSpan; return JSON.stringify([file, start, end]); };

suite('review snapshot projection', () => {
  test('a nested project names declarations by the workspace paths the rest of review uses', async () => {
    const payload = await projectReviewSnapshot({ file: 'données/model.py', cls: 'Net', project: 'données' }, sources, async () => reply());
    const entries = facts(payload.document), nested = (key: string) => key.replace('"model.py"', '"données/model.py"');
    // Git renames and captured constructor calls are keyed by workspace paths,
    // spelled as JSON.stringify spells them; a project-relative key matches neither.
    assert.deepEqual(Object.keys(entries), [fc, seq, `${seq}[0]`].map(nested));
    assert.deepEqual(Object.keys(payload.spanmap).filter(id => id.startsWith('@declaration:')), Object.keys(entries),
      'review navigates to a declaration through the spanmap entry with its key');
    assert.deepEqual(entries[nested(seq)].children, [nested(`${seq}[0]`)]);
    assert.equal(entries[nested(`${seq}[0]`)].parent, nested(seq));
    assert.equal(entries[nested(fc)].constructor?.file, 'données/model.py');
    // Review joins a layer to its declaration by exact location; a miss reports
    // every edited layer twice, once as the layer and once as its declaration.
    assert.equal(location(entries[nested(fc)].span), location(payload.spanmap.fc));
    assert.equal(location(entries[nested(`${seq}[0]`)].span), location(payload.spanmap['linear@Net#0']));
    // Captured review uses a capture only when its workspace dependency digests
    // match this revision's; project-relative names never match.
    assert.deepEqual(payload.dependencies, [{ file: 'données/model.py', digest }]);
  });

  test('a repository-root project keeps the backend declaration keys', async () => {
    const payload = await projectReviewSnapshot({ file: 'model.py', cls: 'Net', project: '' }, sources, async () => reply());
    assert.deepEqual(facts(payload.document), facts(reply().document));
    assert.deepEqual(Object.keys(payload.spanmap), Object.keys(reply().spanmap));
  });
});
