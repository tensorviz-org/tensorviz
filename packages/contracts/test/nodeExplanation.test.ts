import * as assert from 'node:assert/strict';
import {
  isNodeExplanation, isNodeExplanationCitation, isNodeExplanationState,
  isNodeSourceDocumentation, isModelExplanationState, type NodeSourceDocumentation,
  type NodeExplanation, type NodeExplanationCitation, type NodeExplanationState,
} from '@tensorviz/contracts/nodeExplanation';
import { isLocalBrowserMessage } from '@tensorviz/contracts/protocol';

const citation = (update: Partial<NodeExplanationCitation> = {}): NodeExplanationCitation => ({
  id: 'helper', file: 'timejev/asymmetric_video_memory.py', sourceDigest: 'a'.repeat(64),
  startLine: 247, endLine: 251, ...update,
});
const explanation = (update: Partial<NodeExplanation> = {}): NodeExplanation => ({
  title: 'Score candidates', purpose: 'Compare candidate features with the question.',
  inputs: ['Candidate features', 'Question features'], outputs: ['Raw candidate scores'],
  steps: ['Pad and batch candidates.', 'Batch questions and call the scoring head.'],
  uncertainties: ['Tensor shapes have not been recorded.'], citations: [citation()], ...update,
});
const state = (update: Partial<NodeExplanationState> = {}): NodeExplanationState => ({
  requestId: 'request', nodeId: 'python@Model#0', revision: 'revision', status: 'ready',
  explanation: explanation(), ...update,
});
const target = { workspaceFolder: '/workspace', file: 'model.py', cls: 'Model', documentVersion: 1,
  sourceDigest: 'a'.repeat(64), backendRevision: 1 };
const envelope = { protocolVersion: 1, workspaceSessionId: 'session', messageId: 'message', viewId: 'view', target };
const resultMessage = (value: unknown = state()) => ({ ...envelope, type: 'nodeExplanation', state: value });
const requestMessage = (update: Record<string, unknown> = {}) => ({ ...envelope, type: 'nodeExplanationRequest',
  nodeId: 'python@Model#0', requestId: 'request', action: 'retry', ...update });

suite('node explanation contract', () => {
  const documentation = (): Extract<NodeSourceDocumentation, { status: 'resolved' }> => ({
    status: 'resolved', qualifiedName: 'CandidateScorer.compare', kind: 'method',
    signature: 'def compare(self, candidates, /, query, *, normalize=False) -> Scores:',
    docstring: 'Compare candidate representations.\n\nArgs:\n    candidates: Candidate features.',
    truncated: false, citation: citation(),
  });
  test('source documentation is usable independently of an agent result', () => {
    const docs = documentation();
    assert.ok(isNodeSourceDocumentation(docs));
    for (const status of ['queued', 'loading', 'ready', 'unavailable'] as const) {
      const value = state({ status, documentation: docs, ...(status === 'ready' ? {} : { explanation: undefined }) });
      assert.ok(isNodeExplanationState(value), status);
      assert.ok(isLocalBrowserMessage(resultMessage(value)), status);
    }
    assert.ok(isNodeSourceDocumentation({ ...docs, docstring: undefined }));
    assert.ok(isNodeSourceDocumentation({ status: 'unavailable', reason: 'ambiguous' }));
  });
  test('source documentation bounds content, citation authority and resolution shape', () => {
    const docs = documentation();
    for (const update of [{ qualifiedName: '' }, { signature: 'x'.repeat(2049) }, { signature: 'def f():\npass' },
      { docstring: 'x'.repeat(4097) }, { docstring: 'text\u0000' }, { docstring: ' text' },
      { kind: 'runtime' }, { truncated: 'false' }, { execute: 'python' },
      { citation: { ...citation(), file: '/private.py' } }]) assert.equal(isNodeSourceDocumentation({ ...docs, ...update }), false);
    for (const invalid of [{ status: 'unavailable', reason: 'guess' },
      { status: 'unavailable', reason: 'ambiguous', signature: 'invented' }, { status: 'resolved' }]) {
      assert.equal(isNodeSourceDocumentation(invalid), false);
    }
    assert.ok(isNodeSourceDocumentation({ ...docs, docstring: 'x'.repeat(4096) }));
    assert.ok(isNodeSourceDocumentation({ ...docs, docstring: '\u0085author text\u0085' }));
  });
  test('accepts a bounded explanation with source citations and explicit unknowns', () => {
    assert.ok(isNodeExplanation(explanation()));
    assert.ok(isNodeExplanation(explanation({ inputs: [], outputs: [], steps: [], uncertainties: [] })));
    assert.ok(isNodeExplanation(explanation({ citations: Array.from({ length: 8 }, (_, index) => citation({ id: `source-${index}` })) })));
    assert.equal(isNodeExplanation({ ...explanation(), execute: 'model()' }), false);
    assert.equal(isNodeExplanation({ ...explanation(), citations: [] }), false);
    assert.equal(isNodeExplanation({ ...explanation(), citations: Array.from({ length: 9 }, (_, index) => citation({ id: `source-${index}` })) }), false);
    assert.equal(isNodeExplanation({ ...explanation(), citations: [citation(), citation()] }), false);
  });

  for (const [field, maximum] of [['title', 80], ['purpose', 320]] as const) {
    test(`${field} accepts its ceiling and rejects absent, oversized or control text`, () => {
      assert.ok(isNodeExplanation(explanation({ [field]: 'x'.repeat(maximum) })));
      for (const value of [undefined, null, 3, '', ' ', ' padded', 'padded ', 'x'.repeat(maximum + 1), 'line\nbreak', 'tab\there', 'nul\0byte', 'delete\u007f']) {
        assert.equal(isNodeExplanation({ ...explanation(), [field]: value }), false, JSON.stringify(value));
      }
    });
  }

  for (const [field, count, length] of [['inputs', 6, 160], ['outputs', 4, 160], ['steps', 6, 320], ['uncertainties', 4, 240]] as const) {
    test(`${field} bounds both item count and text length`, () => {
      assert.ok(isNodeExplanation(explanation({ [field]: Array(count).fill('x'.repeat(length)) })));
      for (const value of [undefined, null, 'text', Array(count + 1).fill('x'), ['x'.repeat(length + 1)], [''], [' '], ['x\ny'], [false]]) {
        assert.equal(isNodeExplanation({ ...explanation(), [field]: value }), false, JSON.stringify(value));
      }
    });
  }

  test('optional terminology remains bounded explanation data', () => {
    const concept = { term: 'VAE', explanation: 'A variational autoencoder learns a latent representation.' };
    assert.ok(isNodeExplanation(explanation({ concepts: [concept] })));
    assert.ok(isNodeExplanation(explanation({ concepts: [] })));
    assert.ok(isNodeExplanation(explanation({ concepts: Array(4).fill({ term: 'x'.repeat(80), explanation: 'x'.repeat(320) }) })));
    for (const concepts of [null, 'text', Array(5).fill(concept), [{}], [{ ...concept, term: '' }],
      [{ ...concept, term: 'x'.repeat(81) }], [{ ...concept, explanation: 'x'.repeat(321) }],
      [{ ...concept, explanation: 'line\nbreak' }], [{ ...concept, command: 'execute' }]]) {
      assert.equal(isNodeExplanation({ ...explanation(), concepts }), false, JSON.stringify(concepts));
    }
  });

  test('model progress has bounded integral counts and explicit terminal states', () => {
    const progress = { requestId: 'model-request', revision: 'revision', status: 'loading', completed: 0, total: 128 };
    for (const status of ['loading', 'ready', 'unavailable', 'cancelled']) {
      assert.ok(isModelExplanationState({ ...progress, status }));
      assert.ok(isLocalBrowserMessage({ ...envelope, type: 'modelExplanation', state: { ...progress, status } }));
    }
    assert.ok(isModelExplanationState({ ...progress, completed: 10000, total: 10000, detail: 'x'.repeat(320) }));
    for (const update of [{ completed: -1 }, { completed: 129 }, { completed: 0.5 }, { completed: true },
      { completed: Number.NaN }, { total: 10001 }, { total: -1 }, { total: Infinity }, { total: 1.5 },
      { requestId: '' }, { revision: '' }, { status: 'queued' }, { detail: '' }, { detail: 'x'.repeat(321) },
      { command: 'execute' }]) {
      assert.equal(isModelExplanationState({ ...progress, ...update }), false, JSON.stringify(update));
      assert.equal(isLocalBrowserMessage({ ...envelope, type: 'modelExplanation', state: { ...progress, ...update } }), false);
    }
  });

  test('model passes start with bounded distinct nodes and cancel by exact request identity', () => {
    const request = { ...envelope, type: 'modelExplanationRequest', action: 'start', requestId: 'model-request', nodeIds: ['node'] };
    assert.ok(isLocalBrowserMessage(request, 'session'));
    assert.ok(isLocalBrowserMessage({ ...request, nodeIds: Array.from({ length: 128 }, (_, i) => `node-${i}`) }));
    assert.ok(isLocalBrowserMessage({ ...request, action: 'cancel', nodeIds: [] }));
    assert.equal(isLocalBrowserMessage(request, 'another-session'), false);
    for (const update of [{ nodeIds: [] }, { nodeIds: ['node', 'node'] }, { nodeIds: [''] },
      { nodeIds: ['n'.repeat(1025)] }, { nodeIds: Array.from({ length: 129 }, (_, i) => `node-${i}`) },
      { nodeIds: [1] }, { nodeIds: null }, { action: 'cancel' }, { action: 'execute' }, { viewId: '' },
      { target: null }, { requestId: '' }, { requestId: 'r'.repeat(129) }, { command: 'execute' }]) {
      assert.equal(isLocalBrowserMessage({ ...request, ...update }), false, JSON.stringify(update));
    }
  });

  test('citation paths stay relative and spans identify real positive line ranges', () => {
    assert.ok(isNodeExplanationCitation(citation({ startLine: 1, endLine: 1 })));
    assert.ok(isNodeExplanationCitation(citation({ id: 'i'.repeat(128), file: 'f'.repeat(4096) })));
    for (const file of ['/etc/passwd', 'C:\\model.py', 'C:/model.py', '\\\\server\\model.py', '../model.py',
      'pkg/../../model.py', 'pkg\\..\\model.py', 'file:///model.py', 'https://example.com/model.py', 'bad\0.py']) {
      assert.equal(isNodeExplanationCitation(citation({ file })), false, file);
    }
    for (const update of [{ id: '' }, { id: 'i'.repeat(129) }, { file: '' }, { file: 'f'.repeat(4097) },
      { sourceDigest: 'a'.repeat(63) }, { sourceDigest: 'A'.repeat(64) }, { startLine: 0 }, { startLine: 1.5 },
      { startLine: Number.NaN }, { endLine: 246 }, { endLine: Number.POSITIVE_INFINITY }, { endLine: Number.MAX_SAFE_INTEGER + 1 }]) {
      assert.equal(isNodeExplanationCitation(citation(update)), false, JSON.stringify(update));
    }
    assert.equal(isNodeExplanationCitation({ ...citation(), command: 'open model.py' }), false);
    assert.equal(isNodeExplanation({ ...explanation(), citations: [citation({ endLine: 246 })] }), false);
  });

  test('only ready state can carry a validated citation-bearing explanation', () => {
    assert.ok(isNodeExplanationState(state()));
    for (const status of ['queued', 'loading', 'unavailable'] as const) {
      const pending = state({ status });
      delete pending.explanation;
      assert.ok(isNodeExplanationState(pending));
      assert.equal(isNodeExplanationState(state({ status })), false);
    }
    for (const value of [undefined, null, {}, explanation({ citations: [] })]) {
      assert.equal(isNodeExplanationState({ ...state(), explanation: value }), false);
      assert.equal(isLocalBrowserMessage(resultMessage({ ...state(), explanation: value })), false);
    }
    assert.equal(isNodeExplanationState({ ...state(), status: 'finished' }), false);
    assert.equal(isNodeExplanationState({ ...state(), status: { toString: () => 'queued' }, explanation: undefined }), false);
    assert.equal(isNodeExplanationState({ ...state(), autoExecute: true }), false);
  });

  test('state identifiers and optional detail remain bounded plain text', () => {
    for (const [field, maximum] of [['requestId', 128], ['nodeId', 1024], ['revision', 256], ['detail', 320]] as const) {
      assert.ok(isNodeExplanationState(state({ [field]: 'x'.repeat(maximum) })));
      for (const value of ['', ' ', 'x'.repeat(maximum + 1), 'x\ny', null]) {
        assert.equal(isNodeExplanationState({ ...state(), [field]: value }), false, `${field}: ${JSON.stringify(value)}`);
      }
    }
  });

  test('wire messages bind explanations to a complete target and workspace session', () => {
    assert.ok(isLocalBrowserMessage(resultMessage(), 'session'));
    assert.equal(isLocalBrowserMessage(resultMessage(), 'other-session'), false);
    for (const update of [{ viewId: '' }, { viewId: 'v'.repeat(129) }, { target: null },
      { target: { ...target, sourceDigest: 'stale' } }, { unsupported: true }]) {
      assert.equal(isLocalBrowserMessage({ ...resultMessage(), ...update }), false, JSON.stringify(update));
    }
  });

  test('retry and citation reveal requests have distinct exact payloads', () => {
    assert.ok(isLocalBrowserMessage(requestMessage()));
    assert.ok(isLocalBrowserMessage(requestMessage({ action: 'reveal', citationId: 'helper' })));
    for (const update of [{ action: 'execute' }, { action: 'reveal' }, { action: 'reveal', citationId: '' },
      { action: 'reveal', citationId: 'c'.repeat(129) }, { citationId: 'helper' }, { nodeId: '' },
      { nodeId: 'n'.repeat(1025) }, { requestId: '' }, { requestId: 'r'.repeat(129) },
      { viewId: '' }, { target: { ...target, backendRevision: -1 } }, { command: 'model()' }]) {
      assert.equal(isLocalBrowserMessage(requestMessage(update)), false, JSON.stringify(update));
    }
  });
});
