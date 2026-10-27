import * as assert from 'node:assert/strict';
import { executionDiagnosticDetail } from '../../executionDiagnostic';
import { captureRunEvidence } from '@tensorviz/graph-host/mcpRunEvidence';
import type { components } from '@tensorviz/contracts/semanticContracts.generated';
import type { TraceOutcome } from '../../worker';

suite('typed forward diagnostics', () => {
  const shape: components['schemas']['Diagnostic'] = { id: 'forward', phase: 'forward', code: 'invalid_shape',
    field: ['args', 0], names: [], sourceBindingId: null, remedy: 'inspect_failure',
    expectedShape: [null, 5], actualShape: [2, 3] };
  test('the page displays only known dimension constraints', () => {
    assert.match(executionDiagnosticDetail(shape, 'Net')!, /expected \[\?, 5\], received \[2, 3\]/);
    assert.doesNotMatch(executionDiagnosticDetail({ ...shape, code: 'execution_failed', expectedShape: null }, 'Net')!, /expected/);
    assert.match(executionDiagnosticDetail(shape, 'Net', 'classifier.fc')!, /at classifier\.fc/);
    assert.match(executionDiagnosticDetail(shape, 'Net', 'classifier.fc')!, /expected \[\?, 5\], received \[2, 3\]/);
    assert.doesNotMatch(executionDiagnosticDetail(shape, 'Net', 'not a layer')!, /not a layer/);
  });
  test('retained MCP evidence preserves the same operand path and dimensions without exception prose', () => {
    const outcome: TraceOutcome = { ok: false, reason: 'Forward failed', detail: 'Private host detail', exitCode: 4,
      envelope: { protocol_version: 1, spans: {}, worker: {}, result: null,
        error: { type: 'TraceError', message: 'PRIVATE_EXCEPTION', frames: [], diagnostic: shape } } };
    const evidence = captureRunEvidence(outcome, '/workspace');
    assert.deepEqual((evidence.error as any).diagnostic, shape);
    assert.ok(!JSON.stringify(evidence).includes('PRIVATE_EXCEPTION'));
  });
  test('retained stop reasons use host facts rather than prose or worker-failure guesses', () => {
    for (const [fact, code] of [['cancelled', 'CANCELLED'], ['timedOut', 'TIMEOUT']] as const) {
      const result = captureRunEvidence({ ok: false, [fact]: true, reason: 'Reworded host explanation', detail: '' }, '/workspace');
      assert.equal((result.error as any).code, code);
    }
  });
});
