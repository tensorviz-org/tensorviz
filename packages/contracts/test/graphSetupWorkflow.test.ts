import * as assert from 'node:assert/strict';
import { graphSetupWorkflow } from '@tensorviz/contracts/graphSetupWorkflow';
import type { GraphOpeningState } from '@tensorviz/contracts/protocol';

const state = (update: Partial<GraphOpeningState> = {}): GraphOpeningState => ({ id: 'opening',
  target: { workspaceFolder: '/workspace', file: 'model.py', cls: 'Net', documentVersion: 1,
    sourceDigest: 'a'.repeat(64), backendRevision: 1 },
  phase: 'needs_setup', detail: 'Host setup state.', hasCapture: false, ...update });
const readyPython = { reason: 'ready' as const, detail: 'Model libraries and capture tools checked.', candidates: [] };

suite('host-confirmed graph setup workflow', () => {
  test('a deleted setup file needs saving again without erasing the previous captured result', () => {
    const workflow = graphSetupWorkflow(state({ phase: 'ready', hasCapture: true,
      configFileExists: false, environment: readyPython }));
    assert.equal(workflow.stages.python, 'ready');
    assert.equal(workflow.stages.inputs, 'needed');
    assert.equal(workflow.stages.capture, 'ready');
  });
  test('a draft, config file or agent prose cannot attest Python or a captured invocation', () => {
    const workflow = graphSetupWorkflow(state({ configFileExists: true, draft: { profile: { kind: 'model' } },
      detail: 'Python and inputs are ready; setup complete.' }));
    assert.deepEqual(workflow.stages, { python: 'unchecked', inputs: 'needed', capture: 'not_started' });
    assert.deepEqual(workflow.nextStep, { kind: 'inputs', tool: 'suggest_graph_inputs' });
  });
  test('pending Python stays ahead of input preparation', () => {
    const workflow = graphSetupWorkflow(state({ environment: { ...readyPython, reason: 'missing_libraries' } }));
    assert.deepEqual(workflow.stages, { python: 'pending', inputs: 'unchecked', capture: 'not_started' });
    assert.deepEqual(workflow.nextStep, { kind: 'environment', tool: 'configure_graph_environment' });
  });
  test('checked Python continues to inputs in the same opening, without claiming saved setup or capture', () => {
    const workflow = graphSetupWorkflow(state({ environment: readyPython }));
    assert.deepEqual(workflow.stages, { python: 'ready', inputs: 'needed', capture: 'not_started' });
    assert.equal(workflow.nextStep.kind, 'inputs');
  });
  for (const phase of ['resolving', 'saving', 'capturing'] as const) {
    test(`${phase} waits on the host instead of requesting another operation`, () => {
      const workflow = graphSetupWorkflow(state({ phase, environment: readyPython }));
      assert.deepEqual(workflow.nextStep, { kind: 'wait', tool: 'get_graph_setup', waitMs: 10000 });
      assert.equal(workflow.stages.capture, phase === 'capturing' ? 'running' : 'not_started');
      assert.notEqual(workflow.nextStep.kind, 'complete');
    });
  }
  test('Run review is an explicit user decision, with constructor preparation distinct from saved inputs', () => {
    const execution = { kind: 'prepare', phase: 'review' } as NonNullable<GraphOpeningState['execution']>;
    const preparing = graphSetupWorkflow(state({ phase: 'awaiting_execution', execution, environment: readyPython }));
    assert.equal(preparing.nextStep.kind, 'execution_approval');
    assert.equal(preparing.stages.inputs, 'unchecked'); assert.equal(preparing.stages.capture, 'awaiting_approval');
    const capturing = graphSetupWorkflow(state({ phase: 'awaiting_execution', execution: { ...execution, kind: 'capture' } }));
    assert.equal(capturing.nextStep.kind, 'execution_approval'); assert.equal(capturing.stages.inputs, 'saved');
    assert.equal(capturing.stages.python, 'unchecked', 'execution state alone does not attest Python readiness');
  });
  for (const phase of ['failed', 'cancelled'] as const) {
    test(`${phase} is blocked even when Python and a prior partial graph exist`, () => {
      const workflow = graphSetupWorkflow(state({ phase, environment: readyPython, hasCapture: true }));
      assert.deepEqual(workflow.nextStep, { kind: 'blocked', reason: phase });
      assert.equal(workflow.stages.python, 'ready'); assert.equal(workflow.stages.capture, phase);
    });
  }
  test('source-only is an explicit endpoint and never a captured result', () => {
    const workflow = graphSetupWorkflow(state({ phase: 'source_only', environment: readyPython }));
    assert.equal(workflow.nextStep.kind, 'source_only'); assert.equal(workflow.stages.capture, 'not_started');
  });
  test('completion needs both ready host phase and a captured result', () => {
    assert.deepEqual(graphSetupWorkflow(state({ phase: 'ready' })).nextStep, { kind: 'blocked', reason: 'failed' });
    const complete = graphSetupWorkflow(state({ phase: 'ready', hasCapture: true, environment: readyPython }));
    assert.equal(complete.nextStep.kind, 'complete');
    assert.deepEqual(complete.stages, { python: 'ready', inputs: 'saved', capture: 'ready' });
  });
});
