import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { execFileSync } from 'node:child_process';
import type * as vscode from 'vscode';
import { TraceWorker, workerArgs, type WorkerEnvelope } from '../../worker';
import { candidateRequest, candidateEvidence, type CandidateInput, type CandidateRunResult } from '@tensorviz/graph-host/candidateExecution';
import { CandidateReview } from '../../candidateReview';
import { CandidateHistory } from '../../candidateHistory';
import type { CandidateRequest, ExecutionSnapshot } from '../../executionPlan';
import { resolveRetainedPlan } from '../../executionPlan';
import { executionSpecSchema } from '@tensorviz/graph-host/executionSpec';
import { AttemptLedger } from '@tensorviz/graph-host/attemptLedger';
import { readValidationSnapshot } from '../../validationHost';
import { translateProfile } from '@tensorviz/graph-host/validationProfileBridge';
import type { RunSpec } from '@tensorviz/contracts/protocol';
import { isCandidateRecord, candidatePostconditionsText } from '@tensorviz/contracts/protocol';
import { reset, resetEditors, state, openStubEditor, editorState } from '../vscodeStub';

const repo = path.resolve(__dirname, '../../../..'), backend = path.join(repo, 'backend');
const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(repo, 'venv/bin/python');
const transport = require('@tensorviz/graph-host/executionPlanCore') as typeof import('@tensorviz/graph-host/executionPlanCore');
const source = `import torch
class Model(torch.nn.Module):
    def forward(self, h):
        a = h + 1
        b = a + a
        return b * 2
`;
const planner = `import json,sys
sys.path.insert(0, sys.argv[1])
from services.semantic.source_edits import CapturedEditResolver
p=json.load(sys.stdin)
r=CapturedEditResolver(p['capture'], {'model.py':p['source']})
if p.get('operandEdit') and 'configuration' in p['operandEdit']:
    choice=p['operandEdit']
    c=next(cap for cap in r.configuration_literal_capabilities(r.capture.invocations[1].id) if cap.sourceArgumentPath==choice['configuration'])
    edit=dict(kind='set_configuration_literal',capabilityId=c.id,target=c.target.model_dump(mode='json'),
        sourceArgumentPath=c.sourceArgumentPath,expected=c.expected.model_dump(mode='json'),
        value=choice['value'],scope='configuration_definition')
elif p.get('operandEdit') and 'constructor' in p['operandEdit']:
    choice=p['operandEdit']
    c=next(cap for cap in r.constructor_literal_capabilities(r.capture.invocations[1].id) if cap.sourceArgumentPath==choice['constructor'])
    edit=dict(kind='set_constructor_literal',capabilityId=c.id,target=c.target.model_dump(mode='json'),
        sourceArgumentPath=c.sourceArgumentPath,expected=c.expected.model_dump(mode='json'),
        value=choice['value'],scope='source_expression')
elif p.get('operandEdit') and 'delete' in p['operandEdit']:
    c=r.delete_unary_capability(next(op.id for op in r.capture.operations if op.target=='aten.gelu.default'))
    edit=dict(kind='delete_unary_op',capabilityId=c.id,target=c.target.model_dump(mode='json'),
        argumentPath=c.argumentPath,expected=c.expected.model_dump(mode='json'),
        value=c.input.value.model_dump(mode='json'),scope='source_expression')
elif p.get('operandEdit'):
    choice=p['operandEdit']
    op=next(op for op in r.capture.operations if op.target=='aten.add.Tensor' and op.arguments.args[0]==op.arguments.args[1])
    caps=r.insert_unary_capabilities(op.id) if 'activation' in choice else r.reconnect_capabilities(op.id)
    c=next(cap for cap in caps if cap.argumentPath==['args',choice['port']])
    edit=dict(kind='insert_unary_op' if 'activation' in choice else 'reconnect_argument',capabilityId=c.id,target=c.target.model_dump(mode='json'),
        argumentPath=c.argumentPath,expected=c.expected.model_dump(mode='json'),
        value=choice['activation'] if 'activation' in choice else next(item.value.model_dump(mode='json') for item in c.choices if item.name==choice['name']),scope='source_expression')
elif p.get('activation'):
    choice=p['activation']
    c=(r.module_activation_capability(r.capture.invocations[1].id) if choice['kind']=='module' else
       r.activation_capability(next(op.id for op in r.capture.operations if op.target in ('aten.relu.default','aten.relu_.default'))))
    edit=dict(kind='replace_activation',capabilityId=c.id,target=c.target.model_dump(mode='json'),
        expected=c.expected,value=choice['value'],scope='source_expression')
else:
    op=next(op for op in r.capture.operations if op.target=='aten.add.Tensor' and op.arguments.args[1].kind=='int')
    c=r.literal_capability(op.id)
    edit=dict(kind='set_literal',capabilityId=c.id,target=c.target.model_dump(mode='json'),
        argumentPath=c.argumentPath,expected=c.expected.model_dump(mode='json'),value=dict(kind='int',value='2'),scope='source_expression')
after=p['source']
for patch in sorted(r.plan(edit)['edits'],key=lambda item:item['range']['start'],reverse=True):
    start,end=patch['range']['start'],patch['range']['end']; lines=after.split('\\n')
    line=lines[start[0]-1].encode('utf-16-le')
    lines[start[0]-1]=(line[:start[1]*2]+patch['newText'].encode('utf-16-le')+line[end[1]*2:]).decode('utf-16-le')
    after='\\n'.join(lines)
print(json.dumps(dict(intent=edit,reviewedSource=after)))
`;

suite('candidate lifecycle with the real selected interpreter', function () {
  this.timeout(40000);
  let temporary: string[], requests: CandidateRequest[];
  const originalTransport = transport.candidateRequestFile;
  setup(() => {
    reset(); resetEditors(); temporary = []; requests = [];
    assert.ok(existsSync(python), 'Set TENSORVIZ_TEST_PYTHON to a Torch-capable interpreter.');
    transport.candidateRequestFile = async request => {
      const result = await originalTransport(request);
      temporary.push(result.file); requests.push(structuredClone(request)); return result;
    };
  });
  teardown(async () => {
    transport.candidateRequestFile = originalTransport;
    for (const file of temporary) assert.equal(existsSync(path.dirname(file)), false, 'host must remove the full private directory');
    resetEditors(); reset();
  });

  async function fixture(text = source, timeoutMs = 15000, profile = false,
    activation?: { kind: 'functional' | 'module'; value: 'gelu' | 'silu' },
    operandEdit?: { port: 0 | 1; name: string } | { port: 0 | 1; activation: 'gelu' } | { delete: true }
      | { constructor: Array<string | number>; value: { kind: 'int'; value: string } }
      | { configuration: Array<string | number>; value: { kind: 'int'; value: string } }) {
    const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-candidate-host-')));
    const file = path.join(root, 'model.py'); await fs.writeFile(file, text);
    const logs: string[] = [];
    let selected = python;
    const worker = new TraceWorker(backend, { appendLine: (line: string) => logs.push(line) } as unknown as vscode.OutputChannel,
      { interpreter: async () => selected });
    const ledger = new AttemptLedger();
    const target = { workspaceFolder: root, file: 'model.py', cls: 'Model', sourceDigest: 'a'.repeat(64),
      documentVersion: 1, backendRevision: 1 };
    const attempt = ledger.begin('base', target, 'manual', null);
    let snapshot!: ExecutionSnapshot;
    let execution = executionSpecSchema.parse({ inputs: { args: [{ shape: [2, 4], dtype: 'float32', fill: 'ones' }] } });
    let profileSpec: RunSpec | undefined;
    if (profile) {
      await fs.writeFile(path.join(root, 'tensorviz.toml'), `version = 1
[profiles.small]
kind = "model"
model = "model:Model"
seed = 7
mode = "eval"
device = "cpu"
grad_enabled = false
init = {args = [], kwargs = {}}
inputs = {args = [{kind="tensor", shape=[2,4], dtype="float32", requires_grad=false, generator={kind="full", value=1}}], kwargs = {}}
`);
      const prepared = await readValidationSnapshot(python, backend, { root, executable: python, environmentRevision: 0,
        session: 'candidate-test', workspaceId: 'candidate-test-workspace', runId: 'profile', generation: 0,
        profile: 'small', documents: {} }, new AbortController().signal);
      assert.ok(prepared.resolution.status === 'resolved');
      profileSpec = prepared.resolution.spec;
      const translated = translateProfile(profileSpec.profile, 'small'); assert.ok(translated.ok);
      execution = translated.run.execution;
    }
    const base = await worker.run({ workspace: root, file, cls: 'Model', timeoutMs, attemptId: 'base',
      execution, profileSpec,
      onResolvedPlan: value => { snapshot = value as ExecutionSnapshot; attempt.resolved(value); } });
    assert.equal(base.ok, true, JSON.stringify(base));
    attempt.captured(base); attempt.finish('passed');
    const capture = (base.envelope!.result as any).diagnostics.semantic_capture;
    const reviewed = JSON.parse(execFileSync(python, ['-I', '-c', planner, backend], {
      input: JSON.stringify({ capture, source: text, activation, operandEdit }), encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 }));
    const input: CandidateInput = { version: 1, plan: snapshot.plan, capture, intent: reviewed.intent,
      reviewedFile: 'model.py', reviewedSource: reviewed.reviewedSource };
    return { root, file, worker, input, logs, ledger, target, select: (value: string) => { selected = value; },
      run: (signal = new AbortController().signal, current = async () => true) =>
        worker.runCandidate({ workspace: root, file, cls: 'Model', candidate: input, current }, signal),
      dispose: async () => { worker.dispose(); await fs.rm(root, { recursive: true, force: true }); } };
  }

  test('saved setup read verifies the selected environment and dependencies without executing model source', async () => {
    const marker = `from pathlib import Path\nPath(__file__ + '.executed').write_text('imported')\n`;
    const f = await fixture(marker + source);
    try {
      const original = f.ledger.state(f.target), check = (signal = new AbortController().signal) =>
        f.worker.checkCapturedSetup(f.root, f.input.plan, f.input.capture.planDigest!, signal);
      await fs.rm(f.file + '.executed');
      assert.equal(await check(), true);
      assert.equal(existsSync(f.file + '.executed'), false, 'a freshness read must never import model source');
      assert.equal(requests.length, 0); assert.deepEqual(f.ledger.state(f.target), original);
      f.select('/different/python'); assert.equal(await check(), false); f.select(python);
      const cancelled = new AbortController(); cancelled.abort(); assert.equal(await check(cancelled.signal), false);
      await fs.writeFile(path.join(f.root, 'added.py'), '# new dependency'); assert.equal(await check(), false);
      await fs.rm(path.join(f.root, 'added.py'));
      await fs.writeFile(f.file, marker + source + '# changed'); assert.equal(await check(), false);
      assert.equal(existsSync(f.file + '.executed'), false);
    } finally { await f.dispose(); }
  });

  test('saved setup read refuses dirty profile/source and Workspace Trust revocation', async () => {
    const f = await fixture(source, 15000, true);
    try {
      const check = () => f.worker.checkCapturedSetup(f.root, f.input.plan, f.input.capture.planDigest!, new AbortController().signal);
      assert.equal(await check(), true);
      const config = path.join(f.root, 'tensorviz.toml');
      editorState.documents.set(config, await fs.readFile(config, 'utf8')); editorState.dirtyDocuments.add(config);
      assert.equal(await check(), false); editorState.dirtyDocuments.clear();
      editorState.documents.set(f.file, source); editorState.dirtyDocuments.add(f.file);
      assert.equal(await check(), false); editorState.dirtyDocuments.clear();
      state.isTrusted = false; assert.equal(await check(), false);
    } finally { await f.dispose(); }
  });

  test('executes exact reviewed bytes and keeps the live ledger untouched', async () => {
    const f = await fixture();
    const history = new CandidateHistory();
    let result!: CandidateRunResult;
    const review = new CandidateReview(f.input.capture, new AbortController().signal, () => true,
      async signal => (result = await f.run(signal)), { history, requestId: 'real-review' });
    try {
      const ledger = structuredClone(f.ledger.state(f.target));
      const summary = await review.run();
      assert.equal(summary.runtime, 'succeeded'); assert.equal(summary.inputs, 'same'); assert.equal(summary.state, 'same');
      assert.equal(summary.semanticPostconditions, 'verified');
      assert.equal(summary.postconditions?.observedOperations, 1);
      assert.match(candidatePostconditionsText(summary), /Requested literal observed in 1 captured occurrence/);
      assert.match(candidatePostconditionsText(summary), /Other executions are unverified/);
      assert.equal(result.outcome.ok, true, JSON.stringify(result.outcome));
      assert.equal(result.outcome.executionCurrent, true);
      assert.equal(result.evidence?.capture?.operations[0].arguments.args[1].kind, 'int');
      assert.deepEqual(result.evidence!.capture!.operations[0].arguments.args[1], { kind: 'int', value: '2' });
      assert.equal(result.evidence!.execution.id, requests[0].id);
      assert.equal(await result.outcome.recheckExecution!(new AbortController().signal), true);
      assert.deepEqual(f.ledger.state(f.target), ledger);
      assert.equal(await fs.readFile(f.file, 'utf8'), source);
      assert.doesNotMatch(f.logs.join('\n'), /class Model|capabilityId|reviewedSource/);
      await fs.writeFile(path.join(f.root, 'helper.py'), '# a new dependency');
      assert.equal(await result.outcome.recheckExecution!(new AbortController().signal), false);
      assert.equal((await review.refresh()).freshness, 'stale');
    } finally { await review.finish(); await f.dispose(); }
    const record = history.list('real-review')[0];
    assert.equal(record.review, 'historical'); assert.equal(record.summary.freshness, 'stale');
    assert.equal(record.unavailable, null); assert.equal(record.executionId, result.evidence!.execution.id);
    assert.equal(isCandidateRecord(record), true);
    assert.equal(record.summary.semanticPostconditions, 'verified');
    const page = history.inspect('real-review', record.runId, { ...record.capture!,
      path: ['operations', 0, 'arguments', 'args', 1], offset: 0, limit: 50 });
    assert.equal(page.items.find(item => item.key === 'value')!.value, '2');
  });

  test('refuses replay substitutions in actual worker evidence', async () => {
    const f = await fixture();
    try {
      const result = await f.run(); assert.ok(result.outcome.ok && result.evidence);
      for (const change of ['attempt', 'workspace', 'digest', 'input', 'file', 'capture', 'base', 'profile',
        'semanticRun', 'semanticCapture', 'semanticBinding', 'semanticOtherExpression', 'semanticTarget',
        'semanticCount', 'semanticStatus', 'semanticValue', 'semanticMissingProof']) {
        const envelope: WorkerEnvelope = structuredClone(result.outcome.envelope);
        const value = envelope.worker.candidateExecution as any;
        if (change === 'attempt') value.id = value.plan.id = 'candidate:older';
        if (change === 'workspace') value.plan.source.workspaceId = f.input.plan.source.workspaceId;
        if (change === 'digest') value.afterDigest = '0'.repeat(64);
        if (change === 'input') value.plan.execution.seed = 99;
        if (change === 'file') value.plan.source.files[0].path = 'other.py';
        if (change === 'capture') (envelope.result as any).diagnostics.semantic_capture.source = f.input.plan.source;
        if (change === 'base') value.baseCaptureId = 'another';
        if (change === 'profile') value.plan.origin = 'profile';
        const evidence = envelope.worker.candidatePostconditions as any;
        if (change === 'semanticRun') evidence.executionId = 'old-run';
        if (change === 'semanticCapture') evidence.captureId = f.input.capture.id;
        if (change === 'semanticBinding') evidence.bindingId = 'missing-binding';
        if (change === 'semanticOtherExpression') evidence.bindingId = (envelope.result as any).diagnostics.semantic_capture.operations[2].sourceBindingIds[0];
        if (change === 'semanticTarget') (envelope.result as any).diagnostics.semantic_capture.operations[0].target = 'aten.mul.Tensor';
        if (change === 'semanticCount') evidence.observedOperations++;
        if (change === 'semanticStatus') { evidence.status = 'failed'; evidence.reason = 'observed_literal_mismatch'; }
        if (change === 'semanticValue') (envelope.result as any).diagnostics.semantic_capture.operations[0].arguments.args[1].value = '1';
        if (change === 'semanticMissingProof') delete evidence.bindingId;
        assert.equal(candidateEvidence(envelope, requests[0]), undefined, change);
      }
    } finally { await f.dispose(); }
  });

  for (const kind of ['functional', 'module'] as const) for (const inplace of [false, true]) {
    test(`${kind} activation with required import and inplace=${inplace} retains exact semantic witnesses`, async () => {
      const option = inplace ? 'inplace=True' : '';
      const text = kind === 'functional' ? `from torch.nn import Module
from torch.nn.functional import relu as action
class Model(Module):
    def forward(self, h):
        for _ in range(2):
            h = action(h${inplace ? ', ' + option : ''})
        return h
` : `from torch.nn import Module, ReLU as Action
class Model(Module):
    def __init__(self):
        super().__init__()
        self.act = Action(${option})
        self.shared = self.act
    def forward(self, h):
        return self.shared(self.act(h))
`;
      const f = await fixture(text, 15000, false, { kind, value: inplace ? 'silu' : 'gelu' });
      try {
        const result = await f.run(); assert.ok(result.outcome.ok && result.evidence, JSON.stringify(result.outcome));
        const evidence = result.evidence.postconditions!;
        assert.equal(evidence.status, 'verified'); assert.equal(evidence.recipe, `${kind}_activation_v1`);
        assert.equal(evidence.observedOperations, 2); assert.equal(evidence.scope, 'observed_expression');
        assert.match(f.input.reviewedSource, /import torch.nn(?:.functional)? as _tensorviz_/);
        assert.equal(await fs.readFile(f.file, 'utf8'), text);
        for (const change of ['recipe', 'count', 'reason', 'target', 'source', 'effects', 'class', 'boundary']) {
          if (kind !== 'module' && (change === 'class' || change === 'boundary')) continue;
          if (change === 'effects' && !inplace) continue;
          const envelope: WorkerEnvelope = structuredClone(result.outcome.envelope);
          const verdict = envelope.worker.candidatePostconditions as any;
          const capture = (envelope.result as any).diagnostics.semantic_capture;
          const op = capture.operations.find((item: any) => item.target === (inplace ? 'aten.silu_.default' : 'aten.gelu.default'));
          if (change === 'recipe') verdict.recipe = 'binary_scalar_v1';
          if (change === 'count') verdict.observedOperations = 1;
          if (change === 'reason') verdict.reason = 'observed_requested_literal';
          if (change === 'target') op.target = 'aten.relu.default';
          if (change === 'effects') op.effects = [];
          if (change === 'source') {
            const anchor = capture.sourceBindings.find((item: any) => item.id === verdict.bindingId).anchors[0];
            anchor.startLine++; anchor.endLine++;
          }
          if (change === 'class') capture.definitions.find((item: any) => item.qualifiedName.endsWith(inplace ? '.SiLU' : '.GELU')).qualifiedName = 'torch.nn.modules.activation.ReLU';
          if (change === 'boundary') capture.invocations[1].output = capture.invocations[1].arguments.args[0];
          assert.equal(candidateEvidence(envelope, requests[0]), undefined, change);
        }
      } finally { await f.dispose(); }
    });
  }

  for (const port of [0, 1] as const) {
    test(`reconnection retains source-flow evidence for operand ${port + 1} in an actual candidate`, async () => {
      const f = await fixture(source, 15000, false, undefined, { port, name: 'h' });
      try {
        const result = await f.run(); assert.ok(result.outcome.ok && result.evidence, JSON.stringify(result.outcome));
        const evidence = result.evidence.postconditions!;
        assert.equal(evidence.status, 'verified'); assert.equal(evidence.recipe, 'straight_line_tensor_v1');
        assert.equal(evidence.reason, 'observed_requested_argument'); assert.equal(evidence.observedOperations, 1);
        const capture = result.evidence.capture!;
        const op = capture.operations.find(item => item.sourceBindingIds.includes(evidence.bindingId!))!;
        assert.deepEqual(op.arguments.args[port], capture.inputs.args[0]);
        assert.notDeepEqual(op.arguments.args[port], op.arguments.args[1 - port]);
        assert.equal(await fs.readFile(f.file, 'utf8'), source);
        for (const change of ['count', 'recipe', 'reason', 'binding', 'source', 'target', 'argument']) {
          const envelope: WorkerEnvelope = structuredClone(result.outcome.envelope);
          const verdict = envelope.worker.candidatePostconditions as any;
          const recorded = (envelope.result as any).diagnostics.semantic_capture;
          const event = recorded.operations.find((item: any) => item.id === op.id);
          if (change === 'count') verdict.observedOperations++;
          if (change === 'recipe') verdict.recipe = 'binary_scalar_v1';
          if (change === 'reason') verdict.reason = 'observed_requested_literal';
          if (change === 'binding') verdict.bindingId = recorded.operations[0].sourceBindingIds[0];
          if (change === 'source') {
            const anchor = recorded.sourceBindings.find((item: any) => item.id === verdict.bindingId).anchors[0];
            anchor.startLine++; anchor.endLine++;
          }
          if (change === 'target') event.target = 'aten.mul.Tensor';
          if (change === 'argument') event.arguments.args[port] = { kind: 'int', value: '1' };
          assert.equal(candidateEvidence(envelope, requests[0]), undefined, change);
        }
      } finally { await f.dispose(); }
    });
  }

  for (const port of [0, 1] as const) {
    test(`insertion on operand ${port + 1} uses an actual candidate and exact source origin`, async () => {
      const text = port === 0 ? source : source.replace('import torch', 'from torch.nn import Module').replace('torch.nn.Module', 'Module');
      const f = await fixture(text, 15000, false, undefined, { port, activation: 'gelu' });
      try {
        const result = await f.run(); assert.ok(result.outcome.ok && result.evidence, JSON.stringify(result.outcome));
        const evidence = result.evidence.postconditions!;
        assert.equal(evidence.status, 'verified'); assert.equal(evidence.recipe, 'straight_line_unary_insert_v1');
        assert.equal(evidence.reason, 'observed_requested_insertion'); assert.equal(evidence.observedOperations, 1);
        const capture = result.evidence.capture!;
        const parent = capture.operations.find(op => op.sourceBindingIds.includes(evidence.bindingId!))!;
        const unary = capture.operations.find(op => op.target === 'aten.gelu.default')!;
        assert.deepEqual(parent.arguments.args[port], unary.output);
        assert.deepEqual(parent.arguments.args[1 - port], unary.arguments.args[0]);
        if (port === 1) assert.match(f.input.reviewedSource, /import torch.nn.functional as _tensorviz_/);
        assert.equal(await fs.readFile(f.file, 'utf8'), text);
        for (const change of ['recipe', 'reason', 'count', 'source', 'argument']) {
          const envelope: WorkerEnvelope = structuredClone(result.outcome.envelope);
          const verdict = envelope.worker.candidatePostconditions as any;
          const recorded = (envelope.result as any).diagnostics.semantic_capture;
          if (change === 'recipe') verdict.recipe = 'straight_line_tensor_v1';
          if (change === 'reason') verdict.reason = 'observed_requested_argument';
          if (change === 'count') verdict.observedOperations++;
          if (change === 'source') recorded.sourceBindings.find((item: any) => item.id === verdict.bindingId).anchors[0].startColumn++;
          if (change === 'argument') recorded.operations.find((op: any) => op.id === parent.id).arguments.args[port] = { kind: 'int', value: '1' };
          assert.equal(candidateEvidence(envelope, requests[0]), undefined, change);
        }
      } finally { await f.dispose(); }
    });
  }

  for (const shared of [false, true]) {
    test(`deletion verifies surviving flow with shared=${shared} in an actual candidate`, async () => {
      const text = `import torch
from torch.nn import functional as F
class ${shared ? 'Block' : 'Model'}(torch.nn.Module):
    def forward(self, h):
        return F.gelu(h)
` + (shared ? `class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = Block(), Block()
    def forward(self, h):
        return self.b(self.a(h))
` : '');
      const f = await fixture(text, 15000, false, undefined, { delete: true });
      try {
        const result = await f.run(); assert.ok(result.outcome.ok && result.evidence, JSON.stringify(result.outcome));
        const evidence = result.evidence.postconditions!, capture = result.evidence.capture!;
        assert.equal(evidence.status, 'verified'); assert.equal(evidence.recipe, 'straight_line_unary_delete_v1');
        assert.equal(evidence.reason, 'observed_requested_deletion'); assert.equal(evidence.observedOperations, shared ? 2 : 1);
        assert.equal(capture.operations.length, 0); assert.deepEqual(capture.output, capture.inputs.args[0]);
        assert.equal(capture.sourceBindings.find(binding => binding.id === evidence.bindingId)?.kind, 'definition');
        assert.equal(await fs.readFile(f.file, 'utf8'), text);
        for (const change of ['recipe', 'count', 'scope', 'kind', 'origin', 'call', 'definition']) {
          if (change === 'definition' && !shared) continue;
          const envelope: WorkerEnvelope = structuredClone(result.outcome.envelope);
          const verdict = envelope.worker.candidatePostconditions as any;
          const recorded = (envelope.result as any).diagnostics.semantic_capture;
          const binding = recorded.sourceBindings.find((item: any) => item.id === verdict.bindingId);
          if (change === 'recipe') verdict.recipe = 'straight_line_unary_insert_v1';
          if (change === 'count') verdict.observedOperations++;
          if (change === 'scope') binding.anchors[0].scope = 'Other.forward';
          if (change === 'kind') binding.kind = 'call';
          if (change === 'origin') { binding.anchors[0].startLine += 50; binding.anchors[0].endLine += 50; }
          if (change === 'call') {
            const def = recorded.definitions.find((item: any) => item.sourceBindingIds.includes(verdict.bindingId));
            const module = recorded.modules.find((item: any) => item.definitionId === def.id);
            recorded.invocations.find((item: any) => item.moduleId === module.id).status = 'failed';
          }
          if (change === 'definition') {
            const def = recorded.definitions.find((item: any) => item.sourceBindingIds.includes(verdict.bindingId));
            const other = recorded.definitions.find((item: any) => item.id !== def.id && item.sourceBindingIds.length);
            def.sourceBindingIds.push(other.sourceBindingIds[0]);
          }
          assert.equal(candidateEvidence(envelope, requests[0]), undefined, change);
        }
      } finally { await f.dispose(); }
    });
  }

  test('replays the exact retained saved profile and refuses later profile changes', async () => {
    const f = await fixture(source, 15000, true);
    try {
      const result = await f.run();
      assert.equal(result.outcome.executionCurrent, true);
      assert.equal(result.evidence?.execution.plan.origin, 'profile');
      assert.equal(result.evidence?.execution.plan.profileName, 'small');
      assert.deepEqual(result.evidence?.execution.plan.execution, f.input.plan.execution);
      assert.equal(result.evidence?.capture?.inputDigest, f.input.capture.inputDigest);
      await fs.appendFile(path.join(f.root, 'tensorviz.toml'), '# profile revision');
      assert.equal(await result.outcome.recheckExecution!(new AbortController().signal), false);
    } finally { await f.dispose(); }
  });

  for (const slot of [['args', 0], ['args', 1], ['kwargs', 'width']]) {
    test(`constructor entry evidence checks the exact ${slot.join('.')} binding`, async () => {
      const text = `import torch
class Block(torch.nn.Module):
    def __init__(self, width=4, /, *sizes, **options):
        super().__init__()
    def forward(self, h): return h
def block(): return Block(4, 6, width=3)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.unused = block(), block()
    def forward(self, h): return self.a(self.a(h))
`;
      const f = await fixture(text, 15000, false, undefined, { constructor: slot, value: { kind: 'int', value: '8' } });
      let result!: CandidateRunResult;
      const review = new CandidateReview(f.input.capture, new AbortController().signal, () => true,
        async signal => (result = await f.run(signal)));
      try {
        const summary = await review.run();
        assert.equal(summary.runtime, 'succeeded'); assert.equal(summary.semanticPostconditions, 'verified');
        assert.equal(summary.postconditions?.observedInitializers, 2);
        assert.equal(summary.postconditions?.observedOperations, 0);
        assert.match(candidatePostconditionsText(summary), /Requested constructor value observed in 2 retained initializer entries/);
        assert.match(candidatePostconditionsText(summary), /later behavior are unverified/);
        assert.equal(result.evidence!.capture!.operations.length, 0);
        for (const change of ['count', 'scope', 'class', 'entry', 'value', 'binding', 'origin']) {
          const envelope: WorkerEnvelope = structuredClone(result.outcome.envelope!);
          const verdict = envelope.worker.candidatePostconditions as any;
          const capture = (envelope.result as any).diagnostics.semantic_capture;
          const module = capture.modules.find((item: any) => item.constructorBindingIds.includes(verdict.bindingId));
          const binding = capture.sourceBindings.find((item: any) => item.id === verdict.bindingId);
          if (change === 'count') verdict.observedInitializers++;
          if (change === 'scope') { verdict.observedOperations = 2; verdict.observedInitializers = 0; verdict.scope = 'observed_expression'; }
          if (change === 'class') { module.constructorClassVerified = false; module.initializerArguments = null; }
          if (change === 'entry') module.initializerArguments = null;
          if (change === 'value') {
            const value = slot[0] === 'kwargs' ? module.initializerArguments.varKwargs.width
              : slot[1] === 0 ? module.initializerArguments.positional[0].value : module.initializerArguments.varArgs[0];
            value.value = '3';
          }
          if (change === 'binding') binding.kind = 'call';
          if (change === 'origin') { binding.anchors[0].startLine++; binding.anchors[0].endLine++; }
          assert.equal(candidateEvidence(envelope, requests[0]), undefined, change);
        }
      } finally { await review.finish(); await f.dispose(); }
    });
  }

  for (const local of [false, true]) {
    test(`configuration evidence checks shared arguments with ${local ? 'same-line local' : 'module'} definition`, async () => {
      let text = `import torch
class Block(torch.nn.Module):
    def __init__(self, width=4, /, *sizes, **options):
        super().__init__()
    def forward(self, h): return h
amount = 4
def block(): return Block(amount, amount, width=amount)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.unused = block(), block()
        self.b = Block(amount)
    def forward(self, h): return self.b(self.a(self.a(h)))
`;
      if (local) text = text.replace('amount = 4\ndef block(): return', 'def block():\n    label = "🚀"; amount = 4; return')
        .replace('self.b = Block(amount)', 'self.b = block()');
      const f = await fixture(text, 15000, false, undefined, { configuration: ['kwargs', 'width'], value: { kind: 'int', value: '900' } });
      let result!: CandidateRunResult;
      const review = new CandidateReview(f.input.capture, new AbortController().signal, () => true,
        async signal => (result = await f.run(signal)));
      try {
        const summary = await review.run();
        assert.equal(summary.runtime, 'succeeded'); assert.equal(summary.semanticPostconditions, 'verified');
        assert.equal(summary.postconditions?.observedInitializers, 3); assert.equal(summary.postconditions?.observedOperations, 0);
        assert.equal(summary.postconditions?.configurationUses?.length, local ? 3 : 4);
        assert.match(candidatePostconditionsText(summary), /direct constructor arguments across 3 retained initializer entries/);
        assert.match(candidatePostconditionsText(summary), /Other references and later behavior are unverified/);
        assert.equal(result.evidence!.capture!.operations.length, 0);
        for (const change of ['count', 'scope', 'value', 'entry', 'class', 'missing_selected', 'slot', 'binding', 'competing', 'origin']) {
          const envelope: WorkerEnvelope = structuredClone(result.outcome.envelope!);
          const verdict = envelope.worker.candidatePostconditions as any;
          const capture = (envelope.result as any).diagnostics.semantic_capture;
          const use = verdict.configurationUses.at(-1);
          const module = capture.modules.find((item: any) => item.constructorBindingIds.includes(use.bindingId));
          const binding = capture.sourceBindings.find((item: any) => item.id === use.bindingId);
          if (change === 'count') verdict.observedInitializers++;
          if (change === 'scope') verdict.scope = 'observed_initializer_entries';
          if (change === 'value') module.initializerArguments.positional[0].value.value = '4';
          if (change === 'entry') module.initializerArguments = null;
          if (change === 'class') { module.constructorClassVerified = false; module.initializerArguments = null; }
          if (change === 'missing_selected') verdict.configurationUses = verdict.configurationUses.filter((item: any) => item.sourceArgumentPath[0] !== 'kwargs');
          if (change === 'slot') use.sourceArgumentPath = ['args', 99];
          if (change === 'binding') binding.kind = 'call';
          if (change === 'competing') capture.sourceBindings.push({ ...structuredClone(binding), id: 'constructor:competing' });
          if (change === 'origin') binding.anchors[0].digest = 'a'.repeat(64);
          assert.equal(candidateEvidence(envelope, requests[0]), undefined, change);
        }
      } finally { await review.finish(); await f.dispose(); }
    });
  }

  test('preserves a real shape failure as candidate evidence without changing live slots', async () => {
    const text = source.replace('        b = a + a\n        return b * 2',
      '        return h @ torch.ones(int(a[0, 0]) + 2, 2)');
    const f = await fixture(text);
    try {
      const before = structuredClone(f.ledger.state(f.target));
      const result = await f.run();
      assert.equal(result.outcome.ok, false);
      assert.equal(result.outcome.exitCode, 4);
      assert.equal(result.outcome.executionCurrent, true, 'freshness is distinct from runtime success');
      assert.equal(result.evidence?.capture?.status, 'partial');
      assert.ok(result.evidence?.capture?.diagnostics.some(item => item.code === 'invalid_shape'));
      assert.deepEqual(f.ledger.state(f.target), before);
    } finally { await f.dispose(); }
  });

  test('candidate prints and arbitrary failure messages do not enter host logs or summaries', async () => {
    const text = source.replace('import torch', 'import torch\nfrom pathlib import Path')
      .replace('        b = a + a', '        if Path(__file__).parent.name == "workspace":\n            print("private_candidate_stdout_marker")\n            raise RuntimeError("private_candidate_failure_marker")\n        b = a + a');
    const f = await fixture(text);
    try {
      const result = await f.run();
      assert.equal(result.outcome.ok, false);
      assert.doesNotMatch(f.logs.join('\n'), /private_candidate/);
      if (!result.outcome.ok) assert.doesNotMatch(result.outcome.reason + result.outcome.detail, /private_candidate/);
      assert.ok(result.evidence, 'structured candidate diagnostics remain inspectable');
    } finally { await f.dispose(); }
  });

  for (const change of ['dirty', 'config', 'source', 'interpreter', 'trust', 'guard']) {
    test(`${change} prevents candidate execution without applying source`, async () => {
      const f = await fixture();
      try {
        if (change === 'dirty') { openStubEditor(f.file, source + '# buffer'); editorState.dirtyDocuments.add(f.file); }
        if (change === 'config') await fs.writeFile(path.join(f.root, 'tensorviz.toml'), '# changed');
        if (change === 'source') await fs.writeFile(f.file, source + '# changed');
        if (change === 'interpreter') f.select('/not-selected');
        if (change === 'trust') state.isTrusted = false;
        const result = await f.run(undefined, async () => change !== 'guard');
        assert.equal(result.outcome.ok, false);
        assert.equal(result.evidence, undefined);
        assert.equal(requests.length, 0, 'refuse before materializing a candidate transport');
      } finally { await f.dispose(); }
    });
  }

  test('rechecks selected environment immediately before launch', async () => {
    const f = await fixture(); let checks = 0;
    try {
      const result = await f.run(undefined, async () => { if (++checks === 2) f.select('/changed'); return true; });
      assert.equal(result.outcome.ok, false);
      assert.equal(result.evidence, undefined);
      assert.equal(requests.length, 1, 'the prepared private request is still cleaned up');
      assert.doesNotMatch(f.logs.join('\n'), /--candidate-execution/);
    } finally { await f.dispose(); }
  });

  test('buffer version changes after execution cannot regain current status', async () => {
    const f = await fixture(); openStubEditor(f.file, source);
    try {
      const result = await f.run(); assert.equal(result.outcome.executionCurrent, true);
      openStubEditor(f.file, source);
      assert.equal(await result.outcome.recheckExecution!(new AbortController().signal), false);
    } finally { await f.dispose(); }
  });

  test('returned success remains stale when the host target changed during the run', async () => {
    const f = await fixture(); let checks = 0;
    try {
      const result = await f.run(undefined, async () => ++checks < 3);
      assert.equal(result.outcome.ok, true);
      assert.equal(result.outcome.executionCurrent, false);
      assert.equal(result.evidence?.baseCurrent, false);
      result.evidence!.baseCurrent = true;
      assert.equal(await result.outcome.recheckExecution!(new AbortController().signal), false);
    } finally { await f.dispose(); }
  });

  test('candidate request snapshots cannot be changed by the caller', async () => {
    const f = await fixture();
    try {
      const request = candidateRequest(f.input);
      f.input.plan.execution.seed = 99;
      f.input.reviewedSource += '\nprivate_extra_edit()';
      assert.notEqual(request.plan.execution.seed, f.input.plan.execution.seed);
      assert.doesNotMatch(request.reviewedSource, /private_extra_edit/);
      assert.ok(workerArgs({ file: f.file, workspace: f.root, cls: 'Model' }, 'candidate').includes('--candidate-execution'));
      assert.throws(() => workerArgs({ file: f.file, workspace: f.root, cls: 'Model', input: 'private()' }, 'candidate'));
    } finally { await f.dispose(); }
  });

  test('cancellation during final cleanup cannot return an earlier success', async () => {
    const f = await fixture(), controller = new AbortController();
    const outer = transport.candidateRequestFile;
    let entered!: () => void, release!: () => void;
    const cleaning = new Promise<void>(resolve => { entered = resolve; });
    const gate = new Promise<void>(resolve => { release = resolve; });
    transport.candidateRequestFile = async request => {
      const result = await outer(request);
      return { file: result.file, dispose: async () => { entered(); await gate; await result.dispose(); } };
    };
    const pending = f.run(controller.signal);
    try {
      await cleaning; controller.abort(); release();
      const result = await pending;
      assert.ok(!result.outcome.ok && result.outcome.cancelled);
      assert.equal(result.evidence, undefined);
    } finally { controller.abort(); release(); await pending; await f.dispose(); }
  });

  test('failed private-directory cleanup refuses without exposing arbitrary errors', async () => {
    const f = await fixture(), outer = transport.candidateRequestFile;
    let cleanup: (() => Promise<void>) | undefined;
    transport.candidateRequestFile = async request => {
      const result = await outer(request); cleanup = result.dispose;
      return { file: result.file, dispose: async () => { throw new Error('private_cleanup_marker'); } };
    };
    try {
      const result = await f.run();
      assert.ok(!result.outcome.ok);
      assert.match(result.outcome.reason, /cleanup/);
      assert.equal(result.cleanupFailed, true);
      assert.doesNotMatch(result.outcome.reason + result.outcome.detail + f.logs.join('\n'), /private_cleanup_marker/);
      assert.equal(result.evidence, undefined);
      assert.ok(existsSync(path.dirname(temporary[0])));
    } finally { await cleanup?.(); await f.dispose(); }
  });

  test('source changed during final cleanup cannot publish an earlier freshness verdict', async () => {
    const f = await fixture(), outer = transport.candidateRequestFile;
    transport.candidateRequestFile = async request => {
      const result = await outer(request);
      return { file: result.file, dispose: async () => {
        await fs.appendFile(f.file, '# changed during cleanup');
        await result.dispose();
      } };
    };
    try {
      const result = await f.run();
      assert.equal(result.outcome.ok, true, 'the candidate itself completed');
      assert.equal(result.outcome.executionCurrent, false);
      assert.equal(result.evidence?.baseCurrent, false);
      await fs.writeFile(f.file, source);
      assert.equal(await result.outcome.recheckExecution!(new AbortController().signal), false);
    } finally { await f.dispose(); }
  });

  test('a changed saved inventory fails through the retained-plan preflight', async () => {
    const f = await fixture();
    try {
      const before = await resolveRetainedPlan(python, backend, f.root, f.input.plan, new AbortController().signal);
      assert.equal(before.applicabilityDigest, f.input.capture.planDigest);
      await fs.writeFile(path.join(f.root, 'new.py'), 'raise RuntimeError("must not import")');
      await assert.rejects(resolveRetainedPlan(python, backend, f.root, f.input.plan, new AbortController().signal));
    } finally { await f.dispose(); }
  });

  for (const stop of ['cancel', 'timeout'] as const) {
    test(`${stop} stops a real candidate and cleans abandoned staging`, async () => {
      const text = source.replace('import torch', 'import torch\nimport time\nfrom pathlib import Path')
        .replace('        a = h + 1', '        if Path(__file__).parent.name == "workspace":\n            Path(__file__).with_suffix(".started").touch()\n            time.sleep(30)\n        a = h + 1');
      const f = await fixture(text, 4000), controller = new AbortController();
      try {
        const pending = f.run(controller.signal);
        const until = Date.now() + 10000;
        let entered = false;
        while (!entered && Date.now() < until) {
          if (temporary[0]) {
            const directory = path.dirname(temporary[0]);
            for (const name of await fs.readdir(directory).catch(() => [])) {
              entered ||= existsSync(path.join(directory, name, 'workspace/model.started'));
            }
          }
          if (!entered) await new Promise(resolve => setTimeout(resolve, 25));
        }
        assert.equal(entered, true, 'the candidate must actually enter forward before the stop');
        if (stop === 'cancel') controller.abort();
        const result = await pending;
        assert.ok(!result.outcome.ok);
        assert.equal(stop === 'cancel' ? result.outcome.cancelled : result.outcome.timedOut, true);
        assert.equal(result.evidence, undefined);
        assert.equal(await fs.readFile(f.file, 'utf8'), text);
      } finally { controller.abort(); await f.dispose(); }
    });
  }

  test('cancellation terminates an owned child before removing its private workspace', async () => {
    const childCode = 'import signal,time,sys;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(sys.argv[1]).touch();time.sleep(30)';
    const text = source.replace('import torch', 'import torch\nimport subprocess,sys,time\nfrom pathlib import Path')
      .replace('        a = h + 1', `        if Path(__file__).parent.name == "workspace":
            ready = Path(__file__).with_suffix(".ready")
            child = subprocess.Popen([sys.executable, "-c", ${JSON.stringify(childCode)}, str(ready)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while not ready.exists(): time.sleep(0.01)
            Path(__file__).with_suffix(".childpid").write_text(str(child.pid))
            time.sleep(30)
        a = h + 1`);
    const f = await fixture(text), controller = new AbortController();
    const pending = f.run(controller.signal);
    try {
      const until = Date.now() + 10000;
      let pid: number | undefined;
      while (!pid && Date.now() < until) {
        if (temporary[0]) {
          const directory = path.dirname(temporary[0]);
          for (const name of await fs.readdir(directory).catch(() => [])) {
            const file = path.join(directory, name, 'workspace/model.childpid');
            if (existsSync(file)) pid = Number(await fs.readFile(file, 'utf8'));
          }
        }
        if (!pid) await new Promise(resolve => setTimeout(resolve, 25));
      }
      assert.ok(pid, 'candidate must have started its child');
      controller.abort();
      if (process.platform !== 'win32') {
        await new Promise(resolve => setTimeout(resolve, 100));
        assert.ok(existsSync(path.dirname(temporary[0])), 'do not remove files while forced termination is pending');
      }
      const result = await pending;
      assert.ok(!result.outcome.ok && result.outcome.cancelled);
      if (process.platform === 'win32') assert.throws(() => process.kill(pid!, 0));
      else {
        let status = '';
        try { status = execFileSync('ps', ['-o', 'stat=', '-p', String(pid)], { encoding: 'utf8' }).trim(); }
        catch { /* reaped */ }
        assert.ok(!status || status.startsWith('Z'), `owned child is still running: ${status}`);
      }
    } finally { controller.abort(); await pending; await f.dispose(); }
  });

  test('an older cancellation cannot cancel or replace the newer candidate', async () => {
    const text = source.replace('import torch', 'import torch\nimport time\nfrom pathlib import Path')
      .replace('        b = a + a', '        if Path(__file__).parent.name == "workspace" and int(a[0, 0]) == 3:\n            Path(__file__).with_suffix(".started").touch()\n            time.sleep(30)\n        b = a + a');
    const f = await fixture(text), old = new AbortController(), next = new AbortController();
    const first = f.run(old.signal);
    let second: ReturnType<typeof f.run> | undefined;
    try {
      const until = Date.now() + 10000;
      let started = false;
      while (!started && Date.now() < until) {
        if (temporary[0]) for (const name of await fs.readdir(path.dirname(temporary[0])).catch(() => [])) {
          started ||= existsSync(path.join(path.dirname(temporary[0]), name, 'workspace/model.started'));
        }
        if (!started) await new Promise(resolve => setTimeout(resolve, 25));
      }
      assert.ok(started);
      assert.equal(f.input.intent.kind, 'set_literal');
      if (f.input.intent.kind !== 'set_literal') throw new Error('fixture');
      f.input.intent.value = { kind: 'int', value: '3' };
      f.input.reviewedSource = text.replace('h + 1', 'h + 3');
      second = f.run(next.signal);
      while (requests.length < 2 && Date.now() < until) await new Promise(resolve => setTimeout(resolve, 25));
      assert.equal(requests.length, 2);
      old.abort();
      const [obsolete, current] = await Promise.all([first, second]);
      assert.ok(!obsolete.outcome.ok && obsolete.outcome.cancelled);
      assert.equal(obsolete.evidence, undefined);
      assert.equal(current.outcome.ok, true);
      assert.deepEqual(current.evidence?.capture?.operations[0].arguments.args[1], { kind: 'int', value: '3' });
      assert.equal(current.evidence?.execution.id, requests[1].id);
      assert.notEqual(requests[0].id, requests[1].id);
      assert.equal(await fs.readFile(f.file, 'utf8'), text);
    } finally { old.abort(); next.abort(); await Promise.allSettled([first, second]); await f.dispose(); }
  });
});
