import * as assert from 'node:assert/strict';
import { AUTO_FAKE_MIN_PARAMETERS, AutoTraceTier, capturedParameterCount } from '@tensorviz/graph-host/traceTierAuto';
import type { TraceOutcome } from '@tensorviz/graph-host/traceWorkerCore';

const model = { workspace: '/ws', file: '/ws/m.py', cls: 'M' };
const captured = (parameterCount: unknown, extra: Partial<TraceOutcome> = {}): TraceOutcome => ({
  ok: true, exitCode: 0, ...extra,
  envelope: { protocol_version: 1, spans: {}, error: null, worker: {}, result: { diagnostics: { semantic_capture: {
    rootInvocationId: 'invocation:0',
    invocations: [{ id: 'invocation:1', parameterCount: 1 }, { id: 'invocation:0', parameterCount }],
  } } } },
} as TraceOutcome);
const LARGE = AUTO_FAKE_MIN_PARAMETERS, SMALL = AUTO_FAKE_MIN_PARAMETERS - 1;

suite('auto trace tier', () => {
  test('real and fake settings are returned unchanged, whatever is remembered', () => {
    const auto = new AutoTraceTier();
    auto.record(model, 'real', captured(LARGE));
    assert.equal(auto.tier('real', model), 'real');
    assert.equal(auto.tier('fake', model), 'fake');
    const small = new AutoTraceTier();
    small.record(model, 'real', captured(1));
    assert.equal(small.tier('fake', model), 'fake');
  });

  test('the first run of a model is real; a large one runs fake from the next run', () => {
    const auto = new AutoTraceTier();
    assert.equal(auto.tier('auto', model), 'real');
    auto.record(model, 'real', captured(LARGE));
    assert.equal(auto.tier('auto', model), 'fake');
  });

  test('a model below the threshold stays real', () => {
    const auto = new AutoTraceTier();
    auto.record(model, 'real', captured(SMALL));
    assert.equal(auto.tier('auto', model), 'real');
  });

  test('a model that shrinks below the threshold goes back to real', () => {
    const auto = new AutoTraceTier();
    auto.record(model, 'real', captured(LARGE));
    auto.record(model, 'fake', captured(SMALL));
    assert.equal(auto.tier('auto', model), 'real');
  });

  test('a fake run that fell back to real is remembered: the model runs real from then on', () => {
    const auto = new AutoTraceTier();
    auto.record(model, 'real', captured(LARGE));
    auto.record(model, 'fake', captured(LARGE, { fakeTensorFallback: { reason: 'data_dependent', errorType: 'E', operation: '', phase: 'forward' } }));
    assert.equal(auto.tier('auto', model), 'real');
    auto.record(model, 'real', captured(LARGE));
    assert.equal(auto.tier('auto', model), 'real', 'a later real run does not undo it');
  });

  test('a failed run or one without a capture teaches nothing', () => {
    const auto = new AutoTraceTier();
    auto.record(model, 'real', { ok: false, reason: 'r', detail: 'd', envelope: captured(LARGE).envelope } as TraceOutcome);
    auto.record(model, 'real', { ok: true, exitCode: 0, envelope: { protocol_version: 1, spans: {}, error: null, worker: {}, result: {} } } as TraceOutcome);
    assert.equal(auto.tier('auto', model), 'real');
  });

  test('models are remembered separately by workspace, file and class', () => {
    const auto = new AutoTraceTier();
    auto.record(model, 'real', captured(LARGE));
    assert.equal(auto.tier('auto', { ...model, cls: 'Other' }), 'real');
    assert.equal(auto.tier('auto', { ...model, file: '/ws/other.py' }), 'real');
    assert.equal(auto.tier('auto', { ...model, workspace: '/other' }), 'real');
  });

  test('memory is bounded: the least recently recorded model is forgotten first', () => {
    const auto = new AutoTraceTier(10, 2);
    const a = { ...model, cls: 'A' }, b = { ...model, cls: 'B' }, c = { ...model, cls: 'C' };
    auto.record(a, 'real', captured(10)); auto.record(b, 'real', captured(10));
    auto.record(a, 'fake', captured(10));                 // a is now the most recent
    auto.record(c, 'real', captured(10));
    assert.deepEqual([auto.tier('auto', a), auto.tier('auto', b), auto.tier('auto', c)], ['fake', 'real', 'fake']);
    auto.clear();
    assert.equal(auto.tier('auto', a), 'real');
  });

  test('capturedParameterCount reads only the root invocation\'s exact count', () => {
    assert.equal(capturedParameterCount(captured(12_811_778)), 12_811_778);
    for (const bad of [null, -1, 1.5, '7', Number.MAX_SAFE_INTEGER + 1]) assert.equal(capturedParameterCount(captured(bad)), undefined);
    assert.equal(capturedParameterCount({ ok: false, reason: 'r', detail: 'd' } as TraceOutcome), undefined);
  });
});
