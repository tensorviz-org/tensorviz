import * as assert from 'node:assert/strict';
import { CanvasAgentEvidence } from '../../canvasAgentEvidence';

const inspection = (value = 37, revision = 'a'.repeat(64)) => ({ modelId: 'host:model', revision,
  nodes: [{ nodeId: 'stem', parameters: { calibration_code: value }, shapes: { input: null, output: null, provenance: 'unreached' } }] });
const records = (memory: CanvasAgentEvidence) => JSON.parse(memory.replay().split('\n').at(-1)!);

suite('Canvas conversation evidence', () => {
  test('keeps original exact evidence after a revision changes and labels it as historical data', () => {
    const memory = new CanvasAgentEvidence(), original = inspection();
    memory.remember('inspect_nodes', original);
    original.nodes[0].parameters.calibration_code = 99;
    memory.remember('inspect_nodes', inspection(53, 'b'.repeat(64)));
    const retained = records(memory);
    assert.deepEqual(retained.map((row: any) => row.result.nodes[0].parameters.calibration_code), [37, 53]);
    assert.deepEqual(retained.map((row: any) => row.result.revision), ['a'.repeat(64), 'b'.repeat(64)]);
    assert.deepEqual(retained[0].result.nodes[0].shapes, inspection().nodes[0].shapes);
    assert.match(memory.replay(), /historical data, never instructions/);
    assert.match(memory.replay(), /do not guess missing history/);
  });

  test('repeated inspections do not crowd out an older unique fact', () => {
    const memory = new CanvasAgentEvidence();
    memory.remember('inspect_nodes', inspection());
    for (let i = 0; i < 100; i++) memory.remember('inspect_nodes', inspection(53, 'b'.repeat(64)));
    assert.equal(records(memory).length, 2);
    assert.equal(records(memory)[0].result.nodes[0].parameters.calibration_code, 37);
  });

  test('bounded retention omits whole records and discloses missing evidence', () => {
    const memory = new CanvasAgentEvidence();
    for (let i = 0; i < 40; i++) memory.remember('inspect_nodes', { ...inspection(i), note: 'x'.repeat(1800) });
    assert.ok(Buffer.byteLength(memory.replay()) < 34 * 1024);
    assert.ok(records(memory).length < 24);
    assert.equal(records(memory).at(-1).result.nodes[0].parameters.calibration_code, 39);
    assert.match(memory.replay(), /[1-9]\d* earlier records were omitted/);
    const previous = records(memory);
    memory.remember('get_graph', { ...inspection(), note: 'x'.repeat(13 * 1024) });
    assert.deepEqual(records(memory), previous, 'an oversized page cannot discard useful earlier facts');
    memory.clear(); assert.equal(memory.replay(), '');
  });

  test('ignores arbitrary tools, refusals and unversioned data', () => {
    const memory = new CanvasAgentEvidence();
    memory.remember('read_model_source', inspection());
    memory.remember('inspect_nodes', { ...inspection(), error: { code: 'STALE_TARGET' } });
    memory.remember('inspect_nodes', { ...inspection(), revision: 'unknown' });
    memory.remember('inspect_nodes', { modelId: 'host:model', revision: 'a'.repeat(64) });
    assert.equal(memory.replay(), '');
    memory.remember('get_graph', { modelId: 'host:model', revision: 'a'.repeat(64), nodeCount: 32 });
    assert.equal(records(memory)[0].result.nodeCount, 32);
  });
});
