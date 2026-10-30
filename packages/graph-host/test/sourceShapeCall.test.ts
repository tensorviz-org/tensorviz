import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { semanticCapture } from '@tensorviz/graph-host/semanticCapture';
const fixture = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/source-shape-call.json'), 'utf8'));

suite('observed source shape argument correspondence', () => {
  function read(capture: any) {
    const setup = { id: capture.attemptId, source: structuredClone(capture.source), captureLimits: structuredClone(capture.limits) };
    return semanticCapture(capture, capture.attemptId, setup as any, capture.planDigest);
  }
  test('retains the actual CPU capture and keeps absent evidence absent', () => {
    const capture = structuredClone(fixture.portable.capture);
    assert.equal(read(capture), capture);
    delete capture.operations[0].sourceShapeCall;
    assert.equal(read(capture), capture);
    assert.equal(Object.hasOwn(capture.operations[0], 'sourceShapeCall'), false);
  });
  test('legacy dimension mappings remain readable without acquiring tensor-read evidence', () => {
    const capture = structuredClone(fixture.portable.capture);
    for (const dimension of capture.operations[0].sourceShapeCall.dimensions) delete dimension.tensorDimension;
    assert.equal(read(capture), capture);
    assert.ok(capture.operations[0].sourceShapeCall.dimensions.every((dimension: any) => !Object.hasOwn(dimension, 'tensorDimension')));
  });
  for (const item of fixture.cases) test(`rejects ${item.name}`, () => {
    const capture = structuredClone(fixture.portable.capture);
    for (const patch of item.patches) {
      let target = capture;
      for (const key of patch.path.slice(0, -1)) target = target[key];
      target[patch.path[patch.path.length - 1]] = patch.value;
    }
    assert.equal(read(capture), undefined);
  });
});
