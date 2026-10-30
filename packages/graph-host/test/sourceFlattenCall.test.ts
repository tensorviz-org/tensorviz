import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { flattenedShape } from '@tensorviz/contracts/sourceShapeCall';
import { semanticCapture } from '@tensorviz/graph-host/semanticCapture';
import { canonicalFixture } from '../../../tests/fixtures/semantic.cjs';

const fixture = JSON.parse(readFileSync(path.join(__dirname, '../../../tests/fixtures/contracts/source-flatten-call.json'), 'utf8'));
const read = (capture: any) => {
  const { plan } = canonicalFixture(capture.attemptId);
  plan.source = capture.source; plan.captureLimits = capture.limits;
  return semanticCapture(capture, capture.attemptId, plan, capture.planDigest);
};

suite('observed flatten source arguments', () => {
  for (const item of fixture.captures) test(`${item.source.trim()} on ${JSON.stringify(item.inputShape)}`, () => {
    const capture = item.portable.capture;
    assert.equal(read(capture), capture);
    const legacy = structuredClone(capture); delete legacy.operations[0].sourceFlattenCall;
    assert.equal(read(legacy), legacy);
  });
  for (const item of fixture.cases) test(`rejects ${item.name}`, () => {
    const capture = structuredClone(fixture.captures[0].portable.capture);
    for (const patch of item.patches) {
      let target = capture; for (const key of patch.path.slice(0, -1)) target = target[key];
      target[patch.path[patch.path.length - 1]] = patch.value;
    }
    assert.equal(read(capture), undefined);
  });
  for (const [index, item] of fixture.dimensions.entries()) test(`actual flatten dimension oracle ${index}`, () => {
    assert.deepEqual(flattenedShape(item.inputShape, item.startDim, item.endDim), item.outputShape);
  });
  test('unknown and oversized dimensions cannot produce rounded or invented shapes', () => {
    assert.equal(flattenedShape([Number.MAX_SAFE_INTEGER, 2], 0, -1), null);
    assert.equal(flattenedShape(['batch', 4], 0, -1), null);
    assert.equal(flattenedShape([2, 4], 0.5, -1), null);
    assert.equal(flattenedShape(Array(33).fill(1), 0, -1), null);
    assert.deepEqual(flattenedShape([Number.MAX_SAFE_INTEGER, 2, 0], 0, -1), [0]);
  });
});
