import fixture from '../../../../tests/fixtures/contracts/source-shape-call.json';
import { requirePortableCapture } from '../../services/portableCapture';

it('preserves actual portable source-to-shape observations without adding absent evidence', () => {
  const value = JSON.parse(JSON.stringify(fixture.portable));
  expect(requirePortableCapture(value)).toEqual(value);
  delete value.capture.operations[0].sourceShapeCall;
  expect(requirePortableCapture(value)).toEqual(value);
  expect(Object.hasOwn(value.capture.operations[0], 'sourceShapeCall')).toBe(false);
});
it.each(fixture.cases)('refuses $name in offline source-to-shape evidence', item => {
  const value = JSON.parse(JSON.stringify(fixture.portable));
  for (const patch of item.patches) {
    let target = value.capture;
    for (const key of patch.path.slice(0, -1)) target = target[key];
    target[patch.path[patch.path.length - 1]] = patch.value;
  }
  expect(() => requirePortableCapture(value)).toThrow();
});
it('does not manufacture tensor-read provenance in older dimension mappings', () => {
  const value = JSON.parse(JSON.stringify(fixture.portable));
  for (const dimension of value.capture.operations[0].sourceShapeCall.dimensions) delete dimension.tensorDimension;
  expect(requirePortableCapture(value)).toEqual(value);
  expect(value.capture.operations[0].sourceShapeCall.dimensions.every((dimension: any) => !Object.hasOwn(dimension, 'tensorDimension'))).toBe(true);
});
