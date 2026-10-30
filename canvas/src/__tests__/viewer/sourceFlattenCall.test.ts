import fixture from '../../../../tests/fixtures/contracts/source-flatten-call.json';
import { requirePortableCapture } from '../../services/portableCapture';

it.each(fixture.captures)('preserves observed flatten parameters for $inputShape', item => {
  const value = JSON.parse(JSON.stringify(item.portable));
  expect(requirePortableCapture(value)).toEqual(value);
  delete value.capture.operations[0].sourceFlattenCall;
  expect(requirePortableCapture(value)).toEqual(value);
  expect(Object.hasOwn(value.capture.operations[0], 'sourceFlattenCall')).toBe(false);
});

it.each(fixture.cases)('refuses $name in portable flatten evidence', item => {
  const value = JSON.parse(JSON.stringify(fixture.captures[0].portable));
  for (const patch of item.patches) {
    let target = value.capture;
    for (const key of patch.path.slice(0, -1)) target = target[key];
    target[patch.path[patch.path.length - 1]] = patch.value;
  }
  expect(() => requirePortableCapture(value)).toThrow();
});
