import { graphIsReadOnly, registerGraphAccess } from '../../canvas/graphAccess';

it('keeps snapshot architecture read-only independently of layout navigation and other editors', () => {
  const viewer = {}, builder = {};
  const access = { current: true };
  registerGraphAccess(viewer, access);
  expect(graphIsReadOnly(viewer)).toBe(true);
  expect(graphIsReadOnly(builder)).toBe(false);
  access.current = false;
  expect(graphIsReadOnly(viewer)).toBe(false);
});
