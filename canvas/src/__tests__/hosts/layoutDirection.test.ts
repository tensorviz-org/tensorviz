import { autoLayoutDirection, isLayoutChoice } from '../../ide/layoutDirection';

it('resolves Auto from the canvas shape: a wide editor is horizontal, a narrow sidebar vertical', () => {
  expect(autoLayoutDirection(1880, 1020)).toBe('horizontal');
  expect(autoLayoutDirection(380, 900)).toBe('vertical');
  expect(autoLayoutDirection(800, 800)).toBe('horizontal');
});

it('holds the current direction near square so a divider drag cannot flip the layout back and forth', () => {
  // Just past square in either direction is not enough to switch.
  expect(autoLayoutDirection(950, 1000, 'horizontal')).toBe('horizontal');
  expect(autoLayoutDirection(1100, 1000, 'vertical')).toBe('vertical');
  // A clear change of shape switches.
  expect(autoLayoutDirection(860, 1000, 'horizontal')).toBe('vertical');
  expect(autoLayoutDirection(1150, 1000, 'vertical')).toBe('horizontal');
});

it('ignores a hidden zero-size canvas', () => {
  expect(autoLayoutDirection(0, 0, 'vertical')).toBe('vertical');
  expect(autoLayoutDirection(1600, 0, 'vertical')).toBe('vertical');
  expect(autoLayoutDirection(0, 0)).toBeUndefined();
});

it('accepts only the three Layout choices', () => {
  expect(['auto', 'horizontal', 'vertical'].every(isLayoutChoice)).toBe(true);
  expect([undefined, 'LR', 'Horizontal', 1].some(isLayoutChoice)).toBe(false);
});
