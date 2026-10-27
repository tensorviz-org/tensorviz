import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import CanvasSegmentedControl from '../../components/CanvasSegmentedControl';

test('keyboard selection has one tab stop, wraps, and follows the controlled view', () => {
  const changed = jest.fn();
  function Harness() {
    const [value, setValue] = React.useState('diff');
    return <CanvasSegmentedControl label="Graph revision" value={value}
      onChange={next => { changed(next); setValue(next); }}
      options={[{ value: 'before', label: 'Before' }, { value: 'diff', label: 'Diff' }, { value: 'after', label: 'After' }]} />;
  }
  render(<Harness />);
  const before = screen.getByRole('button', { name: 'Before' });
  const diff = screen.getByRole('button', { name: 'Diff' });
  const after = screen.getByRole('button', { name: 'After' });
  expect(diff.tabIndex).toBe(0);
  expect(before.tabIndex).toBe(-1);
  expect(after.tabIndex).toBe(-1);
  fireEvent.keyDown(diff, { key: 'ArrowRight' });
  expect(after).toHaveFocus();
  expect(after).toHaveAttribute('aria-pressed', 'true');
  expect(diff.tabIndex).toBe(-1);
  fireEvent.keyDown(after, { key: 'ArrowRight' });
  expect(before).toHaveFocus();
  fireEvent.keyDown(before, { key: 'End' });
  expect(after).toHaveFocus();
  fireEvent.keyDown(after, { key: 'Home' });
  expect(before).toHaveFocus();
  fireEvent.keyDown(before, { key: 'ArrowLeft', ctrlKey: true });
  expect(changed.mock.calls.map(([value]) => value)).toEqual(['after', 'before', 'after', 'before']);
});
