import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { NodeActionToolbar, nodeToolbarPosition } from '../../canvas/NodeActionToolbar';

const props = { position: { left: 80, top: 60 }, onInspect: jest.fn(), onClear: jest.fn(), onTrace: jest.fn(), tracing: false };

it('offers labelled icon actions directly without repeating node identity or shapes', () => {
  const explain = jest.fn(), discuss = jest.fn(), capture = jest.fn();
  render(<NodeActionToolbar {...props} onExplain={explain} onDiscuss={discuss} onCapture={capture} />);
  const toolbar = screen.getByRole('toolbar', { name: 'Node actions' });
  expect(toolbar.textContent).toBe('');
  fireEvent.click(screen.getByRole('button', { name: 'Explain this component' }));
  expect(explain).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Discuss in chat' }));
  expect(discuss).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Captured calls & tensors' }));
  expect(capture).toHaveBeenCalledTimes(1);
  expect(props.onInspect).not.toHaveBeenCalled();
});

it('explains icons on keyboard focus without invoking any action', () => {
  render(<NodeActionToolbar {...props} />);
  fireEvent.focus(screen.getByRole('button', { name: 'Inspect' }));
  expect(screen.getByRole('tooltip')).toHaveTextContent('Inspect');
  expect(props.onInspect).not.toHaveBeenCalled();
});

it('keeps secondary actions in a labelled overflow instead of an ambiguous cross', () => {
  const edit = jest.fn();
  render(<NodeActionToolbar {...props} onEdit={edit} />);
  expect(screen.queryByRole('button', { name: 'Clear selection' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'More node actions' }));
  fireEvent.click(screen.getByRole('button', { name: 'Edit source' }));
  expect(edit).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'More node actions' }));
  fireEvent.click(screen.getByRole('button', { name: 'Clear selection' }));
  expect(props.onClear).toHaveBeenCalledTimes(1);
});

it('reveals disabled-action reasons on focus and never starts generation', () => {
  const explain = jest.fn();
  render(<NodeActionToolbar {...props} onExplain={explain} explainDisabled="Connect an agent first" />);
  const button = screen.getByRole('button', { name: 'Explain this component' });
  expect(button).toBeDisabled();
  fireEvent.focus(button.parentElement!);
  expect(screen.getByRole('tooltip')).toHaveTextContent('Connect an agent first');
  fireEvent.click(button);
  expect(explain).not.toHaveBeenCalled();
});

it('closes the trace chooser before showing overflow and supports keyboard navigation', () => {
  const trace = jest.fn();
  render(<NodeActionToolbar {...props} tracing onTrace={trace} />);
  fireEvent.keyDown(screen.getByRole('button', { name: 'Inspect' }), { key: 'ArrowRight' });
  expect(screen.getByRole('button', { name: 'Trace' })).toHaveFocus();
  fireEvent.click(screen.getByRole('button', { name: 'More node actions' }));
  expect(trace).toHaveBeenCalledTimes(1);
});

it('keeps the toolbar inside a narrow canvas and flips below a node at the top edge', () => {
  expect(nodeToolbarPosition({ x: 100, y: 120 }, { x: 300, y: 260 }, { w: 900, h: 600 }, 200))
    .toEqual({ left: 100, top: 72 });
  expect(nodeToolbarPosition({ x: 2, y: 3 }, { x: 70, y: 100 }, { w: 320, h: 500 }, 232))
    .toEqual({ left: 8, top: 108 });
  expect(nodeToolbarPosition({ x: 400, y: 200 }, { x: 600, y: 400 }, { w: 320, h: 500 }, 232).left).toBe(80);
});
