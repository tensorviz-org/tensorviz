import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { ExplainNodeButton } from '../../canvas/ExplainNodeButton';

it('visualizes every phase on the same button without status boxes or extra buttons', () => {
  const run = jest.fn();
  const view = render(<ExplainNodeButton action={{ phase: 'idle', run }} />);
  const original = screen.getByRole('button', { name: 'Explain this component' });
  expect(original.querySelector('.lucide-sparkles')).not.toBeNull();
  for (const [phase, label, icon] of [
    ['preparing', 'Stop explaining', '.node-explain-spinner'], ['working', 'Stop explaining', '.node-explain-spinner'],
    ['complete', 'Explain again', '.lucide-sparkles'], ['error', 'Retry AI explanation', '.lucide-circle-alert'],
    ['stopped', 'Explain again', '.lucide-square'],
  ] as const) {
    view.rerender(<ExplainNodeButton action={{ phase, run, detail: phase === 'error' ? 'Context limit exceeded.' : undefined }} />);
    const button = screen.getByRole('button', { name: label });
    expect(button).toBe(original);
    expect(button).toHaveAttribute('data-state', phase);
    expect(button.querySelector(icon)).not.toBeNull();
    expect(screen.getAllByRole('button')).toHaveLength(1);
    expect(screen.queryByText('Explanation ready.')).not.toBeInTheDocument();
  }
  expect(run).not.toHaveBeenCalled();
});

it('retains the AI sparkle on completion and marks its completed appearance without a replacement checkmark', () => {
  const run = jest.fn();
  const view = render(<ExplainNodeButton action={{ phase: 'idle', run }} />);
  const button = screen.getByRole('button', { name: 'Explain this component' });
  const sparkle = button.querySelector('.lucide-sparkles');
  view.rerender(<ExplainNodeButton action={{ phase: 'complete', run }} />);
  expect(screen.getByRole('button', { name: 'Explain again' })).toBe(button);
  expect(button.querySelector('.lucide-sparkles')).toBe(sparkle);
  expect(sparkle).toHaveClass('node-explain-complete');
  expect(button.querySelector('.lucide-check')).toBeNull();
  fireEvent.focus(button);
  expect(screen.getByRole('tooltip')).toHaveTextContent('AI explanation generated.');
  expect(run).not.toHaveBeenCalled();
});

it('keeps cancellation available on the working button and failure details on focus', () => {
  const run = jest.fn();
  const view = render(<ExplainNodeButton action={{ phase: 'working', run }} />);
  fireEvent.click(screen.getByRole('button', { name: 'Stop explaining' }));
  expect(run).toHaveBeenCalledTimes(1);
  view.rerender(<ExplainNodeButton action={{ phase: 'error', run, detail: 'Context limit exceeded.' }} />);
  fireEvent.focus(screen.getByRole('button', { name: 'Retry AI explanation' }));
  expect(screen.getByRole('tooltip')).toHaveTextContent('Context limit exceeded.');
});
