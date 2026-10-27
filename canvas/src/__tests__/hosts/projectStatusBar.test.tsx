import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';

import { ProjectStatusBar, StatusBarToggle } from '../../ide/ProjectStatusBar';

const noop = () => undefined;

function toggles(overrides: Partial<Record<'rail' | 'suggestions', Partial<StatusBarToggle>>> = {}) {
  return [
    { id: 'rail', label: 'Experiments', edge: 'left' as const, open: true, onToggle: noop, ...overrides.rail },
    { id: 'suggestions', label: 'Suggestions', edge: 'right' as const, open: true, onToggle: noop, ...overrides.suggestions },
  ];
}

const props = {
  experiments: 3,
  loadingLabel: null as string | null,
  focusedLabel: 'train',
  toggles: toggles(),
};

describe('the bottom bar says what the canvas is doing', () => {
  it('names what it is loading, where the eye already is when a graph refuses to change', () => {
    render(<ProjectStatusBar {...props} loadingLabel="evaluate" />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading evaluate…');
  });

  it('announces the wait rather than only drawing it', () => {
    // The canvas keeps the previous graph while the host builds a document, so
    // without a live region there is no signal at all that anything happened.
    render(<ProjectStatusBar {...props} loadingLabel="evaluate" />);
    expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite');
  });

  it('rests on what is actually on screen', () => {
    render(<ProjectStatusBar {...props} />);
    expect(screen.getByRole('status')).toHaveTextContent('train · 3 experiments');
  });
});

describe('the bottom bar gives the screen back', () => {
  it('toggles each panel, and says which state it is in', () => {
    const onToggleRail = jest.fn();
    const onToggleSuggestions = jest.fn();
    render(
      <ProjectStatusBar
        {...props}
        toggles={toggles({
          rail: { open: false, onToggle: onToggleRail },
          suggestions: { onToggle: onToggleSuggestions },
        })}
      />,
    );
    const rail = screen.getByTestId('statusbar-toggle-rail');
    // aria-pressed, not a class: a toggle whose state only exists in CSS is a
    // button that reads as "Experiments" and never says whether they are showing.
    expect(rail).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByTestId('statusbar-toggle-suggestions')).toHaveAttribute('aria-pressed', 'true');

    fireEvent.click(rail);
    fireEvent.click(screen.getByTestId('statusbar-toggle-suggestions'));
    expect(onToggleRail).toHaveBeenCalledTimes(1);
    expect(onToggleSuggestions).toHaveBeenCalledTimes(1);
  });

  it('keeps the count on the hidden panel, so hiding it does not hide what is in it', () => {
    render(<ProjectStatusBar {...props} toggles={toggles({ suggestions: { open: false, count: 4 } })} />);
    expect(screen.getByTestId('statusbar-toggle-suggestions')).toHaveTextContent('4');
  });

  it('shows no count when there is nothing waiting', () => {
    render(<ProjectStatusBar {...props} toggles={toggles({ suggestions: { count: 0 } })} />);
    expect(screen.getByTestId('statusbar-toggle-suggestions')).not.toHaveTextContent('0');
  });
});

describe('the bottom bar serves both surfaces', () => {
  it('puts a toggle on the side its panel is docked to', () => {
    // Zed's rule, and the only one that needs no learning: the button for the
    // thing on the left is on the left. The status text sits between the groups.
    const { container } = render(
      <ProjectStatusBar
        {...props}
        toggles={[
          { id: 'suggestions', label: 'Suggestions', edge: 'right', open: true, onToggle: noop },
          { id: 'rail', label: 'Experiments', edge: 'left', open: true, onToggle: noop },
        ]}
      />,
    );
    const order = Array.from(container.querySelectorAll('[data-testid^="statusbar-toggle-"], [role="status"]'))
      .map((el) => el.getAttribute('data-testid') ?? 'status');
    expect(order).toEqual(['statusbar-toggle-rail', 'status', 'statusbar-toggle-suggestions']);
  });

  it('draws whatever panels it is given, in order', () => {
    // The panel inside the editor has a rail and a diagnostics drawer; the browser
    // page has a rail and a suggestions column. One bar, two sets of panels — the
    // component must not know which surface it is on.
    render(
      <ProjectStatusBar
        {...props}
        toggles={[
          { id: 'rail', label: 'Experiments', edge: 'left', open: true, onToggle: noop },
          { id: 'problems', label: 'Problems', edge: 'bottom', open: false, count: 5, onToggle: noop },
        ]}
      />,
    );
    expect(screen.getByTestId('statusbar-toggle-rail')).toBeInTheDocument();
    const problems = screen.getByTestId('statusbar-toggle-problems');
    expect(problems).toHaveAttribute('aria-pressed', 'false');
    expect(problems).toHaveTextContent('5');
    expect(screen.queryByTestId('statusbar-toggle-suggestions')).toBeNull();
  });
});
