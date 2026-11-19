import React from 'react';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import LocalCanvasNavigation from '../../ide/LocalCanvasNavigation';
import LocalBrowserValidate from '../../ide/LocalBrowserValidate';
import { LocalBrowserContext, LocalBrowserContextValue } from '../../ide/localBrowserContext';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE } from '../../ide/localBrowserState';
import { configureCanvasTheme, setCanvasHostTheme } from '../../styles/canvasTheme';
import { configureCodeFocusOn, isCodeFocusOn } from '../../ide/codeFocusOn';
import { DEFAULT_CODE_FOCUS } from '../../ide/protocol.generated';

beforeEach(() => { localStorage.clear(); configureCanvasTheme('host'); configureCodeFocusOn('host'); });

function mount(selected = false, fullscreen = false, compact = false, running = false,
  extra: Pick<React.ComponentProps<typeof LocalCanvasNavigation>, 'layout'> & { reviewing?: boolean; sourceAction?: 'copy'; codeFocus?: boolean;
    feedback?: LocalBrowserContextValue['feedback'] } = {}) {
  const actions = { find: jest.fn(), reveal: jest.fn(), run: jest.fn(), cancel: jest.fn() };
  const context: LocalBrowserContextValue = {
    page: EMPTY_LOCAL_BROWSER_PAGE_STATE,
    reviewing: extra.reviewing,
    sourceAction: extra.sourceAction,
    feedback: extra.feedback,
    codeFocus: extra.codeFocus ? { options: DEFAULT_CODE_FOCUS, parameter: false, detail: '', change: jest.fn() } : undefined,
    status: { phase: 'connected', hostLink: 'connected', error: null },
    validation: { running, disabledReason: null, statusLabel: 'Failed',
      lastResult: { status: 'failed', detail: 'Expected 3 channels; received 4.' }, run: actions.run, cancel: actions.cancel },
  };
  function Harness() {
    const [header, setHeader] = React.useState<HTMLDivElement | null>(null);
    const [footer, setFooter] = React.useState<HTMLDivElement | null>(null);
    return <><div ref={setHeader} data-testid="header" /><div ref={setFooter} data-testid="footer" />
      <LocalBrowserContext.Provider value={{ ...context, reviewing: extra.reviewing, navigation: { header, footer, compact, inputsOpen: false, changeInputs: jest.fn() } }}>
        <LocalCanvasNavigation onFind={actions.find} onReveal={actions.reveal} hasSelection={selected} nodeCount={12} more={null} fullscreen={fullscreen} detailLevel="layers" layout={extra.layout} />
      </LocalBrowserContext.Provider></>;
  }
  render(<Harness />);
  return actions;
}

it.each([false, true])('turns Code focus on in one click, first in the header, when compact is %s', compact => {
  mount(false, false, compact, false, { codeFocus: true });
  const toggle = within(screen.getByTestId('header')).getByRole('button', { name: 'Code focus', exact: true });
  expect(screen.getByRole('group', { name: 'Graph view' }).firstElementChild).toContainElement(toggle);
  expect(screen.queryByRole('group', { name: 'Canvas mode' })).toBeNull();
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute('aria-pressed', 'true');
  expect(isCodeFocusOn()).toBe(true);
});

it('offers no Code focus where nothing can open in an editor or while reviewing changes', () => {
  // The copy-only page has no Code focus client; graph review hides it.
  mount(false, false, false, false, { sourceAction: 'copy' });
  expect(screen.queryByRole('button', { name: 'Code focus', exact: true })).toBeNull();
  cleanup();
  mount(false, false, false, false, { reviewing: true, codeFocus: true });
  expect(screen.queryByRole('button', { name: 'Code focus', exact: true })).toBeNull();
});

it('labels environment preparation accurately and leaves cancellation in the visible recovery panel', () => {
  const context: LocalBrowserContextValue = {
    page: { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, opening: { id: 'opening', phase: 'resolving', hasCapture: false,
      target: { workspaceFolder: '/project', file: 'model.py', cls: 'Net', sourceDigest: 'source', documentVersion: 1, backendRevision: 1 },
      detail: 'Preparing the reviewed library repair.', environment: { reason: 'missing_libraries', candidates: [], detail: 'Missing SciPy.' } } },
    status: { phase: 'connected', hostLink: 'connected', error: null },
    navigation: { header: null, footer: null, compact: false, inputsOpen: true, changeInputs: jest.fn() },
    validation: { running: true, disabledReason: null, statusLabel: 'Not run', run: jest.fn(), cancel: jest.fn() },
  };
  render(<LocalBrowserContext.Provider value={context}><LocalBrowserValidate controls="run" /></LocalBrowserContext.Provider>);
  const button = screen.getByRole('button', { name: 'Preparing environment…' });
  expect(button).toBeDisabled();
  fireEvent.focus(button);
  expect(screen.getByRole('tooltip')).toHaveTextContent('Environment setup is running');
  expect(screen.queryByRole('button', { name: 'Capturing…' })).toBeNull();
  expect(screen.queryByRole('button', { name: 'Cancel' })).toBeNull();
});

it.each([false, true])('renders one set of controls in the header and footer when compact is %s', compact => {
  mount(true, false, compact);
  const header = screen.getByTestId('header'), footer = screen.getByTestId('footer');
  expect(header).toContainElement(screen.getByRole('switch', { name: 'Dark canvas appearance' }));
  // Global getByRole also rejects duplicate controls outside the intended slot.
  for (const name of ['Layers', 'All operations', 'Run model']) {
    expect(header).toContainElement(screen.getByRole('button', { name }));
  }
  for (const name of ['Latest validation: Failed', 'Reveal in editor', 'Zoom out',
    'Fit graph to view', 'Zoom options', 'Zoom in', 'Canvas guide and shortcuts']) {
    expect(footer).toContainElement(screen.getByRole('button', { name }));
  }
  expect(footer).toContainElement(screen.getByText('12 graph nodes'));
  // A compact header drops labels, never Find: there is no toggle to reveal it.
  expect(header).toContainElement(screen.getByRole('button', { name: 'Find a layer' }));
});

it('keeps the direct appearance switch visible in graph review', () => {
  mount(false, false, true, false, { reviewing: true });
  const control = screen.getByRole('switch', { name: 'Dark canvas appearance' });
  expect(screen.getByTestId('header')).toContainElement(control);
  fireEvent.click(control);
  expect(control).toHaveAttribute('aria-checked', 'false');
  expect(screen.queryByRole('button', { name: 'Run model' })).toBeNull();
});

it.each([[false, true], [false, false], [true, false]])('switches appearance directly in fullscreen=%s, compact=%s and preserves the choice', (fullscreen, compact) => {
  const actions = mount(false, fullscreen, compact);
  const control = screen.getByRole('switch', { name: 'Dark canvas appearance' });
  expect(control).toHaveAttribute('aria-checked', 'true');
  fireEvent.focus(control);
  expect(screen.getByRole('tooltip')).toHaveTextContent('Switch the Canvas to Light · Paper.');
  fireEvent.click(control);
  expect(control).toHaveAttribute('aria-checked', 'false');
  fireEvent.focus(control);
  expect(screen.getByRole('tooltip')).toHaveTextContent('Switch the Canvas to Dark · Ink.');
  expect(document.documentElement.dataset.tvTheme).toBe('light');
  expect(localStorage.getItem('tensorviz.canvas.theme.host')).toBe('light');
  expect(screen.queryByRole('menu')).toBeNull();
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(actions.run).not.toHaveBeenCalled();
  act(() => { configureCanvasTheme('host'); setCanvasHostTheme('dark'); });
  expect(control).toHaveAttribute('aria-checked', 'false');
  fireEvent.click(control);
  expect(control).toHaveAttribute('aria-checked', 'true');
  expect(document.documentElement.dataset.tvTheme).toBe('dark');
  expect(localStorage.getItem('tensorviz.canvas.theme.host')).toBe('dark');
});

it('routes header actions to the existing handlers and keeps failure details in the footer', () => {
  const actions = mount();
  const header = within(screen.getByTestId('header')), footer = within(screen.getByTestId('footer'));
  fireEvent.click(header.getByRole('button', { name: 'Find a layer' }));
  fireEvent.click(header.getByRole('button', { name: 'Run model' }));
  expect(actions.find).toHaveBeenCalledTimes(1);
  expect(actions.run).toHaveBeenCalledTimes(1);
  expect(header.queryByText('Failed')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Reveal in editor' })).toBeNull();
  fireEvent.click(footer.getByRole('button', { name: 'Latest validation: Failed' }));
  expect(screen.getByRole('dialog', { name: 'Latest validation result' })).toHaveTextContent('Expected 3 channels; received 4.');
});

it('keeps one labelled Run model action visible in the compact toolbar', () => {
  const actions = mount(false, false, true);
  const button = within(screen.getByTestId('header')).getByRole('button', { name: 'Run model' });
  expect(button).toBeEnabled(); expect(button).toHaveTextContent('Run model');
  fireEvent.focus(button);
  const help = screen.getByRole('tooltip');
  expect(help).toHaveTextContent('Choose the device and review inputs before running.');
  expect(button).toHaveAttribute('aria-describedby', help.id);
  fireEvent.click(button);
  expect(actions.run).toHaveBeenCalledTimes(1);
});

it('says Run model reruns the saved setup when it reopens a model without a current capture', () => {
  const run = jest.fn();
  render(<LocalBrowserContext.Provider value={{ page: EMPTY_LOCAL_BROWSER_PAGE_STATE, status: { phase: 'connected', hostLink: 'connected', error: null },
    validation: { running: false, disabledReason: null, statusLabel: 'Cancelled', lastResult: null, reopens: true, run, cancel: jest.fn() } }}>
    <LocalBrowserValidate controls="run" /></LocalBrowserContext.Provider>);
  const button = screen.getByRole('button', { name: 'Run model' });
  fireEvent.focus(button);
  expect(screen.getByRole('tooltip')).toHaveTextContent('Run the model again with its saved setup.');
  expect(screen.getByRole('tooltip')).not.toHaveTextContent('Choose the device');
  fireEvent.click(button);
  expect(run).toHaveBeenCalledTimes(1);
});

it.each([false, true])('shows one busy Run action when compact is %s and offers one Cancel', compact => {
  const actions = mount(false, false, compact, true);
  const button = screen.getByRole('button', { name: 'Running…' });
  expect(button).toBeDisabled(); expect(button).toHaveAttribute('aria-busy', 'true');
  expect(button).toHaveTextContent('Running…');
  // A text label overflowed the icon-sized button onto the next control.
  expect(screen.getByRole('button', { name: 'Cancel', exact: true })).toHaveTextContent(/^$/);
  fireEvent.click(screen.getByRole('button', { name: 'Cancel', exact: true }));
  expect(actions.cancel).toHaveBeenCalledTimes(1);
  expect(actions.run).not.toHaveBeenCalled();
});

it('keeps source navigation available when selected and dispatches only camera commands from zoom controls', () => {
  const actions = mount(true);
  fireEvent.click(screen.getByRole('button', { name: 'Reveal in editor' }));
  expect(actions.reveal).toHaveBeenCalledTimes(1);
  const commands: string[] = [];
  const listener = (event: Event) => commands.push((event as CustomEvent).detail.command);
  window.addEventListener('builder-command', listener);
  try {
    const navigation = within(screen.getByRole('group', { name: 'Zoom and fit' }));
    fireEvent.click(navigation.getByRole('button', { name: 'Zoom out' }));
    fireEvent.click(navigation.getByRole('button', { name: 'Fit graph to view' }));
    fireEvent.click(navigation.getByRole('button', { name: 'Zoom in' }));
    expect(commands).toEqual(['zoom-out', 'fit-view', 'zoom-in']);
    expect(actions.run).not.toHaveBeenCalled();
  } finally { window.removeEventListener('builder-command', listener); }
});

it('puts Send feedback in the footer just before the Canvas guide, also in full screen, and not while reviewing', () => {
  const client = { state: { phase: 'idle' }, subscribe: () => () => {}, send: jest.fn(), reset: jest.fn(), dispose: jest.fn() };
  const feedback = { client, graph: { model: 'Net', nodes: 12 } } as LocalBrowserContextValue['feedback'];
  mount(false, false, true, false, { feedback });
  const button = screen.getByRole('button', { name: 'Send feedback' });
  expect(screen.getByTestId('footer')).toContainElement(button);
  expect(button.closest('.canvas-tooltip-anchor')!.nextElementSibling)
    .toContainElement(screen.getByRole('button', { name: 'Canvas guide and shortcuts' }));
  cleanup();
  mount(false, true, true, false, { feedback });
  expect(screen.getByRole('button', { name: 'Send feedback' }).closest('.local-canvas-fullscreen-footer')).not.toBeNull();
  cleanup();
  mount(false, false, false, false, { reviewing: true, feedback });
  expect(screen.queryByRole('button', { name: 'Send feedback' })).toBeNull();
});

it('keeps navigation reachable inside the fullscreen canvas instead of behind it', () => {
  const actions = mount(true, true, true);
  expect(screen.getByTestId('header')).toBeEmptyDOMElement();
  expect(screen.getByTestId('footer')).toBeEmptyDOMElement();
  expect(screen.getByRole('button', { name: 'Run model' }).closest('.local-canvas-fullscreen-header')).not.toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Run model' }));
  expect(actions.run).toHaveBeenCalledTimes(1);
  expect(screen.getByRole('group', { name: 'Zoom and fit' }).closest('.local-canvas-fullscreen-footer')).not.toBeNull();
});

it.each([false, true])('offers both detail levels without another capture when compact is %s', compact => {
  const actions = mount(false, false, compact);
  const toggle = within(within(screen.getByTestId('header')).getByRole('group', { name: 'Graph detail' }));
  expect(toggle.getByRole('button', { name: 'Layers' })).toHaveAttribute('aria-pressed', 'true');
  const listener = jest.fn();
  window.addEventListener('builder-command', listener);
  try {
    fireEvent.click(toggle.getByRole('button', { name: 'All operations' }));
    expect(listener.mock.calls[0][0].detail).toEqual({ command: 'set-graph-detail', payload: { level: 'operations' } });
    fireEvent.click(toggle.getByRole('button', { name: 'Layers' }));
    expect(listener.mock.calls[1][0].detail).toEqual({ command: 'set-graph-detail', payload: { level: 'layers' } });
    expect(actions.run).not.toHaveBeenCalled();
  } finally { window.removeEventListener('builder-command', listener); }
});

it('shows the connection guide on demand and restores keyboard focus on Escape', () => {
  mount();
  expect(screen.queryByRole('dialog', { name: 'Canvas guide' })).toBeNull();
  fireEvent.keyDown(document.body, { key: '?' });
  const guide = screen.getByRole('dialog', { name: 'Canvas guide' });
  expect(guide).toHaveTextContent('Residual / skip');
  expect(guide).toHaveTextContent('Mixed ×N');
  fireEvent.keyDown(guide, { key: 'Escape' });
  expect(screen.queryByRole('dialog', { name: 'Canvas guide' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Canvas guide and shortcuts' })).toHaveFocus();
});

it('does not promise that opening always runs the model, since the execution cache can reuse an identical result', () => {
  mount();
  fireEvent.keyDown(document.body, { key: '?' });
  expect(screen.getByRole('dialog', { name: 'Canvas guide' }))
    .toHaveTextContent('Opening a model runs it once with its saved inputs on the saved device, or reuses an identical earlier result.');
});

it('uses arrow keys for detail selection without invoking capture', () => {
  const actions = mount(false, false, true);
  const listener = jest.fn();
  window.addEventListener('builder-command', listener);
  try {
    fireEvent.keyDown(screen.getByRole('button', { name: 'Layers' }), { key: 'ArrowRight' });
    expect(screen.getByRole('button', { name: 'All operations' })).toHaveFocus();
    expect(listener.mock.calls[0][0].detail).toEqual({ command: 'set-graph-detail', payload: { level: 'operations' } });
    expect(actions.run).not.toHaveBeenCalled();
  } finally { window.removeEventListener('builder-command', listener); }
});

it('dismisses the previous popover when help opens from the keyboard', () => {
  mount();
  fireEvent.click(screen.getByRole('button', { name: 'Zoom options' }));
  expect(screen.getByRole('dialog', { name: 'Zoom options' })).toBeVisible();
  fireEvent.keyDown(document.activeElement!, { key: '?' });
  expect(screen.getByRole('dialog', { name: 'Canvas guide' })).toHaveFocus();
  expect(screen.queryByRole('dialog', { name: 'Zoom options' })).toBeNull();
});

it('routes readable zoom through the same command from the footer, guide and shortcut, never from text fields', () => {
  const actions = mount();
  const listener = jest.fn();
  window.addEventListener('builder-command', listener);
  try {
    fireEvent.click(screen.getByRole('button', { name: 'Zoom options' }));
    fireEvent.click(screen.getByRole('button', { name: /Readable detail/ }));
    expect(screen.queryByRole('dialog', { name: 'Zoom options' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Canvas guide and shortcuts' }));
    fireEvent.click(screen.getByRole('button', { name: /Zoom to readable detail/ }));
    fireEvent.keyDown(document.body, { key: '@', code: 'Digit2', shiftKey: true });
    const field = document.createElement('input');
    document.body.appendChild(field);
    fireEvent.keyDown(field, { key: '@', code: 'Digit2', shiftKey: true });
    fireEvent.keyDown(field, { key: '?' });
    field.remove();
    expect(listener.mock.calls.map(([event]) => event.detail.command)).toEqual(['zoom-readable', 'zoom-readable', 'zoom-readable']);
    expect(screen.queryByRole('dialog', { name: 'Canvas guide' })).toBeNull();
    expect(actions.run).not.toHaveBeenCalled();
  } finally { window.removeEventListener('builder-command', listener); }
});

it('offers Auto, Horizontal and Vertical from the footer even with a compact toolbar', () => {
  mount(false, false, true, false, { layout: { choice: 'auto', direction: 'horizontal' } });
  const trigger = screen.getByRole('button', { name: 'Layout direction: Auto (horizontal)' });
  expect(screen.getByTestId('footer')).toContainElement(trigger);
  expect(screen.getByTestId('header')).not.toContainElement(trigger);
  const listener = jest.fn();
  window.addEventListener('builder-command', listener);
  try {
    fireEvent.click(trigger);
    const menu = screen.getByRole('menu', { name: 'Layout' });
    const options = within(menu).getAllByRole('menuitemradio');
    expect(options.map(option => [option.textContent, option.getAttribute('aria-checked')])).toEqual([
      ['AutoHorizontal', 'true'], ['Horizontal', 'false'], ['Vertical', 'false']]);
    fireEvent.click(within(menu).getByRole('menuitemradio', { name: 'Vertical' }));
    expect(listener.mock.calls.map(([event]) => event.detail)).toEqual([
      { command: 'set-layout-direction', payload: { choice: 'vertical' } }]);
    expect(screen.queryByRole('menu', { name: 'Layout' })).toBeNull();
    expect(trigger).toHaveFocus();
  } finally { window.removeEventListener('builder-command', listener); }
});

it('names an explicit layout choice and hides the control where layout cannot change', () => {
  mount(false, false, false, false, { layout: { choice: 'vertical', direction: 'vertical' } });
  fireEvent.click(screen.getByRole('button', { name: 'Layout direction: Vertical' }));
  expect(screen.getByRole('menuitemradio', { name: 'Vertical' })).toHaveAttribute('aria-checked', 'true');
  expect(screen.getByRole('menuitemradio', { name: 'Auto', exact: true })).toHaveTextContent(/^Auto$/);
  cleanup();
  mount(false, false, false, false, { layout: { choice: 'auto', direction: 'vertical' }, reviewing: true });
  expect(screen.queryByRole('button', { name: /Layout direction/ })).toBeNull();
  cleanup();
  mount();
  expect(screen.queryByRole('button', { name: /Layout direction/ })).toBeNull();
});

it('reopens a pending review through Run model, then disables cancelling controls', () => {
  const opening = { id: 'opening', phase: 'awaiting_execution' as const, hasCapture: true, detail: 'Review',
    target: { workspaceFolder: '/project', file: 'model.py', cls: 'Net', sourceDigest: 'a'.repeat(64), documentVersion: 1, backendRevision: 1 },
    execution: { id: 'review', kind: 'capture' as const, phase: 'review' as const, device: 'cpu', configuredDevice: 'cpu', detail: 'Review',
      interpreter: '/python', inputs: '{}', mode: 'eval', gradients: false, stages: [{ label: 'Capture', calls: 1 }] } };
  const context: LocalBrowserContextValue = { page: { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, opening },
    status: { phase: 'connected', hostLink: 'connected', error: null },
    validation: { running: true, disabledReason: null, statusLabel: 'Not run',
      run: jest.fn(), cancel: jest.fn() } };
  const view = render(<LocalBrowserContext.Provider value={context}><LocalBrowserValidate controls="run" /></LocalBrowserContext.Provider>);
  expect(screen.getByRole('button', { name: 'Run model' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Run model' }));
  expect(context.validation!.run).toHaveBeenCalledTimes(1);
  expect(screen.getByRole('button', { name: 'Run model' })).toHaveAttribute('aria-busy', 'false');
  view.rerender(<LocalBrowserContext.Provider value={{ ...context, page: { ...context.page,
    opening: { ...opening, execution: { ...opening.execution, phase: 'cancelling' } } } }}><LocalBrowserValidate controls="run" /></LocalBrowserContext.Provider>);
  expect(screen.getByRole('button', { name: 'Cancelling…' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
});
