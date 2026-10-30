import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import ExecutionPanel from '../../ide/ExecutionPanel';
import type { ExecutionReviewState } from '../../ide/protocol.generated';
beforeAll(() => { global.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} }; });
function Panel(props: Omit<React.ComponentProps<typeof ExecutionPanel>, 'expanded' | 'onExpandedChange'>) {
  const [expanded, setExpanded] = React.useState(false);
  return <ExecutionPanel {...props} expanded={expanded} onExpandedChange={setExpanded} />;
}
const open = () => fireEvent.click(screen.getByRole('button', { name: 'Execution details' }));
const state: ExecutionReviewState = { id: 'review1', kind: 'capture', phase: 'review', configuredDevice: 'cuda:0', device: 'cuda:0',
  detail: 'Not started', interpreter: '/env/bin/python', inputs: '{x: [1, 8]}', mode: 'eval', gradients: false,
  stages: [{ label: 'Capture', calls: 1 }], resources: { sampledAt: Date.now(), host: 'gpu-box', remote: true,
    cpu: 'CPU', cores: 16, cpuPercent: 80, memoryFree: 8 * 1024 ** 3, memoryTotal: 32 * 1024 ** 3,
    devices: [{ id: 'cuda:0', name: 'GPU', utilization: 100, memoryFree: 0, memoryTotal: 24 * 1024 ** 3 }], detail: 'Observed only' } };
it('shows the remote execution host and permits Run on a fully busy GPU', () => {
  const action = jest.fn(); render(<Panel state={state} connected onAction={action} />); open();
  expect(screen.getByText('Remote SSH: gpu-box')).toBeInTheDocument();
  expect(screen.getByRole('meter', { name: 'CPU usage' })).toHaveAttribute('value', '80');
  expect(screen.getByText('8.0 GiB')).toBeInTheDocument();
  expect(screen.queryByRole('meter', { name: 'GPU usage' })).toBeNull();
  expect(screen.queryByText('GPU memory available')).toBeNull();
  const run = screen.getByRole('button', { name: 'Run', exact: true }); expect(run).toBeEnabled();
  expect(action).not.toHaveBeenCalled(); fireEvent.click(run);
  expect(action).toHaveBeenCalledWith({ kind: 'execution', reviewId: 'review1', action: 'run', device: 'cuda:0' });
});
it('Run closes the details and returns focus to the target strip', () => {
  const action = jest.fn(); render(<Panel state={state} connected onAction={action} />); open();
  const trigger = screen.getByRole('button', { name: 'Execution details' });
  fireEvent.click(screen.getByRole('button', { name: 'Run', exact: true }));
  expect(action).toHaveBeenCalledWith({ kind: 'execution', reviewId: 'review1', action: 'run', device: 'cuda:0' });
  expect(trigger).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByRole('button', { name: 'Run', exact: true })).toBeNull();
  expect(trigger).toHaveFocus();
});
it('Run again keeps focus on the target strip when the details close', () => {
  const onRepeat = jest.fn();
  render(<Panel state={{ ...state, phase: 'failed', detail: 'Stopped.' }} connected onAction={jest.fn()} onRepeat={onRepeat} />); open();
  fireEvent.click(screen.getByRole('button', { name: 'Run again' }));
  expect(onRepeat).toHaveBeenCalledTimes(1);
  expect(screen.getByRole('button', { name: 'Execution details' })).toHaveFocus();
});
it('device selection is distinct from execution and environment details start collapsed', () => {
  const action = jest.fn(); render(<Panel state={state} connected onAction={action} />); open();
  fireEvent.change(screen.getByLabelText('Device for this run'), { target: { value: 'cpu' } });
  expect(action).toHaveBeenCalledTimes(1); expect(action.mock.calls[0][0].action).toBe('select');
  expect(screen.getByText('Run settings').parentElement).not.toHaveAttribute('open');
});
it('keeps the target visible when collapsed and cancellation visible until stopping finishes', () => {
  const action = jest.fn(); const view = render(<Panel state={{ ...state, phase: 'cancelling' }} connected onAction={action} />); open();
  expect(screen.getByRole('progressbar')).toHaveAccessibleName('Stopping execution');
  expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Collapse execution details' }));
  expect(screen.queryByRole('progressbar')).toBeNull(); expect(screen.getByText('GPU · cuda:0')).toBeInTheDocument();
  view.rerender(<Panel state={{ ...state, phase: 'cancelled', detail: 'Execution stopped.' }} connected onAction={action} />);
  expect(screen.getByRole('button', { name: 'Execution details' })).not.toHaveTextContent('Cancelled');
  open();
  expect(screen.getByRole('status')).toHaveTextContent('Execution stopped.');
  expect(screen.queryByRole('progressbar')).toBeNull();
});
it('missing metrics and disconnected transport are different states', () => {
  const view = render(<Panel state={{ ...state, resources: undefined }} connected onAction={jest.fn()} />); open();
  expect(screen.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  view.rerender(<Panel state={state} connected={false} onAction={jest.fn()} />);
  expect(screen.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
});
it('starts closed and stays closed across new reviews and resource updates until explicitly reopened', () => {
  const action = jest.fn(); const view = render(<Panel state={state} connected onAction={action} />);
  const trigger = screen.getByRole('button', { name: 'Execution details' });
  expect(trigger).toHaveTextContent('GPU · cuda:0');
  expect(trigger).toHaveTextContent('SSH · gpu-box');
  expect(trigger).not.toHaveTextContent('Not run');
  expect(trigger).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByRole('button', { name: 'Run', exact: true })).toBeNull();
  open(); fireEvent.click(screen.getByRole('button', { name: 'Collapse execution details' }));
  expect(trigger).toHaveFocus();
  view.rerender(<Panel state={{ ...state, id: 'review2', resources: undefined }} connected={false} onAction={action} />);
  view.rerender(<Panel state={{ ...state, id: 'review2', resources: { ...state.resources!, cpuPercent: 12 } }} connected onAction={action} />);
  expect(trigger).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByRole('button', { name: 'Run', exact: true })).toBeNull();
  expect(action).not.toHaveBeenCalled();
  open(); expect(screen.getByRole('button', { name: 'Run', exact: true })).toBeVisible();
  fireEvent.keyDown(trigger, { key: 'Escape' });
  expect(trigger).toHaveAttribute('aria-expanded', 'false');
  expect(action).not.toHaveBeenCalled();
});
it('shows input shapes and values in a table with input data available on demand', () => {
  const inputs = JSON.stringify({ binding: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32', generator: { kind: 'normal', seed: 42 } }], kwargs: { training: false } } });
  render(<Panel state={{ ...state, inputs }} connected onAction={jest.fn()} />); open();
  expect(screen.getByRole('table', { name: 'Inputs' })).toBeVisible();
  const details = screen.getByText('Run settings').parentElement!;
  fireEvent.click(screen.getByText('Run settings'));
  expect(details).toHaveAttribute('open');
  const table = within(screen.getByRole('table', { name: 'Inputs' }));
  expect(table.getByRole('row', { name: 'Argument 1 1 × 8 · float32' })).toBeVisible();
  expect(table.getByRole('row', { name: 'training false' })).toBeVisible();
  expect(screen.getByText('Input recipe').parentElement).not.toHaveAttribute('open');
  fireEvent.click(screen.getByText('Input recipe'));
  expect(screen.getByText(inputs)).toBeVisible();
  expect(screen.queryByText('Observed resources')).toBeNull();
  expect(screen.queryByText('Inputs and environment')).toBeNull();
});
it('identifies the saved profile and navigates each setting or shape without running', () => {
  const action = jest.fn();
  const inputs = JSON.stringify({ binding: { args: [{ kind: 'tensor', shape: [1, 8], dtype: 'float32' }], kwargs: { training: false } } });
  render(<Panel state={{ ...state, inputs, profileName: 'run.v2', resources: { ...state.resources!, remote: false } }} connected onAction={action}
    evidence={[{ field: 'inputs.x', kind: 'source', detail: 'Linear input constraint', file: 'model.py', line: 12 }]} />); open();
  expect(screen.getByText('Local Python worker: gpu-box')).toBeVisible();
  expect(screen.queryByText(/This computer/)).toBeNull();
  expect(screen.getByText('profiles.run.v2')).toBeVisible();
  expect(screen.getByText('How inputs were inferred').parentElement).not.toHaveAttribute('open');
  fireEvent.click(screen.getByText('How inputs were inferred'));
  expect(screen.getByText('Linear input constraint')).toBeVisible();
  expect(screen.getByText('model.py:12')).toBeVisible();
  for (const [name, field] of [['mode', ['mode']], ['gradients', ['grad_enabled']], ['Argument 1', ['inputs', 'args', 0, 'shape']], ['training', ['inputs', 'kwargs', 'training']]] as const) {
    fireEvent.click(screen.getByRole('button', { name: `Edit ${name} in tensorviz.toml` }));
    expect(action).toHaveBeenLastCalledWith({ kind: 'edit_execution', reviewId: 'review1', field });
  }
  expect(action.mock.calls.every(([value]) => value.kind === 'edit_execution')).toBe(true);
});
it('plans one capture call without a latency option', () => {
  render(<Panel state={state} connected onAction={jest.fn()} />); open();
  expect(screen.getByText('One call captures shapes and operations.')).toBeVisible();
  expect(screen.queryByRole('checkbox', { name: 'Measure latency' })).toBeNull();
  expect(screen.queryByRole('spinbutton')).toBeNull();
  expect(screen.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
});
it.each([
  [JSON.stringify({ factory: { file: 'example.py', callable: 'make_inputs' } }), 'Factory example.py:make_inputs()'],
  [JSON.stringify({ fixture: 'inputs.pt' }), 'Fixture inputs.pt'],
])('identifies factory or fixture inputs without inventing tensor dimensions (%s)', (inputs, row) => {
  render(<Panel state={{ ...state, inputs }} connected onAction={jest.fn()} />); open();
  fireEvent.click(screen.getByText('Run settings'));
  expect(within(screen.getByRole('table', { name: 'Inputs' })).getByRole('row', { name: row })).toBeVisible();
});
it('retains a truncated host input summary without pretending to know its shape', () => {
  const inputs = '{"binding":{"args":[{"kind":"tensor","shape":[';
  render(<Panel state={{ ...state, inputs }} connected onAction={jest.fn()} />); open();
  fireEvent.click(screen.getByText('Run settings'));
  expect(screen.queryByRole('table', { name: 'Inputs' })).toBeNull();
  fireEvent.click(screen.getByText('Input recipe'));
  expect(screen.getByText(inputs)).toBeVisible();
});
it('keeps the GPU device selectable without a usage or memory bar', () => {
  render(<Panel state={{ ...state, resources: { ...state.resources!, cpuPercent: null, devices: [{ ...state.resources!.devices[0], utilization: null, memoryFree: null }] } }} connected onAction={jest.fn()} />); open();
  expect(screen.getByRole('option', { name: 'GPU · cuda:0 — GPU' })).toBeInTheDocument();
  expect(screen.queryByRole('meter', { name: 'GPU usage' })).toBeNull();
  expect(screen.queryByText('GPU memory available')).toBeNull();
  expect(screen.getByText('Unknown')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
});
it.each(['succeeded', 'failed', 'cancelled'] as const)('offers a fresh review after %s without replaying the old Run approval', phase => {
  const action = jest.fn(), repeat = jest.fn();
  const view = render(<Panel state={{ ...state, phase }} connected onAction={action} onRepeat={repeat} />); open();
  fireEvent.click(screen.getByRole('button', { name: 'Run again' }));
  expect(repeat).toHaveBeenCalledTimes(1);
  expect(action).not.toHaveBeenCalled();
  view.rerender(<Panel state={{ ...state, phase }} connected onAction={action} onRepeat={repeat} repeatDisabled />);
  expect(screen.getByRole('button', { name: 'Run again' })).toBeDisabled();
  view.rerender(<Panel state={{ ...state, phase }} connected={false} onAction={action} onRepeat={repeat} />);
  expect(screen.getByRole('button', { name: 'Run again' })).toBeDisabled();
});

it('keeps scrolling and configuration keys inside the sheet without preventing native scrolling', () => {
  const wheel = jest.fn(), pointer = jest.fn(), keyboard = jest.fn(), action = jest.fn();
  render(<div onWheel={wheel} onPointerDown={pointer} onKeyDown={keyboard}><Panel state={state} connected onAction={action} /></div>); open();
  const region = screen.getByRole('region', { name: 'Run configuration' });
  expect(region).toHaveAttribute('tabindex', '0');
  expect(fireEvent.wheel(region, { deltaY: 300 })).toBe(true);
  fireEvent.pointerDown(region);
  fireEvent.keyDown(region, { key: 'PageDown' });
  expect(wheel).not.toHaveBeenCalled(); expect(pointer).not.toHaveBeenCalled(); expect(keyboard).not.toHaveBeenCalled();
  expect(action).not.toHaveBeenCalled();
  fireEvent.keyDown(region, { key: 'Escape' });
  expect(screen.queryByRole('region', { name: 'Run configuration' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Execution details' })).toHaveFocus();
});

it('bounds the sheet above the Canvas status bar instead of letting its actions hide behind it', () => {
  const footer = document.createElement('footer');
  const bounds = jest.spyOn(footer, 'getBoundingClientRect').mockReturnValue({ top: 260 } as DOMRect);
  render(<Panel state={state} connected onAction={jest.fn()} bottomBoundary={footer} />); open();
  expect(screen.getByRole('region', { name: 'Run configuration' }).parentElement).toHaveStyle({ maxHeight: '244px' });
  bounds.mockReturnValue({ top: 180 } as DOMRect);
  fireEvent(window, new Event('resize'));
  expect(screen.getByRole('region', { name: 'Run configuration' }).parentElement).toHaveStyle({ maxHeight: '164px' });
});
describe('execution cache: a reused result', () => {
  const capturedAt = new Date(2026, 9, 25, 14, 2).getTime();
  const reused: ExecutionReviewState = { ...state, phase: 'succeeded', detail: 'Reused identical result from 14:02 (model not run again). Validation passed.',
    reused: { capturedAt } };
  it('is labelled on the strip and in the details, and offers Run fresh instead of Run again', () => {
    const onRunFresh = jest.fn(), onRepeat = jest.fn();
    render(<Panel state={reused} connected onAction={jest.fn()} onRepeat={onRepeat} onRunFresh={onRunFresh} />);
    // Serving still runs Python to resolve and recheck the plan: only the model is not run.
    const trigger = screen.getByRole('button', { name: /^Execution details\. Reused identical result from .+ \(model not run again\)$/ });
    expect(within(trigger).getByText('Reused · model not run again')).toBeInTheDocument();
    fireEvent.click(trigger);
    const label = screen.getByText(/^Reused identical result from/, { selector: 'strong' });
    expect(label).toHaveTextContent(/\(model not run again\)$/);
    expect(screen.queryByText(/no Python ran/)).toBeNull();
    expect(label.querySelector('time')).toHaveAttribute('dateTime', new Date(capturedAt).toISOString());
    expect(screen.queryByRole('button', { name: 'Run again' })).toBeNull();
    const fresh = screen.getByRole('button', { name: 'Run fresh' });
    expect(fresh).toHaveAttribute('title', expect.stringMatching(/even though an identical result is available/));
    fresh.focus(); fireEvent.click(fresh);
    expect(onRunFresh).toHaveBeenCalledTimes(1);
    expect(onRepeat).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /^Execution details/ })).toHaveFocus();
  });
  it('explains what made the result reusable without pointing at timings the Canvas no longer shows', () => {
    // 0.1.28 removed every timing display; the note said "Timings shown are from that run".
    render(<Panel state={reused} connected onAction={jest.fn()} onRunFresh={jest.fn()} />); open2();
    expect(screen.getByText('Same saved sources, setup, inputs, environment and tensorViz backend as that run.')).toBeInTheDocument();
    expect(screen.queryByText(/Timings/)).toBeNull();
  });
  it('Run fresh is disabled while disconnected or another run is pending', () => {
    render(<Panel state={reused} connected={false} onAction={jest.fn()} onRunFresh={jest.fn()} />); open2();
    expect(screen.getByRole('button', { name: 'Run fresh' })).toBeDisabled();
  });
  it('a fresh run has no reused label and keeps Run again', () => {
    render(<Panel state={{ ...state, phase: 'succeeded', detail: 'Validation passed.' }} connected onAction={jest.fn()} onRepeat={jest.fn()} onRunFresh={jest.fn()} />);
    open();
    expect(screen.queryByText(/model not run again/)).toBeNull();
    expect(screen.getByRole('button', { name: 'Run again' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run fresh' })).toBeNull();
  });
  const open2 = () => fireEvent.click(screen.getByRole('button', { name: /^Execution details/ }));
});

// A wrapped toolbar can be taller than the inline device button. Its bottom
// owns the sheet position, so measuring only the button would hide Run/Cancel.
it('reserves the whole toolbar height when execution details sit beside the model title', () => {
  const footer = document.createElement('footer');
  jest.spyOn(footer, 'getBoundingClientRect').mockReturnValue({ top: 600 } as DOMRect);
  const view = render(<header className="local-browser-header"><Panel state={state} connected onAction={jest.fn()} bottomBoundary={footer} /></header>);
  const header = view.container.querySelector('header')!;
  const bounds = jest.spyOn(header, 'getBoundingClientRect').mockReturnValue({ bottom: 104 } as DOMRect);
  open();
  const sheet = screen.getByRole('region', { name: 'Run configuration' }).parentElement;
  expect(sheet).toHaveStyle({ maxHeight: '480px' });
  expect(screen.getByRole('button', { name: 'Execution details' })).toHaveAttribute('title', 'GPU · cuda:0 · SSH · gpu-box');
  bounds.mockReturnValue({ bottom: 150 } as DOMRect);
  fireEvent(window, new Event('resize'));
  expect(sheet).toHaveStyle({ maxHeight: '434px' });
});
