import React from 'react';
import { fireEvent, render as renderPanel, screen } from '@testing-library/react';
import GraphOpeningPanel from '../../ide/GraphOpeningPanel';
import type { GraphOpeningState } from '../../ide/protocol.generated';

// Existing binding tests deliberately take the manual route; choice behavior is tested separately.
function render(ui: React.ReactElement) {
  const view = renderPanel(ui);
  const manual = screen.queryByRole('button', { name: 'Set up manually' });
  if (manual) fireEvent.click(manual);
  return view;
}

const state = (): GraphOpeningState => ({ id: 'opening', phase: 'needs_setup', hasCapture: false, detail: 'Complete setup.',
  target: { workspaceFolder: '/workspace', file: 'model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 0 },
  discovery: { model: 'model:Net', sourceDigest: 'a'.repeat(64), init: [{ name: 'width', kind: 'keyword', required: true },
    { name: 'bias', kind: 'keyword', required: false, defaultText: 'True', defaultValue: true }],
    inputs: [{ name: 'x', kind: 'keyword', required: true }, { name: 'mask', kind: 'keyword_only', required: true }],
    examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false } });

test('JAX setup preserves framework and identifies inferred shapes without offering unsupported run modes', () => {
  const opening = state();
  opening.draft = { profile: { kind: 'model', framework: 'jax', model: 'model:Net', mode: 'eval', device: 'cpu',
    seed: 0, grad_enabled: false, init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: {} } } };
  const onAction = jest.fn();
  render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByText(/JAX tracing infers tensor shapes/)).toBeVisible();
  expect(screen.getByLabelText('Mode')).toBeDisabled();
  expect(screen.getByLabelText('Device')).toBeEnabled();
  fireEvent.change(screen.getByLabelText('Device'), { target: { value: 'cuda:1' } });
  expect(screen.getByLabelText('Record gradients')).toBeDisabled();
  fireEvent.change(screen.getByLabelText('width *'), { target: { value: '8' } });
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '2, 8' } });
  fireEvent.change(screen.getByLabelText('mask shape'), { target: { value: '2, 8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.framework).toBe('jax');
  expect(onAction.mock.calls[0][0].draft.profile.device).toBe('cuda:1');
});

test('the model setup form identifies the newly selected environment after creation', () => {
  const opening = state(); opening.environmentName = 'my-model';
  opening.environment = { reason: 'ready', detail: 'Ready', selectedId: 'created', candidates: [{ id: 'created',
    interpreter: '/storage/model-environments/model-123/my-model/bin/python', prefix: '/storage/model-environments/model-123/my-model',
    source: 'Saved for this folder', status: 'libraries_present', python: '3.12.2', torch: '2.7.1' }] };
  const { rerender } = render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByText('my-model')).toBeVisible();
  expect(screen.getByText('Python 3.12.2 · Torch 2.7.1')).toBeVisible();
  rerender(<GraphOpeningPanel state={{ ...opening, phase: 'ready' }} connected onAction={jest.fn()} />);
  expect(screen.queryByText('Setup help and environment')).toBeNull();
});

test('manual and optional agent routes remain visible and preserve entered values without submitting', () => {
  const onAction = jest.fn(); render(<GraphOpeningPanel state={state()} connected onAction={onAction} />);
  fireEvent.click(screen.getByRole('button', { name: 'Setup methods' }));
  expect(screen.getByText('Recommended').closest('button')).toHaveAccessibleName('Set up with agent');
  fireEvent.click(screen.getByRole('button', { name: 'Set up manually' }));
  fireEvent.change(screen.getByLabelText('width *'), { target: { value: '8' } });
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '2, 8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Setup methods' }));
  fireEvent.click(screen.getByRole('button', { name: 'Set up with agent' }));
  expect(onAction).toHaveBeenLastCalledWith(expect.objectContaining({ kind: 'agent_help', draft: expect.objectContaining({ profile: expect.objectContaining({ init: { args: [], kwargs: { width: 8 } }, inputs: { args: [], kwargs: { x: expect.objectContaining({ shape: [2, 8] }) } } }) }) }));
  expect(screen.getByLabelText('width *')).toHaveValue('8');
  expect(screen.getByLabelText('x shape')).toHaveValue('2, 8');
  const edit = screen.getByRole('button', { name: 'Edit tensorviz.toml' });
  fireEvent.click(screen.getByText('Saved setup and source evidence'));
  fireEvent.click(edit);
  expect(onAction.mock.calls.map(call => call[0].kind)).toEqual(['agent_help', 'edit_config']);
});

test('model execution failure offers MCP investigation beside retry rather than asking for new example inputs', () => {
  const onAction = jest.fn();
  render(<GraphOpeningPanel state={{ ...state(), phase: 'failed', failure: { kind: 'execution' }, detail: 'The model raised an exception.' }} connected onAction={onAction} />);
  expect(screen.queryByLabelText('width *')).toBeNull();
  expect(screen.queryByText(/needs an example call/)).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Set up with agent' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'agent_help' });
  expect(screen.getAllByRole('button', { name: 'Retry' })).toHaveLength(1);
});

test('collects constructor and multiple inputs together, preserving defaults without guessing dimensions', () => {
  const onAction = jest.fn(); render(<GraphOpeningPanel state={state()} connected onAction={onAction} />);
  expect(screen.queryByLabelText('Construction')).toBeNull();
  expect(screen.queryByText('Constructor parameters')).toBeNull();
  expect(screen.getByLabelText('Model creation')).toBeVisible();
  expect(screen.getByText('Advanced setup').closest('details')).not.toHaveAttribute('open');
  expect(screen.getByLabelText('x shape')).toHaveValue('');
  expect(screen.getByLabelText('mask shape')).toHaveValue('');
  fireEvent.change(screen.getByLabelText('width *'), { target: { value: '8' } });
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '2, 8' } });
  fireEvent.change(screen.getByLabelText('mask dtype'), { target: { value: 'bool' } });
  expect(screen.getByLabelText('mask shape')).toHaveValue('');
  fireEvent.change(screen.getByLabelText('mask shape'), { target: { value: '2, 8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction).toHaveBeenCalledTimes(1);
  const { draft } = onAction.mock.calls[0][0];
  expect(draft.profile.init).toEqual({ args: [], kwargs: { width: 8 } });
  expect(draft.profile.inputs.kwargs.x.shape).toEqual([2, 8]);
  expect(draft.profile.inputs.kwargs.mask).toMatchObject({ dtype: 'bool', shape: [2, 8], generator: { kind: 'full', value: false } });
});

test('repository examples require a choice and failures retain the submitted values', () => {
  const opening = state();
  opening.discovery!.examples = [{ file: 'train.py', line: 12, init: { args: [], kwargs: { width: 16 } } }];
  const onAction = jest.fn(); const view = render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByLabelText('width *')).toHaveValue('');
  fireEvent.change(screen.getByLabelText('Use an example from your code'), { target: { value: '0' } });
  expect(screen.getByLabelText('width *')).toHaveValue('16');
  const draft = { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
    init: { args: [], kwargs: { width: 16 } }, inputs: { args: [], kwargs: {} } } };
  view.rerender(<GraphOpeningPanel state={{ ...opening, draft, requirements: ['inputs.mask: Required.'] }} connected onAction={onAction} />);
  expect(screen.getByLabelText('width *')).toHaveValue('16');
  expect(screen.getByRole('alert')).toHaveTextContent('inputs.mask');
  expect(onAction).not.toHaveBeenCalled();
});

test('a recovered delegated signature asks for the missing tensor alongside wrapper controls', () => {
  const opening = state();
  opening.discovery!.init = [];
  opening.discovery!.inputs = [{ name: 'x', kind: 'keyword', required: true },
    { name: 'masks', kind: 'keyword', required: false, defaultText: 'None', defaultValue: null },
    { name: 'is_training', kind: 'keyword_only', required: false, defaultText: 'False', defaultValue: false }];
  opening.draft = { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
    init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: {} } } };
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByLabelText('x shape')).toBeRequired();
  expect(screen.getByLabelText('x shape')).toHaveValue('');
  expect(screen.getByLabelText('is_training value')).toHaveValue('');
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '1, 3, 224, 224' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.inputs).toMatchObject({ args: [], kwargs: {
    x: { kind: 'tensor', shape: [1, 3, 224, 224], dtype: 'float32' },
  } });
  expect(onAction.mock.calls[0][0].draft.profile.inputs.kwargs).not.toHaveProperty('is_training');
});

test('creating a factory helper carries entered values and does not execute', () => {
  const onAction = jest.fn(); render(<GraphOpeningPanel state={state()} connected onAction={onAction} />);
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '1, 8' } });
  expect(screen.getByLabelText('Model creation')).toBeVisible();
  fireEvent.change(screen.getByLabelText('Model creation'), { target: { value: 'python' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create starter function' }));
  expect(onAction.mock.calls[0][0]).toMatchObject({ kind: 'factory_template', draft: { profile: { inputs: { kwargs: { x: { shape: [1, 8] } } } } } });
});

test('standalone setup exposes explicit factories without unavailable editor helper or parent controls', () => {
  const opening = state();
  opening.discovery!.parents = [{ cls: 'Parent', file: 'model.py', line: 20, attribute: 'child' }];
  render(<GraphOpeningPanel state={opening} connected standalone onAction={jest.fn()} />);
  expect(screen.getByLabelText('Model creation')).toBeVisible();
  fireEvent.change(screen.getByLabelText('Model creation'), { target: { value: 'python' } });
  expect(screen.getByLabelText('Model builder file')).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Create starter function' })).toBeNull();
  expect(screen.queryByText('Use an enclosing model’s inputs')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Select Python environment' })).toBeNull();
});

test('a selected factory keeps construction visible and only requires the missing input', () => {
  const opening = state(); opening.discovery!.inputs = [{ name: 'x', kind: 'keyword', required: true }];
  opening.draft = { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
    factory: { file: 'setup_model.py', callable: 'make_model', kwargs: { variant: 'small' } }, inputs: { args: [], kwargs: {} } } };
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByRole('heading', { name: 'Set up this model' })).toBeVisible();
  expect(screen.queryByLabelText('width *')).toBeNull();
  expect(screen.getByLabelText('Model builder file')).toBeVisible();
  expect(screen.getByLabelText('x shape')).toBeVisible();
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '1, 8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.factory).toEqual(opening.draft.profile.factory);
  expect(onAction.mock.calls[0][0].draft.profile.init).toBeUndefined();
});

test('visible factory overrides preserve entered inputs and restore the direct settings', () => {
  const onAction = jest.fn(); render(<GraphOpeningPanel state={state()} connected onAction={onAction} />);
  fireEvent.change(screen.getByLabelText('width *'), { target: { value: '8' } });
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '1, 8' } });
  fireEvent.change(screen.getByLabelText('mask shape'), { target: { value: '1, 8' } });
  expect(screen.getByLabelText('Model creation')).toBeVisible();
  fireEvent.change(screen.getByLabelText('Model creation'), { target: { value: 'python' } });
  fireEvent.change(screen.getByLabelText('Model builder file'), { target: { value: 'setup_model.py' } });
  fireEvent.change(screen.getByLabelText('Model builder function'), { target: { value: 'make_model' } });
  expect(screen.getByLabelText('Model builder function')).toBeVisible();
  fireEvent.change(screen.getByLabelText('Model creation'), { target: { value: 'settings' } });
  expect(screen.getByLabelText('width *')).toHaveValue('8');
  expect(screen.getByLabelText('x shape')).toHaveValue('1, 8');
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.init.kwargs).toEqual({ width: 8 });
  expect(onAction.mock.calls[0][0].draft.profile.factory).toBeUndefined();
  fireEvent.change(screen.getByLabelText('Model creation'), { target: { value: 'python' } });
  expect(screen.getByLabelText('Model builder function')).toHaveValue('make_model');
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[1][0].draft.profile.factory).toEqual({ file: 'setup_model.py', callable: 'make_model', kwargs: {} });
  expect(onAction.mock.calls[1][0].draft.profile.inputs.kwargs.x.shape).toEqual([1, 8]);
});

test('an automatically selected config shows its missing values without a construction choice', () => {
  const opening = state();
  opening.discovery!.init = [{ name: 'config', kind: 'keyword', required: true }];
  opening.discovery!.inputs = [];
  opening.discovery!.configs = [{ cls: 'Config', file: 'model.py', parameter: 'config', construction: 'call',
    fields: [{ name: 'width', kind: 'keyword', required: true }] }];
  opening.draft = { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
    init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: {} } }, configObject: { index: 0, parameter: 'config', values: {} } };
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByLabelText('width *')).toBeVisible();
  expect(screen.getByLabelText('Configuration type')).not.toBeVisible();
  expect(screen.getByLabelText('Model parameter for this config')).not.toBeVisible();
  fireEvent.change(screen.getByLabelText('width *'), { target: { value: '12' } });
  expect(screen.getByLabelText('Model creation')).toBeVisible();
  fireEvent.change(screen.getByLabelText('Model creation'), { target: { value: 'python' } });
  fireEvent.change(screen.getByLabelText('Model creation'), { target: { value: 'settings' } });
  expect(screen.getByLabelText('width *')).toHaveValue('12');
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.configObject).toEqual({ index: 0, parameter: 'config', values: { width: 12 } });
});

test('incomplete visible factory fields block execution', () => {
  const opening = state(); opening.discovery!.inputs = [];
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByLabelText('Model creation')).toBeVisible();
  fireEvent.change(screen.getByLabelText('Model creation'), { target: { value: 'python' } });
  expect(screen.getByText('Advanced setup').closest('details')).not.toHaveAttribute('open');
  expect(screen.getByLabelText('Model builder file')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction).not.toHaveBeenCalled();
  expect(screen.getByLabelText('Model builder file')).toBeVisible();
});

test('capture progress exposes cancellation, and disconnected setup cannot submit', () => {
  const onAction = jest.fn(); const view = render(<GraphOpeningPanel state={{ ...state(), phase: 'capturing' }} connected onAction={onAction} />);
  expect(screen.queryByRole('button', { name: 'Open graph' })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'cancel' });
  view.rerender(<GraphOpeningPanel state={state()} connected={false} onAction={onAction} />);
  expect(screen.getByRole('button', { name: 'Set up manually' })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Open graph' })).toBeNull();
});

test('keeps defaults collapsed while required inputs and their generation summary remain visible', () => {
  const opening = state();
  opening.discovery!.init = Array.from({ length: 22 }, (_, i) => ({ name: `option${i}`, kind: 'keyword', required: false, defaultText: '1', defaultValue: 1 }));
  opening.discovery!.inputs = [{ name: 'x', kind: 'keyword', required: true }];
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByText('Model settings · 22 source defaults').closest('details')).not.toHaveAttribute('open');
  expect(screen.getByLabelText('x shape')).not.toHaveValue('1, 3, 224, 224');
  expect(screen.getByLabelText('x shape').closest('details')).toBeNull();
  expect(screen.getByText('float32 · Normal values · seed 0 · synthetic')).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '2, 8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.init.kwargs).toEqual({});
});

test.each([null, [{ name: 'values', kind: 'varargs', required: false }], [
  { name: 'is_training', kind: 'keyword_only', required: false, defaultValue: false },
  { name: 'args', kind: 'varargs', required: false },
]] as any[])('does not turn unresolved arguments into an automatic empty call: %p', inputs => {
  const opening = state(); opening.discovery!.init = []; opening.discovery!.inputs = inputs;
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  const submit = screen.getByRole('button', { name: 'Open graph' });
  expect(submit).toBeDisabled();
  expect(screen.getByLabelText('Advanced profile').closest('details')).toHaveAttribute('open');
  fireEvent.click(screen.getByLabelText('Run this call with no arguments'));
  expect(submit).toBeEnabled();
  fireEvent.click(submit);
  expect(onAction.mock.calls[0][0].draft.profile.inputs).toEqual({ args: [], kwargs: {} });
});

test('an explicitly zero-argument forward needs no extra confirmation', () => {
  const opening = state(); opening.discovery!.init = []; opening.discovery!.inputs = [];
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.queryByLabelText('Run this call with no arguments')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction).toHaveBeenCalledTimes(1);
});

test('a required constructor literal cannot be submitted empty', () => {
  const onAction = jest.fn(); render(<GraphOpeningPanel state={state()} connected onAction={onAction} />);
  expect(screen.getByLabelText('width *')).toBeRequired();
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '1, 8' } });
  fireEvent.change(screen.getByLabelText('mask shape'), { target: { value: '1, 8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction).not.toHaveBeenCalled();
});

test('a failed forward keeps its error visible without reopening the input form', () => {
  const opening = state(); opening.phase = 'failed'; opening.hasCapture = true;
  opening.detail = 'Input width 7 does not match 8.';
  opening.draft = { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu',
    init: { args: [], kwargs: { width: 8 } }, inputs: { args: [], kwargs: {
      x: { kind: 'tensor', shape: [2, 7], dtype: 'float32', generator: { kind: 'normal', seed: 0 } },
    } } } };
  render(<GraphOpeningPanel state={opening} retained connected onAction={jest.fn()} />);
  expect(screen.queryByLabelText('x shape')).toBeNull();
  expect(screen.getByText('Input width 7 does not match 8.')).toBeInTheDocument();
  expect(screen.getByText('Partial capture remains visible up to the failure.')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Change inputs' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Open graph' })).toBeNull();
});


test('prefilled fields stay editable under settings while only missing inputs are prominent', () => {
  const opening = state();
  opening.draft = { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
    init: { args: [], kwargs: { width: 8 } }, inputs: { args: [], kwargs: {
      x: { kind: 'tensor', shape: [1, 8], dtype: 'float32', requires_grad: false, generator: { kind: 'normal', seed: 0 } },
    } } } };
  opening.evidence = [{ field: 'inputs.x', kind: 'example', detail: 'Linear input width; example batch size 1.', file: 'model.py', line: 4 }];
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByLabelText('width *').closest('details')).not.toHaveAttribute('open');
  expect(screen.getByLabelText('x shape').closest('details')).not.toHaveAttribute('open');
  expect(screen.getByLabelText('mask shape').closest('details')).toBeNull();
  fireEvent.change(screen.getByLabelText('mask shape'), { target: { value: '1, 8' } });
  expect(screen.getByLabelText('mask shape').closest('details')).toBeNull();
  fireEvent.click(screen.getByText('Initial value source'));
  const origin = screen.getByText('Initial value source').closest('details')!;
  expect(origin).toHaveTextContent('Linear input width; example batch size 1.');
  fireEvent.click(origin.querySelector('button')!);
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'reveal_setup_source', file: 'model.py', line: 4 });
  onAction.mockClear();
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.inputs.kwargs.x.shape).toEqual([1, 8]);
});

test('the captured summary discloses its synthetic recipe and source assumptions', () => {
  const opening = state(); opening.phase = 'ready'; opening.hasCapture = true;
  opening.draft = { profileName: 'tensorviz_Net', profile: { mode: 'eval', device: 'cpu', inputs: { args: [], kwargs: {
    x: { kind: 'tensor', shape: [1, 8], dtype: 'float32' },
  } } } };
  opening.evidence = [{ field: 'inputs.x', kind: 'example', detail: 'Example batch size 1.' }];
  render(<GraphOpeningPanel state={opening} connected compact onAction={jest.fn()} />);
  expect(screen.getByRole('button', { name: 'Change inputs' })).toHaveTextContent('1 × 8 · float32 · synthetic');
  expect(screen.getByText('Example setup · sources and assumptions').closest('details')).not.toHaveAttribute('open');
  expect(screen.getByText('Example choice')).toBeInTheDocument();
});


test('an unknown source override needs attention even when Python has a default', () => {
  const opening = state();
  opening.discovery!.init![0] = { name: 'width', kind: 'keyword', required: false, defaultText: '8', defaultValue: 8 };
  opening.requirements = ['init.width: the source call uses a value that needs configuration.'];
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByLabelText('width *').closest('details')).toBeNull();
  expect(screen.getByLabelText('width *')).toBeRequired();
});


test.each(['factory', 'constructor_json', 'inputs_json'])('config-default recovery respects an explicit %s edit', edit => {
  const opening = state();
  opening.discovery!.init = null; opening.discovery!.inputs = [];
  opening.draft = { configDefaults: true, profile: { kind: 'model', model: 'model:Net', mode: 'eval',
    device: 'cpu', grad_enabled: false, init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: {} } } };
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  if (edit === 'factory') {
    fireEvent.change(screen.getByLabelText('Model creation'), { target: { value: 'python' } });
    fireEvent.change(screen.getByLabelText('Model builder file'), { target: { value: 'chosen.py' } });
    fireEvent.change(screen.getByLabelText('Model builder function'), { target: { value: 'make_model' } });
  } else {
    const next = { ...opening.draft.profile, [edit === 'constructor_json' ? 'init' : 'inputs']: { args: [], kwargs: { x: 1 } } };
    fireEvent.blur(screen.getByLabelText('Advanced profile'), { target: { value: JSON.stringify(next) } });
  }
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction).toHaveBeenCalledTimes(1);
  expect(onAction.mock.calls[0][0].draft.configDefaults).toBe(edit === 'inputs_json' ? true : undefined);
});

const savedTensor = (shape: number[]) => ({ kind: 'tensor', shape, dtype: 'float32', requires_grad: false,
  generator: { kind: 'normal', mean: 0, std: 1, seed: 7 } });
const savedInputState = (inputs: any): GraphOpeningState => {
  const opening = state(); opening.discovery!.init = null; opening.discovery!.inputs = null;
  opening.draft = { profileName: 'tensorviz_Net', profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu',
    grad_enabled: false, init: { args: [], kwargs: {} }, inputs } };
  return opening;
};

test('an inherited saved call exposes editable tensor shapes without raw JSON or invented constructor requirements', () => {
  const inputs = { args: [], kwargs: { src: savedTensor([8, 1, 512]), tgt: savedTensor([4, 1, 512]) } };
  const opening = savedInputState(inputs), onAction = jest.fn();
  render(<GraphOpeningPanel state={opening} retained connected onAction={onAction} />);
  expect(screen.getByRole('heading', { name: 'Set up this model' })).toBeVisible();
  expect(screen.getByLabelText('src shape')).toBeVisible();
  expect(screen.getByLabelText('tgt shape')).toHaveValue('4, 1, 512');
  expect(screen.getByLabelText('Advanced profile')).not.toBeVisible();
  expect(screen.queryByText(/Model setup needs a custom factory/)).toBeNull();
  expect(screen.queryByText(/Set its explicit inputs in Advanced setup/)).toBeNull();
  fireEvent.change(screen.getByLabelText('tgt shape'), { target: { value: 'three, 1, 512' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText('tgt shape'), { target: { value: '3, 1, 512' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile).toEqual({ ...opening.draft!.profile,
    inputs: { ...inputs, kwargs: { ...inputs.kwargs, tgt: { ...inputs.kwargs.tgt, shape: [3, 1, 512] } } } });
  expect(inputs.kwargs.tgt.shape).toEqual([4, 1, 512]);
});

test('saved positional and extra keyword inputs keep their original binding slots alongside discovered fields', () => {
  const inputs = { args: [{ ...savedTensor([2, 8]), dtype: 'int16', generator: { kind: 'full', value: 2 } }, savedTensor([2, 4])],
    kwargs: { scale: { kind: 'float', value: 1 }, mask: false } };
  const opening = savedInputState(inputs), onAction = jest.fn();
  opening.discovery!.inputs = [{ name: 'x', kind: 'keyword', required: true },
    { name: 'rest', kind: 'varargs', required: false }, { name: 'scale', kind: 'keyword_only', required: false },
    { name: 'options', kind: 'kwargs', required: false }];
  render(<GraphOpeningPanel state={opening} retained connected onAction={onAction} />);
  expect(screen.getAllByLabelText('x shape')).toHaveLength(1);
  expect(screen.getByLabelText('x dtype')).toHaveValue('int16');
  expect(screen.getByLabelText('Argument 2 shape')).toHaveValue('2, 4');
  fireEvent.change(screen.getByLabelText('Argument 2 shape'), { target: { value: '2, 6' } });
  fireEvent.change(screen.getByLabelText('scale value'), { target: { value: '2' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.inputs).toEqual({ args: [inputs.args[0], { ...inputs.args[1], shape: [2, 6] }],
    kwargs: { scale: { kind: 'float', value: 2 }, mask: false } });
});

test('nested conditioning inputs edit leaves while preserving dict, tuple and list contracts', () => {
  const inputs = { args: [], kwargs: { conditioning: { kind: 'dict', items: {
    text: savedTensor([1, 8, 32]), extra: { kind: 'tuple', items: [savedTensor([1, 4]),
      { kind: 'list', items: [{ kind: 'float', value: 1 }, { kind: 'none' }] }] },
  } } } };
  const opening = savedInputState(inputs), onAction = jest.fn();
  opening.draft!.profile.factory = { file: 'setup.py', callable: 'make_model', kwargs: { variant: 'small' } };
  delete opening.draft!.profile.init;
  render(<GraphOpeningPanel state={opening} retained connected onAction={onAction} />);
  expect(screen.getByLabelText('conditioning.text shape')).toBeVisible();
  fireEvent.change(screen.getByLabelText('conditioning.extra[0] shape'), { target: { value: '1, 6' } });
  fireEvent.change(screen.getByLabelText('conditioning.extra[1][0] value'), { target: { value: '2' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  const profile = onAction.mock.calls[0][0].draft.profile;
  expect(profile.factory).toEqual(opening.draft!.profile.factory);
  expect(profile.inputs.kwargs.conditioning).toEqual({ kind: 'dict', items: { text: inputs.kwargs.conditioning.items.text,
    extra: { kind: 'tuple', items: [savedTensor([1, 6]), { kind: 'list', items: [{ kind: 'float', value: 2 }, { kind: 'none' }] }] } } });
  expect(inputs.kwargs.conditioning.items.extra.items[0]).toEqual(savedTensor([1, 4]));
});

test('Change inputs after a failed inherited call retains all inputs for correction', () => {
  const opening = savedInputState({ args: [], kwargs: { query: savedTensor([1, 4, 7]), key: savedTensor([1, 8, 8]) } });
  opening.phase = 'failed'; opening.detail = 'Expected query width 8, got 7.';
  opening.failure = { kind: 'execution' };
  const onAction = jest.fn(), view = render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByText(opening.detail)).toBeVisible();
  expect(screen.queryByLabelText('query shape')).toBeNull();
  expect(onAction).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Change inputs' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'change_inputs' });
  view.rerender(<GraphOpeningPanel state={{ ...opening, phase: 'needs_setup', failure: undefined }} connected onAction={onAction} />);
  fireEvent.click(screen.getByRole('button', { name: 'Set up manually' }));
  expect(screen.getByLabelText('query shape')).toBeVisible();
  expect(screen.getByLabelText('key shape')).toBeVisible();
  fireEvent.change(screen.getByLabelText('query shape'), { target: { value: '1, 4, 8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction).toHaveBeenCalledTimes(2);
  expect(onAction.mock.calls[1][0]).toMatchObject({ kind: 'submit', draft: { profile: { inputs: { args: [], kwargs: {
    query: savedTensor([1, 4, 8]), key: savedTensor([1, 8, 8]),
  } } } } });
});

test('an explicit binding does not satisfy a different missing source argument', () => {
  const opening = savedInputState({ args: [], kwargs: { extra: savedTensor([1, 8]) } });
  opening.discovery!.inputs = [{ name: 'x', kind: 'keyword', required: true }];
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} retained connected onAction={onAction} />);
  expect(screen.getByLabelText('extra shape')).toBeVisible();
  expect(screen.getByLabelText('x shape')).toBeRequired();
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction).not.toHaveBeenCalled();
});

test('a named saved profile cannot silently turn an unknown signature into an empty call', () => {
  render(<GraphOpeningPanel state={savedInputState({ args: [], kwargs: {} })} retained connected onAction={jest.fn()} />);
  expect(screen.getByRole('button', { name: 'Open graph' })).toBeDisabled();
  expect(screen.getByLabelText('Run this call with no arguments')).toBeVisible();
});

test('clearing and retyping a saved shape does not reset its dtype or generator', () => {
  const tensor = { ...savedTensor([1, 8]), dtype: 'int16', generator: { kind: 'full', value: 2 } };
  const opening = savedInputState({ args: [], kwargs: { tokens: tensor } }), onAction = jest.fn();
  render(<GraphOpeningPanel state={opening} retained connected onAction={onAction} />);
  fireEvent.change(screen.getByLabelText('tokens shape'), { target: { value: '' } });
  expect(screen.getByLabelText('tokens shape')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open graph' })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('tokens shape'), { target: { value: '2, 8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.inputs.kwargs.tokens).toEqual({ ...tensor, shape: [2, 8] });
});

test('clearing and retyping a tagged float keeps its Python float type', () => {
  const opening = savedInputState({ args: [], kwargs: { scale: { kind: 'float', value: 1 } } }), onAction = jest.fn();
  render(<GraphOpeningPanel state={opening} retained connected onAction={onAction} />);
  fireEvent.change(screen.getByLabelText('scale value'), { target: { value: '' } });
  fireEvent.change(screen.getByLabelText('scale value'), { target: { value: '2' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.inputs.kwargs.scale).toEqual({ kind: 'float', value: 2 });
});


test('Python example inputs preserve existing bindings when toggled and save the callable', () => {
  const opening = state(); opening.discovery!.init = [];
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '1, 64, 80' } });
  fireEvent.change(screen.getByLabelText('Input source'), { target: { value: 'python' } });
  expect(screen.queryByLabelText('x shape')).toBeNull();
  fireEvent.change(screen.getByLabelText('Input function'), { target: { value: 'graph_inputs' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.inputs).toEqual({ factory: { file: 'model.py', callable: 'graph_inputs', kwargs: {} } });
  fireEvent.change(screen.getByLabelText('Input source'), { target: { value: 'entered' } });
  expect(screen.getByLabelText('x shape')).toHaveValue('1, 64, 80');
  fireEvent.change(screen.getByLabelText('Input source'), { target: { value: 'python' } });
  expect(screen.getByLabelText('Input function')).toHaveValue('graph_inputs');
});

test('saved Python inputs reopen without tensor fields and identify the function in the compact summary', () => {
  const opening = state(); opening.discovery!.init = [];
  opening.draft = { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
    init: { args: [], kwargs: {} }, inputs: { factory: { file: 'model.py', callable: 'graph_inputs', kwargs: { batch: 1 } } } } };
  const onAction = jest.fn(); const view = render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByLabelText('Input function')).toHaveValue('graph_inputs');
  expect(screen.queryByLabelText('x shape')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.inputs).toEqual(opening.draft.profile.inputs);
  view.rerender(<GraphOpeningPanel state={{ ...opening, phase: 'ready' }} connected compact onAction={onAction} />);
  expect(screen.getByText('Inputs: graph_inputs()')).toBeVisible();
});

test('a host update to Python inputs remains editable without treating the factory as tensor bindings', () => {
  const opening = state(); opening.discovery!.init = [];
  const onAction = jest.fn(); const view = render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '1, 8' } });
  const updated = { ...opening, draft: { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
    init: { args: [], kwargs: {} }, inputs: { factory: { file: 'model.py', callable: 'graph_inputs', kwargs: {} } } } } };
  view.rerender(<GraphOpeningPanel state={updated} connected onAction={onAction} />);
  expect(screen.getByLabelText('Input function')).toHaveValue('graph_inputs');
  fireEvent.change(screen.getByLabelText('Input source'), { target: { value: 'entered' } });
  expect(screen.getByLabelText('x shape')).toBeVisible();
  expect(screen.getByLabelText('x shape')).toHaveValue('');
  expect(onAction).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '2, 8' } });
  fireEvent.change(screen.getByLabelText('mask shape'), { target: { value: '2, 8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.profile.inputs).toMatchObject({ args: [], kwargs: {
    x: { kind: 'tensor', shape: [2, 8] }, mask: { kind: 'tensor', shape: [2, 8] },
  } });
  expect(onAction.mock.calls[0][0].draft.profile.inputs).not.toHaveProperty('factory');
});

test('a replacement setup drops its previous configuration object before editing new inputs', () => {
  const opening = state(); opening.discovery!.init = [{ name: 'config', kind: 'keyword', required: true }];
  opening.discovery!.configs = [{ cls: 'Config', file: 'model.py', parameter: 'config', construction: 'call',
    fields: [{ name: 'width', kind: 'keyword', required: true }] }];
  opening.draft = { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
    init: { args: [], kwargs: {} }, inputs: { args: [], kwargs: {} } }, configObject: { index: 0, parameter: 'config', values: {} } };
  const onAction = jest.fn(); const view = render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  fireEvent.change(screen.getByLabelText('width *'), { target: { value: '8' } });
  const replacement = state(); replacement.id = 'updated'; replacement.discovery!.init = [];
  view.rerender(<GraphOpeningPanel state={replacement} connected onAction={onAction} />);
  fireEvent.click(screen.getByRole('button', { name: 'Set up manually' }));
  expect(screen.queryByLabelText('width *')).toBeNull();
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '2, 8' } });
  fireEvent.change(screen.getByLabelText('mask shape'), { target: { value: '2, 8' } });
  expect(onAction).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Open graph' }));
  expect(onAction.mock.calls[0][0].draft.configObject).toBeUndefined();
  expect(onAction.mock.calls[0][0].draft.profile.init).toEqual({ args: [], kwargs: {} });
});

test('agent and environment lead while manual focus is retained across typing and connection updates', () => {
  const opening = state();
  const { rerender, container } = renderPanel(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  const width = screen.getByLabelText('width *');
  expect(screen.getByRole('button', { name: 'Set up with agent' })).toHaveFocus();
  expect(screen.getByRole('region', { name: 'Setup help and environment' }).compareDocumentPosition(container.querySelector('form')!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Set up manually' }));
  expect(width).toHaveFocus();
  fireEvent.change(width, { target: { value: '8' } });
  const shape = screen.getByLabelText('x shape'); shape.focus();
  fireEvent.change(shape, { target: { value: '2, 8' } });
  rerender(<GraphOpeningPanel state={{ ...opening, detail: 'Updated connection.' }} connected={false} onAction={jest.fn()} />);
  rerender(<GraphOpeningPanel state={{ ...opening, detail: 'Reconnected.' }} connected onAction={jest.fn()} />);
  expect(shape).toHaveFocus();
  expect(shape).toHaveValue('2, 8');
});

test('only input settings lead when the constructor is already configured', () => {
  const opening = state();
  opening.draft = { profile: { kind: 'model', model: 'model:Net', mode: 'eval', device: 'cpu', grad_enabled: false,
    init: { args: [], kwargs: { width: 8 } }, inputs: { args: [], kwargs: {} } } };
  const { container } = render(<GraphOpeningPanel state={opening} connected retained onAction={jest.fn()} />);
  expect(container.querySelector('.graph-setup-section')).toHaveAttribute('aria-label', 'Test data');
  expect(screen.getByLabelText('x shape')).toHaveFocus();
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '2, 8' } });
  expect(container.querySelector('.graph-setup-section')).toHaveAttribute('aria-label', 'Test data');
});


test('verified declaration links navigate without saving or losing edited data', () => {
  const opening = state(), onAction = jest.fn();
  opening.discovery!.init![0].source = { file: 'configuration.py', line: 17 };
  opening.discovery!.inputs![0].source = { file: 'base.py', line: 42 };
  render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  fireEvent.change(screen.getByLabelText('width *'), { target: { value: '64' } });
  fireEvent.click(screen.getByTitle('Open declaration of width'));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'reveal_setup_source', file: 'configuration.py', line: 17 });
  expect(screen.getByLabelText('width *')).toHaveValue('64');
  expect(screen.getByTitle('Open declaration of x')).toHaveTextContent('base.py:42');
  expect(screen.getAllByText('Declaration not available').length).toBeGreaterThan(0);
});


test('first setup offers two alternatives without a visible form or submit action', () => {
  const onAction = jest.fn();
  renderPanel(<GraphOpeningPanel state={state()} connected onAction={onAction} />);
  expect(screen.getByRole('region', { name: 'Choose a setup method' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Set up with agent' })).toHaveFocus();
  expect(screen.getByText('Recommended').closest('button')).toHaveAccessibleName('Set up with agent');
  expect(screen.getByLabelText('width *')).not.toBeVisible();
  expect(screen.queryByRole('button', { name: 'Open graph' })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Set up manually' }));
  expect(screen.getByLabelText('width *')).toBeVisible();
  expect(screen.getByLabelText('width *')).toHaveFocus();
  expect(screen.getByRole('button', { name: 'Open graph' })).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Set up with agent' })).toBeNull();
  expect(onAction).not.toHaveBeenCalled();
});

test('switching routes preserves incomplete text and gives the agent the edited valid draft', () => {
  const onAction = jest.fn();
  const view = render(<GraphOpeningPanel state={state()} connected onAction={onAction} />);
  fireEvent.change(screen.getByLabelText('width *'), { target: { value: '64' } });
  fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '2,' } });
  fireEvent.click(screen.getByRole('button', { name: 'Setup methods' }));
  expect(screen.getByRole('button', { name: 'Set up with agent' })).toHaveFocus();
  expect(screen.queryByRole('button', { name: 'Open graph' })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Set up with agent' }));
  expect(onAction).toHaveBeenLastCalledWith(expect.objectContaining({ kind: 'agent_help', draft: expect.objectContaining({
    profile: expect.objectContaining({ init: { args: [], kwargs: { width: 64 } } }) }) }));
  fireEvent.click(screen.getByRole('button', { name: 'Set up manually' }));
  expect(screen.getByLabelText('width *')).toHaveValue('64');
  expect(screen.getByLabelText('x shape')).toHaveValue('2,');
  expect(screen.getByLabelText('x shape')).toBeInvalid();
  view.rerender(<GraphOpeningPanel state={{ ...state(), id: 'another-model' }} connected onAction={onAction} />);
  expect(screen.getByRole('button', { name: 'Set up manually' })).toBeVisible();
  expect(screen.getByLabelText('width *')).not.toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Set up manually' }));
  expect(screen.getByLabelText('x shape')).toHaveValue('');
});
