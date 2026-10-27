import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import GraphOpeningPanel from '../../ide/GraphOpeningPanel';
import type { GraphOpeningState } from '../../ide/protocol.generated';

const state = (): GraphOpeningState => ({ id: 'opening', phase: 'needs_setup', hasCapture: false, detail: 'This model needs torchmetrics, OpenCV and SciPy in the selected environment.',
  target: { workspaceFolder: '/project', file: 'pkg/model.py', cls: 'Net', documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 0 },
  environment: { reason: 'missing_libraries', detail: 'This model needs torchmetrics, OpenCV and SciPy in the selected environment.',
    selectedId: 'saved', requirements: ['torchmetrics', 'opencv-python', 'scipy'], candidates: [
      { id: 'saved', interpreter: '/project/.venv/bin/python', source: 'Saved for this folder', selected: true,
        python: '3.12.13', torch: '2.6.0+cu124', status: 'libraries_present' },
      { id: 'suggestion', interpreter: '/other/bin/python', source: 'Known to Python', pythonSelected: true,
        python: '3.12.10', torch: '2.7.1', status: 'libraries_present' },
      { id: 'gone', interpreter: '/gone/bin/python', source: 'Previously selected', status: 'unavailable' } ] },
  discovery: { model: 'pkg.model:Net', sourceDigest: 'a'.repeat(64), init: [{ name: 'width', kind: 'keyword', required: true }],
    inputs: [], examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false } });

test('native missing model libraries offer environment creation without requiring an agent', () => {
  const opening = state(); opening.environment!.canCreate = true; opening.environment!.requirements = [];
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  fireEvent.click(screen.getByRole('button', { name: 'Create model environment…' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'create_environment', name: 'Net-env' });
  expect(screen.getByText(/Your current environment stays unchanged/)).toBeVisible();
  fireEvent.click(screen.getByText('Investigate with agent'));
  expect(screen.getByRole('button', { name: 'Use Agent' })).toBeVisible();
});

test('creation is available with no existing Python and remains native-host only', () => {
  const opening = state(); opening.environment = { reason: 'unavailable', detail: 'No Python', candidates: [], canCreate: true };
  const onAction = jest.fn(); const { rerender } = render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  fireEvent.click(screen.getByRole('button', { name: 'Create model environment…' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'create_environment', name: 'Net-env' });
  rerender(<GraphOpeningPanel state={opening} standalone connected onAction={onAction} />);
  expect(screen.queryByRole('button', { name: 'Create model environment…' })).toBeNull();
});

test('creation names are editable, preserved on failure, and technical choices stay collapsed', () => {
  const opening = state(); opening.environment!.canCreate = true;
  const onAction = jest.fn(); const { rerender } = render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.queryByRole('list', { name: 'Progress' })).toBeNull();
  expect(screen.getByText('Setup details').parentElement).not.toHaveAttribute('open');
  expect(screen.getAllByRole('radio', { hidden: true })[0]).not.toBeVisible();
  fireEvent.click(screen.getByText('Change environment'));
  fireEvent.change(screen.getByRole('textbox', { name: 'Environment name' }), { target: { value: 'kimi-test' } });
  fireEvent.click(screen.getByRole('button', { name: 'Create model environment…' }));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'create_environment', name: 'kimi-test' });
  rerender(<GraphOpeningPanel state={{ ...opening, phase: 'resolving', detail: 'Installing model libraries…' }} connected onAction={onAction} />);
  expect(screen.getByRole('heading', { name: 'Preparing kimi-test' })).toBeVisible();
  expect(screen.queryByText('Use existing Python (3)')).toBeNull();
  rerender(<GraphOpeningPanel state={{ ...opening, phase: 'failed', detail: 'helper==1 is unavailable for this Python.' }} connected onAction={onAction} />);
  expect(screen.getByRole('heading', { name: 'Environment setup failed' })).toBeVisible();
  expect(screen.getByRole('alert')).toHaveTextContent('helper==1');
  fireEvent.click(screen.getByText('Change environment'));
  expect(screen.getByRole('textbox', { name: 'Environment name' })).toHaveValue('kimi-test');
  expect(onAction).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Inspect source only' }));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'source_only' });
  fireEvent.click(screen.getByRole('button', { name: 'Retry creation' }));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'create_environment', name: 'kimi-test' });
});

test('a reconnect restores the creation name from the host; another model starts with its own name', () => {
  const opening = state(); opening.environment!.canCreate = true;
  opening.phase = 'resolving'; opening.environmentName = 'mamba-test';
  const { rerender } = render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByRole('heading', { name: 'Preparing mamba-test' })).toBeVisible();
  rerender(<GraphOpeningPanel state={{ ...opening, id: 'other', phase: 'needs_setup', environmentName: undefined,
    target: { ...opening.target, cls: 'Other' } }} connected onAction={jest.fn()} />);
  fireEvent.click(screen.getByText('Change environment'));
  expect(screen.getByRole('textbox', { name: 'Environment name' })).toHaveValue('Other-env');
});

test('a failed creation still lets the user explicitly choose another Python instead of inspecting source', () => {
  const opening = state(); opening.environment!.canCreate = true;
  opening.phase = 'failed'; opening.detail = 'triton>=3.5.0 is unavailable for this platform.';
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByRole('button', { name: 'Inspect source only' })).toBeVisible();
  fireEvent.click(screen.getByText('Change environment'));
  fireEvent.click(screen.getAllByRole('radio')[1]);
  fireEvent.click(screen.getByRole('button', { name: 'Use other instead' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'select_environment', candidateId: 'suggestion' });
  expect(onAction).toHaveBeenCalledTimes(1);
});

test('invalid names cannot create; an existing Python still offers additive installation', () => {
  const opening = state(); opening.environment!.canCreate = true;
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  fireEvent.click(screen.getByText('Change environment'));
  fireEvent.change(screen.getByRole('textbox', { name: 'Environment name' }), { target: { value: '../project' } });
  expect(screen.getByRole('button', { name: 'Create model environment…' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Add libraries here…' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'install_dependencies' });
});

test('new environment downloads display host progress with cancellation and no duplicate creation', () => {
  const opening = state(); opening.environment!.canCreate = true; opening.phase = 'resolving'; opening.detail = 'Downloading and installing the reviewed libraries…';
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getAllByRole('status')[0]).toHaveTextContent('Downloading and installing');
  expect(screen.queryByRole('button', { name: 'Create model environment…' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Preparing environment…' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
});

test('repairing existing Python does not label it as a new model environment', () => {
  const opening = state(); opening.environment!.canCreate = true;
  opening.phase = 'resolving'; opening.detail = 'Installing libraries in the selected Python…';
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByRole('heading', { name: 'Preparing project/.venv' })).toBeVisible();
  expect(screen.queryByText('New')).toBeNull();
  expect(screen.getByText('Saved')).toBeVisible();
});

test('a private model class still starts with a valid portable environment name', () => {
  const opening = state(); opening.environment!.canCreate = true; opening.target.cls = '_Net';
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  fireEvent.click(screen.getByText('Change environment'));
  expect(screen.getByRole('textbox', { name: 'Environment name' })).toHaveValue('Net-env');
  expect(screen.getByRole('button', { name: 'Create model environment…' })).toBeEnabled();
});

test('missing libraries replace constructor setup with an explicit review action and accurately labelled environments', () => {
  const onAction = jest.fn(); render(<GraphOpeningPanel state={state()} connected onAction={onAction} />);
  expect(screen.getByRole('heading', { name: 'Add the missing libraries' })).toBeVisible();
  expect(screen.queryByRole('heading', { name: 'Latest update' })).toBeNull();
  expect(screen.getByText(/torchmetrics, OpenCV and SciPy/)).toBeVisible();
  expect(screen.getAllByText('Saved')[0]).toBeVisible(); expect(screen.getByText(/VS Code interpreter/)).toBeVisible();
  expect(screen.getByText(/torch 2.6.0\+cu124/)).toBeVisible();
  expect(screen.queryByLabelText('width *')).toBeNull(); expect(screen.queryByText(/needs an example call/)).toBeNull();
  expect(onAction).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Review missing libraries…' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'install_dependencies' });
});

test('changing a radio is only a suggestion until Use sends its listed ID, never a path', () => {
  const onAction = jest.fn(); render(<GraphOpeningPanel state={state()} connected onAction={onAction} />);
  const radios = screen.getAllByRole('radio'); expect(radios[0]).toBeChecked(); expect(radios[2]).toBeDisabled();
  fireEvent.click(radios[1]); expect(onAction).not.toHaveBeenCalled();
  expect(screen.queryByRole('button', { name: /(?:Review missing|Install missing) libraries/ })).toBeNull();
  expect(screen.getByText(/not used yet/)).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Use other instead' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'select_environment', candidateId: 'suggestion' });
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'refresh_environments' });
});

test('installation progress disables duplicate repair and selection while cancellation stays available', () => {
  const onAction = jest.fn(); render(<GraphOpeningPanel state={{ ...state(), phase: 'resolving', detail: 'Installing reviewed libraries…' }} connected onAction={onAction} />);
  expect(screen.getAllByRole('status')[0]).toHaveTextContent('Installing reviewed libraries');
  expect(screen.queryByRole('alert')).toBeNull();
  expect(screen.getByRole('button', { name: 'Preparing environment…' })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Review missing libraries…' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Stop' })); expect(onAction).toHaveBeenCalledWith({ kind: 'cancel' });
});

test.each([false, true])('a worker failure keeps its diagnostic visible and offers host-specific agent help (plugin=%s)', standalone => {
  const opening = state(); opening.environment = { ...opening.environment!, reason: 'worker_failed', requirements: undefined,
    detail: 'The capture worker could not load.' };
  opening.detail = opening.environment.detail;
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} standalone={standalone} connected onAction={onAction} />);
  expect(screen.queryByRole('button', { name: /(?:Review|Install) missing/ })).toBeNull();
  expect(screen.getByText(standalone ? 'Investigate with MCP' : 'Investigate with agent').parentElement).toHaveAttribute('open');
  expect(screen.getByRole('alert')).toHaveTextContent('The capture worker could not load.');
  fireEvent.click(screen.getByRole('button', { name: standalone ? 'Copy MCP investigation request' : 'Use Agent' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'agent_help' });
  expect(screen.queryByRole('button', { name: 'Browse for Python…' })).toBeNull();
});

test('plugin repair states its existing permission instead of promising a VS Code review', () => {
  const onAction = jest.fn(); render(<GraphOpeningPanel state={state()} standalone connected onAction={onAction} />);
  expect(screen.getByText(/Uses your TensorViz plugin permission/)).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Review missing libraries…' })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Install missing libraries' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'install_dependencies' });
});


test('only the confirmed host ID permits repair even if candidate selected metadata disagrees', () => {
  const opening = state(); opening.environment!.selectedId = undefined;
  const onAction = jest.fn(); const { rerender } = render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.queryByText('Saved')).toBeNull();
  expect(screen.queryByRole('button', { name: /(?:Review missing|Install missing) libraries/ })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Use other' }));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'select_environment', candidateId: 'suggestion' });
  opening.environment!.selectedId = 'suggestion';
  rerender(<GraphOpeningPanel state={{ ...opening }} connected onAction={onAction} />);
  fireEvent.click(screen.getByRole('button', { name: 'Review missing libraries…' }));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'install_dependencies' });
});

test('path details are opt-in and show the full host metadata without changing the choice', () => {
  const opening = state(); Object.assign(opening.environment!.candidates[0], { prefix: '/project/.venv', implementation: 'cpython', abi: 'cp312', architecture: 'arm64' });
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByText('/project/.venv/bin/python')).not.toBeVisible();
  fireEvent.click(screen.getByText('Interpreter details'));
  expect(screen.getByText('Interpreter details').parentElement).toHaveAttribute('open');
  expect(screen.getByText('/project/.venv/bin/python')).toBeVisible();
  expect(screen.getByText('/project/.venv')).toBeVisible();
  expect(screen.getByText('cpython · cp312 · arm64')).toBeVisible();
  expect(screen.getAllByRole('radio')[0]).toBeChecked(); expect(onAction).not.toHaveBeenCalled();
});

test.each([false, true])('empty environments promote one host-appropriate discovery action (plugin=%s)', standalone => {
  const opening = state(); opening.environment!.candidates = []; opening.environment!.selectedId = undefined;
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} standalone={standalone} connected onAction={onAction} />);
  expect(screen.getByRole('heading', { name: 'No usable Python found' })).toBeVisible();
  const primary = screen.getByRole('button', { name: standalone ? 'Refresh list' : 'Browse for Python…' });
  expect(primary).toBeEnabled(); fireEvent.click(primary);
  expect(onAction).toHaveBeenCalledWith({ kind: standalone ? 'refresh_environments' : 'environment' });
  expect(screen.queryByRole('button', { name: 'Use selected environment' })).toBeNull();
});

test('all unavailable candidates stay visible and disabled while browsing remains possible', () => {
  const opening = state(); opening.environment!.candidates = [opening.environment!.candidates[2]]; opening.environment!.selectedId = 'gone';
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByRole('radio')).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Browse for Python…' })).toBeEnabled();
});

test('disconnect disables host actions but local close and Escape do not cancel the attempt', () => {
  const onAction = jest.fn(), onClose = jest.fn();
  render(<GraphOpeningPanel state={state()} connected={false} onAction={onAction} onClose={onClose} />);
  expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Review missing libraries…' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Close environment setup' }));
  fireEvent.keyDown(screen.getByRole('region', { name: 'Python environment for model' }), { key: 'Escape' });
  expect(onClose).toHaveBeenCalledTimes(2); expect(onAction).not.toHaveBeenCalled();
});

test('Torch repair describes exact review only when supplied by the native host', () => {
  const opening = state(); opening.environment!.requirements = ['torch==2.7.1', 'pydantic']; opening.environment!.candidates[0].torch = null;
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByText(/PyTorch can be large/)).toBeVisible();
  expect(screen.getByText('torch==2.7.1')).not.toBeVisible();
  fireEvent.click(screen.getByText('Requested libraries'));
  expect(screen.getByText('torch==2.7.1')).toBeVisible();
  expect(screen.queryByText(/and its device build are kept/)).toBeNull();
});

test('compatibility repair follows the same confirmed-choice gate and preserves exact diagnostics', () => {
  const opening = state(); opening.environment!.reason = 'incompatible'; opening.environment!.compatibilityPackages = ['pydantic==2.12.0'];
  opening.environment!.detail = 'An exact diagnostic\n' + 'architecture mismatch '.repeat(12);
  opening.detail = opening.environment!.detail;
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByText(/An exact diagnostic/)).toBeVisible();
  fireEvent.click(screen.getByText('Packages to align'));
  expect(screen.getByText('pydantic==2.12.0')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Review compatibility copy…' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'install_dependencies' });
  fireEvent.click(screen.getAllByRole('radio')[1]);
  expect(screen.queryByRole('button', { name: 'Review compatibility copy…' })).toBeNull();
});


test('unknown Torch metadata does not become a claim that Torch was not detected', () => {
  const opening = state(); opening.environment!.candidates[0].torch = undefined;
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByText('py 3.12.13')).toBeVisible();
  expect(screen.queryByText(/no torch/)).toBeNull();
});

test.each([
  { interpreter: '/opt/shared/python', prefix: '/research/environments/vision', identity: 'environments/vision' },
  { interpreter: 'C:\\work\\vision\\.venv\\Scripts\\python.exe', prefix: undefined, identity: 'vision/.venv' },
])('environment identity comes from real host paths while confirmation still uses its opaque ID ($identity)', ({ interpreter, prefix, identity }) => {
  const opening = state(); opening.environment!.selectedId = undefined;
  opening.environment!.candidates = [{ ...opening.environment!.candidates[0], id: 'opaque-host-id', interpreter, prefix }];
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByRole('radio', { name: new RegExp(identity.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')) })).toBeChecked();
  fireEvent.click(screen.getByText('Interpreter details'));
  expect(screen.getByText(interpreter)).toBeVisible(); expect(onAction).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: `Use ${identity}` }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'select_environment', candidateId: 'opaque-host-id' });
});


test('a model version conflict without a repair offer is visible before any disclosure is opened', () => {
  const opening = state();
  opening.environment = { ...opening.environment!, reason: 'incompatible', requirements: undefined,
    detail: 'Installed libraries do not match this model: torch 2.7.1 (needs ==2.0.0). Choose a compatible environment.' };
  opening.detail = opening.environment.detail;
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByRole('alert')).toHaveTextContent('torch 2.7.1 (needs ==2.0.0)');
  expect(screen.getByRole('alert')).toBeVisible();
  expect(screen.getByRole('heading', { name: 'Environment needs different libraries' })).toBeVisible();
  expect(screen.queryByText('Active')).toBeNull();
  expect(screen.queryByRole('button', { name: /Review compatibility/ })).toBeNull();
  // Hosts that do not offer the pins (the standalone plugin) show no override.
  expect(screen.queryByRole('button', { name: 'Try with installed versions' })).toBeNull();
  fireEvent.click(screen.getAllByRole('radio')[1]);
  expect(screen.getByRole('alert')).toHaveTextContent('Model check for project/.venv');
  expect(screen.getByRole('alert')).not.toHaveTextContent('Model check for other');
  fireEvent.click(screen.getByRole('button', { name: 'Use other instead' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'select_environment', candidateId: 'suggestion' });
});

test('project pins offered by the host can be tried as installed, only for the saved environment', () => {
  const opening = state();
  opening.environment = { ...opening.environment!, reason: 'incompatible', requirements: undefined, declaredPins: ['torch==2.0.0'],
    detail: 'Installed libraries do not match this model: torch 2.7.1 (needs ==2.0.0). Choose a compatible environment.' };
  opening.detail = opening.environment.detail;
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByRole('heading', { name: 'Continue with project/.venv' })).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Try with installed versions' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'use_installed_versions' });
  // A different local choice is not the environment whose pins were checked.
  fireEvent.click(screen.getAllByRole('radio')[1]);
  expect(screen.queryByRole('button', { name: 'Try with installed versions' })).toBeNull();
});

test('replaceable pins offer an install in the saved environment; the managed one explains why it cannot', () => {
  const opening = state();
  opening.environment = { ...opening.environment!, reason: 'incompatible', requirements: undefined, declaredPins: ['torch==2.0.0'], replaceable: true,
    detail: 'Installed libraries do not match this model: torch 2.7.1 (needs ==2.0.0). Choose a compatible environment.' };
  const onAction = jest.fn(); const { unmount } = render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByText(/checks that a build exists for Python 3\.12\.13/)).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Install in this environment…' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'replace_versions' });
  unmount();
  const managed = state();
  managed.environment = { ...opening.environment, replaceable: undefined,
    candidates: opening.environment.candidates.map(item => item.id === 'saved' ? { ...item, managed: true as const } : item) };
  render(<GraphOpeningPanel state={managed} connected onAction={jest.fn()} />);
  expect(screen.queryByRole('button', { name: 'Install in this environment…' })).toBeNull();
  expect(screen.getByText(/managed environment is shared by every model/)).toBeVisible();
  expect(screen.getByRole('button', { name: 'Try with installed versions' })).toBeVisible();
});

test('a missing model library with no install recipe is visible while capture libraries are present', () => {
  const opening = state(); opening.environment!.requirements = [];
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByRole('alert')).toHaveTextContent('torchmetrics, OpenCV and SciPy');
  expect(screen.getByRole('alert')).toBeVisible();
  expect(screen.getByRole('heading', { name: 'Model libraries are missing' })).toBeVisible();
  expect(screen.queryByRole('button', { name: /(?:Review|Install) missing/ })).toBeNull();
  expect(screen.queryByRole('button', { name: 'Check again with this environment' })).toBeNull();
  expect(screen.getByText(/missing libraries belong to this model/)).not.toBeVisible();
  fireEvent.click(screen.getByText('Setup details'));
  expect(screen.getByText(/missing libraries belong to this model/)).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Use Agent' }));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'agent_help' });
  fireEvent.click(screen.getByRole('button', { name: 'Inspect source only' }));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'source_only' });
  fireEvent.click(screen.getAllByRole('radio')[1]);
  fireEvent.click(screen.getByRole('button', { name: 'Use other instead' }));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'select_environment', candidateId: 'suggestion' });
});

test('unsupported model dependencies expose MCP help in standalone and cannot act while disconnected', () => {
  const opening = state(); opening.environment!.requirements = undefined;
  const onAction = jest.fn(); const { rerender } = render(<GraphOpeningPanel state={opening} connected standalone onAction={onAction} />);
  fireEvent.click(screen.getByRole('button', { name: 'Copy MCP investigation request' }));
  expect(onAction).toHaveBeenLastCalledWith({ kind: 'agent_help' });
  rerender(<GraphOpeningPanel state={opening} connected={false} standalone onAction={onAction} />);
  expect(screen.getByRole('button', { name: 'Copy MCP investigation request' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Inspect source only' })).toBeDisabled();
});

test.each(['selection_required', 'missing_libraries', 'incompatible'] as const)('an environment selection failure replaces stale %s feedback and repair actions', reason => {
  const opening = state(); opening.phase = 'failed'; opening.detail = 'The selected Python changed. Refresh and choose it again.';
  opening.environment!.reason = reason; opening.environment!.compatibilityPackages = ['pydantic==2.12.0'];
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByRole('alert')).toHaveTextContent(opening.detail);
  expect(screen.getByRole('alert')).toBeVisible();
  expect(screen.queryByRole('button', { name: /Review missing|Review compatibility|Install missing/ })).toBeNull();
  expect(screen.queryByText('Blocked · see above')).toBeNull();
});

test('a failed chooser action in needs_setup shows its new error alongside the previous model diagnostic', () => {
  const opening = state(); opening.environment!.reason = 'incompatible';
  opening.environment!.detail = 'Installed torch 2.7.1 does not match the model requirement ==2.0.0.';
  opening.detail = 'Setup could not be saved or resumed. Check unsaved files and tensorviz.toml in VS Code; your form values are retained.';
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByRole('alert')).toHaveTextContent(opening.detail);
  expect(screen.getByText(opening.detail)).toBeVisible();
  expect(screen.getByText(opening.environment!.detail)).not.toBeVisible();
  fireEvent.click(screen.getByText('Setup details'));
  expect(screen.getByText(opening.environment!.detail)).toBeVisible();
});

test('host feedback about a changed candidate is visible without opening Environment details', () => {
  const opening = state(); opening.environment!.reason = 'selection_required';
  opening.environment!.detail = 'This environment changed while the chooser was open. Refresh the list, then select it again.';
  opening.detail = opening.environment!.detail;
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByText(opening.environment!.detail)).toBeVisible();
  expect(screen.queryByRole('heading', { name: /Model check for/ })).toBeNull();
});

test('completed checks reveal their result after scrolling through environments, without interrupting draft selection', () => {
  const opening = state();
  const { rerender } = render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  const body = screen.getByRole('region', { name: 'Python environment for model' }).querySelector<HTMLDivElement>('.graph-environment-body')!;
  body.scrollTop = 240;
  fireEvent.click(screen.getAllByRole('radio')[1]);
  expect(body.scrollTop).toBe(240);
  rerender(<GraphOpeningPanel state={{ ...opening, phase: 'resolving', detail: 'Checking Python environments…' }} connected onAction={jest.fn()} />);
  rerender(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(body.scrollTop).toBe(0);
  body.scrollTop = 240;
  rerender(<GraphOpeningPanel state={{ ...opening, detail: 'The environment could not be saved. Try again.' }} connected onAction={jest.fn()} />);
  expect(body.scrollTop).toBe(0);
});


test('saved-Python repair stays in the footer while alternatives scroll, and creation is secondary', () => {
  const opening = state(); opening.environment!.canCreate = true;
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  const panel = screen.getByRole('region', { name: 'Python environment for model' });
  const body = panel.querySelector('.graph-environment-body')!;
  const repair = screen.getByRole('button', { name: 'Add libraries here…' });
  expect(repair.closest('footer')).not.toBeNull(); expect(body.contains(repair)).toBe(false);
  expect(screen.getByRole('button', { name: 'Create model environment…' })).not.toBeVisible();
  expect(screen.getByText('Model libraries missing · capture tools are installed.').closest('footer')).not.toBeNull();
  fireEvent.click(screen.getByText('Change environment')); body.scrollTop = 900;
  expect(repair).toBeVisible(); fireEvent.click(repair);
  expect(onAction).toHaveBeenCalledWith({ kind: 'install_dependencies' });
});

test('progress and a network failure persist in the footer with a deliberate retry', () => {
  const opening = state(); opening.environment!.canCreate = true;
  const onAction = jest.fn(); const { rerender } = render(<GraphOpeningPanel state={{ ...opening, phase: 'resolving', detail: 'Downloading reviewed libraries…' }} connected onAction={onAction} />);
  expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow');
  expect(screen.getByRole('progressbar').closest('footer')).not.toBeNull();
  rerender(<GraphOpeningPanel state={{ ...opening, phase: 'failed', detail: 'Library download could not connect.', failure: { kind: 'environment', code: 'DOWNLOAD_FAILED' } }} connected onAction={onAction} />);
  expect(screen.getByRole('alert').closest('footer')).not.toBeNull();
  expect(screen.getByRole('alert')).toHaveTextContent('Library download could not connect.');
  expect(screen.queryByRole('progressbar')).toBeNull(); expect(onAction).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Retry library setup…' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'install_dependencies' });
});

test('a platform-unavailable package offers source inspection instead of another primary install or venv', () => {
  const opening = state(); opening.environment!.canCreate = true;
  opening.phase = 'failed'; opening.detail = 'triton>=3.5.0: No distribution for this platform.';
  opening.failure = { kind: 'environment', code: 'PACKAGE_UNAVAILABLE' };
  const onAction = jest.fn(); render(<GraphOpeningPanel state={opening} connected onAction={onAction} />);
  expect(screen.getByRole('alert')).toHaveTextContent('triton>=3.5.0');
  expect(screen.queryByRole('button', { name: 'Retry library setup…' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Create model environment…' })).not.toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Inspect source only' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'source_only' });
});

test('Stop retains the environment and offers an explicit retry without source changes', () => {
  const opening = state(); opening.environment!.canCreate = true; opening.phase = 'cancelled';
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByRole('heading', { name: 'Environment setup stopped' })).toBeVisible();
  expect(screen.getByText('Setup stopped. You can retry when ready.')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Retry library setup…' })).toBeEnabled();
});

// The VS Code host checks every candidate for the model and recommends the one needing the fewest changes.
const ranked = (): GraphOpeningState => {
  const opening = state();
  opening.environment = { reason: 'incompatible', canCreate: true, selectedId: 'saved', recommendedId: 'suggestion',
    detail: 'Installed libraries do not match this model: transformers 4.57.6 (needs >=5). Choose a compatible environment.',
    candidates: [
      { id: 'suggestion', interpreter: '/opt/conda/envs/ml/bin/python', prefix: '/opt/conda/envs/ml', source: 'Known to the Python extension',
        python: '3.12.10', torch: '2.7.1', status: 'libraries_present', modelChecked: true },
      { id: 'saved', interpreter: '/project/.venv/bin/python', source: 'Saved for this folder', selected: true, python: '3.12.13',
        torch: '2.6.0+cu124', status: 'libraries_present', modelChecked: true, modelIncompatible: ['transformers 4.57.6 (needs >=5)'] },
      { id: 'plain', interpreter: '/project/venv/bin/python', source: 'Model’s project folder', python: '3.12.13', torch: '2.6.0',
        status: 'libraries_present', modelChecked: true, modelMissing: ['transformers>=5'] } ] };
  return opening;
};

test('a recommendation opens the list with the environment needing the fewest changes preselected', () => {
  const onAction = jest.fn(); render(<GraphOpeningPanel state={ranked()} connected onAction={onAction} />);
  const radios = screen.getAllByRole('radio');
  expect(radios[0]).toBeVisible(); expect(radios[0]).toBeChecked();
  expect(screen.getByText('Recommended').closest('.graph-environment-entry')).toHaveTextContent('envs/ml');
  expect(screen.getAllByText('Ready for this model')[0]).toBeVisible();
  expect(screen.getByText('To install: transformers>=5')).toBeVisible();
  expect(screen.getByText('To change: transformers 4.57.6 (needs >=5)')).toBeVisible();
  expect(screen.getByRole('heading', { name: 'Create new environment' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Create model environment…' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Use envs/ml instead' }));
  expect(onAction).toHaveBeenCalledWith({ kind: 'select_environment', candidateId: 'suggestion' });
});

test('search filters environments by name, path or version and says when nothing matches', () => {
  render(<GraphOpeningPanel state={ranked()} connected onAction={jest.fn()} />);
  const search = screen.getByRole('searchbox', { name: 'Search environments' });
  fireEvent.change(search, { target: { value: 'conda' } });
  expect(screen.getAllByRole('radio')).toHaveLength(1);
  fireEvent.change(search, { target: { value: 'torch 2.6' } });
  expect(screen.getAllByRole('radio').map(radio => (radio as HTMLInputElement).value)).toEqual(['saved', 'plain']);
  fireEvent.change(search, { target: { value: 'mamba' } });
  expect(screen.queryAllByRole('radio')).toHaveLength(0);
  expect(screen.getByText('No environments match “mamba”.')).toBeVisible();
});

test('without model checks the saved choice stays preselected and the list closed, as for older hosts', () => {
  const opening = ranked(); delete opening.environment!.recommendedId;
  opening.environment!.candidates = opening.environment!.candidates.map(candidate => ({ ...candidate, modelChecked: undefined }));
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getAllByRole('radio', { hidden: true })[1]).toBeChecked();
  expect(screen.getAllByRole('radio', { hidden: true })[1]).not.toBeVisible();
  expect(screen.queryByText('Recommended')).toBeNull();
  fireEvent.click(screen.getByText('Change environment'));
  expect(screen.getAllByText('Capture libraries present').some(node => node.closest('.graph-environment-entry.selected'))).toBe(true);
});

test('a library named by both the capture and model checks is listed once, without its marker', () => {
  const opening = ranked();
  opening.environment!.candidates[2] = { ...opening.environment!.candidates[2], status: 'missing_libraries', torch: null,
    missing: ['torch', 'tomli>=2.0; python_version < "3.11"'], modelMissing: ['torch', 'transformers>=5'] };
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByText('To install: torch, tomli>=2.0, transformers>=5')).toBeVisible();
});

test('the ranked list stays open when the recommended environment is already saved', () => {
  const opening = ranked(); opening.environment!.recommendedId = 'saved';
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByRole('heading', { name: 'Choose Python for this model' })).toBeVisible();
  expect(screen.getByRole('searchbox', { name: 'Search environments' })).toBeVisible();
  expect(screen.getAllByRole('radio')[1]).toBeChecked();
  expect(screen.getByRole('heading', { name: 'Create new environment' })).toBeVisible();
  const panel = screen.getByRole('region', { name: 'Python environment for model' });
  expect(panel.querySelector('.graph-environment-choices')!.compareDocumentPosition(screen.getByRole('heading', { name: 'Create new environment' })) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
});

test('an explicit saved choice keeps the ranked alternatives visible without selecting a ready alternative', () => {
  const opening = ranked(); delete opening.environment!.recommendedId;
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByRole('searchbox', { name: 'Search environments' })).toBeVisible();
  expect(screen.getAllByRole('radio')[1]).toBeChecked();
  expect(screen.getAllByRole('radio')[0]).not.toBeChecked();
  expect(screen.getByText('Ready for this model')).toBeVisible();
});

test('Change environment opens the ranked list even when the saved ready environment remains the recommendation', () => {
  const opening = ranked(); opening.environment!.reason = 'selection_required'; opening.environment!.selectedId = 'suggestion';
  render(<GraphOpeningPanel state={opening} connected onAction={jest.fn()} />);
  expect(screen.getByRole('searchbox', { name: 'Search environments' })).toBeVisible();
  expect(screen.getAllByRole('radio')[0]).toBeChecked();
});
