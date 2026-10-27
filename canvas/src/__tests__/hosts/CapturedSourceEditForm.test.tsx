import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import unaryInsertFixture from '../../../../tests/fixtures/contracts/captured-unary-insert.json';
import unaryImportFixture from '../../../../tests/fixtures/contracts/captured-unary-import.json';
import unaryDeleteFixture from '../../../../tests/fixtures/contracts/captured-unary-delete.json';
import reconnectionFixture from '../../../../tests/fixtures/contracts/captured-reconnection-edit.json';
import fixture from '../../../../tests/fixtures/contracts/captured-source-edit.json';
import activationFixture from '../../../../tests/fixtures/contracts/captured-activation-edit.json';
import moduleFixture from '../../../../tests/fixtures/contracts/captured-module-activation-edit.json';
import activationImportFixture from '../../../../tests/fixtures/contracts/captured-activation-import.json';
import moduleImportFixture from '../../../../tests/fixtures/contracts/captured-module-activation-import.json';
import constructorFixture from '../../../../tests/fixtures/contracts/captured-constructor-edit.json';
import customConstructorFixture from '../../../../tests/fixtures/contracts/captured-custom-constructor.json';
import constructorKeywordFixture from '../../../../tests/fixtures/contracts/captured-constructor-keyword.json';
import configurationFixture from '../../../../tests/fixtures/contracts/captured-configuration-edit.json';
import configurationModuleFixture from '../../../../tests/fixtures/contracts/captured-configuration-module.json';
import activationOptionsFixture from '../../../../tests/fixtures/contracts/captured-activation-options.json';
import moduleOptionsFixture from '../../../../tests/fixtures/contracts/captured-module-activation-options.json';
import { graphDocumentV2ToModelGraph, modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import type { GraphDocumentV2 } from '../../types';
import { SourceEditForm } from '../../ide/SourceEditForm';
import { parseSourceNumber } from '../../ide/CapturedSourceEditForm';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE } from '../../ide/localBrowserState';
import { SourceEditSession } from '../../ide/SourceEditSession';
import type { ProjectionPayload } from '../../ide/protocol.generated';

const editFixtures = { 'unary-insert': unaryInsertFixture, 'unary-import': unaryImportFixture, 'unary-delete': unaryDeleteFixture, reconnection: reconnectionFixture, literal: fixture, activation: activationFixture, module: moduleFixture,
  'activation-import': activationImportFixture, 'module-import': moduleImportFixture,
  constructor: constructorFixture, 'constructor-keyword': constructorKeywordFixture, 'constructor-custom': customConstructorFixture,
  configuration: configurationFixture, 'configuration-module': configurationModuleFixture,
  'activation-options': activationOptionsFixture, 'module-options': moduleOptionsFixture };

function setup(stale = false, data: (typeof editFixtures)[keyof typeof editFixtures] = fixture) {
  const graph = graphDocumentV2ToModelGraph(data.document as unknown as GraphDocumentV2);
  const node = graph.nodes.find(n => n.data.captureRef?.semanticId === data.intent.target.semanticId)!;
  const propose = jest.fn((_edit: unknown) => ({ requestId: 'literal-edit', promise: Promise.resolve({ status: 'applied' as const,
    detail: 'Applied to the VS Code buffer. Runtime validation has not run.' }) }));
  const cancel = jest.fn();
  render(<LocalBrowserContext.Provider value={{ sourceEdits: { propose, cancel, dispose: jest.fn() },
    page: { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, projection: { document: data.document } as ProjectionPayload,
      ...(stale ? { syncState: 'stale' as const } : {}) },
    status: { phase: 'connected', hostLink: 'connected', error: null },
  }}><SourceEditForm nodes={[node]} /></LocalBrowserContext.Provider>);
  return { node, graph, propose, cancel };
}

test('real Python capability survives graph conversion and enables only its exact literal recipe', async () => {
  const { node, graph, propose } = setup();
  const restored = modelGraphToDocumentV2(graph, { name: 'Model', revision: 1 });
  expect(restored.semantic.nodes.find(n => n.node_id === node.id)?.metadata.sourceEditCapabilities).toEqual(node.data.sourceEditCapabilities);
  expect(screen.queryByLabelText('Source operation')).not.toBeInTheDocument();
  expect(screen.getByText('every execution of this source expression')).toBeVisible();
  expect(screen.getByText(/including calls not shown/)).toBeVisible();
  fireEvent.change(screen.getByLabelText('New literal'), { target: { value: '2' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).toHaveBeenCalledWith({ ...fixture.intent, nodeId: node.id });
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Runtime validation has not run'));
  expect(node.data.parameters).not.toHaveProperty('value', 2);
});

test('reconnection distinguishes repeated operands, resets drafts, and submits only the listed value', async () => {
  const { node, graph, propose } = setup(false, reconnectionFixture);
  const restored = modelGraphToDocumentV2(graph, { name: 'Model', revision: 1 });
  expect(restored.semantic.nodes.find(n => n.node_id === node.id)?.metadata.sourceEditCapabilities).toEqual(node.data.sourceEditCapabilities);
  expect(Array.from(screen.getByLabelText('Captured edit').querySelectorAll('option'), o => o.textContent))
    .toEqual(['Reconnect operand 1 (a)', 'Reconnect operand 2 (a)', 'Insert activation on operand 1 (a)', 'Insert activation on operand 2 (a)']);
  expect(screen.getByText(/Only the selected operand changes/)).toBeVisible();
  expect(screen.getByText(/does not guarantee compatible dimensions/)).toBeVisible();
  expect(screen.queryByLabelText('New literal')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('New activation')).not.toBeInTheDocument();
  expect(Array.from(screen.getByLabelText('Available tensor').querySelectorAll('option'), o => o.textContent?.trim()))
    .toEqual(['a — defined on line 12 (current)', 'x — defined on line 9', 'h — defined on line 11']);
  fireEvent.change(screen.getByLabelText('Available tensor'), { target: { value: reconnectionFixture.intent.value.id } });
  fireEvent.change(screen.getByLabelText('Captured edit'), { target: { value: reconnectionFixture.capability.id } });
  expect(screen.getByLabelText('Available tensor')).toHaveValue(reconnectionFixture.intent.expected.id);
  fireEvent.change(screen.getByLabelText('Available tensor'), { target: { value: reconnectionFixture.intent.value.id } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(screen.getByLabelText('Captured edit')).toBeDisabled();
  expect(screen.getByLabelText('Available tensor')).toBeDisabled();
  expect(propose).toHaveBeenCalledWith({ ...reconnectionFixture.intent, nodeId: node.id });
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Runtime validation has not run'));
  fireEvent.change(screen.getByLabelText('Captured edit'), { target: { value: node.data.sourceEditCapabilities![0].id } });
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
  expect(screen.getByLabelText('Available tensor')).toHaveValue(reconnectionFixture.intent.expected.id);
});

test('mixed literal and reconnection controls reset each other and no-op stays exact', async () => {
  const { node, propose } = setup();
  const reconnect = node.data.sourceEditCapabilities!.find(cap => cap.command === 'reconnect_argument')!;
  fireEvent.change(screen.getByLabelText('New literal'), { target: { value: '99' } });
  fireEvent.change(screen.getByLabelText('Captured edit'), { target: { value: reconnect.id } });
  expect(screen.queryByLabelText('New literal')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).toHaveBeenCalledWith({ kind: 'reconnect_argument', nodeId: node.id, capabilityId: reconnect.id,
    target: reconnect.target, argumentPath: ['args', 0], expected: reconnect.expected, value: reconnect.expected, scope: 'source_expression' });
  await waitFor(() => expect(screen.getByRole('status')).toBeInTheDocument());
  fireEvent.change(screen.getByLabelText('Captured edit'), { target: { value: fixture.capability.id } });
  expect(screen.getByLabelText('New literal')).toHaveValue('1');
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});

test('stale reconnection and unavailable tensor choices cannot submit', () => {
  const { propose } = setup(true, reconnectionFixture);
  expect(screen.getByLabelText('Captured edit')).toBeDisabled();
  expect(screen.getByLabelText('Available tensor')).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Available tensor'), { target: { value: 'unavailable' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).not.toHaveBeenCalled();
});

test.each([unaryInsertFixture, unaryImportFixture])('unary insertion selects an exact use, exposes closed activations and discloses its import', async data => {
  const { node, propose } = setup(false, data);
  fireEvent.change(screen.getByLabelText('Captured edit'), { target: { value: data.capability.id } });
  expect(screen.getByText(/adds a nonmutating activation to only the selected operand/)).toBeVisible();
  expect(screen.queryByLabelText('Available tensor')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('New literal')).not.toBeInTheDocument();
  if (data === unaryImportFixture) expect(screen.getByText(/uniquely named Torch import/)).toBeVisible();
  expect(Array.from(screen.getByLabelText('New activation').querySelectorAll('option'), o => o.textContent)).toEqual(['relu', 'gelu', 'silu', 'sigmoid', 'tanh']);
  fireEvent.change(screen.getByLabelText('New activation'), { target: { value: data.intent.value } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).toHaveBeenCalledWith({ ...data.intent, nodeId: node.id });
  expect(screen.getByLabelText('Captured edit')).toBeDisabled();
  expect(screen.getByLabelText('New activation')).toBeDisabled();
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Runtime validation has not run'));
});

test('unary deletion exposes only the original input and resets activation drafts when changing recipe', async () => {
  const { node, propose } = setup(false, unaryDeleteFixture);
  fireEvent.change(screen.getByLabelText('New activation'), { target: { value: 'tanh' } });
  fireEvent.change(screen.getByLabelText('Captured edit'), { target: { value: unaryDeleteFixture.capability.id } });
  expect(screen.getByText(/removes the activation call and uses its original input/)).toHaveTextContent('h');
  expect(screen.queryByLabelText('New activation')).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Available tensor')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).toHaveBeenCalledWith({ ...unaryDeleteFixture.intent, nodeId: node.id });
  expect(screen.getByLabelText('Captured edit')).toBeDisabled();
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Runtime validation has not run'));
  fireEvent.change(screen.getByLabelText('Captured edit'), { target: { value: node.data.sourceEditCapabilities![0].id } });
  expect(screen.getByLabelText('New activation')).toHaveValue('gelu');
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});

test.each([unaryInsertFixture, unaryDeleteFixture])('stale unary fields cannot submit', data => {
  const { propose } = setup(true, data);
  fireEvent.change(screen.getByLabelText('Captured edit'), { target: { value: data.capability.id } });
  expect(screen.getByLabelText('Captured edit')).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).not.toHaveBeenCalled();
});

test.each([activationFixture, moduleFixture, activationImportFixture, moduleImportFixture])('real activation capability renders closed choices and sends only a reviewed activation intent', async data => {
  const { node, graph, propose } = setup(false, data);
  const restored = modelGraphToDocumentV2(graph, { name: 'Model', revision: 1 });
  expect(restored.semantic.nodes.find(n => n.node_id === node.id)?.metadata.sourceEditCapabilities).toEqual(node.data.sourceEditCapabilities);
  if (data.capability.recipe === 'module_activation_v1') expect(screen.getByText(/replaces the module constructor/)).toBeVisible();
  if (data.capability.importPlan) expect(screen.getByText(/uniquely named Torch import/)).toBeVisible();
  expect(screen.queryByLabelText('New literal')).not.toBeInTheDocument();
  expect(screen.getByText('every execution of this source expression')).toBeVisible();
  expect(Array.from(screen.getByLabelText('New activation').querySelectorAll('option'), option => option.textContent)).toEqual(['relu', 'gelu', 'silu', 'sigmoid', 'tanh']);
  fireEvent.change(screen.getByLabelText('New activation'), { target: { value: 'gelu' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).toHaveBeenCalledWith({ ...data.intent, nodeId: node.id });
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Runtime validation has not run'));
});

test.each([constructorFixture, constructorKeywordFixture, customConstructorFixture])('constructor field selection sends the exact source argument path', async data => {
  const { node, graph, propose } = setup(false, data);
  const restored = modelGraphToDocumentV2(graph, { name: 'Model', revision: 1 });
  expect(restored.semantic.nodes.find(n => n.node_id === node.id)?.metadata.sourceEditCapabilities).toEqual(node.data.sourceEditCapabilities);
  expect(screen.getByText(/checkpoint compatibility/)).toBeVisible();
  const labels = screen.getByLabelText('Constructor argument').querySelectorAll('option');
  expect(Array.from(labels, option => option.textContent)).toEqual(data !== constructorKeywordFixture ? ['Argument 1 = 4', 'Argument 2 = 4'] : ['p = 0.25']);
  // A draft for the first field must never be carried into a different field.
  fireEvent.change(screen.getByLabelText('New literal'), { target: { value: '99' } });
  if (data !== constructorKeywordFixture) {
    fireEvent.change(screen.getByLabelText('Constructor argument'), { target: { value: data.capability.id } });
    expect(screen.getByLabelText('New literal')).toHaveValue('4');
  } else expect(screen.getByText(/Observed in 0 operations/)).toBeVisible();
  fireEvent.change(screen.getByLabelText('New literal'), { target: { value: String(data.intent.value.value) } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(screen.getByLabelText('Constructor argument')).toBeDisabled();
  expect(propose).toHaveBeenCalledWith({ ...data.intent, nodeId: node.id });
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Runtime validation has not run'));
  if (data !== constructorKeywordFixture) {
    fireEvent.change(screen.getByLabelText('Constructor argument'), { target: { value: node.data.sourceEditCapabilities![0].id } });
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.getByLabelText('New literal')).toHaveValue('4');
  }
});

test.each([activationOptionsFixture, moduleOptionsFixture])('activation choices preserve the authored inplace option and disclose alias mutation', async data => {
  const { node, propose } = setup(false, data);
  expect(Array.from(screen.getByLabelText('New activation').querySelectorAll('option'), option => option.textContent)).toEqual(['relu', 'silu']);
  expect(screen.getByText(/inplace=True/)).toBeVisible();
  expect(screen.getByText(/effect on aliases/)).toBeVisible();
  fireEvent.change(screen.getByLabelText('New activation'), { target: { value: 'gelu' } });
  expect(screen.getByLabelText('New activation')).not.toHaveValue('gelu');
  fireEvent.change(screen.getByLabelText('New activation'), { target: { value: 'silu' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).toHaveBeenCalledWith({ ...data.intent, nodeId: node.id });
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Runtime validation has not run'));
});

test('GELU approximation never presents incompatible activation choices', () => {
  const data = JSON.parse(JSON.stringify(activationFixture));
  const node = data.document.semantic.nodes.find((n: any) => n.metadata.captureRef.semanticId === data.intent.target.semanticId);
  const cap = node.metadata.sourceEditCapabilities[0];
  cap.expected = 'gelu'; cap.options = { inplace: null, approximate: 'tanh' }; cap.compatibleActivations = ['gelu'];
  setup(false, data);
  expect(Array.from(screen.getByLabelText('New activation').querySelectorAll('option'), option => option.textContent)).toEqual(['gelu']);
  expect(screen.getByText(/approximate=/)).toHaveTextContent('tanh');
  expect(screen.getByText(/Only GELU accepts this option/)).toBeVisible();
});

test('partially published constructor fields remain honest and stale fields cannot submit', () => {
  const data = JSON.parse(JSON.stringify(constructorFixture));
  const node = data.document.semantic.nodes.find((n: any) => n.metadata.captureRef.semanticId === data.intent.target.semanticId);
  node.metadata.sourceEditCapabilities = node.metadata.sourceEditCapabilities.slice(0, 1);
  node.metadata.sourceEditLimitation = 'capability_limit';
  const { propose } = setup(true, data);
  expect(screen.getByText(/Some editable fields are unavailable/)).toBeVisible();
  expect(screen.getByLabelText('Constructor argument')).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).not.toHaveBeenCalled();
});

test.each([configurationFixture, configurationModuleFixture])('configuration fields show definition scope and preserve exact reference identity', async data => {
  const { node, propose } = setup(false, data);
  expect(screen.getByText('all references to this configuration definition')).toBeVisible();
  expect(screen.queryByText('every execution of this source expression')).not.toBeInTheDocument();
  expect(screen.getByText(/known source references; 2 observed constructor sites/)).toBeVisible();
  expect(screen.getByText(/Definition: model.py:/)).toHaveTextContent(data.capability.binding.anchors[0].scope);
  expect(screen.getByText(/checkpoint compatibility/)).toBeVisible();
  const labels = Array.from(screen.getByLabelText('Constructor argument').querySelectorAll('option'), option => option.textContent);
  expect(labels).toEqual(data === configurationFixture
    ? ['Argument 1 = width (4; edit definition)', 'Argument 2 = width (4; edit definition)']
    : ['in_features = width (4; edit definition)', 'out_features = 4']);
  fireEvent.change(screen.getByLabelText('New literal'), { target: { value: '99' } });
  const other = node.data.sourceEditCapabilities!.find(cap => cap.id !== data.capability.id)!;
  fireEvent.change(screen.getByLabelText('Constructor argument'), { target: { value: other.id } });
  fireEvent.change(screen.getByLabelText('Constructor argument'), { target: { value: data.capability.id } });
  expect(screen.getByLabelText('New literal')).toHaveValue('4');
  fireEvent.change(screen.getByLabelText('New literal'), { target: { value: '8' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(screen.getByLabelText('Constructor argument')).toBeDisabled();
  expect(propose).toHaveBeenCalledWith({ ...data.intent, nodeId: node.id });
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Runtime validation has not run'));
  fireEvent.change(screen.getByLabelText('Constructor argument'), { target: { value: other.id } });
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
  expect(screen.getByLabelText('New literal')).toHaveValue('4');
  if (data === configurationModuleFixture) {
    expect(screen.getByText('every execution of this source expression')).toBeVisible();
    expect(screen.queryByText('all references to this configuration definition')).not.toBeInTheDocument();
  }
});

test('stale configuration capability cannot be proposed', () => {
  const { propose } = setup(true, configurationFixture);
  expect(screen.getByLabelText('Constructor argument')).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).not.toHaveBeenCalled();
});

test('literal type is explicit and integer text is never rounded through JavaScript', async () => {
  const { propose } = setup();
  fireEvent.change(screen.getByLabelText('New literal'), { target: { value: '9007199254740993' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose.mock.calls[0][0]).toEqual(expect.objectContaining({ value: { kind: 'int', value: '9007199254740993' } }));
  await waitFor(() => expect(screen.getByRole('status')).toBeInTheDocument());
  fireEvent.change(screen.getByLabelText('Literal number type'), { target: { value: 'float' } });
  fireEvent.change(screen.getByLabelText('New literal'), { target: { value: '-0.0' } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose.mock.calls[1][0]).toEqual(expect.objectContaining({ value: { kind: 'float', value: '-0' } }));
  await waitFor(() => expect(screen.getByRole('status')).toBeInTheDocument());
});

test('invalid numbers and stale graphs cannot submit a captured proposal', () => {
  const { propose } = setup(true);
  expect(screen.getByRole('button', { name: 'Review in VS Code' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  expect(propose).not.toHaveBeenCalled();
  for (const value of ['', 'danger()', '1 + 2', 'NaN', 'Infinity', '1e999', '0x1']) {
    expect(parseSourceNumber(value, 'float')).toBeNull();
  }
  expect(parseSourceNumber('1.5', 'int')).toBeNull();
  expect(parseSourceNumber('2', 'float')).toEqual({ kind: 'float', value: 2 });
});

test.each(Object.entries(editFixtures))('same-model source refresh preserves the pending %s form and its late Apply receipt', async (recipe, data) => {
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); };
  const label = data.intent.kind === 'delete_unary_op' ? 'Captured edit' : data.intent.kind === 'reconnect_argument' ? 'Available tensor' : data.intent.kind === 'replace_activation' || data.intent.kind === 'insert_unary_op' ? 'New activation' : 'New literal';
  const replacement = data.intent.kind === 'delete_unary_op' ? data.capability.id : data.intent.kind === 'reconnect_argument' ? data.intent.value.id : String(data.intent.value instanceof Object ? data.intent.value.value : data.intent.value);
  const graph = graphDocumentV2ToModelGraph(data.document as unknown as GraphDocumentV2);
  const node = graph.nodes.find(n => n.data.captureRef?.semanticId === data.intent.target.semanticId)!;
  let finish!: (value: { status: 'applied'; detail: string }) => void;
  const sourceEdits = { propose: jest.fn(() => ({ requestId: 'pending', promise: new Promise<{ status: 'applied'; detail: string }>(resolve => { finish = resolve; }) })), cancel: jest.fn(), dispose: jest.fn() };
  function Harness({ refreshed = false }) {
    return <LocalBrowserContext.Provider value={{ sourceEdits,
      page: { ...EMPTY_LOCAL_BROWSER_PAGE_STATE, projection: refreshed ? null : { document: data.document } as ProjectionPayload },
      status: { phase: 'connected', hostLink: 'connected', error: null },
    }}><SourceEditSession nodes={refreshed ? [] : [node]} onClose={() => {}} /></LocalBrowserContext.Provider>;
  }
  const mounted = render(<Harness />);
  if (data.intent.kind === 'set_constructor_literal' || data.intent.kind === 'set_configuration_literal') fireEvent.change(screen.getByLabelText('Constructor argument'), { target: { value: data.capability.id } });
  if (['reconnect_argument', 'insert_unary_op', 'delete_unary_op'].includes(data.intent.kind)) fireEvent.change(screen.getByLabelText('Captured edit'), { target: { value: data.capability.id } });
  sourceEdits.cancel.mockClear(); // Field selection disposed the previous, idle field.
  fireEvent.change(screen.getByLabelText(label), { target: { value: replacement } });
  fireEvent.click(screen.getByRole('button', { name: 'Review in VS Code' }));
  mounted.rerender(<Harness refreshed />);
  expect(sourceEdits.cancel).not.toHaveBeenCalled();
  expect(screen.getByLabelText(label)).toHaveValue(replacement);
  finish({ status: 'applied', detail: 'Applied to the VS Code buffer. Runtime validation has not run.' });
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Applied to the VS Code buffer'));
  expect(screen.getByRole('button', { name: 'Review in VS Code' })).toBeDisabled();
  expect(screen.getByText(/no longer displayed/)).toBeVisible();
});
