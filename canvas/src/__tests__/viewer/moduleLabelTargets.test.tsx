import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import CaptureComparisonPanel from '../../viewer/CaptureComparisonPanel';
import * as captureComparison from '../../services/capturedGraphComparison';
import * as offline from '../../viewer/offlineCaptureClient';
import { compareGraphFiles } from '../../services/graphComparison';
import { capturedFile, renameCaptureFile, operationNode } from '../../test/fixtures/capturedGraphFileFixture';
import type { CaptureRecordChange } from '../../ide/captureComparison.generated';

function pair() {
  const before = capturedFile(), after = renameCaptureFile(before);
  const result = compareGraphFiles(before, after).capture!;
  result.changes = []; result.observations = []; result.unresolved = []; result.controls = [];
  result.valueUses.changes = []; result.valueUses.unresolved = [];
  for (const side of ['before', 'after'] as const) {
    for (const kind of Object.keys(result.correspondence.unmatched[side])) {
      (result.correspondence.unmatched[side] as any)[kind] = [];
    }
  }
  const ids = { before: before.capture!.capture.modules[0].id, after: after.capture!.capture.modules[0].id };
  result.correspondence.unmatched.before.module = [ids.before];
  result.correspondence.unmatched.after.module = [ids.after];
  return { before, after, result, onNavigate: jest.fn(), ids };
}
const rows = () => within(screen.getByRole('region', { name: 'Unmatched records' })).getAllByRole('listitem');
function forwardedClients() {
  const create = offline.createOfflineCaptureClient;
  const clients: Array<{ inspect: jest.SpyInstance; dispose: jest.SpyInstance }> = [];
  jest.spyOn(offline, 'createOfflineCaptureClient').mockImplementation(capture => {
    const client = create(capture);
    clients.push({ inspect: jest.spyOn(client, 'inspect'), dispose: jest.spyOn(client, 'dispose') });
    return client;
  });
  return clients;
}
beforeAll(() => { HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); }; });
afterEach(() => { jest.restoreAllMocks(); });

it('omits only discarded module-label resolution while real navigation and module inspection remain intact', async () => {
  const props = pair();
  props.before.capture!.capture.modules[0].paths = ['', 'encoder', 'ignored'];
  props.after.capture!.capture.modules[0].paths = ['', 'encoder', 'ignored'];
  const expectedBefore = captureComparison.capturedRecordTarget(props.before, 'module', props.ids.before);
  const expectedAfter = captureComparison.capturedRecordTarget(props.after, 'module', props.ids.after);
  const targets = jest.spyOn(captureComparison, 'capturedRecordTarget'); // Calls the real resolver.
  const clients = forwardedClients();
  render(<CaptureComparisonPanel {...props} />);
  expect(targets.mock.calls).toEqual([
    [props.before, 'module', props.ids.before], [props.after, 'module', props.ids.after],
    [props.before, 'module', props.ids.before], [props.after, 'module', props.ids.after],
  ]);
  expect(targets.mock.results.map(result => result.value)).toEqual([expectedBefore, expectedAfter, expectedBefore, expectedAfter]);
  const [beforeRow, afterRow] = rows();
  for (const row of [beforeRow, afterRow]) {
    expect(within(row).getByText('Module (root) / encoder')).toBeInTheDocument();
    expect(row).not.toHaveTextContent('ignored');
  }
  fireEvent.click(within(beforeRow).getByRole('button', { name: 'Show before' }));
  fireEvent.click(within(afterRow).getByRole('button', { name: 'Show after' }));
  expect(props.onNavigate.mock.calls).toEqual([['before', expectedBefore], ['after', expectedAfter]]);
  fireEvent.click(within(beforeRow).getByRole('button', { name: 'Inspect before' }));
  const dialog = await screen.findByRole('dialog', { name: 'Captured details: Before · Module (root) / encoder' });
  await waitFor(() => expect(clients[0].inspect).toHaveBeenCalledWith({
    captureId: props.before.capture!.capture.id, sourceId: props.before.capture!.capture.source.id,
    path: ['modules', 0], offset: 0, limit: 25,
  }));
  expect(within(dialog).getByText('Capture.modules[0]')).toBeInTheDocument();
  expect(targets).toHaveBeenCalledTimes(4);
});

it.each(['changes', 'observations', 'unresolved'] as const)('preserves module markup and navigation for %s, including UTF-16 truncation', section => {
  const props = pair();
  props.result.correspondence.unmatched.before.module = []; props.result.correspondence.unmatched.after.module = [];
  const path = 'x'.repeat(192) + '😀tail';
  props.after.capture!.capture.modules[0].paths = [path, 'second', 'ignored'];
  const expected = `Module ${path} / second`.slice(0, 200);
  expect(expected.length).toBe(200); expect(expected.charCodeAt(199)).toBe(0xd83d);
  props.result[section] = [{ kind: 'module', beforeId: props.ids.before, afterId: props.ids.after, fields: [] } as CaptureRecordChange];
  const targets = jest.spyOn(captureComparison, 'capturedRecordTarget');
  render(<CaptureComparisonPanel {...props} />);
  const title = { changes: 'Computation changes', observations: 'Recorded observations', unresolved: 'Unresolved fields' }[section];
  const article = within(screen.getByRole('region', { name: title })).getByRole('article');
  expect(article.getAttribute('aria-label')).toBe(`${title}: ${expected}`);
  expect(within(article).getByRole('heading').textContent).toBe(expected);
  expect(targets).toHaveBeenCalledTimes(4);
  fireEvent.click(within(article).getByRole('button', { name: 'Show after' }));
  expect(props.onNavigate).toHaveBeenLastCalledWith('after', targets.mock.results[1].value);
});

it.each([[], undefined])('keeps the real resolver label fallback for empty or missing paths: %p', paths => {
  const props = pair();
  (props.after.capture!.capture.modules[0] as any).paths = paths;
  const target = captureComparison.capturedRecordTarget(props.after, 'module', props.ids.after)!;
  const node = props.after.document.semantic.nodes.find(node => node.node_id === target.nodeId)!;
  node.metadata.captureLabel = 'Fallback module label';
  const targets = jest.spyOn(captureComparison, 'capturedRecordTarget');
  render(<CaptureComparisonPanel {...props} />);
  expect(within(rows()[1]).getByText('Fallback module label')).toBeInTheDocument();
  // Before: two navigation calls; After: two navigation plus three fallback-label calls.
  expect(targets.mock.calls.filter(([, , id]) => id === props.ids.before)).toHaveLength(2);
  expect(targets.mock.calls.filter(([, , id]) => id === props.ids.after)).toHaveLength(5);
  fireEvent.click(within(rows()[1]).getByRole('button', { name: 'Show after' }));
  expect(props.onNavigate).toHaveBeenLastCalledWith('after', target);
});

it('refreshes module paths and navigation targets after a direct same-reference rerender', () => {
  const props = pair(), clients = forwardedClients();
  const targets = jest.spyOn(captureComparison, 'capturedRecordTarget');
  const { rerender } = render(<CaptureComparisonPanel {...props} />);
  const target = targets.mock.results[1].value!;
  const node = props.after.document.semantic.nodes.find(node => node.node_id === target.nodeId)!;
  node.node_id = 'replacement-module-card';
  props.after.capture!.capture.modules[0].paths.splice(0, 1, 'updated', 'alias', 'ignored');
  targets.mockClear();
  rerender(<CaptureComparisonPanel {...props} />);
  expect(within(rows()[1]).getByText('Module updated / alias')).toBeInTheDocument();
  expect(targets).toHaveBeenCalledTimes(2);
  expect(clients).toHaveLength(2);
  fireEvent.click(within(rows()[1]).getByRole('button', { name: 'Show after' }));
  expect(props.onNavigate).toHaveBeenLastCalledWith('after', { kind: 'node', nodeId: 'replacement-module-card' });
});

it('retains first-match projection resolution and disables ambiguous, absent and missing-module targets', () => {
  const props = pair();
  const target = captureComparison.capturedRecordTarget(props.after, 'module', props.ids.after)!;
  const node = props.after.document.semantic.nodes.find(node => node.node_id === target.nodeId)!;
  props.after.document.semantic.nodes.unshift({ ...node, node_id: 'first-duplicate' });
  const targets = jest.spyOn(captureComparison, 'capturedRecordTarget');
  const { rerender } = render(<CaptureComparisonPanel {...props} />);
  fireEvent.click(within(rows()[1]).getByRole('button', { name: 'Show after' }));
  expect(props.onNavigate).toHaveBeenLastCalledWith('after', { kind: 'node', nodeId: 'first-duplicate' });
  const capture = props.after.capture!.capture;
  capture.invocations.push({ ...capture.invocations[0], id: 'second-call' });
  rerender(<CaptureComparisonPanel {...props} />);
  expect(within(rows()[1]).getByRole('button', { name: 'Show after' })).toBeDisabled();
  capture.invocations.pop();
  props.after.document.semantic.nodes = props.after.document.semantic.nodes.filter(candidate => candidate.metadata.captureRef !== node.metadata.captureRef);
  rerender(<CaptureComparisonPanel {...props} />);
  expect(within(rows()[1]).getByRole('button', { name: 'Show after' })).toBeDisabled();
  capture.invocations.length = 0;
  rerender(<CaptureComparisonPanel {...props} />);
  expect(within(rows()[1]).getByRole('button', { name: 'Show after' })).toBeDisabled();
  props.result.correspondence.unmatched.after.module = ['missing-module'];
  rerender(<CaptureComparisonPanel {...props} />);
  expect(within(rows()[1]).getByText('missing-module')).toBeInTheDocument();
  expect(within(rows()[1]).getByRole('button', { name: 'Show after' })).toBeDisabled();
  expect(targets.mock.calls.some(([, , id]) => id === 'missing-module')).toBe(true);
});

it('preserves ordinary operation, tensor and capture labels and their inspection queries', async () => {
  const props = pair(), clients = forwardedClients();
  const capture = props.after.capture!.capture;
  const op = operationNode(props.after, capture.operations[0].id); op.metadata.captureLabel = 'Ordinary operation';
  props.result.correspondence.unmatched.before.module = []; props.result.correspondence.unmatched.after.module = [];
  Object.assign(props.result.correspondence.unmatched.after, {
    operation: [capture.operations[0].id], value: [capture.values[0].id], capture: [capture.id],
  });
  const targets = jest.spyOn(captureComparison, 'capturedRecordTarget');
  render(<CaptureComparisonPanel {...props} />);
  for (const [kind, id, label, query] of [
    ['operation', capture.operations[0].id, 'Ordinary operation', { entity: { kind: 'operation', id: capture.operations[0].id }, path: [] }],
    ['value', capture.values[0].id, `Tensor ${capture.values[0].id}`, { entity: { kind: 'value', id: capture.values[0].id }, path: [] }],
    ['capture', capture.id, 'Whole capture', { path: [] }],
  ] as const) {
    const row = rows().find(row => within(row).queryByText(`${kind} · ${id}`))!;
    expect(within(row).getByText(label)).toBeInTheDocument();
    fireEvent.click(within(row).getByRole('button', { name: 'Inspect after' }));
    const dialog = await screen.findByRole('dialog', { name: `Captured details: After · ${label}` });
    await waitFor(() => expect(clients[1].inspect).toHaveBeenLastCalledWith(expect.objectContaining(query)));
    fireEvent.click(within(dialog).getByRole('button', { name: 'Close captured details' }));
  }
  expect(targets.mock.calls.filter(([, kind]) => kind === 'capture')).toHaveLength(2);
});

it('still rejects malformed projection references through the real navigation resolver despite valid module paths', () => {
  const props = pair();
  props.before.document.semantic.nodes[0].metadata.captureRef = { version: 99 };
  const targets = jest.spyOn(captureComparison, 'capturedRecordTarget');
  const errors = jest.spyOn(console, 'error').mockImplementation(() => {});
  expect(() => captureComparison.capturedRecordTarget(props.before, 'module', props.ids.before)).toThrow();
  targets.mockClear();
  expect(() => render(<CaptureComparisonPanel {...props} />)).toThrow();
  expect(targets).toHaveBeenCalledWith(props.before, 'module', props.ids.before);
  expect(targets.mock.results.every(result => result.type === 'throw')).toBe(true);
  errors.mockRestore();
});

it('keeps StrictMode client creation and disposal intact', () => {
  const props = pair(), clients = forwardedClients();
  const { unmount } = render(<React.StrictMode><CaptureComparisonPanel {...props} /></React.StrictMode>);
  expect(clients.length).toBeGreaterThanOrEqual(4);
  expect(clients.filter(client => client.dispose.mock.calls.length === 0)).toHaveLength(2);
  unmount();
  for (const client of clients) expect(client.dispose).toHaveBeenCalledTimes(1);
});
