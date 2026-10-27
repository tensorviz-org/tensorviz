import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import ComparisonPanel from '../../viewer/ComparisonPanel';
import { capturedRecordTarget, capturedUseTarget, comparisonCounterpart } from '../../services/capturedGraphComparison';
import { compareGraphFiles } from '../../services/graphComparison';
import { edgeTarget } from '../../services/graphCollaboration';
import { capturedFile, validatedCaptureFile, renameCaptureFile, operationNode, bindOperation, replaceOperand } from '../../test/fixtures/capturedGraphFileFixture';
import { capturePorts } from '../../services/portableCaptureDocument';

beforeAll(() => { HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); }; });

it('compares literal changes across renamed captures and focuses the exact cards', () => {
  const before = capturedFile(), after = renameCaptureFile(before), navigate = jest.fn();
  after.capture!.capture.operations[0].arguments.args[1] = { kind: 'int', value: '2' };
  render(<ComparisonPanel before={before} after={validatedCaptureFile(after)} side="after" onNavigate={navigate} onChoose={() => {}} onClose={() => {}} />);
  expect(screen.getByRole('region', { name: 'Comparison summary' })).toHaveTextContent('Captured computation differs');
  const changes = within(screen.getByRole('region', { name: 'Computation changes' }));
  expect(changes.getByText(/"value": "1"/)).toBeInTheDocument();
  expect(changes.getByText(/"value": "2"/)).toBeInTheDocument();
  fireEvent.click(changes.getByRole('button', { name: 'Show before' }));
  expect(navigate).toHaveBeenLastCalledWith('before', { kind: 'node', nodeId: operationNode(before, 'add1').node_id });
  fireEvent.click(changes.getByRole('button', { name: 'Show after' }));
  const target = { kind: 'node' as const, nodeId: operationNode(after, 'after:add1').node_id };
  expect(navigate).toHaveBeenLastCalledWith('after', target);
  expect(comparisonCounterpart(before, after, 'after', target)).toEqual({ kind: 'node', nodeId: operationNode(before, 'add1').node_id });
});

it('maps a changed second a+a operand to its original destination slot, not the first edge', () => {
  const before = capturedFile(); bindOperation(before, 1);
  const after = renameCaptureFile(before); replaceOperand(after, 1, 1, 'after:h'); validatedCaptureFile(after);
  const result = compareGraphFiles(before, after).capture!;
  expect(result.valueUses.changes).toHaveLength(1);
  const change = result.valueUses.changes[0];
  expect(change.before!.path).toEqual(['arguments', 'args', 1]);
  const beforeTarget = capturedUseTarget(before, change.before!)!, afterTarget = capturedUseTarget(after, change.after!)!;
  expect(beforeTarget).toEqual(edgeTarget(before.document.semantic.edges[2]));
  expect(afterTarget).toEqual(edgeTarget(after.document.semantic.edges[2]));
  expect(comparisonCounterpart(before, after, 'before', beforeTarget)).toEqual(afterTarget);
  expect(comparisonCounterpart(before, after, 'after', afterTarget)).toEqual(beforeTarget);
  const navigate = jest.fn();
  render(<ComparisonPanel before={before} after={after} side="after" onNavigate={navigate} onChoose={() => {}} onClose={() => {}} />);
  fireEvent.click(within(screen.getByRole('region', { name: 'Changed tensor uses' })).getByRole('button', { name: 'Show after' }));
  expect(navigate).toHaveBeenLastCalledWith('after', afterTarget);
});

it('leaves changed dataflow without an anchor unresolved and does not guess node identity', () => {
  const before = capturedFile(), after = renameCaptureFile(before); replaceOperand(after, 1, 1, 'after:h'); validatedCaptureFile(after);
  render(<ComparisonPanel before={before} after={after} side="after" onNavigate={() => {}} onChoose={() => {}} onClose={() => {}} />);
  expect(screen.getByText('Comparison is incomplete')).toBeInTheDocument();
  expect(screen.getByRole('region', { name: 'Unmatched records' })).toHaveTextContent('Unmatched does not mean added or removed');
  expect(screen.queryByRole('region', { name: 'Added layers' })).not.toBeInTheDocument();
  expect(comparisonCounterpart(before, after, 'before', { kind: 'node', nodeId: operationNode(before, 'add2').node_id })).toBeUndefined();
});

it('keeps limited capture coverage explicit even when all retained records match', () => {
  const before = capturedFile(), after = renameCaptureFile(before);
  after.capture!.capture.status = 'partial'; after.capture!.capture.coverage.omissions.push('operation_limit');
  render(<ComparisonPanel before={before} after={validatedCaptureFile(after)} side="after" onNavigate={() => {}} onChoose={() => {}} onClose={() => {}} />);
  expect(screen.getByText('Comparison is incomplete')).toBeInTheDocument();
  expect(screen.getByText(/At least one capture is incomplete/)).toBeInTheDocument();
});

it('ignores layout and notes without claiming numerical output equality', () => {
  const before = capturedFile(), after = renameCaptureFile(before);
  after.document.view.node_positions[after.document.semantic.nodes[0].node_id].x = 1200;
  after.collaboration.walkthrough.title = 'New guide';
  render(<ComparisonPanel before={before} after={validatedCaptureFile(after)} side="after" onNavigate={() => {}} onChoose={() => {}} onClose={() => {}} />);
  expect(screen.getByText('Captured computation matches')).toBeInTheDocument();
  expect(screen.getByText(/does not prove numerical equality/)).toBeInTheDocument();
  expect(capturedRecordTarget(after, 'module', 'after:module:0')).toEqual({ kind: 'node', nodeId: after.document.semantic.nodes[2].node_id });
});

it('bounds field previews and opens the exact retained Before record without changing sides', async () => {
  const before = capturedFile(); bindOperation(before, 0);
  before.capture!.capture.operations[0].arguments.args[1] = { kind: 'str', value: '1'.repeat(4050) };
  capturePorts(operationNode(before, 'add1').metadata.capturePorts)[1].argumentKind = 'str';
  const after = renameCaptureFile(before);
  after.capture!.capture.operations[0].arguments.args[1] = { kind: 'str', value: '2'.repeat(4050) };
  const navigate = jest.fn();
  render(<ComparisonPanel before={validatedCaptureFile(before)} after={validatedCaptureFile(after)} side="after" onNavigate={navigate} onChoose={() => {}} onClose={() => {}} />);
  const changes = within(screen.getByRole('region', { name: 'Computation changes' }));
  expect(changes.getByText(/Values above are previews/)).toBeInTheDocument();
  expect(changes.getAllByText(/preview limited/).every(element => element.textContent!.length < 4100)).toBe(true);
  fireEvent.click(changes.getByRole('button', { name: 'Inspect before' }));
  const dialog = within(await screen.findByRole('dialog', { name: /Captured details: Before/ }));
  fireEvent.click(dialog.getByRole('button', { name: 'Arguments', exact: true }));
  fireEvent.click(await dialog.findByRole('button', { name: /array · 2 items/ }));
  fireEvent.click(await dialog.findByRole('button', { name: /str · 2 fields/ }));
  expect(await dialog.findByText('"' + '1'.repeat(4050) + '"')).toBeInTheDocument();
  expect(navigate).not.toHaveBeenCalled();
});
