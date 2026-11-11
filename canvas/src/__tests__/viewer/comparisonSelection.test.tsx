import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import GraphFileViewer from '../../viewer/GraphFileViewer';
import ComparisonPanel from '../../viewer/ComparisonPanel';
import * as captureComparison from '../../services/capturedGraphComparison';
import * as graphComparison from '../../services/graphComparison';
import { GraphFile } from '../../services/graphFile';
import { initialGraphFileView, revealGraphNode, restoreGraphFileView } from '../../viewer/graphFileView';
import { downloadGraphFile } from '../../ide/graphFileExport';
import { capturedFile, renameCaptureFile, operationNode, replaceOperand } from '../../test/fixtures/capturedGraphFileFixture';

jest.mock('../../ide/graphFileExport', () => ({ downloadGraphFile: jest.fn() }));
const mockFocus = jest.fn();
const mockCapture = jest.fn();
const mockMount = jest.fn();
const mockFiles = new Set<GraphFile>();
function mockFreeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(mockFreeze);
    Object.freeze(value);
  }
  return value;
}
jest.mock('../../viewer/GraphFileCanvas', () => {
  const React = require('react');
  return { __esModule: true, default: React.forwardRef(({ file, initialView, initialTarget, onTargetChange }: any, ref: any) => {
    mockFreeze(file); mockFiles.add(file);
    const [pan, setPan] = React.useState(initialView?.camera.x ?? 0);
    const [selection, setSelection] = React.useState(initialTarget ?? { kind: 'node', nodeId: file.document.semantic.nodes[0].node_id });
    React.useEffect(() => { mockMount({ file, initialView, initialTarget }); }, [file, initialView, initialTarget]);
    React.useImperativeHandle(ref, () => ({
      captureView: () => {
        mockCapture();
        return { camera: { x: pan, y: 0, z: 1 }, nodePositions: file.document.view.node_positions,
          expandedModuleIds: [], selectedNodeIds: selection.kind === 'node' ? [selection.nodeId] : [],
          ...(selection.kind === 'edge' ? { selectedEdge: selection } : {}) };
      },
      focusTarget: (target: any) => { mockFocus({ captureId: file.capture.capture.id, pan, target }); setSelection(target); },
      restoreView: () => {},
    }));
    return <><div data-testid="canvas">{file.capture.capture.id}</div>
      <button onClick={() => setPan(pan + 1)}>Pan {pan}</button>
      <button onClick={() => onTargetChange({ ...selection })}>Select current</button></>;
  }) };
});

const choose = (name: string, file: GraphFile | string) => fireEvent.change(screen.getByLabelText(name), {
  target: { files: [new File([typeof file === 'string' ? file : JSON.stringify(file)], 'model.tensorviz.json')] },
});
function pair() {
  const before = capturedFile(), after = renameCaptureFile(before);
  after.capture!.capture.operations[0].arguments.args[1] = { kind: 'int', value: '2' };
  return { before, after };
}
async function openComparison(before: GraphFile, after: GraphFile) {
  render(<GraphFileViewer />);
  choose('Graph file', after); await screen.findByTestId('canvas');
  fireEvent.click(screen.getByRole('button', { name: 'Compare' }));
  choose('Comparison Before file', before);
  const section = await screen.findByRole('region', { name: 'Computation changes' });
  await within(section).findAllByRole('button', { name: 'Inspect before' });
}
const changes = () => within(screen.getByRole('region', { name: 'Computation changes' }));
beforeAll(() => { HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); }; });
beforeEach(() => { mockFocus.mockReset(); mockCapture.mockReset(); mockMount.mockReset(); mockFiles.clear(); });
afterEach(() => { jest.restoreAllMocks(); });

it('skips unchanged comparison work on selection while keeping the latest canvas handle', async () => {
  const { before, after } = pair();
  const targets = jest.spyOn(captureComparison, 'capturedRecordTarget');
  await openComparison(before, after);
  const calls = targets.mock.calls.length;
  const panel = screen.getByRole('complementary', { name: 'Architecture comparison' });
  const markup = panel.innerHTML;
  const inputs = [...mockFiles].map(file => [file, JSON.stringify(file)] as const);
  fireEvent.click(screen.getByRole('button', { name: 'Select current' }));
  fireEvent.click(screen.getByRole('button', { name: 'Select current' }));
  expect(targets).toHaveBeenCalledTimes(calls);
  expect(panel.innerHTML).toBe(markup);
  fireEvent.click(screen.getByRole('button', { name: 'Pan 0' }));
  fireEvent.click(changes().getByRole('button', { name: 'Show after' }));
  expect(mockFocus).toHaveBeenLastCalledWith({ captureId: after.capture!.capture.id, pan: 1,
    target: { kind: 'node', nodeId: operationNode(after, 'after:add1').node_id } });
  for (const [file, contents] of inputs) expect(JSON.stringify(file)).toBe(contents);
});

it('keeps an open inspection dialog and its selected path through canvas selection', async () => {
  const { before, after } = pair(); await openComparison(before, after);
  fireEvent.click(changes().getByRole('button', { name: 'Inspect before' }));
  const dialog = await screen.findByRole('dialog', { name: /Captured details: Before/ });
  fireEvent.click(within(dialog).getByRole('button', { name: 'Arguments' }));
  await within(dialog).findByRole('button', { name: /array · 2 items/ });
  fireEvent.click(screen.getByRole('button', { name: 'Select current' }));
  expect(screen.getByRole('dialog', { name: /Captured details: Before/ })).toBe(dialog);
  expect(within(dialog).getByText('operation add1.arguments')).toBeInTheDocument();
  fireEvent.click(within(dialog).getByRole('button', { name: 'Close captured details' }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

it('uses replacement Before files and current-side callbacks after previous-side navigation', async () => {
  const { before, after } = pair(); await openComparison(before, after);
  fireEvent.click(changes().getByRole('button', { name: 'Show before' }));
  fireEvent.click(screen.getByRole('button', { name: 'Select current' }));
  fireEvent.click(changes().getByRole('button', { name: 'Show before' }));
  expect(mockFocus).toHaveBeenLastCalledWith(expect.objectContaining({ captureId: before.capture!.capture.id }));
  const replacement = renameCaptureFile(after);
  replacement.capture!.capture.operations[0].arguments.args[1] = { kind: 'int', value: '3' };
  choose('Comparison Before file', replacement);
  await waitFor(() => expect(changes().getByText(/"value": "3"/)).toBeInTheDocument());
  expect(screen.getByTestId('canvas')).toHaveTextContent(after.capture!.capture.id);
  fireEvent.click(changes().getByRole('button', { name: 'Show before' }));
  expect(mockMount.mock.calls.at(-1)![0].initialTarget).toEqual({ kind: 'node', nodeId: operationNode(replacement, 'after:after:add1').node_id });
  fireEvent.click(screen.getByRole('button', { name: 'Close comparison' }));
  expect(mockMount.mock.calls.at(-1)![0].file.capture.capture.id).toBe(after.capture!.capture.id);
  expect(mockMount.mock.calls.at(-1)![0].initialTarget).toBeUndefined();
});

it('preserves navigation errors and does not close Before until its view can be captured', async () => {
  const { before, after } = pair(); await openComparison(before, after);
  mockFocus.mockImplementationOnce(() => { throw new Error('Focus failed'); });
  fireEvent.click(changes().getByRole('button', { name: 'Show after' }));
  expect(screen.getByRole('alert')).toHaveTextContent('Focus failed');
  mockCapture.mockImplementationOnce(() => { throw new Error('View unavailable'); });
  fireEvent.click(screen.getByRole('button', { name: 'View before' }));
  expect(screen.getByRole('button', { name: 'View after' })).toHaveAttribute('aria-pressed', 'true');
  fireEvent.click(screen.getByRole('button', { name: 'View before' }));
  mockCapture.mockImplementationOnce(() => { throw new Error('Still loading'); });
  fireEvent.click(screen.getByRole('button', { name: 'Close comparison' }));
  expect(screen.getByRole('alert')).toHaveTextContent('Still loading');
  expect(screen.getByRole('button', { name: 'View before' })).toHaveAttribute('aria-pressed', 'true');
  fireEvent.click(screen.getByRole('button', { name: 'Close comparison' }));
  expect(screen.queryByRole('complementary', { name: 'Architecture comparison' })).not.toBeInTheDocument();
  expect(mockMount.mock.calls.at(-1)![0].file.capture.capture.id).toBe(after.capture!.capture.id);
});

it('keeps choose wired after selection and refreshes the panel after replacing the owned file with an annotation edit', async () => {
  const { before, after } = pair(); await openComparison(before, after);
  fireEvent.click(screen.getByRole('button', { name: 'Select current' }));
  const input = screen.getByLabelText('Comparison Before file');
  const click = jest.spyOn(input, 'click');
  fireEvent.click(screen.getByRole('button', { name: 'Choose another Before file' }));
  expect(click).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Walkthrough' }));
  fireEvent.click(screen.getByRole('button', { name: 'Edit walkthrough' }));
  fireEvent.click(screen.getByRole('button', { name: 'Compare' }));
  const panel = screen.getByRole('complementary', { name: 'Architecture comparison' });
  const previous = [...mockFiles].at(-1)!;
  const compare = jest.spyOn(graphComparison, 'compareGraphFiles');
  // The other panels stay mounted. Exercise their real immutable edit while Compare stays mounted.
  fireEvent.change(screen.getByLabelText('Walkthrough title'), { target: { value: 'Updated guide' } });
  const current = [...mockFiles].at(-1)!;
  expect(screen.getByText('Unsaved notes')).toBeInTheDocument();
  expect(screen.getByRole('complementary', { name: 'Architecture comparison' })).toBe(panel);
  expect(compare).toHaveBeenCalledWith(expect.anything(), current);
  expect(current).not.toBe(previous);
  expect(current.capture).toBe(previous.capture);
  expect(current.document).toBe(previous.document);
  expect(changes().getByText(/"value": "2"/)).toBeInTheDocument();
  expect(current.collaboration.walkthrough.title).toBe('Updated guide');
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  const saved = [...mockFiles].at(-1)!;
  expect(downloadGraphFile).toHaveBeenLastCalledWith(saved);
  expect(saved).not.toBe(current);
  expect(compare).toHaveBeenLastCalledWith(expect.anything(), saved);
  expect(screen.getByRole('complementary', { name: 'Architecture comparison' })).toBe(panel);
  expect(screen.getByText('No unsaved notes')).toBeInTheDocument();
});

it('retains expanded record state across selection and resets it for a replacement capture', async () => {
  const { before, after } = pair();
  const compare = graphComparison.compareGraphFiles;
  // Repeat a genuine record only to exercise pagination without enlarging the capture fixture.
  jest.spyOn(graphComparison, 'compareGraphFiles').mockImplementation((a, b) => {
    const result = compare(a, b);
    result.capture!.changes = Array.from({ length: 26 }, () => result.capture!.changes[0]);
    return result;
  });
  await openComparison(before, after);
  expect(changes().getAllByRole('article')).toHaveLength(25);
  fireEvent.click(screen.getByRole('button', { name: 'Show more records (first 25 per section shown)' }));
  fireEvent.click(screen.getByRole('button', { name: 'Select current' }));
  expect(changes().getAllByRole('article')).toHaveLength(26);
  choose('Comparison Before file', before);
  await waitFor(() => expect(changes().getAllByRole('article')).toHaveLength(25));
});

it('does not extend viewer memoization to direct callers with in-place projection edits', () => {
  const { before, after } = pair();
  const props = { before, after, side: 'after' as const, onChoose: jest.fn(), onClose: jest.fn(), onNavigate: jest.fn() };
  const { rerender } = render(<ComparisonPanel {...props} />);
  const node = operationNode(after, 'after:add1');
  node.metadata.captureLabel = 'Updated projection label';
  const replacement = { ...node, node_id: 'replacement-card' };
  after.document.semantic.nodes.splice(after.document.semantic.nodes.indexOf(node), 1, replacement);
  rerender(<ComparisonPanel {...props} />);
  expect(changes().getByRole('heading', { name: 'Updated projection label' })).toBeInTheDocument();
  fireEvent.click(changes().getByRole('button', { name: 'Show after' }));
  expect(props.onNavigate).toHaveBeenLastCalledWith('after', { kind: 'node', nodeId: 'replacement-card' });
});

it('keeps owned capture/document data unchanged through real adapter, reveal and restore operations', () => {
  const file = mockFreeze(capturedFile());
  const original = JSON.stringify(file);
  const initial = initialGraphFileView(file);
  const target = operationNode(file, 'add2').node_id;
  const revealed = revealGraphNode(initial, target);
  const view = mockFreeze({ camera: { x: 120, y: 20, z: 0.8 },
    nodePositions: Object.fromEntries(revealed.nodes.map(node => [node.id, { x: node.position.x + 25, y: node.position.y + 50 }])),
    expandedModuleIds: revealed.nodes.filter(node => node.type === 'moduleNode').map(node => node.id), selectedNodeIds: [target] });
  const viewContents = JSON.stringify(view);
  const restored = restoreGraphFileView(revealed, view);
  expect(restored.nodes.some(node => node.id === target)).toBe(true);
  expect(JSON.stringify(file)).toBe(original);
  expect(JSON.stringify(view)).toBe(viewContents);
  const projected = initial.nodes.find(node => node.id === target)!;
  const source = file.document.semantic.nodes.find(node => node.node_id === target)!;
  expect(projected.data.captureRef).not.toBe(source.metadata.captureRef);
  expect(projected.data.capturePorts).not.toBe(source.metadata.capturePorts);
});

it('restores the destination own view when an unmatched record has no confident counterpart', async () => {
  const { before, after } = pair();
  replaceOperand(after, 1, 1, 'after:h');
  await openComparison(before, after);
  fireEvent.click(screen.getByRole('button', { name: 'Pan 0' }));
  fireEvent.click(screen.getByRole('button', { name: 'View before' }));
  fireEvent.click(screen.getByRole('button', { name: 'Pan 0' }));
  fireEvent.click(screen.getByRole('button', { name: 'Pan 1' }));
  fireEvent.click(screen.getByRole('button', { name: 'View after' }));
  const unmatched = within(screen.getByRole('region', { name: 'Unmatched records' }));
  const record = unmatched.getAllByRole('listitem').find(item => within(item).queryByText('operation · after:add2'))!;
  fireEvent.click(within(record).getByRole('button', { name: 'Show after' }));
  expect(mockFocus).toHaveBeenLastCalledWith(expect.objectContaining({ target: { kind: 'node', nodeId: operationNode(after, 'after:add2').node_id } }));
  fireEvent.click(screen.getByRole('button', { name: 'Select current' }));
  fireEvent.click(screen.getByRole('button', { name: 'View before' }));
  const mount = mockMount.mock.calls.at(-1)![0];
  expect(mount.initialTarget).toBeUndefined();
  expect(mount.initialView.camera.x).toBe(2);
  expect(screen.getByRole('button', { name: 'Pan 2' })).toBeInTheDocument();
});
