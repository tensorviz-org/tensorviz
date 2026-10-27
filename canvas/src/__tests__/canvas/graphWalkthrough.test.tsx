import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import GraphFileViewer from '../../viewer/GraphFileViewer';
import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { downloadGraphFile } from '../../ide/graphFileExport';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';
import { restoreGraphFileView } from '../../viewer/graphFileView';

const mockRestore = jest.fn();
jest.mock('../../ide/graphFileExport', () => ({ downloadGraphFile: jest.fn() }));
jest.mock('../../viewer/GraphFileCanvas', () => {
  const React = require('react');
  return { __esModule: true, default: React.forwardRef(({ file }: any, ref: any) => {
    const [pan, setPan] = React.useState(0);
    React.useImperativeHandle(ref, () => ({
      captureView: () => ({ camera: { x: pan * 10, y: 20, z: .75 }, nodePositions: file.document.view.node_positions,
        expandedModuleIds: ['block', 'inner'], selectedNodeIds: ['fc'] }),
      restoreView: mockRestore,
    }));
    return <button onClick={() => setPan(pan + 1)}>Pan canvas {pan}</button>;
  }) };
});
const snapshot = () => createGraphFile(nestedGraph(), { name: 'Net', status: 'static' });
const choose = (contents: string) => fireEvent.change(screen.getByLabelText('Graph file'), { target: { files: [new File([contents], 'Net.tensorviz.json')] } });
async function open() {
  render(<GraphFileViewer />); choose(JSON.stringify(snapshot())); await screen.findByRole('heading', { name: 'Net' });
  fireEvent.click(screen.getByRole('button', { name: 'Walkthrough', exact: true }));
}
beforeEach(() => { mockRestore.mockReset(); (downloadGraphFile as jest.Mock).mockReset(); });
it('captures multiple views, edits/reorders them, and plays the saved camera without resetting exploration', async () => {
  await open(); fireEvent.click(screen.getByRole('button', { name: 'Pan canvas 0' }));
  fireEvent.click(screen.getByRole('button', { name: 'Add current view' }));
  fireEvent.change(screen.getByLabelText('Step title'), { target: { value: 'Encoder' } });
  fireEvent.change(screen.getByLabelText('Step note'), { target: { value: 'Keep this residual.\nIt preserves the input.' } });
  expect(screen.getByRole('button', { name: 'Pan canvas 1' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Pan canvas 1' }));
  fireEvent.click(screen.getByRole('button', { name: 'Add current view' }));
  fireEvent.change(screen.getByLabelText('Step title'), { target: { value: 'Prediction head' } });
  fireEvent.click(screen.getByRole('button', { name: 'Move up' }));
  fireEvent.click(screen.getByRole('button', { name: 'Start walkthrough' }));
  expect(mockRestore.mock.calls.at(-1)[0].camera.x).toBe(20);
  fireEvent.click(screen.getByRole('button', { name: 'Next', exact: true }));
  expect(mockRestore.mock.calls.at(-1)[0].camera.x).toBe(10);
  expect(screen.getByText(/It preserves the input/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Exit walkthrough' }));
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  const downloaded = (downloadGraphFile as jest.Mock).mock.calls[0][0];
  expect(parseGraphFile(JSON.stringify(downloaded)).collaboration.walkthrough.steps.map(s => s.title)).toEqual(['Prediction head', 'Encoder']);
  expect(screen.getByText('No unsaved notes')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Pan canvas 2' })).toBeInTheDocument();
});
it('retains notes after invalid imports and requires an explicit discard for valid replacements', async () => {
  await open(); fireEvent.click(screen.getByRole('button', { name: 'Add current view' }));
  const event = new Event('beforeunload', { cancelable: true }); window.dispatchEvent(event);
  expect(event.defaultPrevented).toBe(true);
  choose('invalid'); expect(await screen.findByRole('alert')).toHaveTextContent('previous graph is still open');
  expect(screen.getByLabelText('Step title')).toHaveValue('fc');
  choose(JSON.stringify(snapshot())); expect(await screen.findByRole('alertdialog')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Keep editing' }));
  expect(screen.getByLabelText('Step title')).toHaveValue('fc');
  choose(JSON.stringify(snapshot())); await screen.findByRole('alertdialog');
  fireEvent.click(screen.getByRole('button', { name: 'Discard changes and open' }));
  expect(screen.getByText('No unsaved notes')).toBeInTheDocument();
  expect(screen.queryByLabelText('Step title')).toBeNull();
});
it('does not claim a save succeeded when validation or download fails', async () => {
  await open(); fireEvent.click(screen.getByRole('button', { name: 'Add current view' }));
  fireEvent.change(screen.getByLabelText('Step title'), { target: { value: '' } });
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('step title');
  expect(downloadGraphFile).not.toHaveBeenCalled(); expect(screen.getByText('Unsaved notes')).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Step title'), { target: { value: 'Encoder' } });
  (downloadGraphFile as jest.Mock).mockImplementationOnce(() => { throw new Error('Download blocked'); });
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Download blocked');
  expect(screen.getByText('Unsaved notes')).toBeInTheDocument();
});
it('restores nested positions and expansion while retaining absent targets for context warnings', () => {
  const graph = nestedGraph();
  const view = { camera: { x: 1, y: 2, z: 1 }, nodePositions: { fc: { x: 330, y: 170 }, removed: { x: 0, y: 0 } }, expandedModuleIds: ['block', 'inner'], selectedNodeIds: ['fc', 'removed'] };
  const restored = restoreGraphFileView(graph, view);
  expect(restored.nodes.find(n => n.id === 'fc')).toMatchObject({ position: { x: 330, y: 170 }, hidden: false });
  expect(restored.nodes.some(n => n.id === 'removed')).toBe(false);
  expect(graph.nodes.find(n => n.id === 'block')?.data.expanded).toBe(false);
});
it('reopens a downloaded walkthrough with the same notes and saved view', async () => {
  await open(); fireEvent.click(screen.getByRole('button', { name: 'Add current view' }));
  fireEvent.change(screen.getByLabelText('Step note'), { target: { value: 'Why this layer exists' } });
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  const saved = (downloadGraphFile as jest.Mock).mock.calls[0][0]; choose(JSON.stringify(saved));
  await waitFor(() => expect(screen.queryByLabelText('Step note')).toBeNull());
  fireEvent.click(screen.getByRole('button', { name: 'Walkthrough', exact: true }));
  fireEvent.click(screen.getByRole('button', { name: 'Start walkthrough' }));
  expect(screen.getByText('Why this layer exists')).toBeInTheDocument();
  expect(mockRestore).toHaveBeenCalledWith(saved.collaboration.walkthrough.steps[0].view);
});

it('cancels an older pending replacement when a newer import fails', async () => {
  await open(); fireEvent.click(screen.getByRole('button', { name: 'Add current view' }));
  const other = snapshot(); other.document.semantic.name = 'Other model';
  choose(JSON.stringify(other)); await screen.findByRole('alertdialog');
  choose('invalid newer choice');
  expect(await screen.findByRole('alert')).toHaveTextContent('valid JSON');
  expect(screen.queryByRole('alertdialog')).toBeNull();
  expect(screen.getByLabelText('Step title')).toHaveValue('fc');
});
