import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import GraphFileViewer from '../../viewer/GraphFileViewer';
import { createGraphFile, parseGraphFile } from '../../services/graphFile';
import { edgeTarget } from '../../services/graphCollaboration';
import { downloadGraphFile } from '../../ide/graphFileExport';
import { cameraForTarget, cameraForViewport, restoreGraphFileView } from '../../viewer/graphFileView';
import { nestedGraph } from '../../test/fixtures/graphFileFixture';

const mockRestore = jest.fn();
const mockOpening = jest.fn();
jest.mock('../../ide/graphFileExport', () => ({ downloadGraphFile: jest.fn() }));
jest.mock('../../viewer/GraphFileCanvas', () => {
  const React = require('react');
  return { __esModule: true, default: React.forwardRef(({ file, initialView }: any, ref: any) => {
    const [pan, setPan] = React.useState(0);
    const [edge, setEdge] = React.useState(false);
    React.useEffect(() => { if (initialView) mockOpening(initialView); }, []);
    React.useImperativeHandle(ref, () => ({
      captureView: () => ({ camera: { x: pan * 10, y: 20, z: .8 }, viewport: { width: 1200, height: 800 },
        nodePositions: file.document.view.node_positions, expandedModuleIds: ['block', 'inner'], selectedNodeIds: edge ? [] : ['fc'],
        ...(edge ? { selectedEdge: {kind:'edge',source:'input',target:'fc',sourcePort:'y',targetPort:'x'} } : {}) }),
      restoreView: mockRestore,
    }));
    return <><button onClick={() => setPan(pan + 1)}>Pan canvas {pan}</button><button onClick={() => setEdge(true)}>Select connection</button></>;
  }) };
});
const snapshot = () => createGraphFile(nestedGraph(), { name: 'Net', status: 'static' });
const choose = (f: any) => fireEvent.change(screen.getByLabelText('Graph file'), { target: { files: [new File([JSON.stringify(f)], 'Net.tensorviz.json')] } });
async function open() {
  render(<GraphFileViewer />); choose(snapshot()); await screen.findByRole('heading', { name: 'Net' });
  fireEvent.click(screen.getByRole('button', { name: 'Saved views', exact: true }));
}
beforeEach(() => { mockRestore.mockReset(); mockOpening.mockReset(); (downloadGraphFile as jest.Mock).mockReset(); });
it('shares a named opening view and preserves notes and exploration through download', async () => {
  await open(); fireEvent.click(screen.getByRole('button', { name: 'Pan canvas 0' }));
  fireEvent.click(screen.getByRole('button', { name: 'Share this view' }));
  fireEvent.change(screen.getByLabelText('View title'), { target: { value: 'Review the encoder' } });
  fireEvent.change(screen.getByLabelText('View note'), { target: { value: 'Check this branch.\nKeep the skip connection.' } });
  expect(screen.getByRole('button', { name: 'Pan canvas 1' })).toBeInTheDocument();
  expect(screen.getByText('Unsaved notes')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  const saved = parseGraphFile(JSON.stringify((downloadGraphFile as jest.Mock).mock.calls[0][0]));
  expect(saved.collaboration.savedViews[0].title).toBe('Review the encoder');
  expect(saved.collaboration.openingViewId).toBe(saved.collaboration.savedViews[0].id);
  expect(saved.collaboration.savedViews[0].view.camera.x).toBe(10);
  expect(screen.getByRole('button', { name: 'Pan canvas 1' })).toBeInTheDocument();
  expect(mockOpening).not.toHaveBeenCalled();
  choose(saved); await waitFor(() => expect(mockOpening).toHaveBeenCalledTimes(1));
  expect(mockOpening).toHaveBeenCalledWith(saved.collaboration.savedViews[0].view);
  expect(screen.getByText(/Keep the skip connection/)).toBeInTheDocument();
  expect(screen.getByText('This view opens first for recipients.')).toBeInTheDocument();
});
it('reorders named views and restores a real edge after sequential edge ids change', async () => {
  await open(); fireEvent.click(screen.getByRole('button', { name: 'Share this view' }));
  fireEvent.click(screen.getByRole('button', { name: 'Select connection' }));
  fireEvent.click(screen.getByRole('button', { name: 'Share this view' }));
  fireEvent.click(screen.getByRole('button', { name: 'Move view up' }));
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  const saved = (downloadGraphFile as jest.Mock).mock.calls[0][0];
  const view = saved.collaboration.savedViews[0].view;
  expect(view.selectedEdge).toEqual(edgeTarget(saved.document.semantic.edges[0]));
  saved.document.semantic.edges.forEach((e: any, i: number) => e.edge_id = `reordered-${i}`);
  choose(saved); await waitFor(() => expect(mockOpening).toHaveBeenCalledWith(view));
  fireEvent.click(screen.getByRole('button', { name: 'Show saved view', exact: true }));
  expect(mockRestore).toHaveBeenCalledWith(view);
});
it('updates and deletes a saved view without leaving a dangling opening reference', async () => {
  await open(); fireEvent.click(screen.getByRole('button', { name: 'Share this view' }));
  fireEvent.click(screen.getByRole('button', { name: 'Pan canvas 0' }));
  fireEvent.click(screen.getByRole('button', { name: 'Update captured view' }));
  fireEvent.click(screen.getByRole('button', { name: 'Show saved view' }));
  expect(mockRestore.mock.calls[0][0].camera.x).toBe(10);
  fireEvent.click(screen.getByRole('button', { name: 'Clear opening view' }));
  fireEvent.click(screen.getByRole('button', { name: 'Make opening view' }));
  fireEvent.click(screen.getByRole('button', { name: 'Delete view' }));
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  const saved = (downloadGraphFile as jest.Mock).mock.calls[0][0];
  expect(saved.collaboration.savedViews).toEqual([]); expect(saved.collaboration.openingViewId).toBeNull();
});
it('does not silently save an invalid view title or clear dirty state', async () => {
  await open(); fireEvent.click(screen.getByRole('button', { name: 'Share this view' }));
  fireEvent.change(screen.getByLabelText('View title'), { target: { value: '' } });
  fireEvent.click(screen.getByRole('button', { name: 'Download graph file' }));
  expect(screen.getByRole('alert')).toHaveTextContent('saved view title');
  expect(downloadGraphFile).not.toHaveBeenCalled(); expect(screen.getByText('Unsaved notes')).toBeInTheDocument();
});
it('restores nested edge endpoints without changing architecture or saved positions', () => {
  const graph = nestedGraph(); const before = JSON.stringify(graph);
  const view = { camera: {x:12,y:20,z:.8}, nodePositions: {fc:{x:333,y:177}}, expandedModuleIds: [], selectedNodeIds: [],
    selectedEdge: {kind:'edge' as const,source:'input',target:'fc',sourcePort:'y',targetPort:'x'} };
  const result = restoreGraphFileView(graph, view);
  expect(result.nodes.find(n => n.id === 'fc')).toMatchObject({hidden:false,position:{x:333,y:177}});
  expect(result.nodes.find(n => n.id === 'inner')?.data.expanded).toBe(true);
  expect(result.nodes.find(n => n.id === 'block')?.data.expanded).toBe(true);
  expect(JSON.stringify(graph)).toBe(before);
});
it('keeps the shared page center across viewport sizes and leaves legacy cameras unchanged', () => {
  const view = {camera:{x:12,y:20,z:.8},nodePositions:{},expandedModuleIds:[],selectedNodeIds:[],viewport:{width:1200,height:800}};
  expect(cameraForViewport(view,view.viewport)).toEqual(view.camera);
  const size = {width:600,height:400}; const camera = cameraForViewport(view,size);
  expect(size.width/(2*camera.z)-camera.x).toBeCloseTo(view.viewport.width/(2*view.camera.z)-view.camera.x);
  expect(size.height/(2*camera.z)-camera.y).toBeCloseTo(view.viewport.height/(2*view.camera.z)-view.camera.y);
  expect(cameraForViewport({...view,viewport:undefined},size)).toEqual(view.camera);
});

it('focuses comparison targets beside the panel without magnifying a layer to fill the screen', () => {
  const bounds = {x:700,y:150,w:220,h:120}; const camera = cameraForTarget(bounds,{width:1440,height:900},452);
  expect(camera.z).toBeLessThanOrEqual(1.25);
  const left = (bounds.x + camera.x) * camera.z; const right = left + bounds.w * camera.z;
  expect(left).toBeGreaterThanOrEqual(48); expect(right).toBeLessThanOrEqual(1440-452-48);
  const wide = {x:0,y:0,w:1600,h:2000}; const fit = cameraForTarget(wide,{width:1440,height:900},452);
  expect(wide.w*fit.z).toBeLessThanOrEqual(1440-452-96); expect(wide.h*fit.z).toBeLessThanOrEqual(900-128);
});
