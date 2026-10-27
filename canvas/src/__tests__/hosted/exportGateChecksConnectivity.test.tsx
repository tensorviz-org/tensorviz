import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import CodeExportDialog from '../../components/CodeExportDialog';
import { ApiService } from '../../services/apiService';
import { ModelGraph } from '../../types';

/**
 * S-M1: the export gate must test that the Input node FEEDS the model, and must
 * offer the fix rather than dead-ending.
 *
 * The gate was a bare `.find` for a node with a shape, so an Input dropped on the
 * canvas and left dangling satisfied it — the export went ahead against a shape
 * that reached none of the layers. And when it did refuse, it refused flatly:
 * "No Input node found. Please add an Input node to define the input shape." — a
 * palette layer the tutorial never mentioned, with no way to act on the message.
 */
jest.mock('../../services/apiService', () => ({
  ApiService: {
    generateCode: jest.fn(),
    exportPackage: jest.fn(),
  },
}));

const layer = (id: string, layerId: string, parameters: Record<string, unknown>) => ({
  id,
  position: { x: 0, y: 0 },
  data: { layerType: { id: layerId, name: layerId, displayName: layerId }, parameters },
});

const danglingInput = {
  nodes: [
    layer('input-1', 'input', { shape: '1, 3, 224, 224' }),
    layer('relu-1', 'relu', { inplace: false }),
  ],
  edges: [],
} as unknown as ModelGraph;

const wiredInput = {
  nodes: danglingInput.nodes,
  edges: [{ id: 'e1', source: 'input-1', target: 'relu-1' }],
} as unknown as ModelGraph;

const noInput = {
  nodes: [layer('relu-1', 'relu', { inplace: false })],
  edges: [],
} as unknown as ModelGraph;

beforeEach(() => {
  jest.clearAllMocks();
  (ApiService.generateCode as jest.Mock).mockResolvedValue({ code: 'import torch\n', unsupported: [] });
  (ApiService.exportPackage as jest.Mock).mockResolvedValue(new Blob(['zip']));
});

const open = (graph: ModelGraph, onAddInputNode?: () => void) =>
  render(
    <CodeExportDialog isOpen onClose={() => undefined} graph={graph} onAddInputNode={onAddInputNode} />
  );

describe('the export gate', () => {
  it('refuses a graph whose Input node is connected to nothing', async () => {
    open(danglingInput);
    await screen.findByText(/not connected to anything/i);
    // And it refuses BEFORE asking the backend for anything — the point is that
    // the shape it would have sent reaches none of the layers.
    expect(ApiService.generateCode).not.toHaveBeenCalled();
    expect(ApiService.exportPackage).not.toHaveBeenCalled();
  });

  it('lets a wired Input through', async () => {
    open(wiredInput);
    await waitFor(() => expect(ApiService.generateCode).toHaveBeenCalled());
    expect(screen.queryByText(/not connected to anything/i)).toBeNull();
  });

  it('offers to add an Input node instead of dead-ending', async () => {
    const onAddInputNode = jest.fn();
    open(noInput, onAddInputNode);
    await screen.findByText(/no Input node/i);
    fireEvent.click(screen.getByRole('button', { name: 'Add an Input node' }));
    expect(onAddInputNode).toHaveBeenCalledTimes(1);
  });

  it('does not offer what it cannot do', async () => {
    // No callback: the message still has to be actionable prose, but there must be
    // no button that does nothing.
    open(noInput);
    await screen.findByText(/no Input node/i);
    expect(screen.queryByRole('button', { name: 'Add an Input node' })).toBeNull();
  });

  it('does not offer an Input node for an unrelated failure', async () => {
    const onAddInputNode = jest.fn();
    const missingParams = {
      nodes: [
        layer('input-1', 'input', { shape: '1, 3, 224, 224' }),
        layer('linear-1', 'linear', { in_features: undefined, out_features: undefined }),
      ],
      edges: [{ id: 'e1', source: 'input-1', target: 'linear-1' }],
    } as unknown as ModelGraph;
    open(missingParams, onAddInputNode);
    await screen.findByText(/missing required parameters/i);
    expect(screen.queryByRole('button', { name: 'Add an Input node' })).toBeNull();
  });
});
