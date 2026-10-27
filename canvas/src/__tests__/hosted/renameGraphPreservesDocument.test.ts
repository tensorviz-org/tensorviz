/**
 * Renaming from a list page commits through the same PATCH the builder uses, so
 * it has to hand the whole document back: modelGraphToDocumentV2 rebuilds the
 * document from what it is given, so a rename that passed only the new name
 * would silently drop the saved canvas arrangement and the operation history —
 * a data loss triggered by fixing a typo in a title.
 */
import { GraphStorageService } from '../../services/graphStorage';
import * as projectsApi from '../../services/projectsApi';
import type { SavedGraph } from '../../types';

jest.mock('../../services/projectsApi');

const ensureDefaultProject = projectsApi.ensureDefaultProject as jest.Mock;
const commitGraph = projectsApi.commitGraph as jest.Mock;
const createGraph = projectsApi.createGraph as jest.Mock;

const operations = [{ op: 'add_node', node_id: 'n1' }];
const layout = { strategy: 'manual', canvas_view_mode: 'expanded', focused_module_id: 'm1' };

const project = {
  id: 'g1',
  name: 'Untitled Model',
  description: 'the one from the pile',
  currentRevision: 7,
  graph: {
    nodes: [{ id: 'n1', type: 'layerNode', position: { x: 40, y: 90 }, data: { layerType: { id: 'relu', name: 'ReLU' }, parameters: {} } }],
    edges: [],
  },
  document: {
    schemaVersion: 2,
    semantic: { graph_id: 'g1', name: 'Untitled Model', revision: 7, inputs: [], nodes: [], edges: [], modules: [], operations, metadata: {} },
    view: { graph_id: 'g1', node_positions: {}, collapsed_modules: [], selection: [], layout },
  },
  createdAt: 't',
  updatedAt: 't',
} as unknown as SavedGraph;

beforeEach(() => {
  jest.clearAllMocks();
  ensureDefaultProject.mockResolvedValue('p1');
  commitGraph.mockImplementation((_pid: string, _gid: string, body: any) => Promise.resolve({
    id: 'g1',
    projectId: 'p1',
    name: body.name,
    document: body.document,
    currentRevision: body.expectedRevision + 1,
    createdAt: 't',
    updatedAt: 't',
  }));
});

it('commits the new name against the revision it was read at', async () => {
  const saved = await GraphStorageService.renameGraph(project, 'MNIST classifier');

  expect(createGraph).not.toHaveBeenCalled();
  expect(commitGraph).toHaveBeenCalledTimes(1);
  const [pid, gid, body] = commitGraph.mock.calls[0];
  expect([pid, gid]).toEqual(['p1', 'g1']);
  expect(body.name).toBe('MNIST classifier');
  // A stale expectedRevision is what makes a concurrent edit a 409 instead of
  // an overwrite, so it must be the row's own revision.
  expect(body.expectedRevision).toBe(7);
  expect(saved.name).toBe('MNIST classifier');
});

it('carries the canvas arrangement and the history through the rename', async () => {
  await GraphStorageService.renameGraph(project, 'MNIST classifier');

  const document = commitGraph.mock.calls[0][2].document;
  expect(document.semantic.operations).toEqual(operations);
  expect(document.view.layout).toMatchObject({
    canvas_view_mode: 'expanded',
    focused_module_id: 'm1',
  });
  // The node the graph had is still in the committed document, in place.
  expect(document.semantic.nodes.map((n: any) => n.node_id)).toEqual(['n1']);
  expect(document.view.node_positions.n1).toEqual({ x: 40, y: 90 });
  // The description is not collateral damage of a rename.
  expect(document.semantic.metadata.description).toBe('the one from the pile');
});
