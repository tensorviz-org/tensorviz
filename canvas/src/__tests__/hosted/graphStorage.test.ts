import { GraphStorageService } from '../../services/graphStorage';
import * as projectsApi from '../../services/projectsApi';
import { localDevelopmentDraftStore } from '../../services/localDraft';
import { modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import {
  graphSaveDocument,
  RevisionedGraphSaveQueue,
} from '../../services/revisionedGraphSaveQueue';
import type { GraphRecord } from '../../services/projectsApi';

jest.mock('../../services/projectsApi');

const ensureDefaultProject = projectsApi.ensureDefaultProject as jest.Mock;
const listGraphs = projectsApi.listGraphs as jest.Mock;
const getGraph = projectsApi.getGraph as jest.Mock;
const createGraph = projectsApi.createGraph as jest.Mock;
const commitGraph = projectsApi.commitGraph as jest.Mock;
const deleteGraph = projectsApi.deleteGraph as jest.Mock;

const emptyGraph = { nodes: [], edges: [] } as any;

function writeDraft(
  id: string,
  graph: any,
  options: {
    baseRevision?: number;
    name?: string;
    description?: string;
    author?: 'user' | 'agent';
    opSummary?: string;
  } = {}
) {
  const baseRevision = options.baseRevision ?? 0;
  const document = modelGraphToDocumentV2(graph, {
    graphId: id,
    name: options.name || 'Recovered draft',
    description: options.description,
    revision: baseRevision,
  });
  localDevelopmentDraftStore.replaceJournal(id, baseRevision, [{
    document,
    author: options.author ?? 'user',
    ...(options.opSummary ? { opSummary: options.opSummary } : {}),
  }]);
}

function record(over: Partial<GraphRecord> = {}): GraphRecord {
  const name = over.name || 'G';
  const description = over.description;
  return {
    id: 'g',
    projectId: 'p1',
    name,
    document: modelGraphToDocumentV2(emptyGraph, {
      graphId: over.id || 'g', name, description, revision: over.currentRevision || 1,
    }),
    currentRevision: 1,
    createdAt: 't',
    updatedAt: 't',
    ...over,
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
  ensureDefaultProject.mockResolvedValue('p1');
});

describe('getAllGraphs', () => {
  it('maps V2 records to canvas projections and sorts newest first', async () => {
    listGraphs.mockResolvedValue([
      record({ id: 'old', name: 'Old', updatedAt: '2023-01-01T00:00:00Z' }),
      record({ id: 'new', name: 'New', updatedAt: '2024-06-01T00:00:00Z' }),
    ]);

    const graphs = await GraphStorageService.getAllGraphs();

    expect(graphs.map((graph) => graph.id)).toEqual(['new', 'old']);
    expect(graphs[0]).toMatchObject({ id: 'new', currentRevision: 1, graph: emptyGraph });
  });
});

describe('saveGraph', () => {
  it('refuses extension projections before creating any saved graph or contacting the server', async () => {
    window.__TENSORVIZ_IDE__ = true;
    try {
      await expect(GraphStorageService.saveGraph(emptyGraph, 'Source model'))
        .rejects.toThrow('Source graphs are managed by VS Code');
      expect(ensureDefaultProject).not.toHaveBeenCalled();
      expect(createGraph).not.toHaveBeenCalled();
      expect(commitGraph).not.toHaveBeenCalled();
    } finally {
      delete window.__TENSORVIZ_IDE__;
    }
  });
  it('creates a native V2 document on first save', async () => {
    createGraph.mockResolvedValue(record({ id: 'g1', name: 'M' }));

    const saved = await GraphStorageService.saveGraph(emptyGraph, 'M');

    expect(createGraph).toHaveBeenCalledTimes(1);
    const sent = createGraph.mock.calls[0][2];
    expect(sent.schemaVersion).toBe(2);
    expect(sent.semantic.name).toBe('M');
    expect(saved.id).toBe('g1');
  });

  it('persists and reopens descriptions through semantic metadata', async () => {
    createGraph.mockResolvedValue(record({ id: 'g1', name: 'M', description: 'Model notes' }));

    const saved = await GraphStorageService.saveGraph(emptyGraph, 'M', 'Model notes');

    expect(createGraph.mock.calls[0][2].semantic.metadata.description).toBe('Model notes');
    expect(saved.description).toBe('Model notes');
  });

  it('attributes an agent-authored first save without a second revision', async () => {
    createGraph.mockResolvedValue(record({ id: 'g1', name: 'AI model' }));

    await GraphStorageService.saveGraph(emptyGraph, 'AI model', undefined, undefined, {
      author: 'agent',
      opSummary: 'AI: create graph',
    });

    expect(createGraph).toHaveBeenCalledTimes(1);
    expect(createGraph.mock.calls[0][3]).toEqual({
      author: 'agent',
      opSummary: 'AI: create graph',
    });
    expect(commitGraph).not.toHaveBeenCalled();
  });

  it('commits with expected revision and advances from the response', async () => {
    commitGraph.mockResolvedValue(record({ id: 'g1', name: 'M2', currentRevision: 4 }));

    const saved = await GraphStorageService.saveGraph(
      emptyGraph,
      'M2',
      undefined,
      'g1',
      { expectedRevision: 3 }
    );

    expect(commitGraph).toHaveBeenCalledTimes(1);
    expect(commitGraph.mock.calls[0][2]).toMatchObject({
      name: 'M2',
      expectedRevision: 3,
    });
    expect(createGraph).not.toHaveBeenCalled();
    expect(saved.currentRevision).toBe(4);
  });

  it('forwards agent attribution with the exact commit', async () => {
    commitGraph.mockResolvedValue(record({ id: 'g1', currentRevision: 2 }));

    await GraphStorageService.saveGraph(emptyGraph, 'M', undefined, 'g1', {
      expectedRevision: 1,
      author: 'agent',
      opSummary: 'AI: 3 operations',
    });

    expect(commitGraph.mock.calls[0][2]).toMatchObject({
      expectedRevision: 1,
      author: 'agent',
      opSummary: 'AI: 3 operations',
    });
  });

  it('propagates a stale conflict and never creates a duplicate graph', async () => {
    commitGraph.mockRejectedValue(new Error('API Error (409): graph_revision_conflict'));

    await expect(
      GraphStorageService.saveGraph(emptyGraph, 'M', undefined, 'g1', { expectedRevision: 1 })
    ).rejects.toThrow('409');

    expect(createGraph).not.toHaveBeenCalled();
  });

  it('requires a revision for an existing server graph', async () => {
    await expect(GraphStorageService.saveGraph(emptyGraph, 'M', undefined, 'g1'))
      .rejects.toThrow('expected revision');
    expect(commitGraph).not.toHaveBeenCalled();
    expect(createGraph).not.toHaveBeenCalled();
  });

  it('treats a local template id as a new graph', async () => {
    createGraph.mockResolvedValue(record({ id: 'g2', name: 'T' }));

    await GraphStorageService.saveGraph(emptyGraph, 'T', undefined, 'template-resnet-123');

    expect(createGraph).toHaveBeenCalledTimes(1);
    expect(commitGraph).not.toHaveBeenCalled();
  });

  it('creates instead of updating when an unsaved recovery draft is retried', async () => {
    createGraph.mockResolvedValue(record({ id: 'g2', name: 'Recovered' }));

    await GraphStorageService.saveGraph(
      emptyGraph,
      'Recovered',
      undefined,
      '__unsaved__',
      {
        expectedRevision: 0,
        author: 'agent',
        opSummary: 'AI: create graph',
        createKey: 'create-1',
      }
    );

    expect(createGraph).toHaveBeenCalledWith(
      'p1',
      'Recovered',
      expect.objectContaining({ schemaVersion: 2 }),
      {
        author: 'agent',
        opSummary: 'AI: create graph',
        idempotencyKey: 'create-1',
      }
    );
    expect(commitGraph).not.toHaveBeenCalled();
  });
});

describe('getGraph and deleteGraph', () => {
  it('returns the authoritative server record', async () => {
    getGraph.mockResolvedValue(record({ id: 'g1', name: 'Server', currentRevision: 6 }));
    const saved = await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);
    expect(saved).toMatchObject({ name: 'Server', currentRevision: 6 });
  });

  it('opens the server authority without consuming an unknown-base legacy draft', async () => {
    const rawLegacy = JSON.stringify({ nodes: [{ id: 'legacy' }], edges: [] });
    localStorage.setItem('canvas-draft-g1', rawLegacy);
    getGraph.mockResolvedValue(record({ id: 'g1', name: 'Server', currentRevision: 6 }));

    const saved = await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);

    expect(saved).toMatchObject({
      name: 'Server',
      currentRevision: 6,
      legacyRecoveryAvailable: true,
    });
    expect(saved?.graph.nodes).toHaveLength(0);
    expect(localStorage.getItem('canvas-draft-g1')).toBe(rawLegacy);
  });

  it('recovers a network outage from the draft\'s recorded base revision', async () => {
    getGraph.mockRejectedValue(new Error('offline'));
    writeDraft(
      'g1',
      { nodes: [{ id: 'n1' }], edges: [] } as any,
      { baseRevision: 5, name: 'Draft' }
    );
    const saved = await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);
    expect(saved?.graph.nodes).toHaveLength(1);
    expect(saved?.currentRevision).toBe(5);
    expect(saved?.name).toBe('Draft');
    expect(saved?.recoveredDraft).toBe(true);
  });

  it('recovers an unsaved draft against the matching server revision', async () => {
    getGraph.mockResolvedValue(record({ id: 'g1', name: 'Server', currentRevision: 5 }));
    writeDraft(
      'g1',
      { nodes: [{ id: 'n1' }], edges: [] } as any,
      { baseRevision: 5, name: 'Server', description: 'Recovered notes' }
    );

    const saved = await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);

    expect(saved).toMatchObject({
      name: 'Server', description: 'Recovered notes', currentRevision: 5, recoveredDraft: true,
    });
    expect(saved?.graph.nodes).toHaveLength(1);
    expect(localDevelopmentDraftStore.readRevisioned('g1')).not.toBeNull();
  });

  it('restores ordered agent and user attribution from the crash journal', async () => {
    const agentDocument = modelGraphToDocumentV2(emptyGraph, {
      graphId: 'g1', name: 'Agent draft', revision: 5,
    });
    const userDocument = modelGraphToDocumentV2(emptyGraph, {
      graphId: 'g1', name: 'User draft', revision: 5,
    });
    localDevelopmentDraftStore.replaceJournal('g1', 5, [
      { document: agentDocument, author: 'agent', opSummary: 'AI: 2 operations' },
      { document: userDocument, author: 'user' },
    ]);
    getGraph.mockResolvedValue(record({ id: 'g1', currentRevision: 5 }));

    const saved = await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);

    expect(saved?.name).toBe('User draft');
    expect(saved?.recoveredCommits).toMatchObject([
      { author: 'agent', opSummary: 'AI: 2 operations' },
      { author: 'user' },
    ]);
  });

  it('drops commits that landed before a crash and rebases the remaining journal', async () => {
    const agentDocument = modelGraphToDocumentV2(emptyGraph, {
      graphId: 'g1', name: 'Agent draft', revision: 5,
    });
    const userDocument = modelGraphToDocumentV2(emptyGraph, {
      graphId: 'g1', name: 'User draft', revision: 5,
    });
    localDevelopmentDraftStore.replaceJournal('g1', 5, [
      { document: agentDocument, author: 'agent', opSummary: 'AI: 1 operation' },
      { document: userDocument, author: 'user' },
    ]);
    const committedAgent = JSON.parse(JSON.stringify(agentDocument));
    committedAgent.semantic.revision = 6;
    getGraph.mockResolvedValue(record({
      id: 'g1', name: 'Agent draft', currentRevision: 6, document: committedAgent,
    }));

    const saved = await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);

    expect(saved).toMatchObject({
      name: 'User draft',
      currentRevision: 6,
      recoveredCommits: [{ author: 'user' }],
    });
    expect(localDevelopmentDraftStore.readRevisioned('g1')).toMatchObject({
      baseRevision: 6,
      pendingCommits: [{ author: 'user' }],
    });
  });

  it('hydrates a brand-new offline canvas from the unsaved draft key', () => {
    writeDraft('__unsaved__', emptyGraph, {
      name: 'Offline AI graph',
      author: 'agent',
      opSummary: 'AI: create graph',
    });

    expect(GraphStorageService.getUnsavedDraft(localDevelopmentDraftStore)).toMatchObject({
      id: '__unsaved__',
      name: 'Offline AI graph',
      currentRevision: 0,
      recoveredDraft: true,
      recoveredCommits: [{ author: 'agent', opSummary: 'AI: create graph' }],
    });
  });

  it('archives an unreconcilable draft and still opens the saved graph', async () => {
    // This used to throw. The draft stayed live, so EVERY later load of the graph
    // threw again: after two tabs edited the same base, the pill told the user to
    // reload and every reload in that browser profile hit the load-error screen,
    // whose only control is a Retry that could never succeed. Opening the saved
    // graph is the fix; silently binning the user's unsent edit is not, so the
    // payload has to survive somewhere read() will not re-apply it.
    getGraph.mockResolvedValue(record({ id: 'g1', currentRevision: 6 }));
    writeDraft('g1', { nodes: [{ id: 'n1' }], edges: [] } as any, { baseRevision: 5, name: 'G' });

    const saved = await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);

    expect(saved?.id).toBe('g1');
    expect(saved?.conflictedDraftAvailable).toBe(true);
    expect(saved?.recoveredDraft).toBeUndefined();
    // No longer returned, so a second load reconciles cleanly instead of throwing.
    expect(localDevelopmentDraftStore.readRevisioned('g1')).toBeNull();
    // But not destroyed.
    const archived = Object.keys(localStorage).filter((k) => k.startsWith('canvas-conflict-v1:'));
    expect(archived).toHaveLength(1);
    expect(localStorage.getItem(archived[0])).toContain('n1');
  });

  it('opens cleanly on a second load once a conflicted draft is archived', async () => {
    getGraph.mockResolvedValue(record({ id: 'g1', currentRevision: 6 }));
    writeDraft('g1', { nodes: [{ id: 'n1' }], edges: [] } as any, { baseRevision: 5, name: 'G' });

    await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);
    const second = await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);

    expect(second?.id).toBe('g1');
    expect(second?.conflictedDraftAvailable).toBeUndefined();
  });

  it('clears a stale draft only when its document already reached the server', async () => {
    writeDraft('g1', emptyGraph, { baseRevision: 5, name: 'G', description: 'Notes' });
    const draft = localDevelopmentDraftStore.readRevisioned('g1')!;
    const committed = JSON.parse(JSON.stringify(draft.document));
    committed.semantic.revision = 6;
    getGraph.mockResolvedValue(record({
      id: 'g1', name: 'G', description: 'Notes', currentRevision: 6, document: committed,
    }));

    const saved = await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);

    expect(saved?.recoveredDraft).toBeUndefined();
    expect(localDevelopmentDraftStore.readRevisioned('g1')).toBeNull();
  });

  it('CAS-commits an offline-reloaded draft once connectivity returns', async () => {
    getGraph.mockRejectedValueOnce(new Error('offline'));
    writeDraft(
      'g1',
      { nodes: [{ id: 'n1' }], edges: [] } as any,
      { baseRevision: 5, name: 'Server', description: 'Recovered notes' }
    );
    const recovered = await GraphStorageService.getGraph('g1', localDevelopmentDraftStore);
    expect(recovered?.recoveredDraft).toBe(true);

    commitGraph.mockImplementation(async (_pid: string, _gid: string, commit: any) => {
      const committed = JSON.parse(JSON.stringify(commit.document));
      committed.semantic.revision = 6;
      return record({
        id: 'g1', name: 'Server', description: 'Recovered notes',
        currentRevision: 6, document: committed,
      });
    });
    const queue = new RevisionedGraphSaveQueue(GraphStorageService.saveGraph.bind(GraphStorageService), {
      onPendingChanged: ({ graphId, revision, requests }) => localDevelopmentDraftStore.replaceJournal(
        graphId!,
        revision,
        requests.map((request) => ({
          document: graphSaveDocument(request, graphId!, revision),
          author: request.author,
          ...(request.opSummary ? { opSummary: request.opSummary } : {}),
        }))
      ),
    });
    queue.configure(recovered!.id, recovered!.currentRevision, null);

    await queue.enqueue({
      graph: recovered!.graph,
      name: recovered!.name,
      description: recovered!.description,
      author: 'user',
      operations: recovered!.document.semantic.operations,
      layout: recovered!.document.view.layout,
    });

    expect(commitGraph).toHaveBeenCalledTimes(1);
    expect(commitGraph.mock.calls[0][2]).toMatchObject({ expectedRevision: 5 });
    expect(queue.authority()).toEqual({ graphId: 'g1', revision: 6 });
    expect(localDevelopmentDraftStore.readRevisioned('g1')).toBeNull();
  });

  it('returns null only for a real 404', async () => {
    getGraph.mockRejectedValue(new Error('API Error (404): graph not found'));
    await expect(GraphStorageService.getGraph('missing', localDevelopmentDraftStore))
      .resolves.toBeNull();
  });

  it('does not discard a local draft when its saved graph returns 404', async () => {
    getGraph.mockRejectedValue(new Error('API Error (404): graph not found'));
    writeDraft('missing', { nodes: [{ id: 'n1' }], edges: [] } as any, { baseRevision: 2 });

    await expect(GraphStorageService.getGraph('missing', localDevelopmentDraftStore))
      .rejects.toThrow('draft was preserved');
    expect(localDevelopmentDraftStore.readRevisioned('missing')).not.toBeNull();
  });

  it('propagates non-404 load failures instead of opening an empty graph', async () => {
    getGraph.mockRejectedValue(new Error('API Error (500): unavailable'));
    writeDraft('g1', { nodes: [{ id: 'n1' }], edges: [] } as any, { baseRevision: 5 });

    await expect(GraphStorageService.getGraph('g1', localDevelopmentDraftStore))
      .rejects.toThrow('500');
  });

  it('propagates document decode failures instead of using a draft', async () => {
    getGraph.mockRejectedValue(new SyntaxError('Unexpected token'));
    writeDraft('g1', { nodes: [{ id: 'n1' }], edges: [] } as any, { baseRevision: 5 });

    await expect(GraphStorageService.getGraph('g1', localDevelopmentDraftStore))
      .rejects.toThrow('Unexpected token');
  });

  it('deletes on the server', async () => {
    deleteGraph.mockResolvedValue(undefined);
    await expect(GraphStorageService.deleteGraph('g1', localDevelopmentDraftStore))
      .resolves.toBe(true);
    expect(deleteGraph).toHaveBeenCalledWith('p1', 'g1');
  });
});

describe('legacy local import', () => {
  it('converts every local ModelGraph to V2 and leaves the source blob intact', async () => {
    const source = JSON.stringify([
      { id: 'graph-1', name: 'A', graph: emptyGraph },
      { id: 'graph-2', name: 'B', graph: emptyGraph },
    ]);
    localStorage.setItem('canvas-saved-graphs', source);
    createGraph.mockResolvedValue(record());

    expect(await GraphStorageService.importLocalGraphs()).toBe(2);
    expect(createGraph).toHaveBeenCalledTimes(2);
    expect(createGraph.mock.calls[0][2].schemaVersion).toBe(2);
    expect(createGraph.mock.calls[0][2].semantic.metadata.migration.legacyPayload).toEqual(emptyGraph);
    expect(localStorage.getItem('canvas-saved-graphs')).toBe(source);
    expect(GraphStorageService.getUnmigratedLocalGraphs()).toHaveLength(0);
  });

  it('persists dismissal without marking or deleting legacy graphs', () => {
    localStorage.setItem(
      'canvas-saved-graphs',
      JSON.stringify([{ id: 'graph-1', name: 'Legacy', graph: emptyGraph }])
    );
    GraphStorageService.dismissLocalGraphImport();
    expect(GraphStorageService.isLocalGraphImportDismissed()).toBe(true);
    expect(localStorage.getItem('canvas-migrated')).toBeNull();
    expect(GraphStorageService.getUnmigratedLocalGraphs()).toHaveLength(1);
  });
});
