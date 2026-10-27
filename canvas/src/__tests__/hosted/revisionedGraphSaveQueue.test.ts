import {
  RevisionedGraphSaveQueue,
  GraphSaveRequest,
  graphSaveDocument,
  graphSaveFingerprint,
} from '../../services/revisionedGraphSaveQueue';
import { modelGraphToDocumentV2 } from '../../services/graphDocumentAdapter';
import {
  localDevelopmentDraftStore,
  UNSAVED_DRAFT_ID,
} from '../../services/localDraft';

const graph = { nodes: [], edges: [] } as any;
const populatedGraph = {
  nodes: [{
    id: 'linear-1',
    type: 'layerNode',
    position: { x: 0, y: 0 },
    data: {
      layerType: {
        id: 'linear', name: 'Linear', displayName: 'Linear', description: '',
        icon: 'Box', category: 'linear', parameters: [],
      },
      parameters: { in_features: 4, out_features: 2 },
    },
  }],
  edges: [],
} as any;

function request(label: string, author: 'user' | 'agent' = 'user'): GraphSaveRequest {
  return {
    graph,
    name: label,
    author,
    opSummary: label,
    operations: author === 'agent' ? [{ op: 'add_layer' }] : [],
    layout: { strategy: 'manual' },
  };
}

function autosave(label: string): GraphSaveRequest {
  return { ...request(label), opSummary: undefined };
}

function saved(id: string, revision: number, label: string) {
  return {
    id,
    name: label,
    graph,
    document: modelGraphToDocumentV2(graph, { graphId: id, name: label, revision }),
    currentRevision: revision,
    createdAt: 't',
    updatedAt: 't',
  };
}

beforeEach(() => {
  localStorage.clear();
});

function crashJournalCallbacks(initialKey: string | null = null) {
  let authorityKey = initialKey;
  return {
    onPendingChanged: ({ graphId, revision, requests }: {
      graphId?: string;
      revision: number;
      requests: GraphSaveRequest[];
    }) => {
      const nextKey = graphId ?? UNSAVED_DRAFT_ID;
      localDevelopmentDraftStore.replaceJournal(
        nextKey,
        revision,
        requests.map((item) => ({
          document: graphSaveDocument(item, nextKey, revision),
          author: item.author,
          ...(item.opSummary ? { opSummary: item.opSummary } : {}),
          ...(item.createKey ? { createKey: item.createKey } : {}),
        }))
      );
      if (authorityKey && authorityKey !== nextKey) {
        localDevelopmentDraftStore.clear(authorityKey);
      }
      authorityKey = nextKey;
    },
  };
}

it('serializes overlapping commits and advances expected revisions', async () => {
  let releaseFirst: (() => void) | undefined;
  const persist = jest.fn()
    .mockImplementationOnce(() => new Promise((resolve) => {
      releaseFirst = () => resolve(saved('g1', 2, 'first'));
    }))
    .mockResolvedValueOnce(saved('g1', 3, 'second'));
  const queue = new RevisionedGraphSaveQueue(persist);
  queue.configure('g1', 1, null);

  const first = queue.enqueue(request('first'));
  const second = queue.enqueue(request('second', 'agent'));
  releaseFirst?.();
  await Promise.all([first, second]);

  expect(persist).toHaveBeenCalledTimes(2);
  expect(persist.mock.calls[0][4]).toMatchObject({ expectedRevision: 1, author: 'user' });
  expect(persist.mock.calls[1][4]).toMatchObject({ expectedRevision: 2, author: 'agent' });
  expect(queue.authority().revision).toBe(3);
});

it('preserves user, agent, and user commit boundaries while a save is in flight', async () => {
  let releaseFirst: (() => void) | undefined;
  const persist = jest.fn()
    .mockImplementationOnce(() => new Promise((resolve) => {
      releaseFirst = () => resolve(saved('g1', 2, 'first-user'));
    }))
    .mockResolvedValueOnce(saved('g1', 3, 'agent'))
    .mockResolvedValueOnce(saved('g1', 4, 'second-user'));
  const queue = new RevisionedGraphSaveQueue(persist);
  queue.configure('g1', 1, null);

  const firstUser = queue.enqueue(request('first-user'));
  const agent = queue.enqueue(request('agent', 'agent'));
  const secondUser = queue.enqueue(request('second-user'));
  releaseFirst?.();
  await Promise.all([firstUser, agent, secondUser]);

  expect(persist).toHaveBeenCalledTimes(3);
  expect(persist.mock.calls.map((call) => call[4])).toEqual([
    expect.objectContaining({ expectedRevision: 1, author: 'user', opSummary: 'first-user' }),
    expect.objectContaining({ expectedRevision: 2, author: 'agent', opSummary: 'agent' }),
    expect.objectContaining({ expectedRevision: 3, author: 'user', opSummary: 'second-user' }),
  ]);
  expect(queue.authority().revision).toBe(4);
});

it('coalesces only compatible pending user autosaves to their latest snapshot', async () => {
  let releaseFirst: (() => void) | undefined;
  const persist = jest.fn()
    .mockImplementationOnce(() => new Promise((resolve) => {
      releaseFirst = () => resolve(saved('g1', 2, 'first'));
    }))
    .mockResolvedValueOnce(saved('g1', 3, 'draft-2'));
  const queue = new RevisionedGraphSaveQueue(persist);
  queue.configure('g1', 1, null);

  const first = queue.enqueue(request('first'));
  const draftOne = queue.enqueue(autosave('draft-1'));
  const draftTwo = queue.enqueue(autosave('draft-2'));
  releaseFirst?.();
  await Promise.all([first, draftOne, draftTwo]);

  expect(persist).toHaveBeenCalledTimes(2);
  expect(persist.mock.calls[1][1]).toBe('draft-2');
  expect(persist.mock.calls[1][4]).toMatchObject({
    expectedRevision: 2,
    author: 'user',
    opSummary: undefined,
  });
});

it('coalesces an identical pending snapshot instead of creating another revision', async () => {
  let release: (() => void) | undefined;
  const persist = jest.fn(() => new Promise((resolve) => {
    release = () => resolve(saved('g1', 2, 'same'));
  }));
  const queue = new RevisionedGraphSaveQueue(persist);
  queue.configure('g1', 1, null);

  const first = queue.enqueue(request('same'));
  const duplicate = queue.enqueue(request('same'));
  release?.();
  await Promise.all([first, duplicate]);

  expect(persist).toHaveBeenCalledTimes(1);
});

it('surfaces conflict without retrying or creating', async () => {
  const states: string[] = [];
  const persist = jest.fn().mockRejectedValue(new Error('API Error (409): graph_revision_conflict'));
  const queue = new RevisionedGraphSaveQueue(persist, { onState: (state) => states.push(state) });
  queue.configure('g1', 4, null);

  await expect(queue.enqueue(request('stale', 'agent'))).rejects.toThrow('409');
  queue.retry();
  await Promise.resolve();

  expect(persist).toHaveBeenCalledTimes(1);
  expect(states).toContain('conflict');
  expect(queue.authority().revision).toBe(4);
});

it('freezes a conflicted head and journals later edits behind it until reconfigure', async () => {
  const journal: string[][] = [];
  const persist = jest.fn().mockRejectedValue(
    new Error('API Error (409): graph_revision_conflict')
  );
  const queue = new RevisionedGraphSaveQueue(persist, {
    onPendingChanged: ({ requests }) => {
      journal.push(requests.map((item) => item.name));
    },
  });
  queue.configure('g1', 4, null);

  await expect(queue.enqueue(request('stale-agent', 'agent'))).rejects.toThrow('409');
  const laterUser = queue.enqueue(autosave('later-user'));
  const latestUser = queue.enqueue(autosave('latest-user'));
  await Promise.resolve();

  expect(persist).toHaveBeenCalledTimes(1);
  expect(journal[journal.length - 1]).toEqual(['stale-agent', 'latest-user']);
  queue.retry();
  await Promise.resolve();
  expect(persist).toHaveBeenCalledTimes(1);

  queue.configure('g1', 5, null);
  await expect(Promise.all([laterUser, latestUser])).resolves.toEqual([null, null]);
});

it('retries an offline commit before later author boundaries', async () => {
  const persist = jest.fn()
    .mockRejectedValueOnce(new Error('offline'))
    .mockResolvedValueOnce(saved('g1', 2, 'agent'))
    .mockResolvedValueOnce(saved('g1', 3, 'later-user'));
  const queue = new RevisionedGraphSaveQueue(persist);
  queue.configure('g1', 1, null);

  await expect(queue.enqueue(request('agent', 'agent'))).rejects.toThrow('offline');
  const laterUser = queue.enqueue(autosave('later-user'));
  expect(persist).toHaveBeenCalledTimes(1);

  queue.retry();
  await laterUser;

  expect(persist).toHaveBeenCalledTimes(3);
  expect(persist.mock.calls.map((call) => call[4])).toEqual([
    expect.objectContaining({ expectedRevision: 1, author: 'agent' }),
    expect.objectContaining({ expectedRevision: 1, author: 'agent' }),
    expect.objectContaining({ expectedRevision: 2, author: 'user' }),
  ]);
  expect(queue.authority().revision).toBe(3);
});

it('rejects an authority boundary immediately when offline recovery is pending', async () => {
  const persist = jest.fn().mockRejectedValue(new Error('offline'));
  const queue = new RevisionedGraphSaveQueue(persist);
  queue.configure('g1', 1, null);

  await expect(queue.enqueue(autosave('draft'))).rejects.toThrow('offline');
  await expect(queue.enqueue(request('ai-boundary'), {
    rejectIfBlocked: true,
  })).rejects.toThrow('Retry the save before starting an AI turn');

  expect(persist).toHaveBeenCalledTimes(1);
});

it('reuses the same idempotency key when the first create response is lost', async () => {
  const firstCreate = { ...autosave('new'), createKey: 'create-stable' };
  const persist = jest.fn()
    .mockRejectedValueOnce(new Error('offline'))
    .mockResolvedValueOnce(saved('server-id', 1, 'new'));
  const queue = new RevisionedGraphSaveQueue(persist);
  queue.configure(undefined, 0, null);

  await expect(queue.enqueue(firstCreate)).rejects.toThrow('offline');
  queue.retry();
  await new Promise((resolve) => setTimeout(resolve, 0));

  expect(persist).toHaveBeenCalledTimes(2);
  expect(persist.mock.calls.map((call) => call[4].createKey)).toEqual([
    'create-stable',
    'create-stable',
  ]);
  expect(persist.mock.calls.map((call) => call[3])).toEqual([undefined, undefined]);
  expect(queue.authority()).toEqual({ graphId: 'server-id', revision: 1 });
});

it('preserves a possibly-sent autosave head before a later autosave on retry', async () => {
  const journal: string[][] = [];
  const persist = jest.fn()
    .mockRejectedValueOnce(new Error('offline'))
    .mockResolvedValueOnce(saved('g1', 2, 'ambiguous-a'))
    .mockResolvedValueOnce(saved('g1', 3, 'later-b'));
  const queue = new RevisionedGraphSaveQueue(persist, {
    onPendingChanged: ({ requests }) => {
      journal.push(requests.map((item) => item.name));
    },
  });
  queue.configure('g1', 1, null);

  await expect(queue.enqueue(autosave('ambiguous-a'))).rejects.toThrow('offline');
  const later = queue.enqueue(autosave('later-b'));

  expect(persist).toHaveBeenCalledTimes(1);
  expect(journal[journal.length - 1]).toEqual(['ambiguous-a', 'later-b']);

  queue.retry();
  await later;

  expect(persist.mock.calls.map((call) => call[1])).toEqual([
    'ambiguous-a',
    'ambiguous-a',
    'later-b',
  ]);
  expect(persist.mock.calls.slice(1).map((call) => call[4].expectedRevision)).toEqual([1, 2]);
  expect(queue.authority().revision).toBe(3);
});

it('skips the hydration baseline exactly', async () => {
  const persist = jest.fn();
  const baseline = request('loaded');
  const queue = new RevisionedGraphSaveQueue(persist);
  queue.configure('g1', 8, graphSaveFingerprint(baseline));

  expect(queue.needsSave(baseline)).toBe(false);
  expect(queue.needsSave(request('changed'))).toBe(true);
  await expect(queue.enqueue(baseline)).resolves.toBeNull();
  expect(persist).not.toHaveBeenCalled();
});

it('does not invent a user preflush between identical sequential agent snapshots', async () => {
  const persist = jest.fn().mockResolvedValue(saved('g1', 2, 'same'));
  const agentSnapshot = request('same', 'agent');
  const queue = new RevisionedGraphSaveQueue(persist);
  queue.configure('g1', 1, null);
  await queue.enqueue(agentSnapshot);

  const attributedAsUser: GraphSaveRequest = {
    ...agentSnapshot,
    author: 'user',
    opSummary: undefined,
  };

  expect(queue.needsSave(attributedAsUser)).toBe(true);
  expect(queue.needsSnapshotSave(attributedAsUser)).toBe(false);
});

it('detects a real user snapshot before the next agent boundary', async () => {
  const persist = jest.fn().mockResolvedValue(saved('g1', 2, 'agent'));
  const queue = new RevisionedGraphSaveQueue(persist);
  queue.configure('g1', 1, null);
  await queue.enqueue(request('agent', 'agent'));

  const editedByUser: GraphSaveRequest = {
    ...request('agent', 'agent'),
    graph: populatedGraph,
    author: 'user',
    opSummary: undefined,
  };

  expect(queue.needsSnapshotSave(editedByUser)).toBe(true);
});

it('treats a description-only edit as a document change', () => {
  const before = { ...autosave('loaded'), description: 'Before' };
  const after = { ...autosave('loaded'), description: 'After' };

  expect(graphSaveFingerprint(after)).not.toBe(graphSaveFingerprint(before));
});

it.each(['user', 'agent'] as const)(
  'persists a %s-authored clear as exactly one empty-graph revision',
  async (author) => {
    const persist = jest.fn().mockResolvedValue(saved('g1', 2, 'G'));
    const queue = new RevisionedGraphSaveQueue(persist);
    const baseline = { ...autosave('G'), graph: populatedGraph };
    queue.configure('g1', 1, graphSaveFingerprint(baseline));
    const cleared: GraphSaveRequest = {
      ...autosave('G'),
      author,
      ...(author === 'agent' ? {
        opSummary: 'AI: clear graph',
        operations: [{ op: 'remove_layer', node_id: 'linear-1' }],
      } : {}),
    };

    await queue.enqueue(cleared);

    expect(persist).toHaveBeenCalledTimes(1);
    expect(persist.mock.calls[0][0]).toEqual(graph);
    expect(persist.mock.calls[0][4]).toMatchObject({ expectedRevision: 1, author });
    expect(queue.authority().revision).toBe(2);
  }
);

it('adopts a server id when a local template is first persisted', async () => {
  const onCreated = jest.fn();
  const persist = jest.fn().mockResolvedValue(saved('server-id', 1, 'template'));
  const queue = new RevisionedGraphSaveQueue(persist, { onCreated });
  queue.configure('template_local', 0, null);

  await queue.enqueue(request('template'));

  expect(onCreated).toHaveBeenCalledWith(expect.objectContaining({ id: 'server-id' }));
  expect(queue.authority()).toEqual({ graphId: 'server-id', revision: 1 });
});

it('keeps the latest crash draft when an earlier FIFO save succeeds and the next fails', async () => {
  let releaseFirst: (() => void) | undefined;
  const persist = jest.fn()
    .mockImplementationOnce(() => new Promise((resolve) => {
      releaseFirst = () => resolve(saved('g1', 2, 'first'));
    }))
    .mockRejectedValueOnce(new Error('offline'));
  const queue = new RevisionedGraphSaveQueue(persist, crashJournalCallbacks('g1'));
  queue.configure('g1', 1, null);

  const first = queue.enqueue(request('first'));
  const secondRequest = request('second');
  const second = queue.enqueue(secondRequest);
  releaseFirst?.();

  await first;
  await expect(second).rejects.toThrow('offline');

  expect(localDevelopmentDraftStore.readRevisioned('g1')).toMatchObject({
    baseRevision: 2,
    document: { semantic: { name: 'second' } },
    pendingCommits: [{ author: 'user', opSummary: 'second' }],
  });
});

it('promotes a newer first-save draft onto the minted server revision', async () => {
  let releaseFirst: (() => void) | undefined;
  const persist = jest.fn()
    .mockImplementationOnce(() => new Promise((resolve) => {
      releaseFirst = () => resolve(saved('g1', 1, 'first'));
    }))
    .mockRejectedValueOnce(new Error('offline'));
  const queue = new RevisionedGraphSaveQueue(
    persist,
    crashJournalCallbacks(UNSAVED_DRAFT_ID)
  );
  queue.configure(undefined, 0, null);

  const first = queue.enqueue(request('first'));
  const secondRequest = request('second');
  const second = queue.enqueue(secondRequest);
  releaseFirst?.();

  await first;
  await expect(second).rejects.toThrow('offline');

  expect(localDevelopmentDraftStore.readRevisioned(UNSAVED_DRAFT_ID)).toBeNull();
  expect(localDevelopmentDraftStore.readRevisioned('g1')).toMatchObject({
    baseRevision: 1,
    document: {
      semantic: { graph_id: 'g1', revision: 1, name: 'second' },
      view: { graph_id: 'g1' },
    },
    pendingCommits: [{ author: 'user', opSummary: 'second' }],
  });
});

it('ignores a stale in-flight completion after authority is reconfigured', async () => {
  let releaseOld: (() => void) | undefined;
  const onCreated = jest.fn();
  const persist = jest.fn()
    .mockImplementationOnce(() => new Promise((resolve) => {
      releaseOld = () => resolve(saved('old-server-id', 1, 'old'));
    }))
    .mockResolvedValueOnce(saved('g2', 8, 'new'));
  const queue = new RevisionedGraphSaveQueue(persist, { onCreated });
  queue.configure(undefined, 0, null);

  const oldSave = queue.enqueue(request('old'));
  queue.configure('g2', 7, null);
  const newSave = queue.enqueue(request('new'));
  releaseOld?.();

  await expect(oldSave).resolves.toBeNull();
  await expect(newSave).resolves.toMatchObject({ id: 'g2', currentRevision: 8 });
  expect(persist).toHaveBeenCalledTimes(2);
  expect(persist.mock.calls[1][3]).toBe('g2');
  expect(persist.mock.calls[1][4]).toMatchObject({ expectedRevision: 7 });
  expect(onCreated).not.toHaveBeenCalled();
  expect(queue.authority()).toEqual({ graphId: 'g2', revision: 8 });
});

it('suppresses every lifecycle callback when an in-flight create finishes after retirement', async () => {
  let releaseCreate: (() => void) | undefined;
  const callbacks = {
    onState: jest.fn(),
    onCreated: jest.fn(),
    onSaved: jest.fn(),
    onPendingChanged: jest.fn(),
  };
  const persist = jest.fn(() => new Promise((resolve) => {
    releaseCreate = () => resolve(saved('server-id', 1, 'new'));
  }));
  const queue = new RevisionedGraphSaveQueue(persist, callbacks);
  queue.configure(undefined, 0, null);

  const creating = queue.enqueue(request('new'));
  Object.values(callbacks).forEach((callback) => callback.mockClear());
  queue.retireCallbacks();
  releaseCreate?.();

  await expect(creating).resolves.toMatchObject({ id: 'server-id', currentRevision: 1 });
  expect(callbacks.onState).not.toHaveBeenCalled();
  expect(callbacks.onCreated).not.toHaveBeenCalled();
  expect(callbacks.onSaved).not.toHaveBeenCalled();
  expect(callbacks.onPendingChanged).not.toHaveBeenCalled();
  expect(queue.authority()).toEqual({ graphId: 'server-id', revision: 1 });
});
