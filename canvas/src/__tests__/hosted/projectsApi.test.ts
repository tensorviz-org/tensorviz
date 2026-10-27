import { ApiService } from '../../services/apiService';
import {
  commitGraph,
  createGraph,
  ensureDefaultProject,
  getProject,
  listProjects,
  resetDefaultProjectCache,
  restoreVersion,
  updateProject,
} from '../../services/projectsApi';
import type { GraphDocumentV2 } from '../../types/graphDocument';

jest.mock('../../services/apiService');

const requestMock = ApiService.request as unknown as jest.Mock;

const document: GraphDocumentV2 = {
  schemaVersion: 2,
  semantic: {
    graph_id: 'g1', name: 'G', revision: 1, inputs: [], nodes: [], edges: [],
    modules: [], operations: [], metadata: {},
  },
  view: {
    graph_id: 'g1', node_positions: {}, collapsed_modules: [], selection: [], layout: {},
  },
};

function graphRow(over: Record<string, unknown> = {}) {
  return {
    id: 'g1', project_id: 'proj1', name: 'G', document,
    current_revision: 1, created_at: 't1', updated_at: 't2', ...over,
  };
}

beforeEach(() => {
  requestMock.mockReset();
  resetDefaultProjectCache();
});

describe('ensureDefaultProject', () => {
  it("creates 'My Models' when the project list is empty, then caches it", async () => {
    requestMock
      .mockResolvedValueOnce({ projects: [] })
      .mockResolvedValueOnce({ project: { id: 'proj1', name: 'My Models', created_at: 't', updated_at: 't' } });

    expect(await ensureDefaultProject()).toBe('proj1');
    expect(JSON.parse(requestMock.mock.calls[1][1].body).name).toBe('My Models');
    expect(await ensureDefaultProject()).toBe('proj1');
    expect(requestMock).toHaveBeenCalledTimes(2);
  });

  it('shares one in-flight default-project resolution', async () => {
    requestMock.mockResolvedValueOnce({
      projects: [{ id: 'proj1', name: 'P', created_at: 't', updated_at: 't' }],
    });

    await expect(Promise.all([ensureDefaultProject(), ensureDefaultProject()]))
      .resolves.toEqual(['proj1', 'proj1']);
    expect(requestMock).toHaveBeenCalledTimes(1);
  });

  it('rejects a pre-reset lookup and cannot poison the next owner cache', async () => {
    let resolveOld: ((value: unknown) => void) | undefined;
    requestMock
      .mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }))
      .mockResolvedValueOnce({
        projects: [{ id: 'project-b', name: 'B', created_at: 't', updated_at: 't' }],
      });

    const oldLookup = ensureDefaultProject();
    resetDefaultProjectCache();
    await expect(ensureDefaultProject()).resolves.toBe('project-b');
    resolveOld?.({
      projects: [{ id: 'project-a', name: 'A', created_at: 't', updated_at: 't' }],
    });
    await expect(oldLookup).rejects.toThrow('superseded by an auth change');
    await expect(ensureDefaultProject()).resolves.toBe('project-b');
    expect(requestMock).toHaveBeenCalledTimes(2);
  });

  it('does not create a project when an empty stale lookup resolves after reset', async () => {
    let resolveOld: ((value: unknown) => void) | undefined;
    requestMock
      .mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }))
      .mockResolvedValueOnce({
        projects: [{ id: 'project-b', name: 'B', created_at: 't', updated_at: 't' }],
      });

    const oldLookup = ensureDefaultProject();
    resetDefaultProjectCache();
    await expect(ensureDefaultProject()).resolves.toBe('project-b');
    resolveOld?.({ projects: [] });

    await expect(oldLookup).rejects.toThrow('superseded by an auth change');
    expect(requestMock).toHaveBeenCalledTimes(2);
    expect(requestMock.mock.calls).not.toEqual(expect.arrayContaining([
      expect.arrayContaining(['/api/projects', expect.objectContaining({ method: 'POST' })]),
    ]));
  });
});

describe('project instructions', () => {
  it('maps legacy projects to empty instructions', async () => {
    requestMock.mockResolvedValueOnce({
      projects: [{ id: 'proj1', name: 'P', created_at: 't', updated_at: 't' }],
    });

    await expect(listProjects()).resolves.toEqual([
      { id: 'proj1', name: 'P', instructions: '', createdAt: 't', updatedAt: 't' },
    ]);
  });

  it('updates project instructions without renaming the project', async () => {
    requestMock.mockResolvedValueOnce({
      project: {
        id: 'proj1', name: 'P', instructions: 'Prefer minimal edits.',
        created_at: 't', updated_at: 't2',
      },
    });

    const project = await updateProject('proj1', { instructions: 'Prefer minimal edits.' });

    expect(requestMock).toHaveBeenCalledWith('/api/projects/proj1', {
      method: 'PATCH',
      body: JSON.stringify({ instructions: 'Prefer minimal edits.' }),
    });
    expect(project.instructions).toBe('Prefer minimal edits.');
  });

  it('loads one project for task settings', async () => {
    requestMock.mockResolvedValueOnce({
      project: {
        id: 'proj1', name: 'P', instructions: 'Keep outputs stable.',
        created_at: 't', updated_at: 't2',
      },
    });

    await expect(getProject('proj1')).resolves.toMatchObject({
      id: 'proj1', instructions: 'Keep outputs stable.',
    });
    expect(requestMock).toHaveBeenCalledWith('/api/projects/proj1', { method: 'GET' });
  });
});

describe('project graph documents', () => {
  it('creates with V2 and maps current_revision', async () => {
    requestMock.mockResolvedValueOnce({ graph: graphRow() });

    const record = await createGraph('proj1', 'G', document);

    const [url, options] = requestMock.mock.calls[0];
    expect(url).toBe('/api/projects/proj1/graphs');
    expect(options.method).toBe('POST');
    expect(JSON.parse(options.body)).toEqual({ name: 'G', document });
    expect(record.document).toEqual(document);
    expect(record.currentRevision).toBe(1);
  });

  it('maps the saved description from semantic metadata', async () => {
    requestMock.mockResolvedValueOnce({
      graph: graphRow({
        document: {
          ...document,
          semantic: {
            ...document.semantic,
            metadata: { description: 'Classifier notes' },
          },
        },
      }),
    });

    const record = await createGraph('proj1', 'G', document);

    expect(record.description).toBe('Classifier notes');
  });

  it('attributes an agent-authored first revision on create', async () => {
    requestMock.mockResolvedValueOnce({ graph: graphRow() });

    await createGraph('proj1', 'G', document, {
      author: 'agent',
      opSummary: 'AI: create graph',
    });

    const body = JSON.parse(requestMock.mock.calls[0][1].body);
    expect(body).toEqual({
      name: 'G',
      document,
      author: 'agent',
      op_summary: 'AI: create graph',
    });
  });

  it('sends the stable first-create idempotency key', async () => {
    requestMock.mockResolvedValueOnce({ graph: graphRow() });

    await createGraph('proj1', 'G', document, { idempotencyKey: 'create-1' });

    expect(JSON.parse(requestMock.mock.calls[0][1].body)).toEqual({
      name: 'G',
      document,
      idempotency_key: 'create-1',
    });
  });

  it('commits V2 with the expected revision and attribution', async () => {
    requestMock.mockResolvedValueOnce({ graph: graphRow({ current_revision: 2 }) });

    const record = await commitGraph('proj1', 'g1', {
      document,
      expectedRevision: 1,
      author: 'agent',
      opSummary: 'AI: 2 operations',
    });

    const [url, options] = requestMock.mock.calls[0];
    expect(url).toBe('/api/projects/proj1/graphs/g1');
    expect(options.method).toBe('PATCH');
    expect(JSON.parse(options.body)).toEqual({
      document,
      expected_revision: 1,
      author: 'agent',
      op_summary: 'AI: 2 operations',
    });
    expect(record.currentRevision).toBe(2);
  });

  it('restores with an expected revision', async () => {
    requestMock.mockResolvedValueOnce({ graph: graphRow({ current_revision: 4 }) });

    await restoreVersion('proj1', 'g1', 2, 3);

    const [url, options] = requestMock.mock.calls[0];
    expect(url).toBe('/api/projects/proj1/graphs/g1/restore/2');
    expect(options.method).toBe('POST');
    expect(JSON.parse(options.body)).toEqual({ expected_revision: 3 });
  });
});
