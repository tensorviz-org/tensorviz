/**
 * Thin typed client for the owner-scoped projects / graphs persistence API.
 *
 * Every call goes through ApiService.request<T> so it inherits the Firebase
 * Bearer token (and admin key) attached by ApiService.buildHeaders.
 *
 * All snake_case <-> camelCase conversion for graph rows lives in the mappers
 * below (mapGraphRow / mapVersionRow). Nothing outside this module touches the
 * raw API field names (project_id, current_revision, created_at, updated_at).
 */
import { ApiService } from './apiService';
import { ModelGraph } from '../types';
import { GraphDocumentV2 } from '../types/graphDocument';
import {
  isGraphDocumentV2,
  legacyModelGraphToDocumentV2,
} from './graphDocumentAdapter';

// --- Client-facing types (camelCase) ---------------------------------------

export interface Project {
  id: string;
  name: string;
  instructions: string;
  createdAt: string;
  updatedAt: string;
}

export interface GraphRecord {
  id: string;
  projectId: string;
  name: string;
  description?: string;
  document: GraphDocumentV2;
  currentRevision: number;
  createdAt: string;
  updatedAt: string;
}

export interface GraphVersion {
  revision: number;
  author: string;
  opSummary: string | null;
  createdAt: string;
}

// --- Raw API row shapes (snake_case) ---------------------------------------

interface ProjectRow {
  id: string;
  name: string;
  instructions?: string;
  created_at: string;
  updated_at: string;
}

interface GraphRow {
  id: string;
  project_id: string;
  name: string;
  document: GraphDocumentV2 | ModelGraph | string | null;
  current_revision: number;
  created_at: string;
  updated_at: string;
}

interface GraphVersionRow {
  revision: number;
  author: string;
  op_summary: string | null;
  created_at: string;
}

// --- Mappers (the only place that reads raw field names) --------------------

function parseDocument(row: GraphRow): GraphDocumentV2 {
  let raw: GraphDocumentV2 | ModelGraph | string | null = row.document;
  if (typeof raw === 'string') {
    raw = raw ? (JSON.parse(raw) as GraphDocumentV2 | ModelGraph) : null;
  }
  if (isGraphDocumentV2(raw)) return raw;
  const legacy = raw && Array.isArray((raw as ModelGraph).nodes) && Array.isArray((raw as ModelGraph).edges)
    ? raw as ModelGraph
    : { nodes: [], edges: [] };
  return legacyModelGraphToDocumentV2(legacy, {
    graphId: row.id,
    name: row.name,
    revision: row.current_revision || 0,
  });
}

function mapProjectRow(row: ProjectRow): Project {
  return {
    id: row.id,
    name: row.name,
    instructions: row.instructions || '',
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

function mapGraphRow(row: GraphRow): GraphRecord {
  const document = parseDocument(row);
  const description = document.semantic.metadata?.description;
  return {
    id: row.id,
    projectId: row.project_id,
    name: row.name,
    document,
    ...(typeof description === 'string' ? { description } : {}),
    currentRevision: row.current_revision,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

function mapVersionRow(row: GraphVersionRow): GraphVersion {
  return {
    revision: row.revision,
    author: row.author,
    opSummary: row.op_summary,
    createdAt: row.created_at,
  };
}

// --- Projects ---------------------------------------------------------------

export async function listProjects(): Promise<Project[]> {
  const { projects } = await ApiService.request<{ projects: ProjectRow[] }>('/api/projects', { method: 'GET' });
  return projects.map(mapProjectRow);
}

export async function getProject(projectId: string): Promise<Project> {
  const { project } = await ApiService.request<{ project: ProjectRow }>(
    `/api/projects/${projectId}`,
    { method: 'GET' },
  );
  return mapProjectRow(project);
}

export async function createProject(name: string): Promise<Project> {
  const { project } = await ApiService.request<{ project: ProjectRow }>('/api/projects', {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
  return mapProjectRow(project);
}

export async function updateProject(
  projectId: string,
  updates: { name?: string; instructions?: string },
): Promise<Project> {
  const { project } = await ApiService.request<{ project: ProjectRow }>(
    `/api/projects/${projectId}`,
    {
      method: 'PATCH',
      body: JSON.stringify(updates),
    },
  );
  return mapProjectRow(project);
}

export async function deleteProject(pid: string): Promise<void> {
  await ApiService.request<void>(`/api/projects/${pid}`, { method: 'DELETE' });
}

// --- Default project ------------------------------------------------------

let defaultProjectId: string | null = null;
let defaultProjectResolution: Promise<string> | null = null;
let defaultProjectCacheEpoch = 0;

function assertDefaultProjectEpoch(expectedEpoch: number): void {
  if (defaultProjectCacheEpoch !== expectedEpoch) {
    throw new Error('Default project resolution was superseded by an auth change');
  }
}

/** Clear the cached default project id (call on sign-out / 401). */
export function resetDefaultProjectCache(): void {
  defaultProjectCacheEpoch += 1;
  defaultProjectId = null;
  defaultProjectResolution = null;
}

/**
 * Resolve the owner's default project id, creating a 'My Models' project the
 * first time. Cached in module scope; prerequisite for creating graphs.
 */
export async function ensureDefaultProject(): Promise<string> {
  if (defaultProjectId) {
    return defaultProjectId;
  }
  if (defaultProjectResolution) return defaultProjectResolution;

  const resolutionEpoch = defaultProjectCacheEpoch;
  const resolution = (async () => {
    const projects = await listProjects();
    assertDefaultProjectEpoch(resolutionEpoch);
    if (projects.length > 0) return projects[0].id;
    const created = await createProject('My Models');
    assertDefaultProjectEpoch(resolutionEpoch);
    return created.id;
  })();
  defaultProjectResolution = resolution;
  try {
    const resolvedId = await resolution;
    assertDefaultProjectEpoch(resolutionEpoch);
    defaultProjectId = resolvedId;
    return resolvedId;
  } finally {
    if (
      defaultProjectCacheEpoch === resolutionEpoch
      && defaultProjectResolution === resolution
    ) {
      defaultProjectResolution = null;
    }
  }
}

// --- Graphs -----------------------------------------------------------------

export async function listGraphs(pid: string): Promise<GraphRecord[]> {
  const { graphs } = await ApiService.request<{ graphs: GraphRow[] }>(`/api/projects/${pid}/graphs`, { method: 'GET' });
  return graphs.map(mapGraphRow);
}

export async function getGraph(pid: string, gid: string): Promise<GraphRecord> {
  const { graph } = await ApiService.request<{ graph: GraphRow }>(`/api/projects/${pid}/graphs/${gid}`, { method: 'GET' });
  return mapGraphRow(graph);
}

export async function createGraph(
  pid: string,
  name: string,
  document?: GraphDocumentV2,
  commit?: { author?: 'user' | 'agent'; opSummary?: string; idempotencyKey?: string }
): Promise<GraphRecord> {
  const { graph: row } = await ApiService.request<{ graph: GraphRow }>(`/api/projects/${pid}/graphs`, {
    method: 'POST',
    timeoutMs: 20_000,
    body: JSON.stringify({
      name,
      ...(document !== undefined ? { document } : {}),
      ...(commit?.author !== undefined ? { author: commit.author } : {}),
      ...(commit?.opSummary !== undefined ? { op_summary: commit.opSummary } : {}),
      ...(commit?.idempotencyKey !== undefined
        ? { idempotency_key: commit.idempotencyKey }
        : {}),
    }),
  });
  return mapGraphRow(row);
}

export async function commitGraph(
  pid: string,
  gid: string,
  commit: {
    document: GraphDocumentV2;
    expectedRevision: number;
    name?: string;
    opSummary?: string;
    author?: 'user' | 'agent';
  }
): Promise<GraphRecord> {
  const body: Record<string, unknown> = {
    document: commit.document,
    expected_revision: commit.expectedRevision,
  };
  if (commit.name !== undefined) body.name = commit.name;
  if (commit.opSummary !== undefined) body.op_summary = commit.opSummary;
  if (commit.author !== undefined) body.author = commit.author;

  const { graph: row } = await ApiService.request<{ graph: GraphRow }>(`/api/projects/${pid}/graphs/${gid}`, {
    method: 'PATCH',
    timeoutMs: 20_000,
    body: JSON.stringify(body),
  });
  return mapGraphRow(row);
}

export async function deleteGraph(pid: string, gid: string): Promise<void> {
  await ApiService.request<void>(`/api/projects/${pid}/graphs/${gid}`, { method: 'DELETE' });
}

// --- Versions ---------------------------------------------------------------

export async function listVersions(pid: string, gid: string): Promise<GraphVersion[]> {
  const { versions } = await ApiService.request<{ versions: GraphVersionRow[] }>(
    `/api/projects/${pid}/graphs/${gid}/versions`,
    { method: 'GET' }
  );
  return versions.map(mapVersionRow);
}

export async function restoreVersion(
  pid: string,
  gid: string,
  rev: number,
  expectedRevision: number
): Promise<GraphRecord> {
  const { graph } = await ApiService.request<{ graph: GraphRow }>(
    `/api/projects/${pid}/graphs/${gid}/restore/${rev}`,
    { method: 'POST', body: JSON.stringify({ expected_revision: expectedRevision }) }
  );
  return mapGraphRow(graph);
}
