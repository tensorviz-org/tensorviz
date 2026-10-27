/**
 * Typed client for the owner-scoped personal-templates ("starter files") API.
 * Mirrors projectsApi conventions: every call goes through ApiService.request<T>
 * and snake_case <-> camelCase mapping lives here.
 */
import { ApiService } from './apiService';
import { ModelGraph } from '../types';
import { ensureDefaultProject, createGraph, GraphRecord } from './projectsApi';
import { modelGraphToDocumentV2 } from './graphDocumentAdapter';

export interface UserTemplate {
  id: string;
  name: string;
  description: string | null;
  graph: ModelGraph;
  createdAt: string;
}

interface UserTemplateRow {
  id: string;
  name: string;
  description: string | null;
  canonical_graph: ModelGraph | string | null;
  created_at: string;
}

function parseGraph(raw: ModelGraph | string | null | undefined): ModelGraph {
  if (typeof raw === 'string') {
    try { return JSON.parse(raw) as ModelGraph; } catch { return { nodes: [], edges: [] }; }
  }
  return raw ?? { nodes: [], edges: [] };
}

function mapRow(row: UserTemplateRow): UserTemplate {
  return { id: row.id, name: row.name, description: row.description, graph: parseGraph(row.canonical_graph), createdAt: row.created_at };
}

export async function saveUserTemplate(name: string, graph: ModelGraph, description?: string): Promise<UserTemplate> {
  const { template } = await ApiService.request<{ template: UserTemplateRow }>('/api/user-templates', {
    method: 'POST',
    body: JSON.stringify({ name, canonical_graph: graph, description }),
  });
  return mapRow(template);
}

export async function listUserTemplates(): Promise<UserTemplate[]> {
  const { templates } = await ApiService.request<{ templates: UserTemplateRow[] }>('/api/user-templates', { method: 'GET' });
  return templates.map(mapRow);
}

export async function deleteUserTemplate(id: string): Promise<void> {
  await ApiService.request(`/api/user-templates/${id}`, { method: 'DELETE' });
}

/** Duplicate-to-project: create a new graph in the default project from a template. */
export async function createGraphFromUserTemplate(template: UserTemplate): Promise<GraphRecord> {
  const pid = await ensureDefaultProject();
  return createGraph(
    pid,
    template.name,
    modelGraphToDocumentV2(template.graph, { name: template.name, revision: 0 })
  );
}
