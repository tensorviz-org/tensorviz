/**
 * Typed client for the model-ingestion API (`/api/ingest/*`).
 *
 * Mirrors projectsApi/userTemplatesApi conventions: every call goes through
 * ApiService.request<T> so it inherits auth headers and the base URL, and the
 * snake_case <-> camelCase mapping lives here rather than leaking into components.
 */
import { ApiService } from './apiService';
import { ModelGraph } from '../types';

export type IngestLibrary = 'torchvision' | 'transformers' | 'diffusers' | 'timm';

export interface CatalogEntry {
  library: string;
  arch: string;
  kind: string;
}

export interface EntrypointCandidate {
  name: string;
  score: number;
  file: string;
  line: number;
  reasons: string[];
  factories: string[];
}

export interface RepeatGroup {
  path: string;
  count: number;
  reason: 'module-list' | 'call-count' | 'fused-parameter';
}

export interface IngestDiagnostics {
  tier: 'fake' | 'real' | 'tree-only';
  rawModules: number;
  emittedNodes: number;
  dataflowEdges: number;
  atenOps: number;
  nodesWithObservedShapes: number;
  repeatGroups: RepeatGroup[];
  tiedParameters: string[][];
  unreachedModules: string[];
  opaqueOps: string[];
  stubbedDependencies: string[];
  exampleInput: number[][];
  provenance: string;
  warnings: string[];
  timingsMs: Record<string, number>;
}

export interface IngestResult {
  ok: boolean;
  name: string;
  graph: ModelGraph;
  diagnostics: IngestDiagnostics;
  candidates: EntrypointCandidate[];
  graphId?: string;
  projectId?: string;
  builderPath?: string;
  error?: string;
}

interface RawDiagnostics {
  tier: IngestDiagnostics['tier'];
  raw_modules: number;
  emitted_nodes: number;
  dataflow_edges: number;
  aten_ops: number;
  nodes_with_observed_shapes: number;
  repeat_groups: RepeatGroup[];
  tied_parameters: string[][];
  unreached_modules: string[];
  opaque_ops: string[];
  stubbed_dependencies: string[];
  example_input: number[][];
  provenance: string;
  warnings: string[];
  timings_ms: Record<string, number>;
}

interface RawResult {
  ok: boolean;
  name: string;
  graph?: ModelGraph;
  diagnostics: RawDiagnostics;
  candidates?: EntrypointCandidate[];
  graph_id?: string;
  project_id?: string;
  builder_path?: string;
  error?: string;
}

function mapDiagnostics(raw: RawDiagnostics): IngestDiagnostics {
  return {
    tier: raw.tier,
    rawModules: raw.raw_modules,
    emittedNodes: raw.emitted_nodes,
    dataflowEdges: raw.dataflow_edges,
    atenOps: raw.aten_ops,
    nodesWithObservedShapes: raw.nodes_with_observed_shapes,
    repeatGroups: raw.repeat_groups || [],
    tiedParameters: raw.tied_parameters || [],
    unreachedModules: raw.unreached_modules || [],
    opaqueOps: raw.opaque_ops || [],
    stubbedDependencies: raw.stubbed_dependencies || [],
    exampleInput: raw.example_input || [],
    provenance: raw.provenance || '',
    warnings: raw.warnings || [],
    timingsMs: raw.timings_ms || {},
  };
}

function mapResult(raw: RawResult): IngestResult {
  return {
    ok: raw.ok,
    name: raw.name,
    graph: raw.graph ?? { nodes: [], edges: [] },
    diagnostics: mapDiagnostics(raw.diagnostics),
    candidates: raw.candidates || [],
    graphId: raw.graph_id,
    projectId: raw.project_id,
    builderPath: raw.builder_path,
    error: raw.error,
  };
}

/** Which model libraries this deployment can import from. */
export async function getLibraries(): Promise<Record<IngestLibrary, boolean>> {
  const { libraries } = await ApiService.request<{
    libraries: Record<IngestLibrary, boolean>;
  }>('/api/ingest/libraries', { method: 'GET' });
  return libraries;
}

/**
 * Paginated architecture index. Paginated deliberately — the full index across all
 * four libraries is thousands of entries.
 */
export async function searchCatalog(params: {
  library?: string;
  q?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; entries: CatalogEntry[] }> {
  const query = new URLSearchParams();
  if (params.library) query.set('library', params.library);
  if (params.q) query.set('q', params.q);
  query.set('limit', String(params.limit ?? 40));
  query.set('offset', String(params.offset ?? 0));
  const res = await ApiService.request<{ total: number; entries: CatalogEntry[] }>(
    `/api/ingest/catalog?${query.toString()}`,
    { method: 'GET' }
  );
  return { total: res.total, entries: res.entries || [] };
}

/** Import an architecture from an installed library. No weights are downloaded. */
export async function ingestModel(body: {
  library: string;
  arch: string;
  depth?: number;
  name?: string;
  overrides?: Record<string, unknown>;
}): Promise<IngestResult> {
  const raw = await ApiService.request<RawResult>('/api/ingest/model', {
    method: 'POST',
    body: JSON.stringify({ depth: 2, save: true, ...body }),
  });
  return mapResult(raw);
}

/**
 * Rank a repository's candidate model classes. Pure static analysis — nothing in the
 * repository is imported or executed by this call.
 */
export async function scanRepo(repo: string): Promise<EntrypointCandidate[]> {
  const { candidates } = await ApiService.request<{ candidates: EntrypointCandidate[] }>(
    '/api/ingest/repo/scan',
    { method: 'POST', body: JSON.stringify({ repo }) }
  );
  return candidates || [];
}

/** Import a model from a local repository. This DOES execute the repository's code. */
export async function ingestRepo(body: {
  repo: string;
  entrypoint?: string;
  depth?: number;
  name?: string;
  overrides?: Record<string, unknown>;
}): Promise<IngestResult> {
  const raw = await ApiService.request<RawResult>('/api/ingest/repo', {
    method: 'POST',
    body: JSON.stringify({ depth: 2, save: true, ...body }),
  });
  return mapResult(raw);
}
