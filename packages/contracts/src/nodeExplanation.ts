/** Generated interpretations are presentation, never source or execution evidence. */
export interface NodeExplanationCitation {
  id: string;
  file: string;
  sourceDigest: string;
  startLine: number;
  endLine: number;
}

export interface NodeExplanation {
  title: string;
  purpose: string;
  inputs: string[];
  outputs: string[];
  steps: string[];
  uncertainties: string[];
  /** General terminology, separate from claims about this implementation. */
  concepts?: { term: string; explanation: string }[];
  /** Resolved by the host from source evidence the agent actually received. */
  citations: NodeExplanationCitation[];
}

export const SOURCE_DOCUMENTATION_REASONS = ['ambiguous', 'dynamic_binding', 'decorated', 'unresolved_import',
  'unresolved_symbol', 'unsupported_callable', 'analysis_limit'] as const;

/** Author-written documentation and a static declaration, never a live Python signature. */
export type NodeSourceDocumentation = {
  status: 'resolved';
  qualifiedName: string;
  kind: 'function' | 'method' | 'class';
  signature: string;
  docstring?: string;
  truncated: boolean;
  citation: NodeExplanationCitation;
} | { status: 'unavailable'; reason: typeof SOURCE_DOCUMENTATION_REASONS[number] };

export interface NodeExplanationState {
  requestId: string;
  nodeId: string;
  /** The graph service revision, including source and capture identity. */
  revision: string;
  status: 'queued' | 'loading' | 'ready' | 'unavailable';
  detail?: string;
  explanation?: NodeExplanation;
  documentation?: NodeSourceDocumentation;
}

/** Progress for an explicit, bounded pass over the current graph. */
export interface ModelExplanationState {
  requestId: string;
  revision: string;
  status: 'loading' | 'ready' | 'unavailable' | 'cancelled';
  completed: number;
  total: number;
  detail?: string;
}

const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const keys = (value: Record<string, unknown>, allowed: readonly string[]): boolean =>
  Object.keys(value).every(key => allowed.includes(key));
const text = (value: unknown, max: number): value is string => typeof value === 'string'
  && value.length > 0 && value.length <= max && value.trim() === value
  && Array.from(value).every(character => { const code = character.charCodeAt(0); return code >= 32 && code !== 127; });
const list = (value: unknown, maxItems: number, maxLength: number): value is string[] =>
  Array.isArray(value) && value.length <= maxItems && value.every(item => text(item, maxLength));

export function isNodeSourceDocumentation(value: unknown): value is NodeSourceDocumentation {
  if (!record(value)) return false;
  if (value.status === 'unavailable') return Object.keys(value).length === 2 && keys(value, ['status', 'reason'])
    && typeof value.reason === 'string' && (SOURCE_DOCUMENTATION_REASONS as readonly string[]).includes(value.reason);
  const docstring = value.docstring;
  return value.status === 'resolved'
    && keys(value, ['status', 'qualifiedName', 'kind', 'signature', 'docstring', 'truncated', 'citation'])
    && text(value.qualifiedName, 512) && text(value.signature, 2048)
    && typeof value.kind === 'string' && ['function', 'method', 'class'].includes(value.kind)
    && typeof value.truncated === 'boolean' && isNodeExplanationCitation(value.citation)
    && (docstring === undefined || (typeof docstring === 'string' && docstring.length > 0 && docstring.length <= 4096
      && docstring.trim() === docstring && Array.from(docstring).every(character => {
        const code = character.charCodeAt(0); return (code >= 32 && code !== 127) || code === 9 || code === 10 || code === 13;
      })));
}

export function isNodeExplanationCitation(value: unknown): value is NodeExplanationCitation {
  return record(value) && Object.keys(value).length === 5
    && keys(value, ['id', 'file', 'sourceDigest', 'startLine', 'endLine'])
    && text(value.id, 128) && text(value.file, 4096)
    && !/^(?:\/|[A-Za-z][A-Za-z0-9+.-]*:|\\)/.test(value.file) && !value.file.split(/[\\/]/).includes('..')
    && typeof value.sourceDigest === 'string' && /^[a-f0-9]{64}$/.test(value.sourceDigest)
    && Number.isSafeInteger(value.startLine) && Number(value.startLine) >= 1
    && Number.isSafeInteger(value.endLine) && Number(value.endLine) >= Number(value.startLine);
}

export function isNodeExplanation(value: unknown): value is NodeExplanation {
  return record(value)
    && keys(value, ['title', 'purpose', 'inputs', 'outputs', 'steps', 'uncertainties', 'citations', 'concepts'])
    && text(value.title, 80) && text(value.purpose, 320)
    && list(value.inputs, 6, 160) && list(value.outputs, 4, 160)
    && list(value.steps, 6, 320) && list(value.uncertainties, 4, 240)
    && (value.concepts === undefined || (Array.isArray(value.concepts) && value.concepts.length <= 4
      && value.concepts.every(concept => record(concept) && keys(concept, ['term', 'explanation'])
        && text(concept.term, 80) && text(concept.explanation, 320))))
    && Array.isArray(value.citations) && value.citations.length >= 1 && value.citations.length <= 8
    && value.citations.every(isNodeExplanationCitation)
    && new Set(value.citations.map(citation => citation.id)).size === value.citations.length;
}

export function isNodeExplanationState(value: unknown): value is NodeExplanationState {
  return record(value) && keys(value, ['requestId', 'nodeId', 'revision', 'status', 'detail', 'explanation', 'documentation'])
    && text(value.requestId, 128) && text(value.nodeId, 1024) && text(value.revision, 256)
    && typeof value.status === 'string' && ['queued', 'loading', 'ready', 'unavailable'].includes(value.status)
    && (value.detail === undefined || text(value.detail, 320))
    && (value.documentation === undefined || isNodeSourceDocumentation(value.documentation))
    && (value.status === 'ready' ? isNodeExplanation(value.explanation) : value.explanation === undefined);
}

export function isModelExplanationState(value: unknown): value is ModelExplanationState {
  return record(value) && keys(value, ['requestId', 'revision', 'status', 'completed', 'total', 'detail'])
    && text(value.requestId, 128) && text(value.revision, 256)
    && typeof value.status === 'string' && ['loading', 'ready', 'unavailable', 'cancelled'].includes(value.status)
    && Number.isSafeInteger(value.completed) && Number.isSafeInteger(value.total)
    && Number(value.completed) >= 0 && Number(value.total) <= 10000 && Number(value.completed) <= Number(value.total)
    && (value.detail === undefined || text(value.detail, 320));
}
