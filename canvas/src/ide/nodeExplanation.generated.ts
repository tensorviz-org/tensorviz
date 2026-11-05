/* GENERATED from packages/contracts/src/nodeExplanation.ts; npm run protocol:sync. */
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
  /** Resolved by the host from source evidence the agent actually received. */
  citations: NodeExplanationCitation[];
}

export interface NodeExplanationState {
  requestId: string;
  nodeId: string;
  /** The graph service revision, including source and capture identity. */
  revision: string;
  status: 'queued' | 'loading' | 'ready' | 'unavailable';
  detail?: string;
  explanation?: NodeExplanation;
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
  return record(value) && Object.keys(value).length === 7
    && keys(value, ['title', 'purpose', 'inputs', 'outputs', 'steps', 'uncertainties', 'citations'])
    && text(value.title, 80) && text(value.purpose, 320)
    && list(value.inputs, 6, 160) && list(value.outputs, 4, 160)
    && list(value.steps, 6, 320) && list(value.uncertainties, 4, 240)
    && Array.isArray(value.citations) && value.citations.length >= 1 && value.citations.length <= 8
    && value.citations.every(isNodeExplanationCitation)
    && new Set(value.citations.map(citation => citation.id)).size === value.citations.length;
}

export function isNodeExplanationState(value: unknown): value is NodeExplanationState {
  return record(value) && keys(value, ['requestId', 'nodeId', 'revision', 'status', 'detail', 'explanation'])
    && text(value.requestId, 128) && text(value.nodeId, 1024) && text(value.revision, 256)
    && typeof value.status === 'string' && ['queued', 'loading', 'ready', 'unavailable'].includes(value.status)
    && (value.detail === undefined || text(value.detail, 320))
    && (value.status === 'ready' ? isNodeExplanation(value.explanation) : value.explanation === undefined);
}
