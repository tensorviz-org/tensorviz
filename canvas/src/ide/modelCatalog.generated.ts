/* GENERATED from packages/contracts/src/modelCatalog.ts; npm run protocol:sync. */
/** The existing source index supplies classes; only host-observed execution may
 * mark a model captured. IDs are host-issued selection handles, never paths.
 */
export interface ModelCatalogEntry {
  id: string;
  cls: string;
  file: string;
  evidence: 'source' | 'captured';
  opened: boolean;
}
export interface ModelCatalog {
  status: 'loading' | 'ready' | 'unavailable';
  models: ModelCatalogEntry[];
  activeId?: string;
  detail?: string;
  truncated?: boolean;
}
export const MODEL_CATALOG_LIMIT = 2000;

const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const keys = (value: Record<string, unknown>, allowed: readonly string[]): boolean =>
  Object.keys(value).every(key => allowed.includes(key));
const text = (value: unknown, max: number): value is string => typeof value === 'string'
  && value.length > 0 && value.length <= max && value.trim() === value
  && Array.from(value).every(character => { const code = character.charCodeAt(0); return code >= 32 && code !== 127; });

export function isModelCatalogEntry(value: unknown): value is ModelCatalogEntry {
  return record(value) && keys(value, ['id', 'cls', 'file', 'evidence', 'opened'])
    && text(value.id, 128) && text(value.cls, 512) && text(value.file, 4096)
    && !/^(?:\/|[A-Za-z][A-Za-z0-9+.-]*:|\\)/.test(value.file) && !value.file.split(/[\\/]/).some(part => !part || part === '.' || part === '..')
    && (value.evidence === 'source' || value.evidence === 'captured') && typeof value.opened === 'boolean';
}
export function isModelCatalog(value: unknown): value is ModelCatalog {
  if (!record(value) || !keys(value, ['status', 'models', 'activeId', 'detail', 'truncated'])
    || typeof value.status !== 'string' || !['loading', 'ready', 'unavailable'].includes(value.status)
    || !Array.isArray(value.models) || value.models.length > MODEL_CATALOG_LIMIT || !value.models.every(isModelCatalogEntry)) return false;
  const ids = new Set(value.models.map(model => model.id));
  return ids.size === value.models.length
    && (value.activeId === undefined || (text(value.activeId, 128) && ids.has(value.activeId)))
    && (value.detail === undefined || text(value.detail, 320))
    && (value.truncated === undefined || typeof value.truncated === 'boolean');
}
