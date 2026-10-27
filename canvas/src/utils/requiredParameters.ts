import { getLayerByType } from '../data/layerTypes';

export interface MissingParameters {
  nodeId: string;
  layerName: string;
  missing: string[];
}

/**
 * Layers on the canvas that are missing a parameter their constructor requires.
 *
 * A single-file import whose __init__ takes required arguments (`nn.Linear(784,
 * hidden_dim)`) cannot know what `hidden_dim` is, so those parameters arrive empty.
 * Nothing said so: the import reported success, the card just read "Linear #fc2"
 * with no parameters, and Export Code went on to emit `nn.Linear()` — code that
 * raises TypeError the moment it is run.
 */
export function findMissingRequiredParameters(
  nodes: Array<{ id: string; data?: { layerType?: any; parameters?: Record<string, any> } }>,
): MissingParameters[] {
  const out: MissingParameters[] = [];
  nodes.forEach((node) => {
    const rawType = node.data?.layerType;
    const typeId = rawType?.id;
    if (!typeId || typeId === 'input' || typeId === 'output') return;
    // Prefer the canonical schema; a node may carry only the parser's stub.
    const schema = getLayerByType(typeId) || rawType;
    const params: Array<{ name: string; required?: boolean }> = schema?.parameters || [];
    if (!params.length) return;
    const values = node.data?.parameters || {};
    const missing = params
      .filter((p) => p.required)
      .filter((p) => {
        const v = values[p.name];
        return v === undefined || v === null || v === '';
      })
      .map((p) => p.name);
    if (missing.length) {
      out.push({ nodeId: node.id, layerName: schema?.name || typeId, missing });
    }
  });
  return out;
}

/** One line naming what is incomplete, for a refusal or a warning. */
export function describeMissingParameters(missing: MissingParameters[]): string {
  return missing
    .map((m) => `${m.layerName} #${m.nodeId} (${m.missing.join(', ')})`)
    .join('; ');
}
