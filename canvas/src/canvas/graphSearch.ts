/**
 * Query the graph the way Model Explorer does: one regex, four fields, hits
 * grouped by the scope they live in.
 *
 * The question a reader actually brings to a graph is "which layer is the
 * 256-wide one" or "where did the rank change". Neither is answerable by eye
 * once the model is past a screenful, and the canvas had no way to ask it.
 *
 * The rollup is the part that matters for a hierarchical graph: a hit inside a
 * CLOSED module is reported on the module, not dropped. Search and collapse
 * would otherwise fight — every query would force the graph open to prove it
 * found anything, which defeats the containers.
 */
import type { ModelNode } from '../types';
import { operationDisplayName } from './operationDisplayName';

export type SearchField = 'label' | 'type' | 'shape' | 'attr';

export interface SearchHit {
  id: string;
  /** The node id relative to nothing — callers scope it for display. */
  field: SearchField;
  /** The text that actually matched, for highlighting the reason. */
  matched: string;
  /** Nearest VISIBLE ancestor, or the node itself when it is on screen. */
  visibleId: string;
}

export interface SearchGroup {
  /** Visible node the hits are reported on: the node itself, or its closed frame. */
  visibleId: string;
  hits: SearchHit[];
}

export interface SearchResult {
  query: string;
  hits: SearchHit[];
  groups: SearchGroup[];
  /** Ids to light up; everything else on screen is dimmed. */
  visibleIds: Set<string>;
  /** A malformed regex is reported, never silently treated as "no matches". */
  error?: string;
}

const EMPTY: SearchResult = { query: '', hits: [], groups: [], visibleIds: new Set() };

/** `[1, 32, 32, 32]` — the same rendering the card shows, so what you see is
 *  what you can search for. */
function shapeText(shape: unknown): string {
  return Array.isArray(shape) ? `[${shape.join(', ')}]` : '';
}

function parseQuery(query: string): { field: SearchField; key?: string; pattern: string } | null {
  const trimmed = query.trim();
  if (!trimmed) return null;
  const typed = /^(type|shape)=(.*)$/i.exec(trimmed);
  if (typed) return { field: typed[1].toLowerCase() as SearchField, pattern: typed[2] };
  const attr = /^attr:([A-Za-z_][\w]*)=(.*)$/i.exec(trimmed);
  if (attr) return { field: 'attr', key: attr[1], pattern: attr[2] };
  return { field: 'label', pattern: trimmed };
}

/** The nearest ancestor that is actually drawn. A hit two frames deep is
 *  reported on the outermost closed one, which is the only card on screen. */
function nearestVisible(node: ModelNode, byId: Map<string, ModelNode>): string {
  const seen = new Set<string>();
  let current: ModelNode | undefined = node;
  while (current && current.hidden) {
    const parent = current.data.parentId ?? current.parentNode;
    if (!parent || seen.has(parent)) break;
    seen.add(parent);
    current = byId.get(parent);
  }
  return current ? current.id : node.id;
}

export function searchGraph(nodes: readonly ModelNode[], query: string): SearchResult {
  const parsed = parseQuery(query);
  if (!parsed) return { ...EMPTY, query };

  let regex: RegExp;
  try {
    regex = new RegExp(parsed.pattern, 'i');
  } catch (error) {
    // Rule 12: a typo in a regex is a user error with an answer, not zero hits.
    return { ...EMPTY, query, error: (error as Error).message };
  }

  const byId = new Map(nodes.map(node => [node.id, node]));
  const hits: SearchHit[] = [];

  for (const node of nodes) {
    const data = node.data;
    const candidates: Array<{ field: SearchField; text: string }> = [];
    const operationName = operationDisplayName(data);
    if (operationName && (parsed.field === 'label' || parsed.field === 'type')) {
      for (const text of [operationName, data.layerType.name, data.layerType.id, data.layerType.displayName]) {
        if (text) candidates.push({ field: parsed.field, text });
      }
    }
    if (parsed.field === 'label') {
      candidates.push({ field: 'label', text: node.id });
      if (data.captureLabel) candidates.push({ field: 'label', text: data.captureLabel });
      const label = (data as { label?: string }).label;
      if (label) candidates.push({ field: 'label', text: label });
    } else if (parsed.field === 'type') {
      const type = data.layerType;
      for (const text of [type?.displayName, type?.name, type?.id]) {
        if (text) candidates.push({ field: 'type', text });
      }
      const cls = data.parameters?._class;
      if (typeof cls === 'string') candidates.push({ field: 'type', text: cls });
    } else if (parsed.field === 'shape') {
      for (const text of [shapeText(data.inputShape), shapeText(data.outputShape),
        ...(data.capturePorts ?? []).map(port => shapeText(port.shape))]) {
        if (text) candidates.push({ field: 'shape', text });
      }
    } else if (parsed.field === 'attr' && parsed.key) {
      const value = data.parameters?.[parsed.key];
      if (value !== undefined) candidates.push({ field: 'attr', text: String(value) });
    }

    const hit = candidates.find(candidate => regex.test(candidate.text));
    if (hit) hits.push({ id: node.id, field: hit.field, matched: hit.text, visibleId: nearestVisible(node, byId) });
  }

  const order: string[] = [];
  const grouped = new Map<string, SearchHit[]>();
  for (const hit of hits) {
    if (!grouped.has(hit.visibleId)) { grouped.set(hit.visibleId, []); order.push(hit.visibleId); }
    grouped.get(hit.visibleId)!.push(hit);
  }

  return {
    query,
    hits,
    groups: order.map(visibleId => ({ visibleId, hits: grouped.get(visibleId)! })),
    visibleIds: new Set(hits.map(hit => hit.visibleId)),
  };
}
