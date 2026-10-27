import type { IdeMultifileSpan } from './protocol.generated';
import type { ModelNode } from '../types';
import type { GraphDocumentV2 } from '../types/graphDocument';
import { getLayerByType } from '../data/layerTypes';

/** One literal `self.<path> = value` in `__init__` (`layer3[0].relu`), recorded as
 * review metadata only. `title` is the source callable, not a registry type; the fact
 * names no runtime module, call count or captured counterpart. */
export interface SourceAssignmentFact {
  target: string; digest: string; span: IdeMultifileSpan;
  ambiguous: boolean; conditional: boolean; title: string; value: string;
  /** AST target segments below `self`; absent on legacy facts. Never parsed from `target`. */
  path?: { kind: 'attribute' | 'index'; key: string | number }[];
}
export interface SourceAssignmentEntry { id: string; before?: SourceAssignmentFact; after?: SourceAssignmentFact; unchanged: boolean }

export function sourceAssignmentsOf(document: unknown): Record<string, SourceAssignmentFact> {
  const facts = ((document as GraphDocumentV2 | undefined)?.semantic?.metadata?.reviewClass as { assignments?: unknown } | undefined)?.assignments;
  return facts && typeof facts === 'object' ? facts as Record<string, SourceAssignmentFact> : {};
}

/** A display-only card: no ports, edges, shapes or edit capability, and never a
 * semantic node. Its id is the metadata key, the literal target path, so a value
 * edit keeps the same card; it is not a runtime ordinal identity. */
export function sourceAssignmentCard(id: string, fact: SourceAssignmentFact): ModelNode {
  return { id, type: 'layerNode', position: { x: 0, y: 0 }, data: {
    layerType: getLayerByType('custom_code')!, parameters: { _class: fact.title, value: fact.value },
    captureLabel: fact.target, sourceAssignment: { target: fact.target, conditional: fact.conditional },
    sourceLocation: { file: fact.span.file, line: fact.span.start[0] },
    inputPorts: [], outputPorts: [], shapeEvidence: 'unknown', sourceEditCapabilities: [],
  } };
}

/** Keep unchanged assignments as context in the next draft too. An ambiguous write
 * has no single value to show; changed uncertainty gets the unresolved row. */
export function sourceAssignmentsForReview(before: unknown, after: unknown): { entries: SourceAssignmentEntry[]; unresolved: boolean } {
  const left = sourceAssignmentsOf(before), right = sourceAssignmentsOf(after);
  const entries: SourceAssignmentEntry[] = [];
  let unresolved = false;
  for (const id of Array.from(new Set([...Object.keys(left), ...Object.keys(right)]))) {
    const unchanged = left[id]?.digest === right[id]?.digest && left[id]?.ambiguous === right[id]?.ambiguous;
    if (left[id]?.ambiguous || right[id]?.ambiguous) { if (!unchanged) unresolved = true; }
    else entries.push({ id, unchanged, ...(left[id] ? { before: left[id] } : {}), ...(right[id] ? { after: right[id] } : {}) });
  }
  return { entries, unresolved };
}
