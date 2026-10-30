import type { IdeMultifileSpan, ProjectionPayload } from './protocol.generated';
import type { GraphDocumentV2 } from '../types/graphDocument';

export interface ReviewDeclarationFact {
  label: string; digest: string; ambiguous: boolean; span: IdeMultifileSpan;
  constructor?: IdeMultifileSpan | null;
  parent?: string; children?: string[]; contextDigest?: string;
}
export function reviewDeclarations(projection: ProjectionPayload): Record<string, ReviewDeclarationFact> {
  return ((projection.document as GraphDocumentV2).semantic.metadata.reviewDeclarations as {
    entries?: Record<string, ReviewDeclarationFact>;
  } | undefined)?.entries ?? {};
}
export const declarationLocation = (span: IdeMultifileSpan) => JSON.stringify([span.file, span.start, span.end]);

export function declarationInventoryComplete(projection: ProjectionPayload): boolean | undefined {
  return ((projection.document as GraphDocumentV2).semantic.metadata.reviewDeclarations as { complete?: boolean } | undefined)?.complete;
}

/** A literal slot survives a replacement only when its entire surrounding
 * constructor is unchanged. This also pins identical unchanged siblings; a
 * shifted/reordered ordinal by itself never establishes correspondence. */
export function compareDeclarations(before: ProjectionPayload, after: ProjectionPayload, renamedFiles?: Record<string, string>) {
  const original = reviewDeclarations(before), beforeKeys = new Map<string, string>(), originalBeforeKeys = new Map<string, string>();
  // Only host-proven Git renames change identity. Keep original keys/spans for
  // before-side source links and exact captured-constructor evidence.
  for (const [key, fact] of Object.entries(original)) {
    const file = fact.span.file, renamed = renamedFiles?.[file];
    const prefix = `@declaration:[${JSON.stringify(file)},`;
    beforeKeys.set(key, renamed && key.startsWith(prefix)
      ? `@declaration:[${JSON.stringify(renamed)},${key.slice(prefix.length)}` : key);
  }
  const normalized: Record<string, ReviewDeclarationFact> = {};
  for (const [key, fact] of Object.entries(original)) {
    const canonical = beforeKeys.get(key)!;
    if (normalized[canonical]) {
      normalized[canonical] = { ...normalized[canonical], ambiguous: true };
      originalBeforeKeys.delete(canonical);
    } else {
      normalized[canonical] = { ...fact, parent: fact.parent && (beforeKeys.get(fact.parent) ?? fact.parent),
        children: fact.children?.map(child => beforeKeys.get(child) ?? child) };
      originalBeforeKeys.set(canonical, key);
    }
  }
  const sides = { before: normalized, after: reviewDeclarations(after) };
  const eligible = new Set<string>(), containers = new Set<string>();
  if (![before, after].every(side => declarationInventoryComplete(side) === true)) return { sides, eligible, containers, originalBeforeKeys };
  for (const key of Array.from(new Set([...Object.keys(sides.before), ...Object.keys(sides.after)]))) {
    const previous = sides.before[key], current = sides.after[key];
    if (previous?.ambiguous || current?.ambiguous) continue;
    if (!(previous?.parent || current?.parent) || (previous && current && previous.contextDigest === current.contextDigest)) eligible.add(key);
  }
  for (const [key, previous] of Object.entries(sides.before)) {
    const current = sides.after[key];
    if (!previous.children?.length || !current || JSON.stringify(previous.children) !== JSON.stringify(current.children)) continue;
    if (previous.children.every(child => sides.before[child]?.digest === sides.after[child]?.digest || eligible.has(child))) {
      containers.add(key);
      previous.children.forEach(child => eligible.add(child));
    }
  }
  return { sides, eligible, containers, originalBeforeKeys };
}

export function declarationLocations(entries: Record<string, ReviewDeclarationFact>, keys: Set<string>) {
  const result = new Map<string, string>();
  for (const [key, fact] of Object.entries(entries)) if (keys.has(key)) {
    result.set(declarationLocation(fact.span), key);
    if (fact.constructor) result.set(declarationLocation(fact.constructor), key);
  }
  return result;
}
