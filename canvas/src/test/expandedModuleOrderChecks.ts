import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Editor, getIndicesBetween } from 'tldraw';
import type { IndexKey, TLParentId, TLShape, TLShapeId, TLShapePartial } from 'tldraw';
import { planExpandedModuleOrder } from '../canvas/expandedModuleOrder';

function spy<T extends (...args: any[]) => any>(fn: T) {
  const calls: Parameters<T>[] = [];
  return Object.assign((...args: Parameters<T>): ReturnType<T> => { calls.push(args); return fn(...args); }, { calls });
}

const sid = (id: string) => `shape:${id}` as TLShapeId;
const page = 'page:test' as TLParentId;
const shape = (id: string, index: string, parentId = page) => ({
  id: sid(id), type: 'module-node', parentId, index: index as IndexKey,
  x: 12, y: -7, rotation: 0, opacity: 1, isLocked: true,
  props: { w: 420, h: 280 }, meta: {},
} as unknown as TLShape);

function scene(shapes: TLShape[], staleFirst = false) {
  const records = new Map(shapes.map(s => [s.id, s]));
  const getShape = spy((id: TLShapeId) => records.get(id));
  const getSortedChildIdsForParent = spy((parentId: TLParentId) => {
    const children = Array.from(records.values()).filter(s => s.parentId === parentId)
      .sort((a, b) => a.index < b.index ? -1 : a.index > b.index ? 1 : 0).map(s => s.id);
    return staleFirst ? [sid('stale'), ...children] : children;
  });
  const updateShapes = (partials: TLShapePartial[]) => {
    for (const partial of partials) {
      const previous = records.get(partial.id);
      if (previous) records.set(partial.id, { ...previous, ...partial } as TLShape);
    }
  };
  return { records, getShape, getSortedChildIdsForParent, updateShapes };
}

function compare(shapes: TLShape[], ids: TLShapeId[], staleFirst = false) {
  const baseline = scene(shapes, staleFirst);
  const candidate = scene(shapes, staleFirst);
  // Exercise the installed public singleton implementation, not a handwritten
  // approximation or no-op recording stub. This is an output differential, not
  // a performance benchmark or a real Editor/store side-effect integration test.
  for (const id of ids) Editor.prototype.sendToBack.call(baseline as unknown as Editor, [id]);
  const changes = planExpandedModuleOrder(candidate, ids);
  assert.deepEqual(Array.from(candidate.records.values()), shapes); // planning is pure
  assert.equal(new Set(changes.map(s => s.id)).size, changes.length);
  candidate.updateShapes(changes);
  // NODE_ENV=test disables fractional-index jitter: compare exact index bytes,
  // all unchanged geometry, props, locks, and every non-target record as well.
  assert.deepEqual(candidate.records, baseline.records);
  return { baseline, candidate, changes };
}

function permutations<T>(items: T[]): T[][] {
  if (items.length === 0) return [[]];
  return items.flatMap((item, i) => permutations(items.filter((_, j) => i !== j))
    .map(rest => [item, ...rest]));
}

describe('expanded module singleton ordering', () => {
  it('matches every initial order and move order, including already-bottom noops', () => {
    const ids = ['outer', 'inner', 'sibling', 'edge'];
    for (const initial of permutations(ids)) {
      const shapes = initial.map((id, i) => shape(id, `a${i}`));
      for (const order of permutations(ids.slice(0, 3))) compare(shapes, order.map(sid));
    }
  });

  it('uses actual shape parents and tolerates missing records and repeated ids', () => {
    const other = sid('actual-parent');
    compare([
      shape('edge', 'a0'), shape('outer', 'a1'), shape('inner', 'a2'),
      shape('child-edge', 'a0', other), shape('child', 'a1', other),
      shape('alone', 'a0', sid('third-parent')),
    ], ['missing', 'inner', 'child', 'outer', 'inner', 'inner', 'alone', 'child', 'missing'].map(sid), true);
  });

  it('retains singleton index allocation and skips only current-bottom noops', () => {
    const first = getIndicesBetween(undefined, 'a0' as IndexKey, 1)[0];
    const second = getIndicesBetween(undefined, first, 1)[0];
    const third = getIndicesBetween(undefined, second, 1)[0];
    const { candidate, changes } = compare([
      shape('edge', 'a0'), shape('a', 'a1'), shape('b', 'a2'),
    ], ['a', 'a', 'b', 'a'].map(sid));
    assert.equal(candidate.records.get(sid('a'))?.index, third);
    assert.equal(candidate.records.get(sid('b'))?.index, second);
    assert.equal(changes.length, 2);
  });

  it('handles empty and entirely missing sequences without reading siblings', () => {
    for (const ids of [[], [sid('missing')]]) {
      const { candidate, changes } = compare([shape('untouched', 'a0')], ids);
      assert.deepEqual(changes, []);
      assert.equal(candidate.getSortedChildIdsForParent.calls.length, 0);
    }
    compare([shape('only', 'a0')], [sid('only'), sid('only')]);
  });

  it('matches deterministic mixed-parent repeated-move sequences', () => {
    let seed = 783;
    const random = (n: number) => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed % n;
    };
    const indices = getIndicesBetween(undefined, undefined, 32);
    for (let trial = 0; trial < 80; trial++) {
      const shapes = Array.from({ length: 32 }, (_, i) => shape(String(i), indices[i],
        `page:${random(4)}` as TLParentId));
      compare(shapes, Array.from({ length: 100 }, () => sid(String(random(36)))));
    }
  });

  it('snapshots siblings once per touched parent, not once per module', () => {
    const indices = getIndicesBetween(undefined, undefined, 1001);
    const shapes = indices.map((index, i) => shape(String(i), index));
    const { baseline, candidate } = compare(shapes, shapes.slice(1).map(s => s.id));
    assert.equal(baseline.getSortedChildIdsForParent.calls.length, 1000);
    assert.equal(candidate.getSortedChildIdsForParent.calls.length, 1);
    assert.equal(candidate.getShape.calls.length, 1001);
  });
});
