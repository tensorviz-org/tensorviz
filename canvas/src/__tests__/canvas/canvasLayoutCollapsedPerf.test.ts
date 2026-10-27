/**
 * WHY: a collapsed module renders as a fixed 250×240 box with its children
 * hidden, so laying its children out is work whose result is discarded. An
 * imported backbone (a Stable Diffusion UNet is ~80 modules) reconciles on every
 * single expand/collapse, so paying a dagre run per collapsed module made the
 * canvas cost scale with the modules the user is NOT looking at.
 *
 * These tests count real `dagre.layout` invocations (the module is mocked around
 * the real implementation so geometry stays truthful) and assert:
 *   1. collapsed modules cost strictly fewer layouts than the same graph expanded;
 *   2. the collapsed cost does not grow at all with the number of modules;
 *   3. the expanded path is unchanged — children are still laid out inside frame.
 */
const mockDagreCalls = { count: 0 };

jest.mock('dagre', () => {
  const actual = jest.requireActual('dagre');
  return {
    ...actual,
    layout: (...args: any[]) => {
      mockDagreCalls.count += 1;
      return (actual.layout as any)(...args);
    },
  };
});

import { reconcileModuleView } from '../../services/canvasLayout';
import { layerCardHeight } from '../../canvas/cardMetrics';
import { ModelNode, ModelEdge } from '../../types';

const CHILD_W = 215;
const MODULE_HEADER = 52;
const CHILDREN_PER_MODULE = 3;

const moduleNode = (id: string, expanded: boolean): ModelNode =>
  ({
    id,
    type: 'moduleNode',
    position: { x: 0, y: 0 },
    data: {
      layerType: { id: 'module', name: 'Module' },
      parameters: { name: id },
      isGroup: true,
      expanded,
      moduleInstanceId: id,
    },
  } as unknown as ModelNode);

const layer = (id: string, parentId?: string): ModelNode =>
  ({
    id,
    type: 'layerNode',
    position: { x: 0, y: 0 },
    data: {
      layerType: { id: 'conv2d', name: 'Conv2d' },
      parameters: { in_channels: 3, out_channels: 8 },
      parentId,
    },
  } as unknown as ModelNode);

const edge = (id: string, source: string, target: string): ModelEdge =>
  ({ id, source, target, data: {} } as unknown as ModelEdge);

/** A chain of `moduleCount` modules, each holding a chain of children. */
const buildGraph = (expanded: boolean, moduleCount: number) => {
  const nodes: ModelNode[] = [layer('input')];
  const edges: ModelEdge[] = [];
  let prev = 'input';
  for (let m = 0; m < moduleCount; m += 1) {
    const mid = `m${m}`;
    nodes.push(moduleNode(mid, expanded));
    edges.push(edge(`e_${prev}_${mid}`, prev, mid));
    prev = mid;
    let prevChild: string | undefined;
    for (let c = 0; c < CHILDREN_PER_MODULE; c += 1) {
      const cid = `${mid}__c${c}`;
      nodes.push(layer(cid, mid));
      if (prevChild) edges.push(edge(`e_${prevChild}_${cid}`, prevChild, cid));
      prevChild = cid;
    }
  }
  nodes.push(layer('output'));
  edges.push(edge(`e_${prev}_output`, prev, 'output'));
  return { nodes, edges };
};

const countLayouts = (expanded: boolean, moduleCount: number) => {
  const { nodes, edges } = buildGraph(expanded, moduleCount);
  mockDagreCalls.count = 0;
  const result = reconcileModuleView(nodes, edges, { relayout: true });
  return { calls: mockDagreCalls.count, result };
};

describe('reconcileModuleView — collapsed modules must not pay layout cost', () => {
  const MODULES = 12;

  it('runs strictly fewer dagre layouts when the modules are collapsed', () => {
    const collapsed = countLayouts(false, MODULES);
    const expanded = countLayouts(true, MODULES);
    expect(collapsed.calls).toBeLessThan(expanded.calls);
    // The saving must be per-module, not a constant: every collapsed module
    // skips its own child layout.
    expect(expanded.calls - collapsed.calls).toBe(MODULES);
  });

  it('keeps the collapsed cost flat as the module count grows', () => {
    const small = countLayouts(false, 2);
    const large = countLayouts(false, 40);
    // Only the top-level layout should run, regardless of how many collapsed
    // modules the graph carries.
    expect(large.calls).toBe(small.calls);
  });
});

describe('reconcileModuleView — expanded modules still lay children out in frame', () => {
  const { result } = countLayouts(true, 3);
  const byId = new Map(result.nodes.map((n) => [n.id, n]));

  it('places every child of an expanded module inside its parent frame', () => {
    for (let m = 0; m < 3; m += 1) {
      const mod = byId.get(`m${m}`)! as any;
      const width = mod.style.width as number;
      const height = mod.style.height as number;
      for (let c = 0; c < CHILDREN_PER_MODULE; c += 1) {
        const child = byId.get(`m${m}__c${c}`)!;
        const h = layerCardHeight(child.data as any);
        expect(child.hidden).toBe(false);
        expect(child.position.x).toBeGreaterThan(0);
        expect(child.position.x + CHILD_W).toBeLessThanOrEqual(width);
        expect(child.position.y).toBeGreaterThanOrEqual(MODULE_HEADER);
        expect(child.position.y + h).toBeLessThanOrEqual(height);
      }
    }
  });

  it('does not stack an expanded module\'s children on one another', () => {
    const coords = new Set(
      [0, 1, 2].map((c) => {
        const p = byId.get(`m0__c${c}`)!.position;
        return `${Math.round(p.x)},${Math.round(p.y)}`;
      })
    );
    expect(coords.size).toBe(CHILDREN_PER_MODULE);
  });
});

describe('reconcileModuleView — nested static modules', () => {
  const nested = (outerExpanded: boolean, innerExpanded: boolean) => {
    const outer = moduleNode('outer', outerExpanded);
    const inner = moduleNode('outer.inner', innerExpanded);
    inner.data.parentId = 'outer';
    const leaf = layer('outer.inner.fc', 'outer.inner');
    return reconcileModuleView(
      [outer, inner, leaf],
      [edge('inner-leaf', 'outer.inner', 'outer.inner.fc')],
      { relayout: true },
    );
  };

  it('sizes each expanded parent around its direct nested module and leaf', () => {
    const result = nested(true, true);
    const byId = new Map(result.nodes.map((node) => [node.id, node]));
    const outer = byId.get('outer')! as any;
    const inner = byId.get('outer.inner')! as any;
    const leaf = byId.get('outer.inner.fc')!;

    expect(inner.parentNode).toBe('outer');
    expect(leaf.parentNode).toBe('outer.inner');
    expect(inner.hidden).toBe(false);
    expect(leaf.hidden).toBe(false);
    expect(inner.style.width).toBeGreaterThan(250);
    expect(outer.style.width).toBeGreaterThanOrEqual(inner.position.x + inner.style.width);
    expect(outer.style.height).toBeGreaterThanOrEqual(inner.position.y + inner.style.height);
  });

  it('hides every descendant when an outer module is collapsed', () => {
    const result = nested(false, true);
    const byId = new Map(result.nodes.map((node) => [node.id, node]));
    expect(byId.get('outer.inner')!.hidden).toBe(true);
    expect(byId.get('outer.inner.fc')!.hidden).toBe(true);
  });
});
