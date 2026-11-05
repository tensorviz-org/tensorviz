import { visibleExplanationNodeIds } from '../../canvas/visibleExplanationScope';
import type { ModelNode } from '../../types';

const data = { layerType: { id: 'custom', name: 'Custom' } as ModelNode['data']['layerType'], parameters: {} };
const nodes: ModelNode[] = [
  { id: 'audio', type: 'inputNode', position: { x: 0, y: 0 }, data },
  { id: 'decoder', type: 'moduleNode', position: { x: 0, y: 0 }, data },
  { id: 'sample-rate', type: 'inputNode', position: { x: 0, y: 0 }, data },
  { id: 'waveform', type: 'outputNode', position: { x: 0, y: 0 }, data },
  { id: 'collapsed-child', type: 'layerNode', position: { x: 0, y: 0 }, hidden: true, data },
];
const positions = new Map([
  ['audio', { x: -80, y: 30, w: 100, h: 70 }],
  ['decoder', { x: 100, y: 30, w: 160, h: 160 }],
  ['sample-rate', { x: 100, y: -60, w: 160, h: 90 }],
  ['waveform', { x: 500, y: 30, w: 120, h: 80 }],
  ['collapsed-child', { x: 120, y: 100, w: 80, h: 80 }],
]);

it('includes partially visible boundaries and excludes off-camera nodes and collapsed contents', () => {
  expect(visibleExplanationNodeIds(nodes, { x: 0, y: 0, w: 400, h: 300 }, id => positions.get(id)))
    .toEqual(['audio', 'decoder', 'sample-rate']);
});

it('takes a fresh scope on another explicit click without modifying the prior snapshot', () => {
  const first = visibleExplanationNodeIds(nodes, { x: 0, y: 0, w: 400, h: 300 }, id => positions.get(id));
  const second = visibleExplanationNodeIds(nodes, { x: 400, y: 0, w: 300, h: 300 }, id => positions.get(id));
  expect(second).toEqual(['waveform']);
  expect(first).toEqual(['audio', 'decoder', 'sample-rate']);
  expect(nodes[4].hidden).toBe(true);
});

it('does not guess a scope while bounds or the viewport are unavailable', () => {
  expect(visibleExplanationNodeIds(nodes, { x: 0, y: 0, w: 0, h: 300 }, id => positions.get(id))).toEqual([]);
  expect(visibleExplanationNodeIds(nodes, { x: 0, y: 0, w: 400, h: 300 }, () => undefined)).toEqual([]);
});
