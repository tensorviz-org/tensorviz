import * as fs from 'fs';
import * as path from 'path';
import { hierarchicalToFlat } from '../../services/hierarchicalToFlat';
import { reconcileModuleView } from '../../services/canvasLayout';
import { ModelNode } from '../../types';

/**
 * End-to-end guard for "templates not working": feed a REAL backend payload
 * (captured from TemplateService.instantiate_popular_model('bert_base')) through
 * the exact frontend pipeline a template takes — flatten → reconcile — and
 * assert the model body survives. Before the fix, the 12 encoder layers were
 * parented into a collapsed, unsized module and vanished; this test fails if
 * that regresses.
 */
const payload = JSON.parse(
  fs.readFileSync(path.join(__dirname, '../../test/fixtures/canvas/bert_base.hierarchical.json'), 'utf8')
);

describe('template pipeline — real bert_base payload', () => {
  const flat = hierarchicalToFlat(payload);

  it('flattens into the module placeholder plus its 12 internal layers', () => {
    const moduleNode = flat.nodes.find((n) => n.id === 'encoder_stack')!;
    expect(moduleNode).toBeDefined();
    expect(moduleNode.type).toBe('moduleNode');
    expect(moduleNode.data.repeatCount).toBe(12);
    expect(moduleNode.data.repeatBlockLabel).toBe('TransformerEncoderLayer');
    const children = flat.nodes.filter((n) => n.data.parentId === 'encoder_stack');
    expect(children.length).toBe(12);
  });

  it('renders collapsed-by-default: module visible, 12 layers hidden, badge intact', () => {
    const { nodes } = reconcileModuleView(
      flat.nodes as ModelNode[],
      flat.edges,
      { relayout: true }
    );
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const moduleNode = byId.get('encoder_stack')!;
    expect(moduleNode.hidden).toBeFalsy();
    expect(moduleNode.data.repeatCount).toBe(12);
    expect(moduleNode.data.moduleLayerCount).toBe(12);

    const children = nodes.filter((n) => n.data.parentId === 'encoder_stack');
    expect(children.length).toBe(12);
    expect(children.every((c) => c.hidden === true)).toBe(true);
  });

  it('lays the top level out so nothing piles at the origin', () => {
    const { nodes } = reconcileModuleView(
      flat.nodes as ModelNode[],
      flat.edges,
      { relayout: true }
    );
    const top = nodes.filter((n) => !n.parentNode);
    const coords = new Set(top.map((n) => `${Math.round(n.position.x)},${Math.round(n.position.y)}`));
    expect(top.length).toBeGreaterThan(1);
    expect(coords.size).toBe(top.length); // all distinct → no origin pile-up
  });

  it('expanding the stack reveals all 12 layers, parented to the module', () => {
    const expandedLogical = (flat.nodes as ModelNode[]).map((n) =>
      n.id === 'encoder_stack' ? { ...n, data: { ...n.data, expanded: true } } : n
    );
    const { nodes } = reconcileModuleView(expandedLogical, flat.edges, { relayout: true });
    const children = nodes.filter((n) => n.data.parentId === 'encoder_stack');
    expect(children.length).toBe(12);
    expect(children.every((c) => c.hidden === false)).toBe(true);
    expect(children.every((c) => c.parentNode === 'encoder_stack' && c.extent === 'parent')).toBe(true);
  });
});
