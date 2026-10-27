import fs from 'fs';
import path from 'path';
import { gunzipSync } from 'zlib';
import { buildGraphReview } from '../../ide/graphReview';
import type { GraphReviewState } from '../../ide/protocol.generated';

interface StressCase {
  id: string; family: string; domain: string; size: number; cls: string; kind: string;
  base: string; head: string; errors?: unknown[]; state: GraphReviewState;
  oracle: { unchanged: boolean; target: string; parameter?: string; old?: unknown; new?: unknown };
  sourceLines: Record<'before' | 'after', Record<string, number>>;
}
const corpus: StressCase[] = JSON.parse(gunzipSync(fs.readFileSync(process.env.GRAPH_REVIEW_STRESS_CORPUS || path.join(__dirname, '../../test/fixtures/canvas/graphReviewStress.json.gz'))).toString());
const reports: unknown[] = [];
afterAll(() => {
  if (process.env.GRAPH_REVIEW_STRESS_REPORT) fs.writeFileSync(process.env.GRAPH_REVIEW_STRESS_REPORT, JSON.stringify(reports, null, 2));
});

it('contains 1000 distinct classes, 100 architecture patterns, ten sizes, and 20 equally sampled edits', () => {
  expect(corpus).toHaveLength(1000);
  expect(new Set(corpus.map(c => c.cls)).size).toBe(1000);
  expect(new Set(corpus.map(c => c.family)).size).toBe(100);
  expect(new Set(corpus.map(c => c.kind)).size).toBe(20);
  for (const family of new Set(corpus.map(c => c.family))) expect(corpus.filter(c => c.family === family)).toHaveLength(10);
  for (const kind of new Set(corpus.map(c => c.kind))) expect(corpus.filter(c => c.kind === kind)).toHaveLength(50);
});

describe.each(corpus)('$id / $kind', item => {
  it('satisfies its source-change oracle and preserves graph integrity in every revision', () => {
    let review: ReturnType<typeof buildGraphReview> | undefined;
    const started = performance.now();
    let reviewMs: number | undefined;
    const failures: string[] = [];
    const check = (name: string, fn: () => void) => { try { fn(); } catch (e) { failures.push(`${name}: ${String(e)}`); } };
    if (item.errors?.length) failures.push(`projection: ${JSON.stringify(item.errors)}`);
    else {
      review = buildGraphReview(item.state);
      reviewMs = Math.round((performance.now() - started) * 100) / 100;
      const baseline = buildGraphReview({ ...item.state, after: item.state.before, changedFiles: [] });
      const result = review;
      check('change visibility', () => {
        if (item.oracle.unchanged) {
          expect(result.counts.added + result.counts.removed + result.counts.modified + result.counts.rewired).toBe(0);
          expect(result.counts.unresolved).toBe(baseline.counts.unresolved);
          expect(result.edgeCounts).toEqual({ added: 0, removed: 0 });
        } else {
          // Existing ambiguity cannot serve as evidence that the actual edit was seen.
          expect(JSON.stringify(result.changes)).not.toBe(JSON.stringify(baseline.changes));
        }
      });
      if (item.oracle.parameter) check('parameter meaning', () => {
        const nodes = Array.from(result.nodes.values());
        expect(nodes.some(n => n.params[item.oracle.parameter!]?.after === item.oracle.new)).toBe(true);
      });
      if (item.kind === 'activation_replacement') check('replacement', () => {
        expect(result.nodes.get('review:after:activation')).toMatchObject({ status: 'modified', beforeName: 'ReLU' });
      });
      if (item.kind === 'output_rewire') check('output connection', () => {
        const graph = result.projections.after.reviewGraph;
        expect(graph.edges.some(e => e.source === 'review:after:dropout' && e.target.includes('output'))).toBe(false);
      });
      check('intended edit target', () => {
        const nodes = result.projections.after.reviewGraph.nodes;
        const records = Array.from(result.nodes.values());
        const incomplete = item.state.before.truncated || item.state.after.truncated || item.state.before.diagnostics?.length || item.state.after.diagnostics?.length;
        const unmatched = (status: string) => incomplete ? 'unresolved' : status;
        const byCall = (side: 'before' | 'after', call: string) => {
          const node = result.projections[side].reviewGraph.nodes.find(n => n.data.parameters._raw_call === call);
          expect(node).toBeDefined();
          expect(result.nodes.get(node!.id)?.status).not.toBe('unchanged');
          return node!;
        };
        if (item.kind === 'constructor_parameter') {
          expect(result.nodes.get(`review:after:${item.oracle.target}`)?.params[item.oracle.parameter!])
            .toEqual({ before: item.oracle.old, after: item.oracle.new });
        } else if (item.kind === 'insert_layer') {
          expect(result.nodes.get('review:after:adapter')?.status).toBe(unmatched('added'));
          expect(nodes.some(n => n.id === 'review:after:adapter')).toBe(true);
        } else if (item.kind === 'remove_layer') {
          expect(result.nodes.get('review:before:dropout')?.status).toBe(unmatched('removed'));
          expect(nodes.some(n => n.id === 'review:after:dropout')).toBe(false);
        } else if (item.kind === 'module_rename') {
          expect(result.nodes.get('review:before:activation')?.status).toBe(unmatched('removed'));
          expect(result.nodes.get('review:after:nonlinearity')?.status).toBe(unmatched('added'));
        } else if (item.kind === 'duplicate_call') {
          const node = byCall('after', 'self.activation(x) + self.activation(x)');
          const incoming = result.projections.after.reviewGraph.edges.filter(e => e.target === node.id);
          expect(incoming).toHaveLength(2);
          expect(incoming.every(e => e.source === 'review:after:activation')).toBe(true);
        } else if (item.kind === 'scalar_literal') {
          expect(byCall('before', 'self.dropout(x) * 1.0').data.parameters.other).toBe(1);
          expect(byCall('after', 'self.dropout(x) * 2.0').data.parameters.other).toBe(2);
        } else if (item.kind === 'operand_order') {
          expect(byCall('before', 'self.dropout(x) - 2.0').data.parameters._scalar_side).toBe('right');
          expect(byCall('after', '2.0 - self.dropout(x)').data.parameters._scalar_side).toBe('left');
        } else if (item.kind === 'functional_activation') {
          byCall('before', 'torch.relu(x)'); byCall('after', 'torch.sigmoid(x)');
        } else if (item.kind === 'helper_body') {
          expect(records.some(n => n.status === 'modified' && n.params._review_helpers && n.summary === 'Helper implementation changed')).toBe(true);
        } else if (item.kind === 'branch_condition') {
          expect(result.nodes.get('review:after:dropout')).toMatchObject({ status: 'modified', summary: 'Execution condition changed',
            params: { _review_guards: { before: [['if self.training']], after: [['if not self.training']] } } });
        } else if (item.kind === 'child_parameter') {
          expect(Array.from(result.nodes).some(([id, n]) => {
            const span = result.projections.after.spanmap[id] as any;
            return span?.file !== 'model.py' && (n.params.p?.before === 0.1 && n.params.p?.after === 0.3
              || n.params.eps?.before === 1e-5 && n.params.eps?.after === 1e-3);
          })).toBe(true);
        }
      });
      for (const mode of ['before', 'diff', 'after'] as const) check(`${mode} integrity`, () => {
        const projection = result.projections[mode];
        const graph = projection.reviewGraph;
        const ids = new Set(graph.nodes.map(n => n.id));
        expect(ids.size).toBe(graph.nodes.length);
        expect(new Set(graph.edges.map(e => e.id)).size).toBe(graph.edges.length);
        for (const node of graph.nodes) {
          expect(Number.isFinite(node.position.x) && Number.isFinite(node.position.y)).toBe(true);
          expect(result.projections.diff.reviewGraph.nodes.find(n => n.id === node.id)?.position).toEqual(node.position);
          if (node.data.parentId) expect(ids.has(node.data.parentId)).toBe(true);
          const span = projection.spanmap[node.id] as any;
          const record = result.nodes.get(node.id);
          if (span && record) {
            const side = mode === 'before' || !record.after ? 'before' : 'after';
            expect(span.start[0]).toBeGreaterThan(0);
            expect(span.end[0]).toBeGreaterThanOrEqual(span.start[0]);
            expect(span.end[0]).toBeLessThanOrEqual(item.sourceLines[side][span.file]);
          }
        }
        for (const edge of graph.edges) {
          expect(ids.has(edge.source) && ids.has(edge.target)).toBe(true);
          if (mode !== 'diff') expect(edge.data?.diffStatus).toBeUndefined();
        }
        if (mode !== 'diff') expect(graph.nodes.length - 1).toBe((item.state[mode].document as any).semantic.nodes.length);
      });
      check('reverse symmetry', () => {
        const reversed = buildGraphReview({ ...item.state, before: item.state.after, after: item.state.before });
        expect(reversed.counts.added).toBe(result.counts.removed);
        expect(reversed.counts.removed).toBe(result.counts.added);
        expect(reversed.edgeCounts).toEqual({ added: result.edgeCounts.removed, removed: result.edgeCounts.added });
      });
    }
    reports.push({ id: item.id, family: item.family, domain: item.domain, size: item.size, kind: item.kind, base: item.base, head: item.head,
      passed: failures.length === 0, failures, counts: review?.counts, edges: review?.edgeCounts,
      changes: review?.changes, diagnostics: { before: item.state.before?.diagnostics, after: item.state.after?.diagnostics },
      reviewMs, verificationMs: Math.round(performance.now() - started) });
    expect(failures).toEqual([]);
  });
});
