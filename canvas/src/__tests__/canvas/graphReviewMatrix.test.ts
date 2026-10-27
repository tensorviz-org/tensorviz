import fs from 'fs';
import path from 'path';
import { gunzipSync } from 'zlib';
import { buildGraphReview } from '../../ide/graphReview';
import type { GraphReviewState } from '../../ide/protocol.generated';

interface Case {
  id: string; family: string; size: number; kind: string; cls: string;
  errors?: { side: string; status: number; detail: string }[];
  state: GraphReviewState;
  sourceLines: Record<'before' | 'after', Record<string, number>>;
  validity: Record<'before' | 'after', { parameters: number; outputShape: number[] }>;
}
const corpus: Case[] = JSON.parse(gunzipSync(fs.readFileSync(path.join(__dirname, '../../test/fixtures/canvas/graphReviewMatrix.json.gz'))).toString());
const results: unknown[] = [];
afterAll(() => {
  if (process.env.GRAPH_REVIEW_MATRIX_REPORT) fs.writeFileSync(process.env.GRAPH_REVIEW_MATRIX_REPORT, JSON.stringify(results, null, 2));
});

it('contains 100 distinct, valid model PRs spanning 20 families and five sizes', () => {
  expect(corpus).toHaveLength(100);
  expect(new Set(corpus.map(c => c.cls)).size).toBe(100);
  expect(new Set(corpus.map(c => c.family)).size).toBe(20);
  expect(new Set(corpus.map(c => c.size)).size).toBe(5);
  for (const family of new Set(corpus.map(c => c.family))) expect(corpus.filter(c => c.family === family)).toHaveLength(5);
  for (const kind of new Set(corpus.map(c => c.kind))) expect(corpus.filter(c => c.kind === kind)).toHaveLength(10);
  for (const item of corpus) for (const side of ['before', 'after'] as const) {
    expect(item.validity[side].parameters).toBeGreaterThan(0);
    expect(item.validity[side].outputShape.every(d => d > 0)).toBe(true);
  }
});

describe.each(corpus)('$id / $kind', item => {
  it('reports the intended change and preserves revision, layout, and source integrity', () => {
    const started = performance.now();
    let result: ReturnType<typeof buildGraphReview> | undefined;
    try {
      expect(item.errors).toBeUndefined();
      result = buildGraphReview(item.state);
      const clean = ['keyword_equivalence', 'comments_only', 'import_alias'].includes(item.kind);
      if (clean) {
        expect(result.changes).toEqual([]);
        expect(result.edgeCounts).toEqual({ added: 0, removed: 0 });
      } else if (item.kind === 'dropout_probability' || item.kind === 'compound_insertion') {
        expect(result.nodes.get('review:after:dropout')).toMatchObject({ status: 'modified', params: {
          p: { before: 0.1, after: item.kind === 'compound_insertion' ? 0.25 : 0.35 },
        } });
      } else if (item.kind === 'activation_replacement') {
        expect(result.nodes.get('review:after:activation')).toMatchObject({ status: 'modified', beforeName: 'ReLU' });
      } else if (item.kind === 'insert_adapter') {
        // The ModuleList loop has incomplete source coverage; an insertion must
        // stay uncertain there, while complete projections confirm it.
        expect(result.nodes.get('review:after:adapter')?.status).toBe(item.family === 'container' ? 'unresolved' : 'added');
      } else if (item.kind === 'remove_dropout') {
        expect(result.nodes.get('review:before:dropout')?.status).toBe('removed');
      } else if (item.kind === 'attribute_value') {
        expect(Array.from(result.nodes.values()).filter(node => node.params.other)).toEqual([
          expect.objectContaining({ status: 'modified', params: expect.objectContaining({ other: { before: 1, after: 2 } }) }),
        ]);
      } else {
        expect(item.kind).toBe('output_rewire');
        expect(result.edgeCounts).toEqual({ added: 1, removed: 2 });
        expect(result.projections.after.reviewGraph.edges.some(e => e.source === 'review:after:dropout' || e.target === 'review:after:dropout')).toBe(false);
      }
      if (item.kind === 'compound_insertion') expect(result.nodes.get('review:after:adapter')?.status).toBe('added');
      if (result.edgeCounts.added + result.edgeCounts.removed > 0) expect(result.changes.length).toBeGreaterThan(0);
      for (const mode of ['before', 'diff', 'after'] as const) {
        const projection = result.projections[mode];
        const graph = projection.reviewGraph;
        const ids = new Set(graph.nodes.map(n => n.id));
        expect(ids.size).toBe(graph.nodes.length);
        for (const node of graph.nodes) {
          expect(Number.isFinite(node.position.x) && Number.isFinite(node.position.y)).toBe(true);
          if (node.data.parentId) expect(ids.has(node.data.parentId)).toBe(true);
          expect(result.projections.diff.reviewGraph.nodes.find(n => n.id === node.id)?.position).toEqual(node.position);
          const span = projection.spanmap[node.id] as any;
          const record = result.nodes.get(node.id);
          if (span && record) {
            const side = mode === 'before' || !record.after ? 'before' : 'after';
            expect(span.start[0]).toBeGreaterThan(0);
            expect(span.end[0]).toBeLessThanOrEqual(item.sourceLines[side][span.file]);
          }
        }
        for (const edge of graph.edges) {
          expect(ids.has(edge.source)).toBe(true);
          expect(ids.has(edge.target)).toBe(true);
          if (mode !== 'diff') expect(edge.data?.diffStatus).toBeUndefined();
        }
        if (mode !== 'diff') expect(graph.nodes.length - 1).toBe((item.state[mode].document as any).semantic.nodes.length);
      }
      const reversed = buildGraphReview({ ...item.state, before: item.state.after, after: item.state.before });
      expect(reversed.counts.added).toBe(result.counts.removed);
      expect(reversed.counts.removed).toBe(result.counts.added);
      expect(reversed.edgeCounts).toEqual({ added: result.edgeCounts.removed, removed: result.edgeCounts.added });
      results.push({ id: item.id, family: item.family, size: item.size, kind: item.kind, passed: true,
        counts: result.counts, edges: result.edgeCounts, nodes: result.projections.diff.reviewGraph.nodes.length,
        comparisonMs: Math.round(performance.now() - started), validity: item.validity });
    } catch (error) {
      results.push({ id: item.id, family: item.family, size: item.size, kind: item.kind, passed: false,
        counts: result?.counts, edges: result?.edgeCounts, error: String(error) });
      throw error;
    }
  });
});
