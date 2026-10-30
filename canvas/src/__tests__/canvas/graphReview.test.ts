import { buildGraphReview } from '../../ide/graphReview';
import type { GraphReviewState } from '../../ide/protocol.generated';
import fixture from '../../test/fixtures/canvas/graphReview.json';

const state = (): GraphReviewState => ({ ...JSON.parse(JSON.stringify(fixture)), id: 'review-1', beforeLabel: 'base', afterLabel: 'head', changedFiles: ['model.py'], stale: false });
describe('Git graph review from real source projections', () => {
  it('ignores display provenance in comparison and uses the readable title in navigation', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    for (const side of ['before', 'after'] as const) {
      const node = (review[side].document as any).semantic.nodes.find((node: any) => node.node_id === 'activation');
      node.op = 'Python'; node.params = { _raw_call: 'torch.where(mask, token, x)' };
      if (side === 'after') node.params._source_result_call = { name: 'where', target: 'torch.where' };
    }
    expect(buildGraphReview(review).changes).toEqual([]);
    (review.after.document as any).semantic.edges.find((edge: any) => edge.target_node_id === 'activation').source_node_id = 'input';
    expect(buildGraphReview(review).changes).toEqual([expect.objectContaining({ label: 'Select by condition' })]);
  });
  it.each(['head', 'output'])('distinguishes changed inputs on an unchanged %s from source edits', target => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    const document = review.after.document as typeof fixture.before.document;
    document.semantic.edges.find(edge => edge.target_node_id === target)!.source_node_id = 'proj';
    const result = buildGraphReview(review);
    const id = `review:after:${target}`;
    expect(result.nodes.get(id)).toMatchObject({ status: 'rewired', params: {}, inputChanges: { added: 1, removed: 1 } });
    expect(result.counts).toMatchObject({ modified: 0, rewired: 1 });
    expect(result.changes).toEqual([expect.objectContaining({ id, detail: 'Input connections changed (+1 / −1). Source and parameters unchanged.' })]);
    expect(result.projections.diff.reviewGraph.nodes.find(node => node.id === id)?.data.diffStatus).toBe('rewired');
    for (const side of ['before', 'after'] as const) {
      expect(result.projections[side].reviewGraph.nodes.find(node => node.id === id)?.data.review).toMatchObject({ status: 'unchanged', inputChanges: undefined });
    }
    expect(result.nodes.get('review:after:proj')?.status).toBe('unchanged');
  });
  it('retains both source and input changes without counting one node twice', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    const document = review.after.document as typeof fixture.before.document;
    document.semantic.edges.find(edge => edge.target_node_id === 'head')!.source_node_id = 'proj';
    (document.semantic.nodes.find(node => node.node_id === 'head')!.params as any).out_features = 7;
    const result = buildGraphReview(review);
    expect(result.nodes.get('review:after:head')).toMatchObject({ status: 'modified', inputChanges: { added: 1, removed: 1 } });
    expect(result.counts).toMatchObject({ modified: 1, rewired: 0 });
    expect(result.changes).toEqual([expect.objectContaining({ detail: 'out_features · Input connections changed (+1 / −1)' })]);
  });
  it('compares symbolic source arguments as visible parameter changes', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    for (const side of ['before', 'after'] as const) {
      const node = (review[side].document as any).semantic.nodes.find((node: any) => node.node_id === 'activation');
      node.op = 'torch.reshape';
      node.params = { _source_arguments: { shape: side === 'before' ? '(1, M, dim)' : '(1, N, dim)' } };
    }
    const result = buildGraphReview(review);
    expect(result.nodes.get('review:after:activation')?.params.shape).toEqual({ before: '(1, M, dim)', after: '(1, N, dim)' });
    expect(result.changes[0].label).toBe('Reshape');
  });
  it('reports constructor-default edits even when both source graphs are identical', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    for (const side of ['before', 'after'] as const) {
      (review[side].document as any).semantic.metadata.reviewClass = {
        defaults: { ffn_layer: side === 'before' ? "'mlp'" : "'swiglu'" }, digest: side,
      };
    }
    const result = buildGraphReview(review);
    expect(result.changes).toEqual([{ id: 'review:root', label: review.after.cls,
      detail: "ffn_layer default: 'mlp' → 'swiglu'" }]);
    expect(result.nodes.get('review:root')).toMatchObject({ before: '@class', after: '@class', status: 'modified' });
  });
  it('does not claim equivalence when source changed outside the lowered graph', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    for (const side of ['before', 'after'] as const) {
      (review[side].document as any).semantic.metadata.reviewClass = { defaults: {}, digest: side };
    }
    expect(buildGraphReview(review).changes).toEqual([expect.objectContaining({ detail: expect.stringContaining('runtime impact is unknown') })]);
    (review.after.document as any).semantic.metadata.reviewClass.digest = 'before';
    expect(buildGraphReview(review).changes).toEqual([]);
  });
  it('shows width edits, replacement and insertion without guessing different identities', () => {
    const result = buildGraphReview(state());
    expect(result.counts).toMatchObject({ added: 1, removed: 0, modified: 3, rewired: 0, unresolved: 0 });
    expect(result.nodes.get('review:after:proj')?.params.out_features).toEqual({ before: 32, after: 64 });
    expect(result.nodes.get('review:after:activation')).toMatchObject({ status: 'modified', before: 'activation', after: 'activation', beforeName: 'ReLU' });
    expect(result.nodes.get('review:after:norm')).toMatchObject({ status: 'added', before: undefined });
    expect(result.nodes.get('review:after:head')).toMatchObject({ status: 'modified', inputChanges: { added: 1, removed: 1 } });
    expect(result.edgeCounts).toEqual({ added: 2, removed: 1 });
    const graph = result.projections.diff.reviewGraph;
    expect(graph.edges.find(edge => edge.source === 'review:after:activation' && edge.target === 'review:after:head')?.data?.diffStatus).toBe('removed');
    expect(result.projections.before.reviewGraph.nodes.some(node => node.id === 'review:after:norm')).toBe(false);
    expect(result.projections.after.reviewGraph.edges.some(edge => edge.data?.diffStatus)).toBe(false);
  });
  it('keeps corresponding tiles in exactly the same positions in all three views', () => {
    const { projections } = buildGraphReview(state());
    for (const node of projections.before.reviewGraph.nodes) {
      expect(projections.diff.reviewGraph.nodes.find(item => item.id === node.id)?.position).toEqual(node.position);
      expect(projections.after.reviewGraph.nodes.find(item => item.id === node.id)?.position).toEqual(node.position);
    }
  });
  it('keeps a removed layer and its old source location available in Diff', () => {
    const original = state();
    const result = buildGraphReview({ ...original, before: original.after, after: original.before });
    expect(result.counts).toMatchObject({ added: 0, removed: 1 });
    expect(result.projections.diff.reviewGraph.nodes.find(node => node.id === 'review:before:norm')?.data.sourceLocation).toEqual({ file: 'model.py', line: 10 });
  });
  it('ignores file line movement, view layout and serialized edge IDs', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    const document = review.after.document as typeof fixture.before.document;
    for (const node of document.semantic.nodes) { (node.params as any)._lineno = 300; document.view.node_positions[node.node_id as keyof typeof document.view.node_positions] = { x: 900, y: 400 }; }
    document.semantic.edges.forEach(edge => edge.edge_id += '-changed');
    expect(buildGraphReview(review).changes).toEqual([]);
  });
  it('does not call an unmatched node added when either projection is partial', () => {
    const review = state(); review.before.truncated = true;
    const result = buildGraphReview(review);
    expect(result.counts.added).toBe(0);
    expect(result.nodes.get('review:after:norm')?.status).toBe('unresolved');
    expect(result.nodes.get('review:after:head')?.inputChanges).toEqual({ added: 0, removed: 1 });
  });
  it('does not pair changed ordinal operations just because their generated IDs agree', () => {
    const review = state();
    for (const side of ['before', 'after'] as const) {
      const document = review[side].document as typeof fixture.before.document;
      const node = document.semantic.nodes.find(node => node.node_id === 'activation')!;
      node.node_id = 'relu@MlpTiny#0';
      const span = review[side].spanmap.activation as any;
      review[side].spanmap[node.node_id] = { ...span, kind: 'call' };
      delete review[side].spanmap.activation;
      document.semantic.edges.forEach(edge => { if (edge.source_node_id === 'activation') edge.source_node_id = node.node_id; if (edge.target_node_id === 'activation') edge.target_node_id = node.node_id; });
    }
    const result = buildGraphReview(review);
    expect(result.nodes.get('review:after:relu@MlpTiny#0')).toMatchObject({ status: 'unresolved', before: undefined });
    expect(result.nodes.get('review:before:relu@MlpTiny#0')?.status).toBe('unresolved');
    expect(result.nodes.get('review:before:relu@MlpTiny#0')?.after).toBeUndefined();
  });
  it('compares duplicate operand connections as a multiset', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    const document = review.after.document as typeof fixture.before.document;
    document.semantic.edges.push({ ...document.semantic.edges[0], edge_id: 'second-use' });
    const result = buildGraphReview(review);
    expect(result.edgeCounts).toEqual({ added: 1, removed: 0 });
    expect(result.nodes.get('review:after:proj')).toMatchObject({ status: 'rewired', inputChanges: { added: 1, removed: 0 } });
  });
  it('detects changed input ports and reverses connection counts without marking source modified', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    const document = review.after.document as typeof fixture.before.document;
    document.semantic.edges[0].target_port = 'other';
    const result = buildGraphReview(review);
    expect(result.nodes.get('review:after:proj')).toMatchObject({ status: 'rewired', inputChanges: { added: 1, removed: 1 } });
    expect(result.counts.modified).toBe(0);
    document.semantic.edges.push({ ...document.semantic.edges[0], edge_id: 'second-input' });
    const forward = buildGraphReview(review), reverse = buildGraphReview({ ...review, before: review.after, after: review.before });
    expect(forward.nodes.get('review:after:proj')?.inputChanges).toEqual({ added: 2, removed: 1 });
    expect(reverse.nodes.get('review:after:proj')?.inputChanges).toEqual({ added: 1, removed: 2 });
  });
  it('pairs edited unique assignments and exposes the changed expression, independent of generated ordinals', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    for (const side of ['before', 'after'] as const) {
      const document = review[side].document as typeof fixture.before.document;
      const node = document.semantic.nodes.find(node => node.node_id === 'activation')!;
      node.node_id = side === 'before' ? 'python@MlpTiny#0' : 'python@MlpTiny#3';
      node.params = { _review_binding: ['MlpTiny', 'forward', 'student_output'],
        _raw_call: side === 'before' ? 'F.normalize(student_output)' : 'F.normalize(student_output.float())' } as any;
      review[side].spanmap[node.node_id] = { ...(review[side].spanmap.activation as any), kind: 'call', digest: side };
      delete review[side].spanmap.activation;
      document.semantic.edges.forEach(edge => { if (edge.source_node_id === 'activation') edge.source_node_id = node.node_id; if (edge.target_node_id === 'activation') edge.target_node_id = node.node_id; });
    }
    const result = buildGraphReview(review);
    expect(result.counts).toMatchObject({ modified: 1, unresolved: 0 });
    expect(result.nodes.get('review:after:python@MlpTiny#3')).toMatchObject({
      before: 'python@MlpTiny#0', status: 'modified', summary: 'Source expression changed',
      params: { _raw_call: { before: 'F.normalize(student_output)', after: 'F.normalize(student_output.float())' } },
    });
    expect(result.projections.before.reviewGraph.nodes.find(node => node.id === 'review:after:python@MlpTiny#3')?.position)
      .toEqual(result.projections.after.reviewGraph.nodes.find(node => node.id === 'review:after:python@MlpTiny#3')?.position);
    // Even malformed duplicate binding evidence must not pair arbitrary statements.
    const document = review.before.document as typeof fixture.before.document;
    const original = document.semantic.nodes.find(node => node.node_id === 'python@MlpTiny#0')!;
    document.semantic.nodes.push({ ...original, node_id: 'python@MlpTiny#1' });
    review.before.spanmap['python@MlpTiny#1'] = review.before.spanmap[original.node_id];
    expect(buildGraphReview(review).counts.unresolved).toBe(3);
  });
  it('describes constructor expression changes without claiming a layer replacement', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    for (const side of ['before', 'after'] as const) {
      const document = review[side].document as typeof fixture.before.document;
      const node = document.semantic.nodes.find(node => node.node_id === 'proj')!;
      (node.params as any)._review_call = { target: 'Linear', arguments: { out_features: { expression: side === 'before' ? 'width()' : 'width(2)' } } };
    }
    expect(buildGraphReview(review).changes).toEqual([
      expect.objectContaining({ id: 'review:after:proj', detail: 'Constructor arguments changed' }),
    ]);
  });
  it('compares omitted and explicit constructor defaults equally, while retaining actual default edits', () => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    for (const side of ['before', 'after'] as const) {
      const node = (review[side].document as typeof fixture.before.document).semantic.nodes.find(n => n.node_id === 'proj')!;
      (node.params as any)._review_call = { target: 'Linear', arguments: { in_features: 8, out_features: 32, bias: true } };
      if (side === 'after') (node.params as any).bias = true;
      else delete (node.params as any).bias;
    }
    expect(buildGraphReview(review).changes).toEqual([]);
    const node = (review.after.document as typeof fixture.before.document).semantic.nodes.find(n => n.node_id === 'proj')!;
    (node.params as any).bias = false;
    (node.params as any)._review_call.arguments.bias = false;
    expect(buildGraphReview(review).nodes.get('review:after:proj')?.params.bias).toEqual({ before: true, after: false });
  });
  it.each([
    ['_review_guards', [['if self.training']], [['if not self.training']], 'Execution condition changed'],
    ['_review_helpers', { finish: 'relu' }, { finish: 'sigmoid' }, 'Helper implementation changed'],
  ])('explains %s changes on their mapped tile', (key, before, after, detail) => {
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    for (const side of ['before', 'after'] as const) {
      const node = (review[side].document as typeof fixture.before.document).semantic.nodes.find(n => n.node_id === 'proj')!;
      (node.params as any)[key as string] = side === 'before' ? before : after;
    }
    expect(buildGraphReview(review).changes).toEqual([expect.objectContaining({ id: 'review:after:proj', detail })]);
  });
  it('lists an unlowered member with several call writes once, not again as its declaration', () => {
    // `if d in ACTS: self.act = ACTS[d]()` / `else: self.act = ACTS[0]()`, with the else branch
    // edited; forward never calls self.act. Spans are the first write, as the backend gives them.
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    const span = { file: 'model.py', start: [13, 12], end: [13, 32], kind: 'decl' };
    const other = { file: 'model.py', start: [4, 12], end: [4, 40], kind: 'decl' };
    for (const side of ['before', 'after'] as const) {
      const metadata = (review[side].document as typeof fixture.before.document).semantic.metadata as any;
      metadata.reviewClass = { defaults: {}, digest: side, members: { act: { digest: side, span, ambiguous: true } } };
      metadata.reviewDeclarations = { complete: true, entries: {
        '@declaration:["model.py","MlpTiny","act"]': { label: 'MlpTiny.act', digest: side, ambiguous: true, span,
          constructor: { ...span, start: [13, 23] } },
        // Another class's attribute is a separate change, even with the same name.
        '@declaration:["model.py","Block","act"]': { label: 'Block.act', digest: side, ambiguous: true, span: other,
          constructor: { ...other, start: [4, 23] } },
      } };
      Object.assign(review[side].spanmap, { act: span, '@declaration:["model.py","MlpTiny","act"]': span,
        '@declaration:["model.py","Block","act"]': other });
    }
    const result = buildGraphReview(review);
    expect(result.changes).toEqual([
      { id: 'review:member:act', label: 'act', detail: 'Declaration has multiple assignments' },
      { id: 'review:declaration:@declaration:["model.py","Block","act"]', label: 'Block.act', detail: 'Declaration has multiple assignments' },
    ]);
    expect(result.counts.unresolved).toBe(2);
    expect(result.nodes.get('review:member:act')).toMatchObject({ status: 'unresolved', ambiguousDeclaration: true });
  });
  it('lets an ambiguous member row stand only for its own declaration of a chained write', () => {
    // `if d: self.a = self.b = ACTS[d]()` / `else: self.a = ACTS[0]()`, with ACTS[d] edited. Only a has
    // several member writes, but the chained statement makes both declarations ambiguous.
    const review = state(); review.after = JSON.parse(JSON.stringify(review.before));
    for (const side of ['before', 'after'] as const) {
      const span = { file: 'model.py', start: [9, 12], end: [9, side === 'before' ? 39 : 43], kind: 'decl' };
      const metadata = (review[side].document as typeof fixture.before.document).semantic.metadata as any;
      metadata.reviewClass = { defaults: {}, digest: side, members: {
        a: { digest: side, span, ambiguous: true }, b: { digest: side, span, ambiguous: false } } };
      const keys = ['a', 'b'].map(name => [name, `@declaration:["model.py","MlpTiny","${name}"]`]);
      metadata.reviewDeclarations = { complete: true, entries: Object.fromEntries(keys.map(([name, key]) => [key,
        { label: `MlpTiny.${name}`, digest: side, ambiguous: true, span, constructor: { ...span, start: [9, 30] } }])) };
      Object.assign(review[side].spanmap, Object.fromEntries(keys.flat().map(id => [id, span])));
    }
    expect(buildGraphReview(review).changes).toEqual([
      { id: 'review:member:a', label: 'a', detail: 'Declaration has multiple assignments' },
      { id: 'review:member:b', label: 'b', detail: 'Declaration modified' },
      // The chained write is a caution that b's own member row does not give.
      { id: 'review:declaration:@declaration:["model.py","MlpTiny","b"]', label: 'MlpTiny.b', detail: 'Declaration has multiple assignments' },
    ]);
  });
});
