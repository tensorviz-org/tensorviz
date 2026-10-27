import * as assert from 'node:assert/strict';
import { reviewAnalysisError } from '@tensorviz/graph-host/graphReviewCore';

suite('review analysis errors', () => {
  test('a syntax error names the file and line', () => {
    const body = JSON.stringify({ detail: { error: 'syntax_error', message: 'Function parameters cannot be parenthesized', line: 15, column: 16 } });
    assert.equal(reviewAnalysisError('examples/graph_ui/residual.py', 400, body), 'could not parse examples/graph_ui/residual.py:15');
  });

  test('other failures stay a source-analysis error without the response body', () => {
    assert.match(reviewAnalysisError('model.py', 500, 'secret traceback'), /Source analysis failed \(500\)/);
    assert.doesNotMatch(reviewAnalysisError('model.py', 500, 'secret traceback'), /secret/);
  });
});
