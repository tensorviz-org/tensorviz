import React from 'react';
import type { ReviewNode } from '../ide/graphReview';
import { graphReviewLabel } from '../ide/graphReviewLabels';
export function GraphReviewBadge({ review }: { review?: ReviewNode }) {
  if (!review || review.status === 'unchanged') return null;
  return <span className="graph-review-badge" data-status={review.status} title={review.status === 'unresolved' ? 'No reliable correspondence across revisions. This is not a confirmed addition or removal.' : review.summary}
    aria-label={`Graph change: ${graphReviewLabel(review).replace(/^[+−~?] /, '')}`}>{graphReviewLabel(review)}</span>;
}
export function GraphReviewValue({ before, after, format = String }: { before: unknown; after: unknown; format?(value: any): React.ReactNode }) {
  return <span className="graph-review-value"><del>{before === undefined ? '∅' : format(before)}</del><span aria-hidden="true">→</span><ins>{after === undefined ? '∅' : format(after)}</ins></span>;
}
