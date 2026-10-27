import type { ReviewNode } from './graphReview';

/** One vocabulary for interactive cards, accessibility and SVG exports. */
export function graphReviewLabel(review: ReviewNode): string {
  switch (review.status) {
    case 'added': return '+ added';
    case 'removed': return '− removed';
    case 'modified': return review.inputChanges ? '~ modified + inputs' : '~ modified';
    case 'rewired': return 'inputs changed';
    case 'unresolved': return '? unresolved';
    case 'unchanged': return '';
  }
}
