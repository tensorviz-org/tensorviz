/**
 * Layer category -> accent colour token.
 *
 * Shared by canvas layer cards, graph previews and palette items.
 */
const CATEGORY_VAR: Record<string, string> = {
  io: 'var(--syntax-type)',
  module: 'var(--cat-module)',
  conv: 'var(--cat-conv)',
  linear: 'var(--cat-linear)',
  activation: 'var(--cat-activation)',
  pool: 'var(--cat-pool)',
  norm: 'var(--cat-norm)',
  recurrent: 'var(--cat-recurrent)',
  embedding: 'var(--cat-embedding)',
  dropout: 'var(--cat-regularization)',
  tensor: 'var(--cat-utility)',
  loss: 'var(--error)',
  misc: 'var(--cat-utility)',
};

export function categoryColor(category?: string): string {
  return CATEGORY_VAR[category ?? ''] || 'var(--cat-utility)';
}
