export type DetailTier = 'overview' | 'working' | 'inspection';

/** Screen-space thresholds with a 12px dead band, independent of graph geometry. */
export function detailTier(width: number, previous?: DetailTier): DetailTier {
  if (previous === 'overview' && width < 122) return 'overview';
  if (previous === 'inspection' && width > 178) return 'inspection';
  if (previous === 'working' && width >= 98 && width <= 202) return 'working';
  return width < 110 ? 'overview' : width < 190 ? 'working' : 'inspection';
}
