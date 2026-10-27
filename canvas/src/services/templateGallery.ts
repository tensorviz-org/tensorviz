import { ApiService } from './apiService';

/**
 * How many starter architectures the template gallery actually offers.
 *
 * Home used to describe the "Start from a template" band with the count of the
 * user's OWN saved templates, so a first run read "0 starter blocks" on the
 * exact screen the gallery was serving dozens of starters from. The categories
 * endpoint already carries a per-category count (it is what the gallery's tabs
 * render), so the honest number is one GET away.
 */
export async function galleryTemplateCount(): Promise<number> {
  const res = await ApiService.request<{ categories?: Array<{ count?: number }> }>(
    '/api/templates/categories'
  );
  return sumCategoryCounts(res.categories);
}

/** Split out so the arithmetic is testable without a server. */
export function sumCategoryCounts(categories?: Array<{ count?: number }>): number {
  return (categories ?? []).reduce((sum, category) => sum + (Number(category.count) || 0), 0);
}
