type QuickJumpNode = { id: string; title: string; path?: string; searchText?: string };

/** An exact identity must remain reachable when a parent has many descendants. */
export function findQuickJumpNodes(nodes: QuickJumpNode[], query: string, limit = 20): QuickJumpNode[] {
  const q = query.trim().toLowerCase();
  if (!q) return nodes.slice(0, limit);
  return nodes
    .filter(node => node.id.toLowerCase().includes(q) || node.title.toLowerCase().includes(q) || node.path?.toLowerCase().includes(q) || node.searchText?.toLowerCase().includes(q))
    .sort((a, b) => Number(b.id.toLowerCase() === q) - Number(a.id.toLowerCase() === q))
    .slice(0, limit);
}
