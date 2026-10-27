/**
 * What a card must still say about itself once its module frame is on screen.
 *
 * The document already carries the hierarchy: the backend emits a `moduleNode`
 * for `stage1` and gives every child `data.parentId = 'stage1'` plus the dotted
 * id `stage1.conv1`. The card then printed that whole id as its subtitle, so an
 * open frame drew its own name twice — once on the frame header and again, as a
 * prefix, on all five children ("#stage1.conv1", "#stage1.norm1", …).
 *
 * The frame supplies the scope, so the child only owes the reader the part the
 * frame does not already say. Nested frames strip in the same single step
 * because ids are fully qualified: inside `stage1`, `stage1.block.conv` becomes
 * `block.conv`, and inside `stage1.block` it becomes `conv`.
 *
 * A node whose id does not actually sit under its parent keeps its full id.
 * That is not a defensive branch for an impossible case — `repeatBlocks` and
 * `autoModularize` both wrap EXISTING nodes in a frame whose id they invent, so
 * their children genuinely do not carry the parent's prefix, and silently
 * trimming a non-prefix would rename the node.
 */
export function scopedNodeLabel(nodeId: string, parentId?: string | null): string {
  if (!parentId) return nodeId;
  const prefix = `${parentId}.`;
  return nodeId.startsWith(prefix) && nodeId.length > prefix.length
    ? nodeId.slice(prefix.length)
    : nodeId;
}
