import { atom, computed, Editor } from 'tldraw';
import { shapeIdToNodeId } from './canvasShapeId';
import { isModelShape } from './canvasShapePolicy';
import { FocusEdge, indexConnections, seedEdgeDirections, TraceDirection, traceConnections } from './neighborhood';
import type { ModelNode } from '../types';

function createFocus(editor: Editor) {
  const index = atom('Graph adjacency', indexConnections([]));
  const members = atom('Visible module contents', new Map<string, string[]>());
  const trace = atom<{ key: string; direction: TraceDirection }>('Graph trace', { key: '', direction: 'local' });
  const focus = computed('Graph neighborhood', () => {
    const selected = editor.getSelectedShapeIds().filter(id => isModelShape(editor.getShape(id))).map(shapeIdToNodeId);
    const key = JSON.stringify(selected);
    const direction = trace.get().key === key ? trace.get().direction : 'local';
    const seeds = selected.flatMap(id => [id, ...(members.get().get(id) ?? [])]);
    const traced = traceConnections(index.get(), seeds, direction);
    const { incoming, outgoing } = seedEdgeDirections(index.get(), seeds, traced.edgeIds);
    return { ...traced, incomingEdgeIds: incoming, outgoingEdgeIds: outgoing, active: selected.length > 0, selected, key, direction };
  });
  return { index, members, trace, focus, signature: '' };
}
const states = new WeakMap<Editor, ReturnType<typeof createFocus>>();
export function graphFocus(editor: Editor) {
  let state = states.get(editor);
  if (!state) { state = createFocus(editor); states.set(editor, state); }
  return state;
}
/** Called at graph sync, never during a camera move or from an individual edge. */
export function updateFocusConnections(editor: Editor, edges: FocusEdge[], nodes: readonly ModelNode[]) {
  const state = graphFocus(editor);
  const signature = JSON.stringify([edges, nodes.map(node => [node.id, node.data.parentId ?? node.parentNode, node.hidden])]);
  if (state.signature === signature) return;
  state.signature = signature;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const members = new Map<string, string[]>();
  for (const node of nodes) if (!node.hidden) {
    let parent = node.data.parentId ?? node.parentNode;
    const seen = new Set<string>([node.id]);
    while (parent && !seen.has(parent)) {
      seen.add(parent);
      if (!members.has(parent)) members.set(parent, []);
      members.get(parent)!.push(node.id);
      const ancestor = byId.get(parent);
      parent = ancestor?.data.parentId ?? ancestor?.parentNode;
    }
  }
  state.members.set(members);
  state.index.set(indexConnections(edges));
  state.trace.set({ key: '', direction: 'local' });
}
export function setTraceDirection(editor: Editor, direction: TraceDirection) {
  const state = graphFocus(editor);
  state.trace.set({ key: state.focus.get().key, direction });
}
