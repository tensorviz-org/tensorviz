import { act, renderHook } from '@testing-library/react';
import { useRef, useState } from 'react';
import type { Editor } from 'tldraw';
import { captureNativePresentation, restoreNativePresentation, useNativeCanvasPresentation } from '../../ide/nativeCanvasPresentation';
import type { ModelNode, ModelEdge } from '../../types';

jest.mock('../../ide/bridge', () => ({ postToHost: jest.fn() }));

test('sidebar recreation restores revealed calls and regenerates their display connectors', () => {
  window.history.replaceState({}, '', '?view=canvas');
  const nodes = [
    { id: 'source', type: 'inputNode', position: { x: 0, y: 0 }, data: { parameters: {} } },
    { id: 'root', type: 'moduleNode', position: { x: 250, y: 0 }, data: { expanded: true, parameters: {} } },
    { id: 'noop', type: 'moduleNode', position: { x: 20, y: 52 }, hidden: true, data: { parentId: 'root', parameters: {},
      capturePassThrough: { valueId: 'v', sourceNodeId: 'source', sourcePortId: 'out', inputPortId: 'in' } } },
  ] as ModelNode[];
  restoreNativePresentation({ graphId: 'native-noop', camera: { x: 1, y: 2, z: 1 }, selected: [],
    nodes: [{ id: 'noop', passThroughShown: true, hidden: false }] });
  const editor = { getCamera: () => ({ x: 1, y: 2, z: 1 }), setCamera: jest.fn(), getSelectedShapeIds: () => [],
    select: jest.fn(), store: { listen: () => () => {} } } as unknown as Editor;
  const hook = renderHook(() => {
    const [current, setNodes] = useState(nodes), [edges, setEdges] = useState<ModelEdge[]>([]);
    const initialFit = useRef<string | null>(null);
    useNativeCanvasPresentation('native-noop', current, setNodes, editor, initialFit, edges, setEdges);
    return { nodes: current, edges };
  });
  expect(hook.result.current.nodes.find(node => node.id === 'noop')).toMatchObject({ hidden: false, data: { passThroughShown: true } });
  expect(hook.result.current.edges).toEqual([expect.objectContaining({ id: 'passthrough:noop', hidden: false })]);
  act(() => { expect(captureNativePresentation()!.nodes.find(node => node.id === 'noop')!.passThroughShown).toBe(true); });
  hook.unmount();
  window.history.replaceState({}, '', '/');
});
