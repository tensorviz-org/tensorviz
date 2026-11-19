import React from 'react';
import { act, renderHook, render, screen } from '@testing-library/react';
import { NodeUnderstanding } from '../../canvas/NodeUnderstanding';
import { NodeExplanationContext } from '../../canvas/NodeExplanationContext';
import { useNodeExplanationAction } from '../../canvas/useNodeExplanationAction';

const ready = (id = 'a', revision = 'revision') => ({ requestId: 'selection', nodeId: id, revision, status: 'ready',
  explanation: { title: 'Role', purpose: 'Source-backed purpose.', inputs: [], outputs: [], steps: [], uncertainties: [], citations: [] } });
let value: any;
beforeEach(() => { value = { scope: 'scope-a', states: {}, annotations: {}, retry: jest.fn(), reveal: jest.fn(),
  model: { start: jest.fn(() => 'pass-a'), cancel: jest.fn(), toggle: jest.fn() } }; });
const wrapper = ({ children }: { children: React.ReactNode }) => <NodeExplanationContext.Provider value={value}>{children}</NodeExplanationContext.Provider>;

it('restores independent in-flight node ownership after the action remounts', () => {
  value = { ...value, nodeJobs: { currentRequestIds: { a: 'pass-a', b: 'pass-b' },
    requests: {
      'pass-a': { state: { requestId: 'pass-a', revision: 'revision', status: 'loading', completed: 0, total: 1 } },
      'pass-b': { state: { requestId: 'pass-b', revision: 'revision', status: 'loading', completed: 0, total: 1 } },
    }, start: jest.fn(), cancel: jest.fn() } };
  const { result, rerender } = renderHook(({ id }) => useNodeExplanationAction(id), { wrapper, initialProps: { id: 'a' } });
  expect(result.current.phase).toBe('working');
  const stopA = result.current.run;
  rerender({ id: 'b' });
  expect(result.current.phase).toBe('working');
  act(() => stopA());
  expect(value.nodeJobs.cancel).toHaveBeenCalledWith('pass-a');
  expect(result.current.phase).toBe('working');
  expect(value.model.cancel).not.toHaveBeenCalled();
});

it('stays idle on selection and exposes preparation immediately only after a click', () => {
  const { result } = renderHook(() => useNodeExplanationAction('a'), { wrapper });
  expect(result.current.phase).toBe('idle'); expect(value.model.start).not.toHaveBeenCalled();
  act(() => result.current.run());
  expect(value.model.start).toHaveBeenCalledWith(['a']);
  expect(result.current.phase).toBe('preparing');
});

it('follows the matching request through working and verified completion', () => {
  const { result, rerender } = renderHook(() => useNodeExplanationAction('a'), { wrapper });
  act(() => result.current.run());
  value = { ...value, model: { ...value.model, state: { requestId: 'pass-a', revision: 'revision', status: 'loading', completed: 0, total: 1 } } };
  rerender(); expect(result.current.phase).toBe('working');
  value = { ...value, model: { ...value.model, state: { ...value.model.state, status: 'ready', completed: 1 } } };
  rerender(); expect(result.current.phase).not.toBe('complete');
  value = { ...value, annotations: { a: ready() }, verifiedResults: { a: { requestId: 'pass-a', result: { ...ready(), requestId: 'pass-a' }, adoption: ready() } } };
  rerender(); expect(result.current.phase).toBe('complete');
});

it('keeps operation state on its originating node while the user explores another node', () => {
  const { result, rerender } = renderHook(({ id }) => useNodeExplanationAction(id), { wrapper, initialProps: { id: 'a' } });
  act(() => result.current.run());
  rerender({ id: 'b' }); expect(result.current.phase).toBe('idle');
  value = { ...value, annotations: { a: ready() }, verifiedResults: { a: { requestId: 'pass-a', result: { ...ready(), requestId: 'pass-a' }, adoption: ready() } }, model: { ...value.model,
    state: { requestId: 'pass-a', revision: 'revision', status: 'ready', completed: 1, total: 1 } } };
  rerender({ id: 'b' }); expect(result.current.phase).toBe('idle');
  rerender({ id: 'a' }); expect(result.current.phase).toBe('complete');
  value = { ...value, scope: 'scope-b', states: {}, annotations: {}, model: { ...value.model, state: undefined } };
  rerender({ id: 'a' }); expect(result.current.phase).toBe('idle');
});

it('represents failure on the action without borrowing a retained result as new success', () => {
  const { result, rerender } = renderHook(() => useNodeExplanationAction('a'), { wrapper });
  act(() => result.current.run());
  value = { ...value, annotations: { a: ready() }, model: { ...value.model,
    state: { requestId: 'pass-a', revision: 'revision', status: 'unavailable', detail: 'Context limit exceeded.', completed: 0, total: 1 } } };
  rerender(); expect(result.current.phase).toBe('error'); expect(result.current.detail).toBe('Context limit exceeded.');
});

it('stops through the same action and does not resurrect completion from a late receipt', () => {
  const { result, rerender } = renderHook(() => useNodeExplanationAction('a'), { wrapper });
  act(() => result.current.run());
  act(() => result.current.run()); expect(value.model.cancel).toHaveBeenCalledTimes(1);
  expect(result.current.phase).toBe('stopped');
  value = { ...value, annotations: { a: ready() }, model: { ...value.model,
    state: { requestId: 'pass-a', revision: 'revision', status: 'ready', completed: 1, total: 1 } } };
  rerender(); expect(result.current.phase).toBe('stopped');
  expect(value.model.start).toHaveBeenCalledTimes(1);
});

it('settles interrupted work on the button and never restarts it after reconnect', () => {
  const { result, rerender } = renderHook(() => useNodeExplanationAction('a'), { wrapper });
  act(() => result.current.run());
  value = { ...value, model: { ...value.model, state: undefined, disabledReason: 'Reconnect first.' } };
  rerender(); expect(result.current.phase).toBe('error');
  value = { ...value, annotations: { a: ready() }, model: { ...value.model, disabledReason: undefined,
    state: { requestId: 'pass-a', revision: 'revision', status: 'ready', completed: 1, total: 1 } } };
  rerender(); expect(result.current.phase).toBe('error');
  expect(value.model.start).toHaveBeenCalledTimes(1);
});

it('does not let an old same-revision reading certify a new pass or borrow its metadata', () => {
  const previous = ready();
  value = { ...value, states: { a: previous }, annotations: { a: previous } };
  const { result, rerender } = renderHook(() => useNodeExplanationAction('a'), { wrapper });
  act(() => result.current.run());
  value = { ...value, model: { ...value.model, state: { requestId: 'pass-a', revision: 'revision', status: 'ready', completed: 1, total: 1,
    execution: { provider: 'codex', model: 'new-model', effort: 'high' } } } };
  rerender();
  expect(result.current.phase).not.toBe('complete');
  expect(result.current.execution).toBeUndefined();
});

it('omits unbound execution metadata from a retained visible reading', () => {
  const previous = ready();
  value = { ...value, states: { a: previous }, annotations: { a: previous } };
  const node: any = { id: 'a', type: 'layerNode', position: { x: 0, y: 0 }, data: { layerType: { id: 'custom', name: 'ActualCall', category: 'misc' }, parameters: {} } };
  const view = render(<NodeExplanationContext.Provider value={value}><NodeUnderstanding node={node} hideActions
    explanationExecution={{ provider: 'codex', model: 'new-model', effort: 'high' }} /></NodeExplanationContext.Provider>);
  expect(screen.getByText(previous.explanation.purpose)).toBeInTheDocument();
  expect(view.container.querySelector('.node-ai-metadata')).toBeNull();
});

it('handles preflight failure and blocked actions without inventing a graph revision', () => {
  const { result, rerender } = renderHook(() => useNodeExplanationAction('a'), { wrapper });
  act(() => result.current.run());
  value = { ...value, model: { ...value.model, startFailure: { requestId: 'pass-a', detail: 'Revision verification failed.' } } };
  rerender(); expect(result.current.phase).toBe('error'); expect(result.current.detail).toBe('Revision verification failed.');
  value = { ...value, model: { ...value.model, disabledReason: 'Reconnect first.' } };
  rerender(); act(() => result.current.run());
  expect(value.model.start).toHaveBeenCalledTimes(1);
});
