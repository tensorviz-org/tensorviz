import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { GraphRunNotice, graphFailureNode } from '../../ide/GraphRunNotice';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE } from '../../ide/localBrowserState';
import type { LocalBrowserPageState } from '../../ide/localBrowserState';

const page = (): LocalBrowserPageState => ({ ...EMPTY_LOCAL_BROWSER_PAGE_STATE,
  opening: { id: 'run', phase: 'failed', hasCapture: true, detail: 'Expected width 8, received 4.', failure: { kind: 'execution' },
    target: { workspaceFolder: '/workspace', cls: 'Net', file: 'model.py', documentVersion: 2, backendRevision: 3, sourceDigest: 'a'.repeat(64) } },
  projection: { graphId: 'graph', revision: 3, cls: 'Net', file: 'model.py', sourceDigest: 'a'.repeat(64), traced: true,
    greyed: false, viewRestored: false, spanmap: {}, document: { semantic: { nodes: [{ node_id: 'failed' }],
      metadata: { capture: { failureNodeId: 'failed' } } } } },
});
test('a shape failure names the layer that stopped the capture', () => {
  const p = page();
  (p.projection!.document as any).semantic.nodes = [{ node_id: 'failed', params: { _module_path: 'classifier.fc' } }];
  render(<GraphRunNotice page={p} connected onAction={jest.fn()} />);
  expect(screen.getByText('Capture stopped at classifier.fc')).toBeVisible();
  expect(screen.getByText('Expected width 8, received 4.')).toBeVisible();
});

test('one action focuses the actual failure, without reopening input setup', () => {
  const listener = jest.fn(); window.addEventListener('builder-command', listener);
  const action = jest.fn(); render(<GraphRunNotice page={page()} connected onAction={action} />);
  fireEvent.click(screen.getByRole('button', { name: 'Show failure' }));
  expect(listener.mock.calls[0][0].detail).toEqual({ command: 'focus-node', payload: { nodeId: 'failed' } });
  expect(action).not.toHaveBeenCalled(); window.removeEventListener('builder-command', listener);
});
test('timeout and stale captures cannot blame a historical node', () => {
  const p = page(); p.opening!.hasCapture = false; p.opening!.failure = { kind: 'capture' };
  render(<GraphRunNotice page={p} connected onAction={jest.fn()} />);
  expect(screen.queryByRole('button', { name: 'Show failure' })).toBeNull();
  expect(graphFailureNode({ ...p.projection!, greyed: true })).toBeUndefined();
});
test.each(['execution', 'capture', 'environment', 'source'] as const)('%s failure offers explicit MCP help without retrying', kind => {
  const p = page(); p.opening!.failure = { kind };
  const onAction = jest.fn(); const view = render(<GraphRunNotice page={p} connected onAction={onAction} />);
  fireEvent.click(screen.getByRole('button', { name: 'Use Agent' }));
  expect(onAction).toHaveBeenCalledTimes(1);
  expect(onAction).toHaveBeenCalledWith({ kind: 'agent_help' });
  view.rerender(<GraphRunNotice page={p} connected={false} onAction={onAction} />);
  expect(screen.getByRole('button', { name: 'Use Agent' })).toBeDisabled();
});

test.each([true, false])('explains retained historical context for traced=%s when it is displayed', traced => {
  const p = page();
  p.projection = { ...p.projection!, traced, debug: { previousDocument: p.projection!.document, nodePairs: [], changed: [] },
    ...(!traced ? { document: { semantic: { nodes: [] } } } : {}) };
  render(<GraphRunNotice page={p} connected onAction={jest.fn()} />);
  expect(screen.getByText(/Dashed context is from the previous run/)).toBeVisible();
});
