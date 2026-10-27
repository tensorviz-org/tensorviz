import React, { useState } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { PassThroughToggle } from '../../components/PassThroughToggle';
import { showPassThroughCalls } from '../../services/passThroughPresentation';
import { revealGraphNode, restoreGraphFileView } from '../../viewer/graphFileView';
import type { ModelNode } from '../../types';

const nodes = [
  { id: 'root', type: 'moduleNode', position: { x: 0, y: 0 }, data: { expanded: false, parameters: {} } },
  { id: 'noop', type: 'moduleNode', position: { x: 0, y: 0 }, data: { parentId: 'root', parameters: {}, capturePassThrough:
    { valueId: 'tensor', inputPortId: 'in', sourceNodeId: 'root', sourcePortId: 'out' } } },
] as ModelNode[];

test('the reversible control exposes the hidden count and does not appear for ordinary graphs', () => {
  function Control() {
    const [current, setCurrent] = useState(nodes);
    return <PassThroughToggle nodes={current} onToggle={shown => setCurrent(showPassThroughCalls(current, shown))} />;
  }
  const view = render(<Control />);
  fireEvent.click(screen.getByRole('button', { name: 'Show unchanged calls (1)', pressed: false }));
  fireEvent.click(screen.getByRole('button', { name: 'Hide unchanged calls (1)', pressed: true }));
  expect(screen.getByRole('button', { name: 'Show unchanged calls (1)' })).toBeInTheDocument();
  view.rerender(<PassThroughToggle nodes={nodes.slice(0, 1)} onToggle={jest.fn()} />);
  expect(screen.queryByRole('button')).not.toBeInTheDocument();
});

test('offline search and a walkthrough selection reveal the exact hidden call', () => {
  const graph = { nodes, edges: [] };
  for (const revealed of [revealGraphNode(graph, 'noop'), restoreGraphFileView(graph, {
    camera: { x: 0, y: 0, z: 1 }, nodePositions: {}, expandedModuleIds: [], selectedNodeIds: ['noop'],
  })]) {
    expect(revealed.nodes.find(node => node.id === 'noop')).toMatchObject({ hidden: false, data: { passThroughShown: true } });
    expect(revealed.nodes.find(node => node.id === 'root')!.data.expanded).toBe(true);
  }
  expect(nodes[1].data.passThroughShown).toBeUndefined();
});

test('the canvas menu exposes unchanged-call visibility as a checked menu item', () => {
  const onToggle = jest.fn();
  const view = render(<PassThroughToggle menuItem nodes={nodes} onToggle={onToggle} />);
  fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Show unchanged calls (1)', checked: false }));
  expect(onToggle).toHaveBeenCalledWith(true);
  view.rerender(<PassThroughToggle menuItem nodes={showPassThroughCalls(nodes, true)} onToggle={onToggle} />);
  expect(screen.getByRole('menuitemcheckbox', { name: 'Hide unchanged calls (1)', checked: true })).toBeInTheDocument();
});
