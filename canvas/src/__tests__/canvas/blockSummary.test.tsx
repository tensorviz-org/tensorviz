import React from 'react';
import { render, screen } from '@testing-library/react';
import { BlockSummary, FamilyLegend } from '../../canvas/BlockSummary';
import { BlockIdentity } from '../../canvas/BlockIdentity';
import type { ModelEdge, ModelNode } from '../../types';

const captured = (id: string, name: string, extra: Partial<ModelNode['data']> = {}, hidden = false) => ({
  id, hidden, position: { x: 0, y: 0 }, type: 'moduleNode',
  data: { captureRef: { kind: 'invocation' }, layerType: { id: 'module', name, displayName: name.split('.').pop() }, parameters: {}, ...extra },
} as unknown as ModelNode);
const edge = (source: string, target: string) => ({ id: `${source}-${target}`, source, target } as ModelEdge);

const block = captured('block', 'models.blocks.ConvNormAct', { moduleTemplateId: 'ConvNormAct' });
const children = [captured('block.act', 'torch.nn.modules.activation.SiLU', { parentId: 'block' }, true),
  captured('block.conv', 'torch.nn.modules.conv.Conv2d', { parentId: 'block' }, true),
  captured('block.bn', 'torch.nn.modules.batchnorm.BatchNorm2d', { parentId: 'block' }, true)];
const edges = [edge('block.conv', 'block.bn'), edge('block.bn', 'block.act')];

it('names a composite module neutrally, with its exact class and verified composition', () => {
  const { container } = render(<BlockSummary node={block} nodes={[block, ...children]} edges={edges} />);
  expect(screen.getByText('Composite module')).toBeInTheDocument();
  expect(screen.getByText('models.blocks.ConvNormAct')).toBeInTheDocument();
  expect(container.querySelector('.graph-block-composition')).toHaveTextContent('Convolution → Normalization → Activation');
});

it('shows a recognised module family without inventing a composition', () => {
  const { container } = render(<BlockSummary node={children[2]} nodes={[block, ...children]} edges={edges} />);
  expect(screen.getByText('Normalization')).toBeInTheDocument();
  expect(screen.getByText('torch.nn.modules.batchnorm.BatchNorm2d')).toBeInTheDocument();
  expect(container.querySelector('.graph-block-composition')).toBeNull();
});

it('names a card by its family glyph in the family colour, with no separate marker bar', () => {
  const { container } = render(<BlockIdentity block={{ kind: 'convolution', label: 'Convolution', modifier: 'Lazy' }} />);
  expect(screen.getByRole('img', { name: 'Convolution · Lazy' })).toHaveAttribute('title', 'Convolution · Lazy');
  expect(container.querySelector('svg[data-block="convolution"]')).toHaveStyle({ color: 'var(--tv-family-convolution)' });
  expect(container.querySelector('.block-marker')).toBeNull();
});

it('lists only the families shown in the current graph', () => {
  render(<FamilyLegend nodes={[block, ...children, { ...children[1], id: 'shown', hidden: false }]} />);
  expect(screen.getByText('Blocks').closest('summary')).toHaveAttribute('aria-label', 'Block families: Convolution, Composite module');
  expect(screen.queryByText('Normalization')).not.toBeInTheDocument();
});
