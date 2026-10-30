import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { NodeSourceLink } from '../../canvas/NodeSourceControls';
import { reportNodeClick } from '../../ide/projection';
jest.mock('../../ide/projection', () => ({ localBrowserMode: () => true, currentProjection: () => ({ spanmap: { fc: {} } }), reportNodeClick: jest.fn() }));

beforeEach(() => { (reportNodeClick as jest.Mock).mockClear(); });

test('the source link opens the exact node, and an unmapped node offers none', () => {
  const data = { captureLabel: 'encoder.fc', parameters: {}, sourceLocation: { file: 'models/encoder.py', line: 42 } } as any;
  const view = render(<NodeSourceLink nodeId="fc" data={data} />);
  fireEvent.click(screen.getByRole('button', { name: 'models/encoder.py:42' }));
  expect(screen.getByRole('button', { name: 'models/encoder.py:42' })).toHaveTextContent('encoder.py:42');
  expect(view.container.querySelector('.node-source-line')).toHaveTextContent(':42');
  expect(reportNodeClick).toHaveBeenCalledWith('fc');
  view.rerender(<NodeSourceLink nodeId="unmapped" data={data} />);
  expect(screen.queryByRole('button')).toBeNull();
});
