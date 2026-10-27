import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { NodeEditButton, NodeEditContext, NodeSourceLink } from '../../canvas/NodeSourceControls';
import { reportNodeClick } from '../../ide/projection';
jest.mock('../../ide/projection', () => ({ localBrowserMode: () => true, currentProjection: () => ({ spanmap: { fc: {} } }), reportNodeClick: jest.fn() }));

test('node controls use the exact node identity, and unavailable captured edits are disabled', () => {
  const open = jest.fn();
  const data = { captureLabel: 'encoder.fc', parameters: {}, sourceLocation: { file: 'models/encoder.py', line: 42 } } as any;
  const view = render(<NodeEditContext.Provider value={open}><NodeEditButton nodeId="fc" data={data} /><NodeSourceLink nodeId="fc" data={data} /></NodeEditContext.Provider>);
  fireEvent.click(screen.getByRole('button', { name: 'Edit encoder.fc' }));
  expect(open).toHaveBeenCalledWith('fc');
  fireEvent.click(screen.getByRole('button', { name: 'models/encoder.py:42' }));
  expect(screen.getByRole('button', { name: 'models/encoder.py:42' })).toHaveTextContent('encoder.py:42');
  expect(view.container.querySelector('.node-source-line')).toHaveTextContent(':42');
  expect(reportNodeClick).toHaveBeenCalledWith('fc');
  view.rerender(<NodeEditContext.Provider value={open}><NodeEditButton nodeId="fc" data={{ ...data, captureRef: {} }} /></NodeEditContext.Provider>);
  expect(screen.getByRole('button')).toBeDisabled();
  fireEvent.focus(screen.getByRole('button').parentElement!);
  expect(screen.getByRole('tooltip')).toHaveTextContent('No supported source edit for this call');
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(screen.queryByRole('tooltip')).toBeNull();
  view.rerender(<NodeSourceLink nodeId="unmapped" data={data} />);
  expect(screen.queryByRole('button')).toBeNull();
});
