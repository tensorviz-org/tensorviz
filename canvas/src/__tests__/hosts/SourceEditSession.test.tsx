import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { SourceEditSession } from '../../ide/SourceEditSession';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { EMPTY_LOCAL_BROWSER_PAGE_STATE } from '../../ide/localBrowserState';
import type { ModelNode } from '../../types';

test.each(['Escape', 'close button'])('the inline form uses native review and closes with %s', method => {
  const cancel = jest.fn();
  function Harness() {
    const [open, setOpen] = React.useState(false);
    return <LocalBrowserContext.Provider value={{ page: EMPTY_LOCAL_BROWSER_PAGE_STATE,
      status: { phase: 'connected', hostLink: 'connected', error: null }, sourceEdits: { propose: jest.fn(), cancel, dispose: jest.fn() } }}>
      <button onClick={() => setOpen(true)}>Edit fc</button>
      {open && <SourceEditSession nodes={[{ id: 'fc', data: { parameters: { out_features: 8 } } }] as ModelNode[]} onClose={() => setOpen(false)} />}
    </LocalBrowserContext.Provider>;
  }
  render(<Harness />);
  const opener = screen.getByRole('button', { name: 'Edit fc' }); opener.focus(); fireEvent.click(opener);
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(screen.getByRole('region', { name: 'Edit node source' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Review in VS Code' })).toBeEnabled();
  const key = jest.fn(); document.addEventListener('keydown', key);
  try {
    if (method === 'Escape') fireEvent.keyDown(screen.getByLabelText('Parameter value'), { key: 'Escape' });
    else fireEvent.click(screen.getByRole('button', { name: 'Close source editor' }));
    expect(key).not.toHaveBeenCalled();
    expect(screen.queryByRole('region', { name: 'Edit node source' })).toBeNull();
    expect(opener).toHaveFocus(); expect(cancel).toHaveBeenCalledTimes(1);
  } finally { document.removeEventListener('keydown', key); }
});
