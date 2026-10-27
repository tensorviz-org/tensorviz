/** Canvas chrome stays independent of website routes. */
import React from 'react';
import { render, screen } from '@testing-library/react';

import WorkbenchShell from '../../components/ui2/WorkbenchShell';

jest.mock('react-router-dom', () => ({
  useNavigate: () => jest.fn(),
  useLocation: () => ({ pathname: '/builder' }),
}));
jest.mock('../../components/auth/UserMenu', () => ({
  __esModule: true,
  default: () => <button aria-label="Account menu" />,
}));
jest.mock('../../services/useBackendEnvironment', () => ({
  useBackendEnvironment: () => ({ environment: null, loading: false, error: null }),
}));

describe('the IDE chrome', () => {
  const shell = (chrome: 'ide') =>
    render(
      <WorkbenchShell chrome={chrome}>
        <div />
      </WorkbenchShell>,
    );

  it('offers no way out of the canvas', () => {
    shell('ide');

    expect(screen.queryByLabelText('Navigation menu')).toBeNull();
    expect(screen.queryByLabelText('Account menu')).toBeNull();
  });

  it('keeps the canvas chrome it is there to frame', () => {
    shell('ide');

    expect(document.querySelector('.ui2-topbar.h-10')).not.toBeNull();
    expect(document.querySelector('aside.ui2-sidebar')).toBeNull();
  });

  it('mounts the local canvas without router, account or persistence providers', () => {
    render(<WorkbenchShell chrome="local"><div>Local canvas</div></WorkbenchShell>);
    expect(screen.getByText('Local canvas')).toBeInTheDocument();
    expect(screen.queryByLabelText('Account menu')).toBeNull();
    expect(screen.queryByLabelText('Navigation menu')).toBeNull();
  });
});
