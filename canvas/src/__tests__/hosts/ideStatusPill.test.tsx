/**
 * The IDE header shows the run, and the save pill is gone (task.md M4/M8).
 *
 * `ideStatus.test.ts` pins what the four states SAY. This pins that the shell
 * shows them, and — the half that matters — that it stops showing the web
 * product's save pill when it does. Two pills disagreeing about the same corner
 * of the screen would be worse than the single wrong one this replaces.
 */
import React from 'react';
import { render, screen } from '@testing-library/react';

import WorkbenchShell from '../../components/ui2/WorkbenchShell';

jest.mock('react-router-dom', () => ({
  useNavigate: () => jest.fn(),
  useLocation: () => ({ pathname: '/builder' }),
}));
jest.mock('../../components/auth/UserMenu', () => ({ __esModule: true, default: () => null }));
jest.mock('../../services/useBackendEnvironment', () => ({
  useBackendEnvironment: () => ({ environment: null, loading: false, error: null }),
}));

const shell = (props: Record<string, unknown>) =>
  render(
    <WorkbenchShell chrome="minimal" {...props}>
      <div />
    </WorkbenchShell>,
  );

describe('the IDE header', () => {
  it('reports the run instead of a save that never happens', () => {
    shell({
      // The state a user saw: the backend had gone away, so the web pill latched
      // to "Offline — changes queued". Nothing was queued; there is no project
      // store behind an IDE canvas.
      saveState: 'offline',
      ideStatus: {
        tone: 'warn',
        text: 'Run failed — shapes are incomplete',
        detail: 'mat1 and mat2 shapes cannot be multiplied (line 13).',
      },
    });

    expect(screen.getByTestId('ide-status-pill')).toHaveTextContent('Run failed');
    expect(screen.queryByText(/changes queued/)).toBeNull();
    // The message is the useful part and does not fit in a pill.
    expect(screen.getByTestId('ide-status-pill')).toHaveAttribute(
      'title',
      expect.stringContaining('mat1 and mat2'),
    );
  });

  it('leaves the web product exactly as it was', () => {
    // No ideStatus means this is not an IDE canvas, and the save pill is the
    // truth there — a graph really is autosaved to the project store.
    shell({ saveState: 'offline' });

    expect(screen.queryByTestId('ide-status-pill')).toBeNull();
    expect(screen.getByText(/changes queued/)).toBeInTheDocument();
  });
});
