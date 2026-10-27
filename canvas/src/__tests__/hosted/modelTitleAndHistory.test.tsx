/**
 * The builder must show the model's real name, let you rename it, and expose
 * version history.
 *
 * The title was a hardcoded <span>Model Graph</span>: every model displayed the same
 * literal string whatever it was called, and double-clicking it did nothing because
 * it was not an input. Version history existed only as a command-palette entry, so
 * there was no version/history affordance anywhere in the builder chrome — a new
 * user following the guide could not reach it. "New Project" in the UI also
 * contradicted the guide's "New model", for a button that creates a model.
 *
 * The RENAME half is a behaviour test. It used to be three 2200-character windows
 * cut out of WorkbenchShell.tsx — one of which then took a further 120-character
 * slice and asserted the absence of the word `onRename` inside it, which passes
 * just as well if the window shrinks. WorkbenchShell imports no tldraw, so it
 * mounts: the assertions below drive the title the way a user does, and one of
 * them (Escape) covers a case the old absence-in-a-window trick could not state.
 *
 * The WIRING half — that the name comes from the builder snapshot, and that renaming goes
 * through the manual-save path — still reads ModelBuilder.tsx and BuilderPageV2.tsx
 * as text, because both pull in tldraw and cannot be mounted under the CRA jest
 * transform (see qa_builder_ux.test.ts). Those assertions are whole-file
 * `toContain`s on single statements rather than character-offset windows.
 *
 * Behaviour was measured in a browser: the title showed "mlp_tiny", double-click
 * opened an input, Enter renamed it to "QA Renamed Model", the name survived a
 * reload, and Escape cancelled without committing.
 */
import React from 'react';
import fs from 'fs';
import path from 'path';
import { fireEvent, render, screen } from '@testing-library/react';
import WorkbenchShell from '../../components/ui2/WorkbenchShell';

// The shell's account chrome and its backend probe are irrelevant to the title.
// The repo-wide react-router-dom stub has no useLocation, which the shell reads
// to highlight the active nav item; supply one rather than widen the shared stub.
jest.mock('react-router-dom', () => ({
  useNavigate: () => jest.fn(),
  useLocation: () => ({ pathname: '/builder' }),
}));
jest.mock('../../components/auth/UserMenu', () => ({ __esModule: true, default: () => null }));
jest.mock('../../services/useBackendEnvironment', () => ({
  useBackendEnvironment: () => ({ environment: null, loading: false, error: null }),
}));

const read = (...p: string[]) => fs.readFileSync(path.join(__dirname, '../..', ...p), 'utf8');
const PAGE = read('components', 'ui2', 'BuilderPageV2.tsx');
const BUILDER = read('components', 'ModelBuilder.tsx');

const renderTitle = (props: { modelName?: string; onRenameModel?: (name: string) => void }) =>
  render(
    <WorkbenchShell chrome="minimal" {...props}>
      <div />
    </WorkbenchShell>,
  );

describe('the builder title is the model name', () => {
  it('shows the model it was given, not a hardcoded string', () => {
    renderTitle({ modelName: 'mlp_tiny', onRenameModel: jest.fn() });
    expect(screen.getByText('mlp_tiny')).toBeInTheDocument();
    expect(screen.queryByText('Model Graph')).not.toBeInTheDocument();
  });

  it('takes the name from the builder snapshot, which reports the live identity', () => {
    expect(BUILDER).toContain('graphName: graphIdentityRef.current.name');
    expect(PAGE).toContain('modelName={snapshot.graphName}');
  });

  it('renames on double-click and commits on Enter', () => {
    const onRenameModel = jest.fn();
    renderTitle({ modelName: 'mlp_tiny', onRenameModel });
    fireEvent.doubleClick(screen.getByText('mlp_tiny'));
    const input = screen.getByLabelText('Model name') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'QA Renamed Model' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onRenameModel).toHaveBeenCalledWith('QA Renamed Model');
  });

  it('cancels on Escape without committing', () => {
    const onRenameModel = jest.fn();
    renderTitle({ modelName: 'mlp_tiny', onRenameModel });
    fireEvent.doubleClick(screen.getByText('mlp_tiny'));
    const input = screen.getByLabelText('Model name') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'Discard me' } });
    fireEvent.keyDown(input, { key: 'Escape' });
    expect(onRenameModel).not.toHaveBeenCalled();
    // And the old name is back on screen — an abandoned draft must not linger.
    expect(screen.getByText('mlp_tiny')).toBeInTheDocument();
  });

  it('does not fire a rename for an unchanged or blank name', () => {
    // Committing the same name would queue a pointless save; a blank one would
    // leave the model with no title at all.
    const onRenameModel = jest.fn();
    renderTitle({ modelName: 'mlp_tiny', onRenameModel });
    fireEvent.doubleClick(screen.getByText('mlp_tiny'));
    const input = screen.getByLabelText('Model name') as HTMLInputElement;
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onRenameModel).not.toHaveBeenCalled();

    fireEvent.doubleClick(screen.getByText('mlp_tiny'));
    const reopened = screen.getByLabelText('Model name') as HTMLInputElement;
    fireEvent.change(reopened, { target: { value: '   ' } });
    fireEvent.keyDown(reopened, { key: 'Enter' });
    expect(onRenameModel).not.toHaveBeenCalled();
  });

  it('stays read-only when no rename handler is supplied', () => {
    // Other routes use this shell and must not offer a rename that goes nowhere.
    renderTitle({ modelName: 'mlp_tiny' });
    fireEvent.doubleClick(screen.getByText('mlp_tiny'));
    expect(screen.queryByLabelText('Model name')).not.toBeInTheDocument();
  });

  it('routes the rename through the manual-save path', () => {
    // handleSave cancels the autosave debounce and adopts the identity BEFORE
    // building the request, so a later autosave cannot revert the name.
    const cmd = BUILDER.slice(BUILDER.indexOf("if (command === 'rename-model')"), BUILDER.indexOf("if (command === 'rename-model')") + 400);
    expect(cmd).toContain('handleSave(name,');
    expect(cmd).toContain('graphIdentityRef.current.name) return');
  });
});

describe('version history is reachable', () => {
  it('has a button in the builder chrome, not only the command palette', () => {
    expect(PAGE).toContain('data-testid="ui2-toggle-history"');
    expect(PAGE).toContain('aria-label="Version history"');
  });

  it('opens the panel it already rendered', () => {
    const btn = PAGE.slice(PAGE.indexOf('data-testid="ui2-toggle-history"') - 400, PAGE.indexOf('data-testid="ui2-toggle-history"'));
    expect(btn).toContain('setHistoryOpen(true)');
  });
});

describe('the new-model action is called the same thing everywhere', () => {
  it('says "New model", matching the guide', () => {
    // Both buttons navigate to /builder, which creates a model, not a project.
    expect(read('components', 'ui2', 'ProjectsPageV2.tsx')).not.toContain('New Project\n');
    expect(read('components', 'ui2', 'OverviewPageV2.tsx')).toContain('New model');
  });
});
