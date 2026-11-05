/**
 * The palette's Workspace section (task.md M6).
 *
 * The user's own `nn.Module` classes are offered beside nn.* layers and paper
 * blocks — from `/api/ide/workspace/classes`, which reads the folder by AST and
 * imports nothing. Three things are pinned, and each of them is a promise the
 * product would otherwise break:
 *
 * 1. In the hosted web product there is no workspace, so there is NO section —
 *    absent, not an empty box or a spinner that never resolves.
 * 2. In the IDE the section lists what the scan found, keyed by the class name a
 *    person would search for.
 * 3. Inserting one produces a `custom_code` node carrying `_class` and
 *    `_import_path`. That is what keeps user code out of the canonical registry
 *    (the template sweep enumerates it) while still generating a real import.
 */
import React from 'react';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import BuilderPaletteV2 from '../../components/ui2/BuilderPaletteV2';
import { LocalBrowserContext } from '../../ide/localBrowserContext';
import { setWorkspaceRoot } from '../../ide/workspaceClasses';

jest.mock('../../ide/projection', () => ({
  localBrowserMode: () => (globalThis as { __TENSORVIZ_LOCAL_BROWSER__?: boolean }).__TENSORVIZ_LOCAL_BROWSER__ === true,
}));

jest.mock('../../ide/bridge', () => ({
  bridgeActive: () => (global as { __bridge?: boolean }).__bridge === true,
  postToHost: jest.fn(),
}));

jest.mock('../../services/apiService', () => ({
  ApiService: { getPaperTemplateInventory: jest.fn().mockResolvedValue({ templates: [] }) },
}));

const CLASSES = [
  { name: 'MyBlock', file: 'models/blocks.py', line: 5, bases: ['nn.Module'], ctor_required: 0 },
  { name: 'Gate', file: 'models/gate.py', line: 12, bases: ['nn.Module'], ctor_required: 1 },
];

function answerRootWith(root: string | null): void {
  // The palette asks the shell which folder is open; the shell answers on the
  // window. Without a reply the request times out and the section stays absent.
  const { postToHost } = jest.requireMock('../../ide/bridge') as { postToHost: jest.Mock };
  postToHost.mockImplementation((message: { type: string }) => {
    if (message.type !== 'workspaceRoot') return;
    window.dispatchEvent(new MessageEvent('message', { data: { type: 'workspaceRoot', root } }));
  });
}

describe('the palette offers the workspace', () => {
  afterEach(() => {
    delete (window as { __TENSORVIZ_LOCAL_BROWSER__?: boolean }).__TENSORVIZ_LOCAL_BROWSER__;
  });

  it('asks for a declaration before offering source review controls', () => {
    (window as { __TENSORVIZ_LOCAL_BROWSER__?: boolean }).__TENSORVIZ_LOCAL_BROWSER__ = true;
    const { container } = render(<BuilderPaletteV2 onAddLayer={jest.fn()} />);
    expect(container).toHaveTextContent('Select a declared layer');
    expect(screen.queryByText('Blocks')).toBeNull();
    expect(screen.queryByLabelText('Search layers, blocks and templates')).toBeNull();
  });

  it('offers supported local additions through source proposals', async () => {
    (window as { __TENSORVIZ_LOCAL_BROWSER__?: boolean }).__TENSORVIZ_LOCAL_BROWSER__ = true;
    const onAddLayer = jest.fn();
    const propose = jest.fn(() => ({ promise: Promise.resolve({ status: 'refused', detail: 'Unsafe forward expression.' }) }));
    render(<LocalBrowserContext.Provider value={{ sourceEdits: { propose }, page: {} } as any}>
      <BuilderPaletteV2 onAddLayer={onAddLayer} sourceNode={{ id: 'fc', data: {
        layerType: { id: 'linear' }, parameters: { out_features: 8 }, sourceLocation: { file: 'net.py', line: 4 },
      } as any }} />
    </LocalBrowserContext.Provider>);
    fireEvent.click(screen.getByText('Review insertion'));
    expect(propose).toHaveBeenCalledWith({ kind: 'add_layer', afterNodeId: 'fc', nodeId: 'new_layer', layerType: 'dropout', params: { p: 0.5 } });
    expect(onAddLayer).not.toHaveBeenCalled();
    expect(await screen.findByText('Unsafe forward expression.')).toBeInTheDocument();
  });

  beforeEach(() => {
    setWorkspaceRoot(null);
    (global as { __bridge?: boolean }).__bridge = false;
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ root: '/w', classes: CLASSES, file_count: 2 }),
    }) as unknown as typeof fetch;
  });

  it('renders no Workspace section at all when there is no workspace', async () => {
    render(<BuilderPaletteV2 onAddLayer={jest.fn()} />);
    await screen.findByTestId('template-blocks-section');
    expect(screen.queryByTestId('workspace-classes-section')).toBeNull();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('renders no section when the endpoint is unavailable', async () => {
    (global as { __bridge?: boolean }).__bridge = true;
    answerRootWith('/w');
    global.fetch = jest.fn().mockResolvedValue({ ok: false, status: 503 }) as unknown as typeof fetch;
    render(<BuilderPaletteV2 onAddLayer={jest.fn()} />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    await waitFor(() => expect(screen.queryByTestId('workspace-classes-section')).toBeNull());
  });

  it('lists the classes the scan found, under Workspace', async () => {
    (global as { __bridge?: boolean }).__bridge = true;
    answerRootWith('/w');
    render(<BuilderPaletteV2 onAddLayer={jest.fn()} />);

    const section = await screen.findByTestId('workspace-classes-section');
    expect(section).toHaveTextContent('Workspace');
    expect(section).toHaveTextContent('MyBlock');
    expect(section).toHaveTextContent('models/blocks.py:5');
    // The scan is asked about the folder the host named, and nothing else.
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/ide/workspace/classes'),
      expect.objectContaining({ body: JSON.stringify({ root: '/w' }) }),
    );
  });

  it('inserts a class as a custom_code node that knows where to import from', async () => {
    (global as { __bridge?: boolean }).__bridge = true;
    answerRootWith('/w');
    const onAddLayer = jest.fn();
    render(<BuilderPaletteV2 onAddLayer={onAddLayer} />);

    const section = await screen.findByTestId('workspace-classes-section');
    fireEvent.click(await within(section).findByText('MyBlock'));

    expect(onAddLayer).toHaveBeenCalledWith('custom_code', {
      parameterOverrides: {
        _class: 'MyBlock',
        _import_path: 'models.blocks',
        _source_file: 'models/blocks.py',
        _source_line: 5,
      },
    });
  });

  it('searches the workspace with the same box as everything else', async () => {
    (global as { __bridge?: boolean }).__bridge = true;
    answerRootWith('/w');
    render(<BuilderPaletteV2 onAddLayer={jest.fn()} />);
    const section = await screen.findByTestId('workspace-classes-section');
    expect(section).toHaveTextContent('Gate');

    fireEvent.change(screen.getByLabelText('Search layers, blocks and templates'), {
      target: { value: 'myblo' },
    });
    expect(section).toHaveTextContent('MyBlock');
    expect(section).not.toHaveTextContent('models/gate.py');
  });

  it('does not offer the placeholder ops as layers you can insert', async () => {
    // `python` and `custom_code` are in layerTypes.ts so that a projected node
    // resolves to an OpSpec. Offering them here would let someone insert a
    // "Custom Code" node with no class behind it: strict validation refuses it
    // for having no shape, and codegen cannot emit it. The Workspace section is
    // where a real class comes from.
    render(<BuilderPaletteV2 onAddLayer={jest.fn()} />);
    await screen.findByTestId('template-blocks-section');

    expect(screen.queryByText('Custom Code')).toBeNull();
    expect(screen.queryByText('Python')).toBeNull();
    // The search box reaches the same list, so it must not be a way back in.
    fireEvent.change(screen.getByLabelText('Search layers, blocks and templates'), {
      target: { value: 'custom' },
    });
    expect(screen.queryByText('Custom Code')).toBeNull();
  });
});
