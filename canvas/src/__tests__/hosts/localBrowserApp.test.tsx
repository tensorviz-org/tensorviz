import reviewFixture from '../../test/fixtures/canvas/graphReview.json';
import captureDocument from '../../test/fixtures/canvas/capture-document.json';
import draftFixture from '../../test/fixtures/canvas/draftResNet.json';
import nestedFixture from '../../test/fixtures/canvas/nestedAssignmentCapture.json';
import deepDeclarationFixture from '../../test/fixtures/canvas/deepDeclarationReview.json';
/** Step-5 local browser page, mounted: the canvas the user gets, the strip
 *  that tells the truth about it, and the ended screen when the session dies.
 *
 * The canvas itself is mocked — App pulls in tldraw, which the CRA jest
 * transform cannot run (see qa_builder_ux.test.ts) — but the delivery seam is
 * NOT: projections travel through the real receiveProjection onto the real
 * 'builder-command' window bus the canvas consumes, and the real module's
 * currentProjection() is asserted, so what is pinned here is the page's own
 * contract end to end short of tldraw itself: broker traffic becomes canvas
 * deliveries and strip labels, and a dead session becomes a full-page answer.
 *
 * Step 6's navigation rides the same rig in both directions: a canvas
 * activation leaves as a `revealNode` naming the identity on screen, and an
 * editor's cursor focus arrives as the same `focus-node` builder-command the
 * framed bridge dispatches.
 */

import React from 'react';
import { execFileSync } from 'child_process';
import * as path from 'path';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';

import LocalBrowserApp from '../../ide/LocalBrowserApp';
import * as bridge from '../../ide/bridge';
import { currentProjection, reportNodeClick, reportProjectExpand } from '../../ide/projection';
import {
  LOCAL_BROWSER_PROTOCOL_VERSION,
  LocalBrowserTargetIdentity,
  ProjectionPayload,
} from '../../ide/protocol.generated';
import { LocalBrowserContextValue } from '../../ide/localBrowserContext';
import {
  LocalBrowserViewMessage,
  LocalBrowserViewStatus,
  LocalBrowserViewTransport,
} from '../../ide/localBrowserViewTransport';

beforeAll(() => { global.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} }; });

/** Capture the graph controls supplied to the canvas subtree. */
let mockCapturedLocalContext: LocalBrowserContextValue | null = null;

jest.mock('../../App', () => { throw new Error('Canvas must not load hosted or saved-graph routes'); });

jest.mock('../../components/ui2/BuilderPageV2', () => {
  const actualReact = jest.requireActual<typeof import('react')>('react');
  const contextModule = jest.requireActual<typeof import('../../ide/localBrowserContext')>('../../ide/localBrowserContext');
  // Render the actual navigation so tests exercise the portals and host requests.
  const navigation = jest.requireActual<typeof import('../../ide/LocalCanvasNavigation')>('../../ide/LocalCanvasNavigation').default;
  return {
    __esModule: true,
    default: () => {
      mockCapturedLocalContext = actualReact.useContext(contextModule.LocalBrowserContext);
      const verdict = mockCapturedLocalContext?.validation?.lastResult ?? null;
      return actualReact.createElement(
        'div',
        { 'data-testid': 'canvas-app' },
        'canvas',
        actualReact.createElement(navigation, { onFind: jest.fn(), onReveal: jest.fn(), hasSelection: false, nodeCount: 5, more: null }),
        // The toolbar toasts this; the stand-in renders it plainly so the specs can
        // still assert the sentence a user is shown.
        verdict ? actualReact.createElement('div', null, verdict.detail) : null,
      );
    },
  };
});

/** Capture what the real receiveProjection puts on the canvas's shared bus:
 *  every apply-projection builder-command's payload, in order. */
function captureCanvasDeliveries(): { deliveries: ProjectionPayload[]; stop(): void } {
  const deliveries: ProjectionPayload[] = [];
  const listener = (event: Event) => {
    const detail = (event as CustomEvent).detail as
      | { command?: string; payload?: { projection?: ProjectionPayload } }
      | undefined;
    if (detail?.command === 'apply-projection' && detail.payload?.projection) {
      deliveries.push(detail.payload.projection);
    }
  };
  window.addEventListener('builder-command', listener);
  return { deliveries, stop: () => window.removeEventListener('builder-command', listener) };
}

interface TransportRig {
  transport: LocalBrowserViewTransport;
  emit(message: LocalBrowserViewMessage): void;
  setStatus(next: LocalBrowserViewStatus): void;
}

/** A scripted transport: whatever the test emits is what the broker said. */
function makeTransport(): TransportRig {
  const messageListeners = new Set<(message: LocalBrowserViewMessage) => void>();
  const statusListeners = new Set<(status: LocalBrowserViewStatus) => void>();
  let status: LocalBrowserViewStatus = { phase: 'connecting', hostLink: 'unknown', error: null };
  const rig: TransportRig = {
    transport: {
      get status() {
        return status;
      },
      session: null,
      protocolViolations: 0,
      connect: jest.fn(async () => undefined),
      send: jest.fn(() => 'echo'),
      onMessage(listener) {
        messageListeners.add(listener);
        return () => messageListeners.delete(listener);
      },
      onStatusChange(listener) {
        statusListeners.add(listener);
        return () => statusListeners.delete(listener);
      },
      dispose: jest.fn(),
    },
    emit(message) {
      for (const listener of [...messageListeners]) listener(message);
    },
    setStatus(next) {
      status = next;
      for (const listener of [...statusListeners]) listener(status);
    },
  };
  return rig;
}

function netTarget(): LocalBrowserTargetIdentity {
  return {
    workspaceFolder: '/ws',
    file: 'models/net.py',
    cls: 'Net',
    documentVersion: 11,
    sourceDigest: 'digest-1',
    backendRevision: 1,
  };
}

function netProjection(extra: Partial<ProjectionPayload> = {}): ProjectionPayload {
  return {
    graphId: 'g-Net-1',
    revision: 1,
    cls: 'Net',
    file: 'models/net.py',
    sourceDigest: 'digest-1',
    traced: false,
    greyed: false,
    document: { view: {}, semantic: { nodes: [] } },
    viewRestored: false,
    spanmap: {},
    ...extra,
  };
}

let sequence = 0;
function hostMessage(type: string, rest: Record<string, unknown>): LocalBrowserViewMessage {
  sequence += 1;
  return {
    type,
    protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
    workspaceSessionId: 'ws-test',
    messageId: `m-${sequence}`,
    hostSequence: sequence,
    ...rest,
  } as LocalBrowserViewMessage;
}

function readyValidationPage() {
  const rig = makeTransport();
  render(<LocalBrowserApp createTransport={() => rig.transport} />);
  act(() => {
    rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
    rig.emit(hostMessage('activeTarget', { target: netTarget() }));
    rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection(), traced: false, greyed: false }));
    rig.emit(hostMessage('syncState', { target: netTarget(), state: 'source-only' }));
  });
  return rig;
}
function readyCapturedPage() {
  const rig = readyValidationPage();
  act(() => {
    rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false }));
    rig.emit(hostMessage('syncState', { target: netTarget(), state: 'runtime-traced' }));
  });
  return rig;
}
function validationRequest(rig: TransportRig): { requestId: string; target: LocalBrowserTargetIdentity } {
  return (rig.transport.send as jest.Mock).mock.calls.find(([message]) => message.type === 'validateRequest')![0];
}

describe('LocalBrowserApp', () => {
  it.each([true, false])('shows input evidence only for the reviewed saved profile (matching=%s)', matching => {
    const rig = readyValidationPage();
    const execution = { id: 'reviewed', phase: 'review', kind: 'capture', device: 'cpu', configuredDevice: 'cpu', profileName: 'actual',
      detail: 'Review', interpreter: '/env/python', inputs: '{}', mode: 'train', gradients: true, stages: [{ label: 'Capture', calls: 1 }] };
    act(() => rig.emit(hostMessage('graphOpening', { state: { id: 'opening', target: netTarget(), phase: 'awaiting_execution', hasCapture: false,
      detail: 'Review', execution, draft: { profileName: matching ? 'actual' : 'previous', profile: {} },
      evidence: [{ field: 'inputs.x', kind: 'source', detail: 'Known input constraint', file: 'model.py', line: 12 }] } })));
    fireEvent.click(screen.getByRole('button', { name: 'Execution details' }));
    expect(screen.getByText('profiles.actual')).toBeVisible();
    expect(!!screen.queryByText('How inputs were inferred')).toBe(matching);
    (rig.transport.send as jest.Mock).mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Edit mode in tensorviz.toml' }));
    expect(rig.transport.send).toHaveBeenCalledTimes(1);
    expect(rig.transport.send).toHaveBeenLastCalledWith(expect.objectContaining({ type: 'graphSetupRequest', openingId: 'opening', target: netTarget(),
      action: { kind: 'edit_execution', reviewId: 'reviewed', field: ['mode'] } }));
  });
  it('Run again requests a fresh review and waits for its new approval', () => {
    const rig = readyCapturedPage();
    const execution = { id: 'completed-review', phase: 'succeeded', kind: 'capture', device: 'cpu', configuredDevice: 'cpu',
      detail: 'Finished.', interpreter: '/env/bin/python', inputs: '{}', mode: 'eval', gradients: false,
      stages: [{ label: 'Capture', calls: 1 }] };
    const opening = { id: 'completed-opening', target: netTarget(), phase: 'ready', hasCapture: true, detail: 'Finished.', execution };
    act(() => rig.emit(hostMessage('graphOpening', { state: opening })));
    fireEvent.click(screen.getByRole('button', { name: 'Execution details' }));
    (rig.transport.send as jest.Mock).mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Run again' }));
    expect(screen.getByRole('button', { name: 'Run again' })).toBeDisabled();
    const messages = (rig.transport.send as jest.Mock).mock.calls.map(([message]) => message);
    expect(messages.filter(message => message.type === 'validateRequest')).toHaveLength(1);
    expect(messages.find(message => message.type === 'validateRequest')).not.toHaveProperty('measureLatency');
    expect(messages.some(message => message.action?.action === 'run')).toBe(false);
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'awaiting_execution',
      execution: { ...execution, id: 'fresh-review', phase: 'review' } } })));
    expect(screen.queryByRole('button', { name: 'Run again' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Run', exact: true }));
    expect(rig.transport.send).toHaveBeenLastCalledWith(expect.objectContaining({ type: 'graphSetupRequest',
      action: { kind: 'execution', reviewId: 'fresh-review', action: 'run', device: 'cpu' } }));
  });

  it('after a reused result, Run fresh asks the host for a fresh run that still waits for its own approval', () => {
    const rig = readyCapturedPage();
    const execution = { id: 'served-review', phase: 'succeeded', kind: 'capture', device: 'cpu', configuredDevice: 'cpu',
      detail: 'Reused identical result from 14:02 (no Python ran). Validation passed.', interpreter: '/env/bin/python', inputs: '{}',
      mode: 'eval', gradients: false, stages: [{ label: 'Capture', calls: 1 }], reused: { capturedAt: 1761400000000 } };
    const opening = { id: 'served-opening', target: netTarget(), phase: 'ready', hasCapture: true, detail: execution.detail, execution };
    act(() => rig.emit(hostMessage('graphOpening', { state: opening })));
    fireEvent.click(screen.getByRole('button', { name: /^Execution details\. Reused identical result/ }));
    (rig.transport.send as jest.Mock).mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Run fresh' }));
    const messages = (rig.transport.send as jest.Mock).mock.calls.map(([message]) => message);
    const requests = messages.filter(message => message.type === 'validateRequest');
    expect(requests).toHaveLength(1);
    expect(requests[0].fresh).toBe(true);
    expect(requests[0]).not.toHaveProperty('measureLatency');
    expect(messages.some(message => message.action?.action === 'run')).toBe(false);
  });

  it('opens execution details only on request and keeps a dismissed preparation closed for the capture review', () => {
    const rig = readyValidationPage();
    const execution = { id: 'prepare', phase: 'review', kind: 'prepare', device: 'cpu', configuredDevice: 'cpu',
      detail: 'Not started', interpreter: '/env/bin/python', inputs: '{}', mode: 'eval', gradients: false,
      stages: [{ label: 'Model construction to infer inputs', calls: 0 }] };
    const opening = { id: 'opening', target: netTarget(), phase: 'awaiting_execution', hasCapture: false, detail: 'Review', execution };
    act(() => rig.emit(hostMessage('graphOpening', { state: opening })));
    expect(screen.getByRole('button', { name: 'Execution details' })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('button', { name: 'Prepare on CPU' })).toBeNull();
    (rig.transport.send as jest.Mock).mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Run model' }));
    expect(screen.getByRole('button', { name: 'Prepare on CPU' })).toBeVisible();
    expect(rig.transport.send).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Prepare on CPU' }));
    expect(rig.transport.send).toHaveBeenLastCalledWith(expect.objectContaining({
      type: 'graphSetupRequest', action: { kind: 'execution', reviewId: 'prepare', action: 'run', device: 'cpu' },
    }));
    expect(screen.getByRole('button', { name: 'Execution details' })).toHaveAttribute('aria-expanded', 'false');
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'capturing', execution: { ...execution, phase: 'running' } } })));
    expect(screen.queryByRole('button', { name: 'Collapse execution details' })).toBeNull();
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'resolving', execution: undefined } })));
    const capture = { ...opening, execution: { ...execution, id: 'capture', kind: 'capture', stages: [{ label: 'Capture', calls: 1 }] } };
    act(() => rig.emit(hostMessage('graphOpening', { state: capture })));
    act(() => rig.setStatus({ phase: 'reconnecting', hostLink: 'unknown', error: null }));
    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('graphOpening', { state: capture }));
    });
    expect(screen.getByRole('button', { name: 'Execution details' })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('button', { name: 'Run', exact: true })).toBeNull();
    (rig.transport.send as jest.Mock).mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Run model' }));
    expect(screen.getByRole('button', { name: 'Run', exact: true })).toBeVisible();
    expect(rig.transport.send).not.toHaveBeenCalled();
  });

  it('reviews one capture in the Run model panel and keeps it closed across host updates after dismissal', () => {
    const rig = readyCapturedPage();
    fireEvent.click(screen.getByRole('button', { name: 'Run model' }));
    expect(validationRequest(rig)).not.toHaveProperty('measureLatency');
    const execution = { id: 'review', phase: 'review', kind: 'capture', device: 'cpu', configuredDevice: 'cpu',
      detail: 'Not started', interpreter: '/env/bin/python', inputs: '{}', mode: 'eval', gradients: false,
      stages: [{ label: 'Capture shapes and operations', calls: 1 }] };
    const opening = { id: 'review-opening', target: netTarget(), phase: 'awaiting_execution', hasCapture: true, detail: 'Review', execution };
    act(() => rig.emit(hostMessage('graphOpening', { state: opening })));
    expect(screen.getByRole('heading', { name: 'Run model' })).toBeVisible();
    expect(screen.getByText('1 forward call')).toBeVisible();
    expect(screen.queryByRole('checkbox', { name: 'Measure latency' })).toBeNull();
    expect((rig.transport.send as jest.Mock).mock.calls.some(([message]) => message.action?.action === 'run')).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Run', exact: true }));
    expect(rig.transport.send).toHaveBeenLastCalledWith(expect.objectContaining({ type: 'graphSetupRequest',
      action: { kind: 'execution', reviewId: 'review', action: 'run', device: 'cpu' } }));
    expect(screen.getByRole('button', { name: 'Execution details' })).toHaveAttribute('aria-expanded', 'false');
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'capturing', execution: { ...execution, phase: 'running' } } })));
    expect(screen.getByRole('button', { name: 'Running…' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeEnabled();
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, execution: { ...execution, id: 'review-refreshed' } } })));
    expect(screen.queryByRole('button', { name: 'Run', exact: true })).toBeNull();
  });

  it('a run that starts at once from Run model or Run again leaves the details closed', () => {
    const rig = readyValidationPage();
    const execution = { id: 'stopped', phase: 'cancelled', kind: 'capture', device: 'cpu', configuredDevice: 'cpu',
      detail: 'Stopped.', interpreter: '/env/bin/python', inputs: '{}', mode: 'eval', gradients: false, stages: [{ label: 'Capture', calls: 1 }] };
    const opening = { id: 'stopped-opening', target: netTarget(), phase: 'cancelled', hasCapture: false, detail: 'Stopped.', execution };
    act(() => rig.emit(hostMessage('graphOpening', { state: opening })));
    const retries = () => (rig.transport.send as jest.Mock).mock.calls.filter(([message]) => message.type === 'graphSetupRequest' && message.action.kind === 'retry');
    fireEvent.click(screen.getByRole('button', { name: 'Run model' }));
    expect(retries()).toHaveLength(1);
    expect(screen.getByRole('button', { name: 'Execution details' })).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(screen.getByRole('button', { name: 'Execution details' }));
    fireEvent.click(screen.getByRole('button', { name: 'Run again' }));
    expect(retries()).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Execution details' })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByRole('button', { name: 'Execution details' })).toHaveFocus();
  });

  it('the connected empty native Canvas offers both entry actions without mounting or executing a graph', () => {
    const location = window.location.href;
    const originalMedia = window.matchMedia;
    window.matchMedia = jest.fn(() => ({ matches: true, addEventListener() {}, removeEventListener() {} })) as any;
    window.history.replaceState(null, '', '?view=canvas');
    const post = jest.spyOn(bridge, 'postToHost').mockImplementation(() => {});
    try {
      const rig = makeTransport();
      render(<LocalBrowserApp createTransport={() => rig.transport} />);
      expect(screen.getByRole('heading', { name: 'Connecting to Canvas' })).toBeVisible();
      expect(screen.queryByRole('button', { name: 'Open PR' })).not.toBeInTheDocument();
      act(() => rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null }));
      expect(screen.getByRole('heading', { name: 'Explore your model' })).toBeVisible();
      expect(screen.queryByTestId('canvas-app')).not.toBeInTheDocument();
      expect(rig.transport.send).not.toHaveBeenCalled();
      fireEvent.click(screen.getByRole('button', { name: 'Open as Graph' }));
      expect(post).toHaveBeenCalledWith({ type: 'openAsGraph' });
      fireEvent.click(screen.getByRole('button', { name: 'Open PR' }));
      expect(rig.transport.send).toHaveBeenCalledWith({ type: 'graphReviewRequest', action: 'openPR' });
      act(() => rig.setStatus({ phase: 'connected', hostLink: 'disconnected', error: null }));
      expect(screen.getByRole('heading', { name: 'Reconnect to Canvas' })).toBeVisible();
      expect(screen.queryByRole('button', { name: 'Open as Graph' })).not.toBeInTheDocument();
    } finally {
      cleanup(); post.mockRestore(); window.matchMedia = originalMedia;
      window.history.replaceState(null, '', location);
    }
  });

  it('an empty model review keeps coverage and explicit source/return actions rather than claiming a graph', () => {
    const rig = makeTransport();
    const emptyProjection = { ...reviewFixture.before, spanmap: {}, document: { ...reviewFixture.before.document,
      semantic: { ...reviewFixture.before.document.semantic, inputs: [], nodes: [], edges: [], modules: [], operations: [], metadata: {} },
      view: { ...reviewFixture.before.document.view, node_positions: {}, collapsed_modules: [], selection: [], layout: {} } } };
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('graphReviewState', { state: { ...reviewFixture, models: [], before: emptyProjection, after: emptyProjection,
        notes: ['Dynamic source could not be resolved.'],
        id: 'empty-review', stale: false, changedFiles: ['model.py'], beforeLabel: 'HEAD', afterLabel: 'PR' } }));
    });
    expect(screen.getByRole('heading', { name: 'No model identified' })).toBeVisible();
    expect(screen.queryByTestId('canvas-app')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Coverage limits (1)' })).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'View code changes' }));
    expect(rig.transport.send).toHaveBeenCalledWith({ type: 'graphReviewRequest', action: 'reveal', reviewId: 'empty-review', side: 'after' });
    fireEvent.click(screen.getByRole('button', { name: 'Return to Canvas' }));
    expect(rig.transport.send).toHaveBeenCalledWith({ type: 'graphReviewRequest', action: 'close', reviewId: 'empty-review' });
  });

  it('standalone failure help copies the exact opening context without retrying or contacting an agent', async () => {
    const original = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
    const writeText = jest.fn(async (_text: string) => undefined);
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    try {
      const rig = makeTransport();
      render(<LocalBrowserApp hostKind="standalone" createTransport={() => rig.transport} />);
      act(() => {
        rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
        rig.emit(hostMessage('activeTarget', { target: netTarget() }));
        rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection(), traced: false, greyed: false }));
        rig.emit(hostMessage('graphOpening', { state: { id: 'failed-exact-opening', target: netTarget(), phase: 'failed', hasCapture: false,
          failure: { kind: 'execution' }, detail: 'Forward raised an exception.' } }));
      });
      const calls = (rig.transport.send as jest.Mock).mock.calls.length;
      await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Copy MCP investigation request' })));
      expect(writeText).toHaveBeenCalledTimes(1);
      const request = writeText.mock.calls[0][0];
      expect(request).toContain('failed-exact-opening');
      expect(request).toContain(netTarget().sourceDigest);
      expect(request).toContain(netTarget().file);
      expect(request).toContain('MCP directly');
      expect((rig.transport.send as jest.Mock).mock.calls.length).toBe(calls);
      expect(screen.getByText('MCP investigation request copied. Paste it into your coding agent.')).toBeVisible();
    } finally {
      if (original) Object.defineProperty(navigator, 'clipboard', original);
      else delete (navigator as any).clipboard;
    }
  });

  it('standalone setup help shares the pending draft with MCP and the copied request without submitting it', async () => {
    const original = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
    const writeText = jest.fn(async (_text: string) => undefined);
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    try {
      const rig = makeTransport();
      render(<LocalBrowserApp hostKind="standalone" createTransport={() => rig.transport} />);
      act(() => {
        rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
        rig.emit(hostMessage('activeTarget', { target: netTarget() }));
        rig.emit(hostMessage('graphOpening', { state: { id: 'pending-setup', target: netTarget(), phase: 'needs_setup', hasCapture: false,
          detail: 'Complete inputs.', discovery: { model: 'model:Net', sourceDigest: netTarget().sourceDigest,
            init: [], inputs: [{ name: 'x', kind: 'keyword', required: true }], examples: [], factories: [], configs: [],
            parents: [], constraints: [], truncated: false } } }));
      });
      fireEvent.click(screen.getByRole('button', { name: 'Set up manually' }));
      fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '3, 12' } });
      fireEvent.change(screen.getByLabelText('x dtype'), { target: { value: 'float64' } });
      fireEvent.click(screen.getByRole('button', { name: 'Setup methods' }));
      await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Copy setup request' })));
      const request = writeText.mock.calls[0][0];
      expect(screen.getByText('Setup request copied. Paste it into your coding agent.')).toBeVisible();
      expect(screen.queryByText(`Copied: ${request}`)).not.toBeInTheDocument();
      expect(request).toContain('pending-setup');
      expect(request).toContain('float64');
      expect(request).toContain('configure_graph_setup');
      const actions = (rig.transport.send as jest.Mock).mock.calls.map(([message]) => message).filter(message => message.type === 'graphSetupRequest');
      expect(actions).toHaveLength(1);
      expect(actions[0].action).toEqual({ kind: 'agent_help', draft: expect.objectContaining({ profile: expect.objectContaining({
        inputs: { args: [], kwargs: { x: expect.objectContaining({ shape: [3, 12], dtype: 'float64' }) } },
      }) }) });
      fireEvent.click(screen.getByRole('button', { name: 'Set up manually' }));
      expect(screen.getByLabelText('x shape')).toHaveValue('3, 12');
    } finally {
      if (original) Object.defineProperty(navigator, 'clipboard', original);
      else delete (navigator as any).clipboard;
    }
  });
  it('acknowledges only the rendered review in its exact view and hides native source actions in standalone mode', () => {
    const rig = makeTransport();
    Object.defineProperty(rig.transport, 'session', { value: { viewId: 'exact-browser' } });
    render(<LocalBrowserApp hostKind="standalone" createTransport={() => rig.transport} />);
    const state = { ...reviewFixture, id: 'standalone-review', workspaceRoot: '/ws', beforeLabel: 'HEAD', afterLabel: 'Working',
      changedFiles: ['model.py'], stale: false };
    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('graphReviewState', { state }));
    });
    expect(screen.queryByRole('button', { name: 'Open native code diff' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy reviewed source reference' })).toBeVisible();
    const receipts = () => (rig.transport.send as jest.Mock).mock.calls.filter(([message]) => message.type === 'graphReviewApplied');
    expect(receipts()).toHaveLength(0);
    act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: { ...currentProjection() } })));
    expect(receipts()).toHaveLength(0);
    act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: currentProjection() })));
    expect(receipts()[0][0]).toEqual({ type: 'graphReviewApplied', viewId: 'exact-browser', reviewId: state.id, mode: 'diff' });
    act(() => rig.emit(hostMessage('graphReviewState', { state: { ...state, stale: true } })));
    act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: currentProjection() })));
    expect(receipts()).toHaveLength(1);
    act(() => rig.emit(hostMessage('graphReviewState', { state: null })));
    expect(screen.getByText('No model is open.')).toBeVisible();
    expect(document.querySelector('.graph-opening-canvas')).toBeNull();
  });

  it('shows changes on the full capture without graph choices or explanatory banners', () => {
    const rig = readyValidationPage();
    const live = netProjection({ traced: true, document: captureDocument });
    act(() => rig.emit(hostMessage('projection', { target: netTarget(), projection: live, traced: true, greyed: false })));
    const state = { ...reviewFixture, id: 'retained-review', beforeLabel: 'HEAD', afterLabel: 'Working', changedFiles: ['models/net.py'], stale: false,
      liveContext: { workspaceFolder: '/ws', file: live.file, cls: live.cls } };
    act(() => rig.emit(hostMessage('graphReviewState', { state })));
    expect(screen.queryByRole('button', { name: 'Captured graph' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Source graph' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Before$/ })).not.toBeInTheDocument();
    expect((currentProjection() as any).capturedContext).toBe(true);
    expect((currentProjection() as any).reviewGraph.nodes.map((node: any) => node.id)).toEqual(captureDocument.semantic.nodes.map(node => node.node_id));
    expect(screen.queryByText(/Last captured architecture|Retained capture|not been re-executed/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next graph change' })).toBeEnabled();
    act(() => rig.emit(hostMessage('graphReviewState', { state: null })));
    expect((currentProjection() as any).reviewGraph).toBeUndefined();
  });

  it.each([false, true])('a usable git-review capture remains frozen when a newer run arrives (greyed: %s)', greyed => {
    const rig = readyValidationPage();
    const live = netProjection({ traced: true, document: captureDocument, greyed });
    act(() => rig.emit(hostMessage('projection', { target: netTarget(), projection: live, traced: true, greyed })));
    const state = { ...reviewFixture, id: 'frozen-review', beforeLabel: 'HEAD', afterLabel: 'Working', changedFiles: ['models/net.py'], stale: false,
      liveContext: { workspaceFolder: '/ws', file: live.file, cls: live.cls } };
    act(() => rig.emit(hostMessage('graphReviewState', { state })));
    const frozen = currentProjection();
    expect((frozen as any).capturedContext).toBe(true);
    const newer = netProjection({ traced: true, document: captureDocument, revision: 900 });
    act(() => rig.emit(hostMessage('projection', { target: netTarget(), projection: newer, traced: true, greyed: false })));
    expect(mockCapturedLocalContext?.page.projection?.revision).toBe(900);
    expect(currentProjection()).toBe(frozen);
  });

  it.each([false, true])('focuses an ordinary Git change once its exact projection renders (captured: %s)', captured => {
    const commands: any[] = [];
    const listener = (event: Event) => commands.push((event as CustomEvent).detail);
    window.addEventListener('builder-command', listener);
    try {
      const rig = readyValidationPage();
      const document = JSON.parse(JSON.stringify(captureDocument));
      const call = document.semantic.nodes.find((node: any) => node.metadata.captureRef.kind === 'invocation');
      document.semantic.metadata.sourceCorrespondence = [{ nodeId: 'head', basis: 'registered_path',
        captureId: call.metadata.captureRef.captureId, invocationIds: [call.metadata.captureRef.semanticId] }];
      const live = netProjection({ traced: true, document, sourceDigest: reviewFixture.after.sourceDigest,
        dependencies: [{ file: 'model.py', digest: reviewFixture.after.sourceDigest }] });
      if (captured) act(() => rig.emit(hostMessage('projection', { target: netTarget(), projection: live, traced: true, greyed: false })));
      const state = { ...reviewFixture, after: { ...reviewFixture.after, dependencies: live.dependencies }, id: 'initial-git-focus',
        beforeLabel: 'HEAD', afterLabel: 'Working', changedFiles: ['model.py'], stale: false,
        liveContext: { workspaceFolder: '/ws', file: live.file, cls: live.cls } };
      act(() => rig.emit(hostMessage('graphReviewState', { state })));
      const applied = currentProjection()!;
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: { ...applied } })));
      expect(commands.some(command => command.command === 'focus-selection')).toBe(false);
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: applied })));
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: applied })));
      const focus = commands.filter(command => command.command === 'focus-selection');
      expect(focus).toHaveLength(1);
      expect(focus[0].payload.target).toEqual({ file: applied.file, cls: applied.cls,
        sourceDigest: applied.sourceDigest, backendRevision: applied.revision });
      if (captured) {
        expect(focus[0].payload.nodeIds).toEqual([call.node_id]);
        expect(screen.getByText('4 / 4')).toBeVisible();
        expect(globalThis.document.querySelector('.graph-review-current')!.textContent).not.toContain('Not located in this capture');
      }
      act(() => rig.emit(hostMessage('graphReviewState', { state: { ...state, id: 'stale-review', stale: true } })));
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: currentProjection() })));
      expect(commands.filter(command => command.command === 'focus-selection')).toHaveLength(1);
    } finally { window.removeEventListener('builder-command', listener); }
  });

  it('selecting a highlighted captured call shows its source change in the navigator', () => {
    const rig = readyValidationPage();
    const document = JSON.parse(JSON.stringify(captureDocument));
    const call = document.semantic.nodes.find((node: any) => node.metadata.captureRef.kind === 'invocation');
    document.semantic.metadata.sourceCorrespondence = [{ nodeId: 'head', basis: 'registered_path',
      captureId: call.metadata.captureRef.captureId, invocationIds: [call.metadata.captureRef.semanticId] }];
    const live = netProjection({ traced: true, document, sourceDigest: reviewFixture.after.sourceDigest,
      dependencies: [{ file: 'model.py', digest: reviewFixture.after.sourceDigest }] });
    act(() => rig.emit(hostMessage('projection', { target: netTarget(), projection: live, traced: true, greyed: false })));
    const state = { ...reviewFixture, after: { ...reviewFixture.after, dependencies: live.dependencies }, id: 'mapped-review',
      beforeLabel: 'HEAD', afterLabel: 'Working', changedFiles: ['model.py'], stale: false,
      liveContext: { workspaceFolder: '/ws', file: live.file, cls: live.cls } };
    act(() => rig.emit(hostMessage('graphReviewState', { state })));
    const caption = globalThis.document.querySelector('.graph-review-current')!;
    expect(screen.getByText('1 / 4')).toBeVisible();
    expect(caption.textContent).toContain('Not located in this capture');
    act(() => reportNodeClick(call.node_id));
    expect(screen.getByText('4 / 4')).toBeVisible();
    expect(caption.textContent).not.toContain('Not located in this capture');
    expect(rig.transport.send).toHaveBeenLastCalledWith({ type: 'graphReviewRequest', action: 'reveal', reviewId: 'mapped-review', side: 'after', nodeId: 'head' });
  });

  /** A Git review over a capture that calls the changed `head` module twice. */
  const twoCallReview = (rig: TransportRig) => {
    const document = JSON.parse(JSON.stringify(captureDocument));
    const root = document.semantic.nodes.find((node: any) => node.metadata.captureRef.kind === 'invocation');
    const calls = [0, 1].map(i => ({ ...JSON.parse(JSON.stringify(root)), node_id: `invocation:head:${i}`, metadata: {
      ...root.metadata, captureLabel: `head · call ${i + 1}/2`, captureRef: { ...root.metadata.captureRef, semanticId: `call:head:${i}` } } }));
    document.semantic.nodes.push(...calls);
    document.semantic.metadata.sourceCorrespondence = [{ nodeId: 'head', basis: 'registered_path',
      captureId: root.metadata.captureRef.captureId, invocationIds: calls.map(call => call.metadata.captureRef.semanticId) }];
    const live = netProjection({ traced: true, document, sourceDigest: reviewFixture.after.sourceDigest,
      dependencies: [{ file: 'model.py', digest: reviewFixture.after.sourceDigest }] });
    act(() => rig.emit(hostMessage('projection', { target: netTarget(), projection: live, traced: true, greyed: false })));
    const state = { ...reviewFixture, after: { ...reviewFixture.after, dependencies: live.dependencies }, id: 'two-call-review',
      beforeLabel: 'HEAD', afterLabel: 'Working', changedFiles: ['model.py'], stale: false,
      liveContext: { workspaceFolder: '/ws', file: live.file, cls: live.cls } };
    act(() => rig.emit(hostMessage('graphReviewState', { state })));
    return calls.map(call => call.node_id);
  };

  it('moving the editor cursor onto a change used by two captured calls focuses a call in step with the navigator', () => {
    jest.useFakeTimers();
    const focused: string[] = [];
    const listener = (event: Event) => {
      const detail = (event as CustomEvent).detail;
      if (detail?.command === 'focus-node') focused.push(detail.payload.nodeId);
    };
    window.addEventListener('builder-command', listener);
    try {
      const rig = readyValidationPage();
      const calls = twoCallReview(rig);
      const cursorOnHead = () => {
        act(() => rig.emit(hostMessage('graphReviewFocus', { reviewId: 'two-call-review', side: 'after', nodeIds: ['head'] })));
        act(() => jest.advanceTimersByTime(100));
      };
      expect(screen.getByText('1 / 5')).toBeVisible();
      cursorOnHead();
      expect(screen.getByText('4 / 5')).toBeVisible();
      expect(screen.getByText('Linear · call 1/2')).toBeVisible();
      expect(focused).toEqual([calls[0]]);
      // Next reveals the second call's source, which moves the editor cursor onto
      // the same line. That echo must keep the second call, or Next could never leave it.
      fireEvent.click(screen.getByRole('button', { name: 'Next graph change' }));
      expect(rig.transport.send).toHaveBeenLastCalledWith({ type: 'graphReviewRequest', action: 'reveal', reviewId: 'two-call-review', side: 'after', nodeId: 'head' });
      cursorOnHead();
      expect(screen.getByText('5 / 5')).toBeVisible();
      expect(screen.getByText('Linear · call 2/2')).toBeVisible();
      expect(focused).toEqual([calls[0], calls[1]]);
    } finally { window.removeEventListener('builder-command', listener); jest.useRealTimers(); }
  });

  it('selecting the second captured call of a change selects that call, so Next moves on', () => {
    const rig = readyValidationPage();
    const calls = twoCallReview(rig);
    act(() => reportNodeClick(calls[1]));
    expect(screen.getByText('5 / 5')).toBeVisible();
    expect(screen.getByText('Linear · call 2/2')).toBeVisible();
    expect(rig.transport.send).toHaveBeenLastCalledWith({ type: 'graphReviewRequest', action: 'reveal', reviewId: 'two-call-review', side: 'after', nodeId: 'head' });
    fireEvent.click(screen.getByRole('button', { name: 'Next graph change' }));
    expect(screen.getByText('1 / 5')).toBeVisible();
    act(() => reportNodeClick(calls[0]));
    expect(screen.getByText('4 / 5')).toBeVisible();
    expect(screen.getByText('Linear · call 1/2')).toBeVisible();
  });

  it('moving the editor cursor onto a block declaration lowered once per call focuses the call the navigator shows', () => {
    // Real capture: `first` and `last` share one Sequential line whose ReLU became GELU.
    // The host matches by line, so both lowered copies and their unchanged neighbours
    // arrive with the change's calls; only those calls are navigable.
    jest.useFakeTimers();
    const data = deepDeclarationFixture.expanded;
    const labels = new Map((data.capture.document.semantic.nodes as { node_id: string; metadata: { captureLabel?: string } }[])
      .map(node => [node.node_id, node.metadata.captureLabel]));
    const focused: (string | undefined)[] = [];
    const listener = (event: Event) => {
      const detail = (event as CustomEvent).detail;
      if (detail?.command === 'focus-node') focused.push(labels.get(detail.payload.nodeId));
    };
    window.addEventListener('builder-command', listener);
    try {
      const rig = makeTransport();
      render(<LocalBrowserApp createTransport={() => rig.transport} />);
      const target = { ...netTarget(), file: 'model.py', cls: 'Model' };
      act(() => {
        rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
        rig.emit(hostMessage('activeTarget', { target }));
        rig.emit(hostMessage('projection', { target, projection: data.capture, traced: true, greyed: false }));
        rig.emit(hostMessage('graphReviewState', { state: { id: 'expanded-review', beforeLabel: 'HEAD', afterLabel: 'Working',
          changedFiles: ['model.py'], stale: false, before: data.before, after: data.after,
          liveContext: { workspaceFolder: '/ws', file: 'model.py', cls: 'Model' } } }));
      });
      const cursorOnSequential = () => {
        const nodeIds = Object.entries(data.after.spanmap).filter(([, span]) => span.start[0] === 6).map(([id]) => id);
        act(() => rig.emit(hostMessage('graphReviewFocus', { reviewId: 'expanded-review', side: 'after', nodeIds })));
        act(() => jest.advanceTimersByTime(100));
      };
      expect(screen.getByText('1 / 2')).toBeVisible();
      cursorOnSequential();
      expect(screen.getByText('ReLU → GELU · call 1/2')).toBeVisible();
      expect(focused).toEqual(['first.proj.1']);
      fireEvent.click(screen.getByRole('button', { name: 'Next graph change' }));
      cursorOnSequential();
      expect(screen.getByText('2 / 2')).toBeVisible();
      expect(screen.getByText('ReLU → GELU · call 2/2')).toBeVisible();
      expect(focused).toEqual(['first.proj.1', 'last.proj.1']);
    } finally { window.removeEventListener('builder-command', listener); jest.useRealTimers(); }
  });

  it('reviews immutable revisions in Canvas, links removed source, and restores the live projection on close', () => {
    const rig = readyValidationPage();
    const liveProjection = netProjection({ traced: true, document: { view: {}, semantic: { nodes: [], metadata: {} } } });
    act(() => rig.emit(hostMessage('projection', { target: netTarget(), projection: liveProjection, traced: true, greyed: false })));
    // Older captures can still carry timing; the Canvas no longer shows it.
    expect(screen.queryByRole('button', { name: 'Latency details' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run model' })).toBeVisible();
    const state = { ...reviewFixture, id: 'git-review', beforeLabel: 'base', afterLabel: 'HEAD', changedFiles: ['model.py'], stale: false };
    act(() => rig.emit(hostMessage('graphReviewState', { state })));
    expect(screen.getByRole('region', { name: 'Graph diff review' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Run model' })).not.toBeInTheDocument();
    expect((currentProjection() as any).reviewGraph.nodes.some((node: any) => node.id === 'review:after:norm')).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: /^Before$/ }));
    expect((currentProjection() as any).reviewGraph.nodes.some((node: any) => node.id === 'review:after:norm')).toBe(false);
    act(() => reportNodeClick('review:after:proj'));
    expect(rig.transport.send).toHaveBeenLastCalledWith({ type: 'graphReviewRequest', action: 'reveal', reviewId: 'git-review', side: 'before', nodeId: 'proj' });
    fireEvent.click(screen.getByRole('button', { name: /^After$/ }));
    act(() => reportNodeClick('review:after:norm'));
    expect(rig.transport.send).toHaveBeenLastCalledWith({ type: 'graphReviewRequest', action: 'reveal', reviewId: 'git-review', side: 'after', nodeId: 'norm' });
    act(() => rig.emit(hostMessage('graphReviewState', { state: { ...state, stale: true } })));
    expect(screen.getByText(/Source changed since this review/)).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Close graph diff' }));
    expect(rig.transport.send).toHaveBeenLastCalledWith({ type: 'graphReviewRequest', action: 'close', reviewId: 'git-review' });
    act(() => rig.emit(hostMessage('graphReviewState', { state: null })));
    expect(screen.queryByRole('region', { name: 'Graph diff review' })).not.toBeInTheDocument();
    expect((currentProjection() as any).reviewGraph).toBeUndefined();
    expect(screen.getByRole('button', { name: 'Run model' })).toBeVisible();
  });

  it('keeps revisions and their actions in the page header so the diff toolbar spends one row on changes', () => {
    const rig = readyValidationPage();
    const state = { ...reviewFixture, id: 'compact-review', title: 'Staged changes', beforeLabel: 'base · 06f83e2', afterLabel: 'HEAD · 201117c',
      changedFiles: ['model.py'], stale: true };
    act(() => rig.emit(hostMessage('graphReviewState', { state })));
    const header = screen.getByTestId('local-browser-strip');
    const toolbar = screen.getByRole('region', { name: 'Graph diff review' });
    expect(within(header).getByText('Staged changes')).toBeVisible();
    expect(within(header).getByText('base · 06f83e2 → HEAD · 201117c')).toBeVisible();
    for (const name of ['Open native code diff', 'Refresh graph diff', 'Close graph diff']) {
      expect(within(header).getByRole('button', { name })).toBeVisible();
      expect(within(toolbar).queryByRole('button', { name })).not.toBeInTheDocument();
    }
    expect(toolbar.textContent).not.toContain('06f83e2');
    // The current change and the stale notice share the stepper's row instead of adding rows.
    const row = toolbar.querySelector('.graph-review-controls')!;
    expect(row.contains(toolbar.querySelector('.graph-review-current'))).toBe(true);
    expect(row.contains(within(toolbar).getByText(/Source changed since this review/))).toBe(true);
    expect(toolbar.querySelectorAll(':scope > div')).toHaveLength(1);
    fireEvent.click(within(header).getByRole('button', { name: 'Refresh graph diff' }));
    expect(rig.transport.send).toHaveBeenLastCalledWith({ type: 'graphReviewRequest', action: 'refresh', reviewId: 'compact-review' });
  });

  it('requests a review when Run model is clicked and retains cancellation ownership', async () => {
    const rig = readyCapturedPage();
    expect(rig.transport.send).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'validateRequest' }));
    fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
    expect(validationRequest(rig)).not.toHaveProperty('measureLatency');
    expect(screen.getByRole('button', { name: 'Running…' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Running…' }));
    expect((rig.transport.send as jest.Mock).mock.calls.filter(([m]) => m.type === 'validateRequest')).toHaveLength(1);
    await act(async () => { mockCapturedLocalContext?.validation?.cancel(); });
    expect(rig.transport.send).toHaveBeenCalledWith(expect.objectContaining({ type: 'validateCancel', requestId: validationRequest(rig).requestId }));
    expect(screen.getByRole('button', { name: 'Run model' })).toBeEnabled();
  });

  it('offers the same Run model entry on a source preview before any capture', () => {
    const rig = readyValidationPage();
    expect(screen.getByRole('button', { name: 'Run model' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Run model' }));
    expect(validationRequest(rig)).not.toHaveProperty('measureLatency');
    expect((rig.transport.send as jest.Mock).mock.calls.some(([message]) => message.action?.action === 'run')).toBe(false);
  });

  it('warns only for confirmed absence, retains measurements and routes optional setup actions', () => {
    const rig = readyValidationPage();
    const opening = { id: 'config-warning', target: netTarget(), phase: 'ready', hasCapture: true,
      detail: 'Captured.', draft: { profileName: 'net_cpu', profile: {} } };
    act(() => {
      rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false }));
      rig.emit(hostMessage('graphOpening', { state: opening }));
    });
    expect(screen.queryByRole('status', { name: 'Missing model configuration' })).toBeNull();
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, configFileExists: false } })));
    const warning = screen.getByRole('status', { name: 'Missing model configuration' });
    expect(warning).toHaveTextContent('previous measurements');
    expect(warning).toHaveTextContent('or edit the file manually');
    const modelMenu = screen.getByRole('button', { name: 'Model and file actions' });
    fireEvent.click(modelMenu);
    expect(screen.queryByText('Saved as')).toBeNull();
    fireEvent.click(modelMenu);
    fireEvent.click(within(warning).getByRole('button', { name: 'Use Agent' }));
    expect(screen.getByRole('region', { name: 'Canvas chat' })).toBeVisible();
    expect(screen.getByRole('textbox', { name: 'Message to coding agent' })).toHaveValue('');
    fireEvent.click(within(warning).getByRole('button', { name: 'Create tensorviz.toml' }));
    for (const kind of ['agent_help', 'edit_config']) expect(rig.transport.send).toHaveBeenCalledWith(expect.objectContaining({
      type: 'graphSetupRequest', openingId: opening.id, target: opening.target, action: { kind },
    }));
    expect(screen.getByTestId('canvas-app')).toBeVisible();
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, configFileExists: true } })));
    expect(screen.queryByRole('status', { name: 'Missing model configuration' })).toBeNull();
    fireEvent.click(modelMenu);
    expect(screen.getByText('Saved as')).toHaveTextContent('net_cpu');
    fireEvent.click(modelMenu);
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'source_only', hasCapture: false, configFileExists: false } })));
    expect(screen.getByRole('status', { name: 'Missing model configuration' })).toHaveTextContent('Source inspection is available');
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'saving', hasCapture: false, configFileExists: false } })));
    expect(screen.queryByRole('status', { name: 'Missing model configuration' })).toBeNull();
    expect((rig.transport.send as jest.Mock).mock.calls.some(([message]) => message.type === 'validateRequest')).toBe(false);
  });
  it('keeps the source preview visible during setup and replaces it with the captured graph without Validate', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    const opening = { id: 'opening', target: netTarget(), phase: 'capturing', hasCapture: false, detail: 'Running one example.' };
    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('snapshot', { role: 'host', state: { activeTarget: null, projection: null,
        traced: false, greyed: false, syncState: null, connectionState: 'connected', opening } }));
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection(), traced: false, greyed: false }));
    });
    expect(screen.getByText('Running one example.')).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Open captured graph' })).toBeNull();
    expect(screen.getByTestId('canvas-app')).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Run model' })).toBeNull();
    act(() => {
      rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false }));
      rig.emit(hostMessage('syncState', { target: netTarget(), state: 'runtime-traced' }));
      rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'ready', hasCapture: true, detail: 'Captured.',
        draft: { profileName: 'net_cpu', profile: {} } } }));
    });
    expect(screen.getByTestId('canvas-app')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Run model' })).toBeEnabled();
    // The saved setup sits in the model menu, not in a header row.
    expect(screen.queryByText('Saved as')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Model and file actions' }));
    expect(screen.getByText('Saved as')).toHaveTextContent('net_cpu');
    fireEvent.click(screen.getByRole('button', { name: 'Open tensorviz.toml' }));
    expect(rig.transport.send).toHaveBeenCalledWith(expect.objectContaining({ type: 'graphSetupRequest', openingId: 'opening', action: { kind: 'edit_config' } }));
    fireEvent.click(screen.getByRole('button', { name: 'Change inputs' }));
    expect(rig.transport.send).toHaveBeenCalledWith(expect.objectContaining({ type: 'graphSetupRequest', openingId: 'opening', action: { kind: 'change_inputs' } }));
  });
  it('a rejected source-only preview shows recovery beside the toolbar instead of a blank canvas', () => {
    const rig = makeTransport();
    const target = { ...netTarget(), cls: 'KimiK3Model', file: 'torchtitan/models/kimi_k3/model.py' };
    const reason = 'The source could not be projected as a supported PyTorch module.';
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('activeTarget', { target }));
      rig.emit(hostMessage('graphOpening', { state: { id: 'kimi-opening', target, phase: 'source_only', hasCapture: false, detail: 'Source only.' } }));
      rig.emit(hostMessage('syncState', { target, state: 'stale', reason }));
    });
    const recovery = screen.getByRole('region', { name: 'Graph preview unavailable' });
    expect(recovery).toBeVisible(); expect(within(recovery).getByText(reason)).toBeVisible();
    fireEvent.click(within(recovery).getByRole('button', { name: 'Retry graph' }));
    expect(rig.transport.send).toHaveBeenCalledWith(expect.objectContaining({ type: 'graphSetupRequest', openingId: 'kimi-opening', target, action: { kind: 'retry' } }));
    fireEvent.click(within(recovery).getByRole('button', { name: 'Set up model' }));
    expect(rig.transport.send).toHaveBeenCalledWith(expect.objectContaining({ type: 'graphSetupRequest', openingId: 'kimi-opening', target, action: { kind: 'change_inputs' } }));
    act(() => {
      rig.emit(hostMessage('projection', { target, projection: netProjection({ cls: target.cls, file: target.file }), traced: false, greyed: false }));
      rig.emit(hostMessage('syncState', { target, state: 'source-only' }));
    });
    expect(screen.queryByRole('region', { name: 'Graph preview unavailable' })).toBeNull();
    expect(screen.getByTestId('canvas-app')).toBeVisible();
  });

  it('allows saved-profile Refresh when a library model has no current source-only projection', () => {
    const rig = readyValidationPage();
    act(() => {
      rig.emit(hostMessage('graphOpening', { state: { id: 'saved-library', target: netTarget(),
        phase: 'ready', hasCapture: true, detail: 'Captured.' } }));
      rig.emit(hostMessage('syncState', { target: netTarget(), state: 'stale',
        reason: 'The imported library class has no static projection after a helper edit.' }));
    });
    const refresh = screen.getByRole('button', { name: 'Run model' });
    expect(refresh).toBeEnabled();
    fireEvent.click(refresh);
    expect(rig.transport.send).toHaveBeenCalledWith(expect.objectContaining({
      type: 'graphSetupRequest', openingId: 'saved-library', action: { kind: 'retry' }, target: netTarget(),
    }));
    expect((rig.transport.send as jest.Mock).mock.calls.some(([message]) => message.type === 'validateRequest')).toBe(false);
    act(() => rig.setStatus({ phase: 'connected', hostLink: 'disconnected', error: null }));
    expect(screen.getByRole('button', { name: 'Run model' })).toBeDisabled();
  });
  it('keeps the AST preview without labelling it captured when a ready message arrives first', () => {
    const rig = readyValidationPage();
    act(() => rig.emit(hostMessage('graphOpening', { state: { id: 'missing-capture', target: netTarget(), phase: 'ready', hasCapture: true, detail: 'Run finished.' } })));
    expect(screen.getByTestId('canvas-app')).toBeVisible();
    expect(screen.getByText('Waiting for the captured graph')).toBeInTheDocument();
    expect(screen.queryByText('Captured graph')).toBeNull();
  });
  it('retains the same measured canvas while inputs are changed and a new capture runs', () => {
    const rig = readyValidationPage();
    const opening = { id: 'edit-inputs', target: netTarget(), phase: 'needs_setup', hasCapture: false, detail: 'Choose inputs.' };
    const canvas = screen.getByTestId('canvas-app');
    act(() => {
      rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false }));
      rig.emit(hostMessage('graphOpening', { state: opening }));
    });
    expect(screen.getByTestId('canvas-app')).toBe(canvas);
    expect(canvas).toBeVisible();
    expect(screen.getByText(/Previous capture stays visible/)).toBeInTheDocument();
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'capturing' } })));
    expect(canvas).toBeVisible();
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'ready', hasCapture: true } })));
    expect(screen.getByTestId('canvas-app')).toBe(canvas);
    expect(screen.queryByText(/Previous capture stays visible/)).toBeNull();
    expect(screen.getByRole('button', { name: 'Change inputs' })).toBeInTheDocument();
  });
  it('closes and reopens inputs without losing the draft or running the model', () => {
    const rig = readyValidationPage();
    const opening = { id: 'draft-inputs', target: netTarget(), phase: 'needs_setup', hasCapture: false, detail: 'Choose inputs.',
      discovery: { model: 'model:Net', sourceDigest: 'a'.repeat(64), init: [], inputs: [{ name: 'x', kind: 'keyword', required: true }],
        examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false } };
    act(() => rig.emit(hostMessage('graphOpening', { state: opening })));
    fireEvent.click(screen.getByRole('button', { name: 'Set up manually' }));
    fireEvent.change(screen.getByLabelText('x shape'), { target: { value: '4, 8' } });
    const sentBefore = (rig.transport.send as jest.Mock).mock.calls.length;
    fireEvent.click(screen.getByRole('button', { name: 'Close inputs' }));
    expect(screen.getByLabelText('x shape')).not.toBeVisible();
    expect(screen.getByTestId('canvas-app')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Change inputs' })).toHaveFocus();
    fireEvent.click(screen.getByRole('button', { name: 'Change inputs' }));
    expect(screen.getByLabelText('x shape')).toBeVisible();
    expect(screen.getByLabelText('x shape')).toHaveValue('4, 8');
    expect((rig.transport.send as jest.Mock).mock.calls.length).toBe(sentBefore);
    fireEvent.keyDown(screen.getByLabelText('x shape'), { key: 'Escape' });
    expect(screen.getByLabelText('x shape')).not.toBeVisible();
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, id: 'new-opening' } })));
    expect(screen.getByRole('button', { name: 'Set up manually' })).toBeVisible();
    expect(screen.getByLabelText('x shape')).not.toBeVisible();
  });
  it('keeps the Canvas when Use Agent publishes a draft the setup form cannot edit', () => {
    const rig = readyValidationPage();
    const opening = { id: 'agent-setup', target: netTarget(), phase: 'needs_setup', hasCapture: false, detail: 'Choose inputs.',
      discovery: { model: 'model:Net', sourceDigest: 'a'.repeat(64), init: [], inputs: [{ name: 'x', kind: 'keyword', required: true }],
        examples: [], factories: [], configs: [], parents: [], constraints: [], truncated: false } };
    act(() => rig.emit(hostMessage('graphOpening', { state: opening })));
    fireEvent.click(screen.getByRole('button', { name: 'Set up with agent' }));
    // The host publishes an agent's draft while the backend checks it; this one omits inputs.args.
    const profile = { kind: 'model', model: 'model:Net', init: { args: [], kwargs: {} },
      inputs: { kwargs: { x: { kind: 'tensor', shape: [4, 8], dtype: 'float32' } } } };
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'saving', draft: { profile } } })));
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening, draft: { profile }, requirements: ['inputs.args: Field required'] } })));
    expect(screen.getByTestId('canvas-app')).toBeVisible();
    expect(screen.getByRole('region', { name: 'Canvas chat' })).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Change inputs' }));
    const setup = screen.getByRole('region', { name: 'Open captured graph' });
    expect(within(setup).getByRole('heading', { name: 'This setup can’t be shown in the form' })).toBeVisible();
    expect(within(setup).getByRole('alert')).toHaveTextContent('inputs.args: Field required');
    // The agent's corrected draft restores the editable form.
    const x = { ...profile.inputs.kwargs.x, requires_grad: false, generator: { kind: 'normal', mean: 0, std: 1, seed: 0 } };
    act(() => rig.emit(hostMessage('graphOpening', { state: { ...opening,
      draft: { profile: { ...profile, inputs: { args: [], kwargs: { x } } } } } })));
    expect(screen.getByLabelText('x shape')).toHaveValue('4, 8');
  });
  it('does not reuse a same-named capture from another workspace during opening', () => {
    const rig = readyValidationPage();
    const other = { ...netTarget(), workspaceFolder: '/other' };
    act(() => {
      rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false }));
      rig.emit(hostMessage('activeTarget', { target: other }));
      rig.emit(hostMessage('graphOpening', { state: { id: 'other-workspace', target: other, phase: 'needs_setup', hasCapture: false, detail: 'Complete setup.' } }));
    });
    expect(screen.getByTestId('canvas-app')).not.toBeVisible();
    expect(screen.queryByText(/Previous capture stays visible/)).toBeNull();
  });
  it('clears the previous verdict while resolving a new opening and attributes the next result to its run', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('snapshot', { role: 'host', state: { activeTarget: netTarget(),
        projection: netProjection({ traced: true }), traced: true, greyed: false,
        syncState: 'runtime-traced', connectionState: 'connected',
        validation: { target: netTarget(), runId: 'previous', status: 'passed', detail: 'Previous input passed.' } } }));
    });
    expect(mockCapturedLocalContext?.validation?.lastResult?.runId).toBe('previous');
    const opening = { id: 'new-opening', target: netTarget(), phase: 'resolving', hasCapture: false, detail: 'Resolving setup.' };
    act(() => rig.emit(hostMessage('graphOpening', { state: opening })));
    expect(mockCapturedLocalContext?.validation?.lastResult).toBeNull();
    act(() => {
      rig.emit(hostMessage('validationState', { target: netTarget(), runId: 'next', status: 'failed', detail: 'New input failed.' }));
      rig.emit(hostMessage('graphOpening', { state: { ...opening, phase: 'failed', hasCapture: true, detail: 'New input failed.' } }));
    });
    expect(mockCapturedLocalContext?.validation?.lastResult).toBeNull(); // Persistent run notice owns the failure, without a duplicate toast.
    expect(mockCapturedLocalContext?.validation?.latestResult).toMatchObject({ runId: 'next', status: 'failed', detail: 'New input failed.' });
    expect(screen.getByText('New input failed.')).toBeInTheDocument();
    expect(screen.queryByText('Previous input passed.')).toBeNull();
  });
  it.each(['legacy', 'semantic'] as const)('shows timeout setup separately from earlier %s measurements after reconnect', format => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('snapshot', { role: 'host', state: {
        activeTarget: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false,
        syncState: 'runtime-traced', connectionState: 'connected',
        validation: { runId: 'timeout', target: netTarget(), status: 'failed', detail: 'The deadline expired.',
          evidence: { origin: 'automatic', attemptStatus: 'timed_out', sourceId: 'source:new', planDigest: 'a'.repeat(64),
            lastCompleteCaptureId: null, lastPartialCaptureId: 'capture:old',
            displayedCapture: { id: 'capture:old', attemptId: 'old', format, status: 'partial',
              forwardStatus: 'passed', sourceId: 'source:old', planDigest: 'b'.repeat(64), unavailable: null },
            setup: { origin: 'profile', profileName: 'tiny', device: 'cpu', mode: 'eval' } } },
      } }));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Latest validation: Failed' }));
    const dialog = screen.getByRole('dialog', { name: 'Latest validation result' });
    expect(within(dialog).getByText('Started by trace on save · Timed out')).toBeInTheDocument();
    expect(within(dialog).getByText('Attempt setup: profile tiny · cpu · eval')).toBeInTheDocument();
    expect(within(dialog).getByText('Displayed measurements: an earlier attempt · forward passed')).toBeInTheDocument();
    expect(within(dialog).getByText('Detailed operation capture is incomplete.')).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Close result' }));
    act(() => rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection(), traced: false, greyed: false })));
    fireEvent.click(screen.getByRole('button', { name: 'Latest validation: Failed' }));
    expect(screen.getByText('No measured capture is displayed.')).toBeInTheDocument();
    expect(screen.queryByText('Displayed measurements: an earlier attempt · forward passed')).toBeNull();
  });

  it('restores a failed attempt separately from retained measurements and keeps its explanation available', () => {
    jest.useFakeTimers();
    const rig = makeTransport();
    const events = jest.spyOn(window, 'dispatchEvent');
    try {
      render(<LocalBrowserApp createTransport={() => rig.transport} />);
      act(() => {
        rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
        rig.emit(hostMessage('snapshot', { role: 'host', state: {
          activeTarget: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false,
          syncState: 'runtime-traced', connectionState: 'connected',
          validation: { runId: 'timeout-run', target: netTarget(), status: 'failed', detail: 'Validation timed out. Old capture retained.' },
        } }));
        jest.advanceTimersByTime(10_000);
      });
      expect(screen.getByText('Previous measurements retained')).toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Latest validation: Failed' }));
      const dialog = screen.getByRole('dialog', { name: 'Latest validation result' });
      expect(within(dialog).getByText('Validation timed out. Old capture retained.')).toBeInTheDocument();
      expect(dialog).toHaveFocus();
      fireEvent.keyDown(dialog, { key: 'Escape' });
      expect(screen.queryByRole('dialog', { name: 'Latest validation result' })).toBeNull();
      expect(screen.getByRole('button', { name: 'Latest validation: Failed' })).toHaveFocus();
      fireEvent.click(screen.getByRole('button', { name: 'Latest validation: Failed' }));
      fireEvent.click(within(screen.getByRole('dialog', { name: 'Latest validation result' })).getByRole('button', { name: 'Close result' }));
      expect(screen.getByRole('button', { name: 'Latest validation: Failed' })).toHaveFocus();
      expect(events.mock.calls.filter(([event]) => (event as CustomEvent).detail?.command === 'validation-complete')).toHaveLength(0);
    } finally { events.mockRestore(); jest.useRealTimers(); }
  });

  it('restores a successful result without replaying the sweep and labels it after a source edit', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    const events = jest.spyOn(window, 'dispatchEvent');
    try {
      act(() => {
        rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
        rig.emit(hostMessage('snapshot', { role: 'host', state: {
          activeTarget: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false,
          syncState: 'runtime-traced', connectionState: 'connected',
          validation: { runId: 'passed-run', target: netTarget(), status: 'passed', detail: 'Passed for these inputs.', projectionRevision: 1 },
        } }));
      });
      expect(screen.getByRole('button', { name: 'Latest validation: Passed' })).toBeInTheDocument();
      expect(events.mock.calls.filter(([event]) => (event as CustomEvent).detail?.command === 'validation-complete')).toHaveLength(0);
      act(() => rig.emit(hostMessage('activeTarget', { target: { ...netTarget(), sourceDigest: 'edited', documentVersion: 12 } })));
      fireEvent.click(screen.getByRole('button', { name: 'Latest validation: Not run for this version' }));
      expect(within(screen.getByRole('dialog', { name: 'Latest validation result' })).getByText(/does not verify the current graph.*Passed for these inputs/)).toBeInTheDocument();
    } finally { events.mockRestore(); }
  });

  it('sweeps a manual broadcast once and keeps a newer result over its delayed private receipt', async () => {
    const events = jest.spyOn(window, 'dispatchEvent');
    const sweeps = () => events.mock.calls.filter(([event]) => (event as CustomEvent).detail?.command === 'validation-complete');
    try {
      const rig = readyValidationPage();
      fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
      const request = validationRequest(rig);
      const target = { ...netTarget(), backendRevision: 2 };
      await act(async () => {
        rig.emit(hostMessage('validationState', { runId: request.requestId, target: request.target, status: 'running', detail: 'Tracing.' }));
        rig.emit(hostMessage('projection', { target, projection: netProjection({ revision: 2, traced: true }), traced: true, greyed: false }));
        rig.emit(hostMessage('validationState', { runId: request.requestId, target, status: 'passed', detail: 'Measured once.', projectionRevision: 2 }));
      });
      expect(sweeps()).toHaveLength(1);
      await act(async () => {
        rig.emit(hostMessage('validationState', { runId: 'newer', target, status: 'refused', detail: 'Save the configuration first.' }));
        rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult', status: 'passed', detail: 'Old private result.' }));
      });
      expect(sweeps()).toHaveLength(1);
      expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Could not run');
      fireEvent.click(screen.getByRole('button', { name: 'Latest validation: Could not run' }));
      expect(within(screen.getByRole('dialog', { name: 'Latest validation result' })).getByText('Save the configuration first.')).toBeInTheDocument();
    } finally { events.mockRestore(); }
  });

  it('retains the restored host result over a pending request and explains stale refresh without claiming a source edit', async () => {
    const rig = readyValidationPage();
    fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
    const request = validationRequest(rig);
    await act(async () => {
      rig.emit(hostMessage('snapshot', { role: 'host', state: {
        activeTarget: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false,
        syncState: 'runtime-traced', connectionState: 'connected',
        validation: { runId: 'restored', target: netTarget(), status: 'failed', detail: 'Host timed out.' },
      } }));
      rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult', status: 'passed', detail: 'Delayed local completion.' }));
    });
    act(() => rig.emit(hostMessage('syncState', { target: netTarget(), state: 'stale', reason: 'Refresh service unavailable.' })));
    fireEvent.click(screen.getByRole('button', { name: 'Latest validation: Not run for this version' }));
    expect(within(screen.getByRole('dialog', { name: 'Latest validation result' })).getByText('This result does not verify the current graph. Host timed out.')).toBeInTheDocument();
  });

  for (const scenario of ['invalid', 'selected_failure', 'snapshot_failure', 'dismiss', 'cancel_resolution']) {
    it(`shows actual snapshot/host ${scenario} feedback and retains earlier measurements without a new sweep`, async () => {
      // Native Node loads the compiled host, whose snapshot spawns real Python -I.
      // The host's execution seam is recorded; it must never be reached in these cases.
      const repo = path.resolve(process.cwd(), '..');
      const run = JSON.parse(execFileSync(process.execPath, [
        '-r', path.join(repo, 'extension/out/test/bootstrap.js'), '-e',
        `require(${JSON.stringify(path.join(repo, 'extension/out/test/modelProfileFixture.js'))})
          .runModelProfileFixture(process.argv[1]).then(value => process.stdout.write(JSON.stringify(value)))
          .catch(() => process.exit(1))`, scenario,
      ], { cwd: repo, timeout: 20000 }).toString());
      expect(run.executions).toEqual([]);
      expect(run.sourceExecuted).toBe(false);
      const rig = makeTransport();
      render(<LocalBrowserApp createTransport={() => rig.transport} />);
      const target = run.result.target;
      const measured = netProjection({ file: target.file, cls: target.cls, sourceDigest: target.sourceDigest, traced: true,
        graph: { nodes: [{ id: 'fc1', type: 'layerNode', position: { x: 0, y: 0 },
          data: { outputShape: [3, 4], outputTensor: { shape: [3, 4], dtype: 'float32', logicalBytes: 48 } } }], edges: [] } });
      act(() => {
        rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
        rig.emit(hostMessage('activeTarget', { target }));
        rig.emit(hostMessage('projection', { target, projection: measured, traced: true, greyed: false }));
        rig.emit(hostMessage('syncState', { target, state: 'runtime-traced' }));
      });
      const previous = currentProjection();
      const events = jest.spyOn(window, 'dispatchEvent');
      try {
        fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
        const request = validationRequest(rig);
        await act(async () => rig.emit(hostMessage('validateResult', { ...run.result, requestId: request.requestId })));
        expect(screen.getByText(run.result.detail)).toBeInTheDocument();
        expect(screen.getByText('Previous measurements retained')).toBeInTheDocument();
        expect(screen.queryByText('Traced — shapes measured')).toBeNull();
        expect(mockCapturedLocalContext?.validation?.statusLabel).toBe(run.result.status === 'refused' ? 'Could not run' : 'Cancelled');
        expect(currentProjection()).toBe(previous);
        expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeEnabled();
        expect(events.mock.calls.filter(([event]) => (event as CustomEvent).detail?.command === 'validation-complete')).toHaveLength(0);
        expect((rig.transport.send as jest.Mock).mock.calls.filter(([m]) => m.type === 'validateRequest')).toHaveLength(1);
      } finally { events.mockRestore(); }
    });
  }

  it('keeps the host run action in the header and the result in the footer', async () => {
    const rig = readyValidationPage();

    const strip = document.querySelector('.local-browser-header');
    // The header adapts to its width; there is no toggle to find controls behind.
    expect(screen.queryByRole('button', { name: 'Expand toolbar' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Find a layer' })).toBeVisible();
    expect(strip!.querySelector('[data-testid="local-browser-validate"]')).not.toBeNull();

    const button = screen.getByRole('button', { name: 'Run model', exact: true });
    expect(screen.getByTestId('canvas-app').contains(button)).toBe(false);
    expect(document.querySelector('.local-browser-footer')?.querySelector('[data-testid="local-browser-validate"]')).toBeNull();

    // And the Inspector line reads the run rather than a literal: it said "Not run"
    // for the life of the page, including after a pass.
    expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Not run');
    fireEvent.click(button);
    expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Running…');
    const request = validationRequest(rig);
    await act(async () => {
      rig.emit(hostMessage('validateResult', {
        ...request, type: 'validateResult', status: 'passed', detail: 'Validation passed for this input.',
      }));
    });
    expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Passed');
  });

  it('shows Validate, routes the click to the host, and preserves success across its own measured projection', async () => {
    const rig = readyValidationPage();
    fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
    expect(screen.getByRole('button', { name: 'Running…' })).toBeDisabled();
    const request = validationRequest(rig);
    expect(request.target).toEqual(netTarget());
    expect(Object.keys(request).sort()).toEqual(['requestId', 'target', 'type']);
    expect((rig.transport.send as jest.Mock).mock.calls.filter(([m]) => m.type === 'validateRequest')).toHaveLength(1);
    const measured = { ...netTarget(), backendRevision: 2 };
    await act(async () => {
      rig.emit(hostMessage('projection', { target: measured, projection: netProjection({ revision: 2, traced: true }), traced: true, greyed: false }));
      rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult', status: 'passed', detail: 'Validation passed for this input.' }));
    });
    expect(screen.getByText('Validation passed for this input.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeEnabled();
    act(() => rig.emit(hostMessage('projection', { target: { ...measured, backendRevision: 3 },
      projection: netProjection({ revision: 3 }), traced: false, greyed: false })));
    expect(screen.queryByText('Validation passed for this input.')).toBeNull();
    // The verdict is a transient toast now, so a stale receipt is not a sentence
    // parked in the header — it is the Inspector refusing to call it current.
    expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Not run for this version');
  });

  it('says so in the verdict when nobody chose the input the run measured', async () => {
    /** "Validation passed for the traced input" is equally true of the tensor the
     *  file declares and of one the backend's shape search found for itself, and the
     *  sentence alone cannot tell them apart — which is the whole defect. The
     *  projection already carries the answer on the Input node, so the verdict adds
     *  it rather than leaving it to a card the reader may not be looking at. */
    const withInput = (params: Record<string, unknown>) => netProjection({
      traced: true,
      document: {
        view: {},
        semantic: { nodes: [{ node_id: 'input', kind: 'input', op: 'Input', params }] },
      },
    });
    const runWith = async (projection: ProjectionPayload, status: 'passed' | 'failed' | 'refused') => {
      // Four pages in one case, so each starts from an empty document: RTL unmounts
      // between `it`s, not within one.
      cleanup();
      const rig = makeTransport();
      render(<LocalBrowserApp createTransport={() => rig.transport} />);
      act(() => {
        rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
        rig.emit(hostMessage('activeTarget', { target: netTarget() }));
        rig.emit(hostMessage('projection', { target: netTarget(), projection, traced: true, greyed: false }));
        rig.emit(hostMessage('syncState', { target: netTarget(), state: 'runtime-traced' }));
      });
      fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
      const request = validationRequest(rig);
      await act(async () => {
        rig.emit(hostMessage('validateResult', {
          ...request, type: 'validateResult', status, detail: 'Validation passed for the traced input.',
        }));
      });
      return mockCapturedLocalContext?.validation?.lastResult?.detail ?? '';
    };

    const guessed = await runWith(
      withInput({ shape: '1,8', _input_provenance: 'seed-from-first-layer' }), 'passed',
    );
    expect(guessed).toContain('Validation passed for the traced input.');
    expect(guessed).toContain("first layer's width");

    // A failed forward on an input nobody picked is the case where this matters most:
    // the failure may be about the guess rather than about the model.
    const failed = await runWith(
      withInput({ shape: '1,3,224,224', _input_provenance: 'failed-candidate' }), 'failed',
    );
    expect(failed).toContain('failed during the forward pass');
    expect(failed).not.toContain('survived a forward pass');

    // A declared input is unremarkable: the verdict is the host's sentence and nothing
    // else, or every run of every correct model grows a paragraph that says nothing.
    const declared = await runWith(
      withInput({ shape: '2,8', _input_provenance: 'declared' }), 'passed',
    );
    expect(declared).toBe('Validation passed for the traced input.');

    // And a run that checked nothing names no input: there was none.
    const refused = await runWith(
      withInput({ shape: '1,8', _input_provenance: 'seed-from-first-layer' }), 'refused',
    );
    expect(refused).toBe('Validation passed for the traced input.');
  });

  it('shows MCP validation progress and reuses the sweep only for its measured target', async () => {
    const events = jest.spyOn(window, 'dispatchEvent');
    const sweeps = () => events.mock.calls.filter(([event]) => (event as CustomEvent).detail?.command === 'validation-complete');
    try {
      const rig = readyValidationPage();
      act(() => rig.emit(hostMessage('validationState', { runId: 'run-1', target: netTarget(), status: 'running', detail: 'Tracing.' })));
      expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeDisabled();
      expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Running…');
      expect(sweeps()).toHaveLength(0);
      const target = { ...netTarget(), backendRevision: 2 };
      await act(async () => {
        rig.emit(hostMessage('projection', { target, projection: netProjection({ revision: 2, traced: true,
          traceError: { nodeId: 'fc', message: 'Forward failed.' } }), traced: true, greyed: false }));
        rig.emit(hostMessage('validationState', { runId: 'run-1', target, status: 'failed', detail: 'Partial shapes measured.', projectionRevision: 2 }));
      });
      expect(sweeps()).toHaveLength(1);
      expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Failed');
      expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeEnabled();
      await act(async () => {
        rig.emit(hostMessage('projection', { target: { ...target, documentVersion: 99, backendRevision: 3 },
          projection: netProjection({ revision: 3 }), traced: false, greyed: false }));
        rig.emit(hostMessage('validationState', { runId: 'old-run', target, status: 'passed', detail: 'Old MCP pass', projectionRevision: 2 }));
      });
      expect(sweeps()).toHaveLength(1);
      expect(screen.queryByText('Old MCP pass')).toBeNull();
    } finally { events.mockRestore(); }
  });

  it('MCP cancellation and refusal never reuse a previous successful trace sweep', async () => {
    const events = jest.spyOn(window, 'dispatchEvent');
    try {
      const rig = readyValidationPage();
      await act(async () => {
        rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false }));
        rig.emit(hostMessage('validationState', { runId: 'run-1', target: netTarget(), status: 'cancelled', detail: 'Stopped.' }));
      });
      expect(events.mock.calls.filter(([event]) => (event as CustomEvent).detail?.command === 'validation-complete')).toHaveLength(0);
      expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Cancelled');
    } finally { events.mockRestore(); }
  });

  it('a dismissed Run model review shows the earlier result again without replaying its sweep', async () => {
    const events = jest.spyOn(window, 'dispatchEvent');
    const sweeps = () => events.mock.calls.filter(([event]) => (event as CustomEvent).detail?.command === 'validation-complete');
    try {
      const rig = readyValidationPage();
      const target = { ...netTarget(), backendRevision: 2 };
      const saved = { runId: 'save-run', target, status: 'passed', detail: 'Passed.', projectionRevision: 2 };
      await act(async () => {
        rig.emit(hostMessage('projection', { target, projection: netProjection({ revision: 2, traced: true }), traced: true, greyed: false }));
        rig.emit(hostMessage('validationState', saved));
      });
      expect(sweeps()).toHaveLength(1);
      fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
      const request = validationRequest(rig);
      // The host republishes the result it still holds, then cancels the request that never ran.
      await act(async () => {
        rig.emit(hostMessage('validationState', saved));
        rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult', status: 'cancelled', detail: 'Validation cancelled.' }));
      });
      expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Passed');
      expect(sweeps()).toHaveLength(1);
    } finally { events.mockRestore(); }
  });

  it.each(['passed', 'failed'] as const)('delivers %s completion only after graph opening releases the measured canvas', status => {
    const rig = readyValidationPage();
    const target = { ...netTarget(), backendRevision: 2 };
    const opening = { id: 'refresh', target, phase: 'capturing', hasCapture: false, detail: 'Capturing.' };
    const delivered: ProjectionPayload[] = [];
    const listener = (event: Event) => {
      if ((event as CustomEvent).detail?.command === 'validation-complete') delivered.push(currentProjection()!);
    };
    window.addEventListener('builder-command', listener);
    try {
      act(() => rig.emit(hostMessage('graphOpening', { state: opening })));
      act(() => {
        rig.emit(hostMessage('projection', { target, projection: netProjection({ revision: 2, traced: true }), traced: true, greyed: false }));
        rig.emit(hostMessage('validationState', { runId: 'refresh-run', target, status, detail: 'Capture finished.', projectionRevision: 2 }));
      });
      expect(currentProjection()?.greyed).toBe(true);
      expect(delivered).toHaveLength(0);
      const settled = { ...opening, phase: status === 'passed' ? 'ready' : 'failed', hasCapture: true };
      act(() => rig.emit(hostMessage('graphOpening', { state: settled })));
      expect(delivered).toEqual([expect.objectContaining({ revision: 2, traced: true, greyed: false })]);
      act(() => rig.emit(hostMessage('graphOpening', { state: { ...settled, detail: 'Current capture.' } })));
      expect(delivered).toHaveLength(1);
    } finally { window.removeEventListener('builder-command', listener); }
  });

  it('replays the existing canvas sweep only after a manual run publishes its measured projection', async () => {
    const events = jest.spyOn(window, 'dispatchEvent');
    const commands = () => events.mock.calls.map(([event]) => (event as CustomEvent).detail?.command).filter(Boolean);
    try {
      const rig = readyValidationPage();
      expect(commands()).not.toContain('validation-complete');
      for (const revision of [2, 3]) {
        fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
        const request = (rig.transport.send as jest.Mock).mock.calls.filter(([m]) => m.type === 'validateRequest').slice(-1)[0][0];
        const measured = { ...netTarget(), backendRevision: revision };
        await act(async () => {
          rig.emit(hostMessage('projection', { target: measured, projection: netProjection({ revision, traced: true }), traced: true, greyed: false }));
          rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult', status: 'passed', detail: 'Passed.' }));
        });
        expect(commands().slice(-2)).toEqual(['apply-projection', 'validation-complete']);
      }
      expect(commands().filter(command => command === 'validation-complete')).toHaveLength(2);
      fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
      const request = (rig.transport.send as jest.Mock).mock.calls.filter(([m]) => m.type === 'validateRequest').slice(-1)[0][0];
      await act(async () => rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult', status: 'failed', detail: 'Worker could not start.' })));
      expect(commands().filter(command => command === 'validation-complete')).toHaveLength(2);
    } finally { events.mockRestore(); }
  });

  it('sweeps partial failure evidence but does not replay stale results after editing', async () => {
    const events = jest.spyOn(window, 'dispatchEvent');
    const sweeps = () => events.mock.calls.filter(([event]) => (event as CustomEvent).detail?.command === 'validation-complete');
    try {
      const rig = readyValidationPage();
      fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
      const request = validationRequest(rig);
      const measured = { ...netTarget(), backendRevision: 2 };
      await act(async () => {
        rig.emit(hostMessage('projection', { target: measured, projection: netProjection({ revision: 2, traced: true,
          traceError: { nodeId: 'fc', message: 'Forward failed.' } }), traced: true, greyed: false }));
        rig.emit(hostMessage('syncState', { target: measured, state: 'runtime-traced' }));
        rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult', status: 'failed', detail: 'Partial forward failed.' }));
      });
      expect(sweeps()).toHaveLength(1);
      expect(screen.getByText('Forward failed — partial shapes measured')).toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
      const retry = (rig.transport.send as jest.Mock).mock.calls.filter(([m]) => m.type === 'validateRequest').slice(-1)[0][0];
      await act(async () => {
        rig.emit(hostMessage('projection', { target: { ...measured, sourceDigest: 'changed', documentVersion: 2, backendRevision: 3 },
          projection: netProjection({ revision: 3, sourceDigest: 'changed' }), traced: false, greyed: false }));
        rig.emit(hostMessage('validateResult', { ...retry, type: 'validateResult', status: 'passed', detail: 'Old pass' }));
      });
      expect(sweeps()).toHaveLength(1);
      expect(screen.queryByText('Old pass')).toBeNull();
    } finally { events.mockRestore(); }
  });

  it('shows save-first refusal, supports retry/cancel, and ignores a late success', async () => {
    const rig = readyValidationPage();
    fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
    const request = validationRequest(rig);
    await act(async () => rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult',
      status: 'refused', detail: 'Save the model and dependencies first.' })));
    expect(screen.getByText('Save the model and dependencies first.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
    const second = (rig.transport.send as jest.Mock).mock.calls.filter(([m]) => m.type === 'validateRequest')[1][0];
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Cancel', exact: true })));
    expect(rig.transport.send).toHaveBeenCalledWith({ type: 'validateCancel', requestId: second.requestId, target: second.target });
    expect(screen.getByText('Validation cancelled.')).toBeInTheDocument();
    await act(async () => rig.emit(hostMessage('validateResult', { ...second, type: 'validateResult', status: 'passed', detail: 'Late success' })));
    expect(screen.queryByText('Late success')).toBeNull();
  });

  it('never accepts a result for another source version and resolves a lost connection', async () => {
    const rig = readyValidationPage();
    fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
    const request = validationRequest(rig);
    await act(async () => rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult',
      target: { ...request.target, documentVersion: 999 }, status: 'passed', detail: 'Wrong source' })));
    expect(screen.queryByText('Wrong source')).toBeNull();
    expect(screen.getByRole('button', { name: 'Running…' })).toBeDisabled();
    await act(async () => rig.setStatus({ phase: 'reconnecting', hostLink: 'disconnected', error: null }));
    expect(screen.getByText(/connection ended during validation/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeDisabled();
  });

  it('keeps a long profile running past the former browser cutoff and accepts its measured completion', async () => {
    jest.useFakeTimers();
    const events = jest.spyOn(window, 'dispatchEvent');
    try {
      const rig = readyValidationPage();
      fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
      const request = validationRequest(rig);
      await act(async () => { jest.advanceTimersByTime(240_000); });
      expect(screen.getByRole('button', { name: 'Running…' })).toBeDisabled();
      expect((rig.transport.send as jest.Mock).mock.calls.some(([m]) => m.type === 'validateCancel')).toBe(false);
      expect(events.mock.calls.some(([event]) => (event as CustomEvent).detail?.command === 'validation-complete')).toBe(false);
      await act(async () => {
        rig.emit(hostMessage('projection', { target: { ...netTarget(), backendRevision: 2 },
          projection: netProjection({ revision: 2, traced: true }), traced: true, greyed: false }));
        rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult', status: 'passed', detail: 'Measured completion.' }));
      });
      expect(screen.getByText('Measured completion.')).toBeInTheDocument();
      expect(events.mock.calls.filter(([event]) => (event as CustomEvent).detail?.command === 'validation-complete')).toHaveLength(1);
    } finally { events.mockRestore(); jest.useRealTimers(); }
  });

  it('shows the host timeout as a failure and rejects late or superseded receipts', async () => {
    jest.useFakeTimers();
    try {
      const rig = readyValidationPage();
      act(() => {
        rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection({ traced: true }), traced: true, greyed: false }));
        rig.emit(hostMessage('syncState', { target: netTarget(), state: 'runtime-traced' }));
      });
      fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
      const first = validationRequest(rig);
      await act(async () => {
        jest.advanceTimersByTime(300_000);
        rig.emit(hostMessage('validateResult', { ...first, type: 'validateResult', status: 'failed', detail: 'Validation timed out and was stopped.' }));
      });
      expect(screen.getByText('Validation timed out and was stopped.')).toBeInTheDocument();
      expect(screen.getByText('Previous measurements retained')).toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
      const second = (rig.transport.send as jest.Mock).mock.calls.filter(([m]) => m.type === 'validateRequest')[1][0];
      await act(async () => {
        rig.emit(hostMessage('validateResult', { ...first, type: 'validateResult', status: 'passed', detail: 'Late timeout success' }));
        jest.advanceTimersByTime(240_000);
      });
      expect(screen.getByRole('button', { name: 'Running…' })).toBeDisabled();
      await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Cancel', exact: true })));
      expect(rig.transport.send).toHaveBeenCalledWith({ type: 'validateCancel', requestId: second.requestId, target: second.target });
      await act(async () => rig.emit(hostMessage('validateResult', { ...second, type: 'validateResult', status: 'passed', detail: 'Late cancelled success' })));
      expect(screen.getByText('Validation cancelled.')).toBeInTheDocument();
      expect(screen.queryByText(/Late .* success/)).toBeNull();
    } finally { jest.useRealTimers(); }
  });

  it('mounts the canvas, connects, and names the class once the host anchors it', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);

    expect(screen.getByTestId('canvas-app')).toBeInTheDocument();
    expect(rig.transport.connect).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Connecting…')).toBeInTheDocument();

    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
    });

    expect(screen.getByTitle('Net · models/net.py')).toBeInTheDocument();
    expect(screen.getByText('Projecting Net…')).toBeInTheDocument();
    expect(screen.queryByText('Connecting…')).toBeNull();
  });

  it('a projection reaches the canvas through the real builder-command bus and the label says how true it is', () => {
    const rig = makeTransport();
    const bus = captureCanvasDeliveries();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);

    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', {
        target: netTarget(),
        projection: netProjection(),
        traced: false,
        greyed: false,
      }));
      rig.emit(hostMessage('syncState', { target: netTarget(), state: 'source-only' }));
    });

    expect(bus.deliveries).toHaveLength(1);
    expect(bus.deliveries[0]).toEqual(netProjection());
    expect(currentProjection()).toEqual(netProjection());
    expect(screen.getByText('Not run — shapes read from the source')).toBeInTheDocument();

    // The traced tier re-labels without a second page, and the canvas gets the
    // measured projection as its own delivery.
    act(() => {
      rig.emit(hostMessage('projection', {
        target: { ...netTarget(), backendRevision: 2 },
        projection: netProjection({ revision: 2, traced: true }),
        traced: true,
        greyed: false,
      }));
      rig.emit(hostMessage('syncState', { target: { ...netTarget(), backendRevision: 2 }, state: 'runtime-traced' }));
    });

    expect(bus.deliveries).toHaveLength(2);
    expect(bus.deliveries[1].traced).toBe(true);
    expect(currentProjection()?.traced).toBe(true);
    expect(screen.getByText('Traced — shapes measured')).toBeInTheDocument();
    bus.stop();
  });

  it('shows incomplete capture coverage even when the forward passed', () => {
    const rig = makeTransport();
    const captured = netProjection({ traced: true });
    (captured.document as any).semantic.metadata = { capture: { status: 'partial', omissions: ['byte_limit'] } };
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', { target: netTarget(), projection: captured, traced: true, greyed: false }));
      rig.emit(hostMessage('syncState', { target: netTarget(), state: 'runtime-traced' }));
      rig.emit(hostMessage('validationState', { runId: 'partial-run', target: netTarget(),
        status: 'passed', detail: 'Forward passed.', projectionRevision: 1 }));
    });
    expect(screen.getByText('Partial capture — recording size limit reached')).toBeInTheDocument();
    expect(screen.queryByText('Traced — shapes measured')).toBeNull();
    expect(screen.getByRole('button', { name: 'Latest validation: Passed' })).toBeEnabled();
  });

  it('a broken buffer dims the graph on screen and says why', () => {
    const rig = makeTransport();
    const bus = captureCanvasDeliveries();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    const reason = 'models/net.py:4 does not parse (expected colon)';

    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', {
        target: netTarget(),
        projection: netProjection(),
        traced: false,
        greyed: false,
      }));
    });
    expect(bus.deliveries).toHaveLength(1);

    // One broker event later: the dimmed last-valid graph replaces the bright
    // one on the canvas, and the strip names the failing line.
    act(() => {
      rig.emit(hostMessage('staleState', {
        target: netTarget(),
        lastValidProjection: netProjection({ greyed: true }),
        reason,
      }));
    });

    expect(bus.deliveries).toHaveLength(2);
    expect(bus.deliveries[1].greyed).toBe(true);
    expect(currentProjection()?.greyed).toBe(true);
    expect(screen.getByText(`Stale — ${reason}`)).toBeInTheDocument();
    bus.stop();
  });

  it('shows an actionable first-load failure through reload, then enables validation after recovery', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    const reason = 'Graph could not refresh (HTTP 500). Run “tensorViz: Open as Graph” in VS Code to retry.';
    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('syncState', { target: netTarget(), state: 'stale', reason }));
    });
    expect(screen.queryByText('Projecting Net…')).toBeNull();
    expect(screen.getByText(`Graph unavailable — ${reason}`)).toBeVisible();
    expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeDisabled();
    expect(mockCapturedLocalContext?.validation?.disabledReason).toBe(reason);
    act(() => rig.emit(hostMessage('snapshot', { state: {
      activeTarget: netTarget(), projection: null, traced: false, greyed: true,
      syncState: 'stale', reason, connectionState: 'connected',
    } })));
    expect(screen.getByText(`Graph unavailable — ${reason}`)).toBeVisible();
    act(() => {
      rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection(), traced: false, greyed: false }));
      rig.emit(hostMessage('syncState', { target: netTarget(), state: 'source-only' }));
    });
    expect(screen.queryByText(`Graph unavailable — ${reason}`)).toBeNull();
    expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeEnabled();
  });

  it('hides an old pass on refresh failure, keeps the actionable cancellation visible, and recovers', async () => {
    const rig = readyValidationPage();
    const reason = 'The latest trace could not be merged. Graph could not refresh (HTTP 500). Reopen from VS Code.';
    fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
    const request = validationRequest(rig);
    const measured = { ...netTarget(), backendRevision: 2 };
    const traced = netProjection({ revision: 2, traced: true });
    await act(async () => {
      rig.emit(hostMessage('projection', { target: measured, projection: traced, traced: true, greyed: false }));
      rig.emit(hostMessage('syncState', { target: measured, state: 'runtime-traced' }));
      rig.emit(hostMessage('validateResult', { ...request, type: 'validateResult', status: 'passed', detail: 'Previous pass.' }));
    });
    expect(screen.getByText('Previous pass.')).toBeVisible();
    act(() => rig.emit(hostMessage('staleState', { target: measured, lastValidProjection: { ...traced, greyed: true }, reason })));
    expect(screen.queryByText('Previous pass.')).toBeNull();
    expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Not run for this version');
    expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeDisabled();
    expect(currentProjection()?.greyed).toBe(true);

    const fresh = { ...measured, backendRevision: 3 };
    act(() => {
      rig.emit(hostMessage('projection', { target: fresh, projection: netProjection({ revision: 3 }), traced: false, greyed: false }));
      rig.emit(hostMessage('syncState', { target: fresh, state: 'source-only' }));
    });
    expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
    await act(async () => rig.emit(hostMessage('staleState', { target: fresh,
      lastValidProjection: netProjection({ revision: 3, greyed: true }), reason })));
    expect(mockCapturedLocalContext?.validation?.statusLabel).toBe('Cancelled');
    expect(mockCapturedLocalContext?.validation?.lastResult?.detail).toBe(reason);
    expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeDisabled();
    expect(screen.queryByText('Previous pass.')).toBeNull();
  });

  it('shows fresh accepted source without stale styling or running validation after source cancels the old capture', () => {
    const rig = readyValidationPage();
    const target = { ...netTarget(), backendRevision: 2, sourceDigest: 'accepted-source', documentVersion: 2 };
    const source = netProjection({ revision: 2, sourceDigest: target.sourceDigest });
    act(() => {
      rig.emit(hostMessage('graphOpening', { state: { id: 'after-accept', target, phase: 'cancelled',
        cancellationReason: 'source_changed', hasCapture: false, detail: 'Source changed. Refresh the saved model to capture it.' } }));
      rig.emit(hostMessage('projection', { target, projection: source, traced: false, greyed: false }));
      rig.emit(hostMessage('syncState', { target, state: 'source-only' }));
    });
    expect(currentProjection()).toMatchObject({ sourceDigest: target.sourceDigest, traced: false, greyed: false });
    expect(screen.getByText('Not run — shapes read from the source')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Run model', exact: true })).toBeEnabled();
    expect((rig.transport.send as jest.Mock).mock.calls.some(([message]) => message.type === 'validateRequest' || message.type === 'graphSetupRequest')).toBe(false);
  });

  it('keeps current source visible when capture cannot start, without claiming historical context is drawn', () => {
    const rig = readyValidationPage();
    const source = netProjection({ document: draftFixture.after.document,
      debug: { previousDocument: captureDocument, nodePairs: [], changed: [] } });
    act(() => {
      rig.emit(hostMessage('projection', { target: netTarget(), projection: source, traced: false, greyed: false }));
      rig.emit(hostMessage('graphOpening', { state: { id: 'environment-failed', target: netTarget(), phase: 'failed',
        hasCapture: false, failure: { kind: 'environment' }, detail: 'Python environment is unavailable.' } }));
    });
    expect(currentProjection()).toMatchObject({ traced: false, greyed: false, document: source.document });
    expect(screen.getByText('Environment needs attention')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Python environment' })).toBeEnabled();
    expect(screen.queryByText(/Dashed context is from the previous run/)).toBeNull();
    expect(screen.queryByRole('button', { name: 'Show failure' })).toBeNull();
    expect((rig.transport.send as jest.Mock).mock.calls.some(([message]) => message.type === 'validateRequest' || message.type === 'graphSetupRequest')).toBe(false);
  });

  it('ignores a superseded target failure while validation runs on the current graph', async () => {
    const rig = readyValidationPage();
    fireEvent.click(screen.getByRole('button', { name: 'Run model', exact: true }));
    await act(async () => rig.emit(hostMessage('staleState', {
      target: { ...netTarget(), cls: 'OldModel' }, lastValidProjection: netProjection({ greyed: true }), reason: 'Old failure',
    })));
    expect(screen.getByRole('button', { name: 'Running…' })).toBeDisabled();
    expect(screen.queryByText(/Old failure/)).toBeNull();
    expect((rig.transport.send as jest.Mock).mock.calls.some(([message]) => message.type === 'validateCancel')).toBe(false);
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Cancel', exact: true })));
  });

  it('a dropped connection says so instead of pretending the graph is live', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);

    act(() => {
      rig.setStatus({ phase: 'reconnecting', hostLink: 'unknown', error: null });
    });

    expect(screen.getByText('Connection lost — reconnecting…')).toBeInTheDocument();

    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'disconnected', error: null });
    });

    expect(screen.getByText('The VS Code window closed or detached')).toBeInTheDocument();
  });

  it('an ended session replaces the canvas with the way back', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);

    act(() => {
      rig.setStatus({ phase: 'ended', hostLink: 'unknown', error: 'CREDENTIAL_EXPIRED' });
    });

    expect(screen.queryByTestId('canvas-app')).toBeNull();
    expect(
      screen.getByText(/session ended/, { exact: false }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Open Canvas in Browser/, { exact: false })).toBeInTheDocument();
  });

  it('unmounting disposes the transport: nothing leaks a broker connection', () => {
    const rig = makeTransport();
    const { unmount } = render(<LocalBrowserApp createTransport={() => rig.transport} />);

    unmount();

    expect(rig.transport.dispose).toHaveBeenCalledTimes(1);
  });

  it('keeps code-focus and selection requests current after a late setup update', () => {
    const rig = makeTransport();
    (rig.transport as { session: unknown }).session = { viewId: 'focus-view', workspaceSessionId: 'ws-test', capabilities: {} };
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    const target = { ...netTarget(), backendRevision: 2 };
    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', { target, projection: netProjection({ revision: 2 }), traced: false, greyed: false }));
      rig.emit(hostMessage('graphOpening', { state: {
        id: 'opening', target: netTarget(), phase: 'source_only', hasCapture: false, detail: 'Source preview',
      } }));
    });
    act(() => {
      window.dispatchEvent(new CustomEvent('tensorviz-selection', { detail: { nodeIds: ['n-1'], edgeIds: [] } }));
      mockCapturedLocalContext!.codeFocus!.change({ enabled: true, scope: 'node', opacity: 25, references: 'both' });
      reportNodeClick('n-1');
    });
    for (const type of ['viewSelection', 'codeFocusRequest', 'revealNode']) {
      expect(rig.transport.send).toHaveBeenCalledWith(expect.objectContaining({ type, target }));
    }
    const sent = (rig.transport.send as jest.Mock).mock.calls.map(call => call[0].type as string);
    const requestAt = sent.lastIndexOf('codeFocusRequest');
    expect(sent.lastIndexOf('viewSelection')).toBeLessThan(requestAt);
    act(() => rig.emit(hostMessage('codeFocusState', { viewId: 'focus-view', target,
      options: { enabled: true, scope: 'node', opacity: 25, references: 'both' }, parameter: false, detail: 'Selected source context.' })));
    expect(mockCapturedLocalContext!.codeFocus!.detail).toBe('Selected source context.');
  });

  it('a canvas activation leaves as a revealNode naming the exact identity on screen', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);

    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', {
        target: netTarget(),
        projection: netProjection(),
        traced: false,
        greyed: false,
      }));
    });

    act(() => {
      reportNodeClick('n-1');
    });

    expect(rig.transport.send).toHaveBeenCalledWith({
      type: 'revealNode',
      target: netTarget(),
      nodeId: 'n-1',
    });

    // The identity rolls forward with each projection; a reveal must name the
    // CURRENT identity, never what the page showed a revision ago, or the host
    // refuses the navigation as stale.
    const rolled = { ...netTarget(), backendRevision: 2, documentVersion: 12 };
    act(() => {
      rig.emit(hostMessage('projection', {
        target: rolled,
        projection: netProjection({ revision: 2 }),
        traced: false,
        greyed: false,
      }));
    });

    act(() => {
      reportNodeClick('n-2');
    });

    expect(rig.transport.send).toHaveBeenLastCalledWith({
      type: 'revealNode',
      target: rolled,
      nodeId: 'n-2',
    });
  });

  it("an editor's cursor focus arrives as the focus-node builder-command the framed bridge dispatches", () => {
    const rig = makeTransport();
    const focused: string[] = [];
    const listener = (event: Event) => {
      const detail = (event as CustomEvent).detail as
        | { command?: string; payload?: { nodeId?: string } }
        | undefined;
      if (detail?.command === 'focus-node' && detail.payload?.nodeId) {
        focused.push(detail.payload.nodeId);
      }
    };
    window.addEventListener('builder-command', listener);
    try {
      render(<LocalBrowserApp createTransport={() => rig.transport} />);

      act(() => {
        rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
        rig.emit(hostMessage('activeTarget', { target: netTarget() }));
        rig.emit(hostMessage('focusNode', { target: netTarget(), nodeId: 'n-1' }));
      });

      expect(focused).toEqual(['n-1']);
      // The strip is unmoved: focus carries no graph state and paints nothing.
      expect(screen.getByTitle('Net · models/net.py')).toBeInTheDocument();
    } finally {
      window.removeEventListener('builder-command', listener);
    }
  });

  it('shows structured resolution warnings and navigates only through their node id', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    const projection = netProjection({
      diagnostics: [{
        code: 'UNRESOLVED_DYNAMIC',
        severity: 'warning',
        file: 'models/dynamic.py',
        line: 9,
        import_text: 'self.inner = block_cls()',
        detail: 'The class is selected dynamically, so this instance remains opaque.',
        node: 'dyn.inner',
      }],
    });

    act(() => {
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', {
        target: netTarget(), projection, traced: false, greyed: false,
      }));
    });

    const toggle = screen.getByRole('button', { name: '1 source note' });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('Dynamic module')).toBeInTheDocument();
    expect(screen.getByText(/remains opaque/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Show source' }));
    expect(rig.transport.send).toHaveBeenCalledWith({
      type: 'revealNode',
      target: netTarget(),
      nodeId: 'dyn.inner',
    });
    const dispatch = jest.spyOn(window, 'dispatchEvent');
    try {
      fireEvent.click(screen.getByRole('button', { name: 'Locate in graph' }));
      expect(dispatch).toHaveBeenCalledWith(expect.objectContaining({ detail: {
        command: 'focus-node', payload: { nodeId: 'dyn.inner' },
      } }));
      expect(toggle).toHaveAttribute('aria-expanded', 'false');
      expect(toggle).toHaveFocus();
    } finally { dispatch.mockRestore(); }
  });

  it('hosts the canvas in the bounded content column, not a viewport fragment', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    // h-screen here was the failure: 100vh of canvas under the strip grew the
    // document and pushed the canvas toolbar out of reach of the window.
    expect(screen.getByTestId('canvas-app').closest('.local-browser-content')).toBeInTheDocument();
  });

  it('closes the warning drawer on Escape and hands focus back to its toggle', () => {
    const rig = makeTransport();
    render(<LocalBrowserApp createTransport={() => rig.transport} />);
    const projection = netProjection({
      diagnostics: [{
        code: 'UNRESOLVED_DYNAMIC',
        severity: 'warning',
        file: 'models/dynamic.py',
        line: 9,
        import_text: 'self.inner = block_cls()',
        detail: 'The class is selected dynamically, so this instance remains opaque.',
        node: 'dyn.inner',
      }],
    });

    act(() => {
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', {
        target: netTarget(), projection, traced: false, greyed: false,
      }));
    });

    const toggle = screen.getByRole('button', { name: '1 source note' });
    fireEvent.click(toggle);
    expect(screen.getByLabelText('Static graph resolution warnings')).toBeInTheDocument();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByLabelText('Static graph resolution warnings')).not.toBeInTheDocument();
    expect(toggle).toHaveFocus();
  });

  it('automatically focuses the outermost matching layer without opening an instance list', () => {
    const rig = makeTransport();
    const focused: string[] = [];
    const listener = (event: Event) => {
      const detail = (event as CustomEvent).detail as { command?: string; payload?: { nodeId?: string } };
      if (detail.command === 'focus-node' && detail.payload?.nodeId) focused.push(detail.payload.nodeId);
    };
    window.addEventListener('builder-command', listener);
    try {
      render(<LocalBrowserApp createTransport={() => rig.transport} />);
      act(() => {
        rig.emit(hostMessage('activeTarget', { target: netTarget() }));
        rig.emit(hostMessage('projection', { target: netTarget(), traced: true, greyed: false,
          projection: netProjection({ document: { semantic: { nodes: [
            { node_id: 'operation:a', kind: 'op', metadata: { parentId: 'invocation:b' } },
            { node_id: 'invocation:b', kind: 'module', metadata: { parentId: 'invocation:z' } },
            { node_id: 'invocation:z', kind: 'module', metadata: { parentId: null } },
          ] } } }),
        }));
      });
      act(() => {
        rig.emit(hostMessage('focusNode', { target: netTarget(), nodeId: 'operation:a',
          nodeIds: ['operation:a', 'invocation:b', 'invocation:z'] }));
      });

      expect(focused).toEqual(['invocation:z']);
      expect(screen.queryByText(/module instances/)).toBeNull();
      expect(screen.queryByRole('button', { name: 'invocation:z' })).toBeNull();
    } finally {
      window.removeEventListener('builder-command', listener);
    }
  });

  it('unmounting unregisters the reveal sink: a dead page navigates nothing', () => {
    const rig = makeTransport();
    const { unmount } = render(<LocalBrowserApp createTransport={() => rig.transport} />);

    act(() => {
      rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', {
        target: netTarget(),
        projection: netProjection(),
        traced: false,
        greyed: false,
      }));
    });

    unmount();

    // `currentProjection()` still holds this page's last delivery (module state
    // outlives the component), so only the sink cleanup can stop this send —
    // which is exactly what is being pinned.
    act(() => {
      reportNodeClick('n-1');
    });

    expect(rig.transport.send).not.toHaveBeenCalled();
  });

  it('shares graph controls without creating a built-in chat client', () => {
    const rig = readyValidationPage();
    expect(mockCapturedLocalContext?.page.activeTarget).toEqual(netTarget());
    expect(mockCapturedLocalContext).not.toHaveProperty('client');
    expect((rig.transport.send as jest.Mock).mock.calls.some(([message]) => message.type === 'askRequest')).toBe(false);
  });

});

it('offers Export graph in the toolbar once the current model opens, and no graph file viewer link', () => {
  const rig = makeTransport();
  render(<LocalBrowserApp createTransport={() => rig.transport} />);
  expect(screen.getByRole('button', { name: 'Export graph' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Model and file actions' }));
  expect(screen.queryByRole('link', { name: 'Open graph file viewer' })).not.toBeInTheDocument();
  const commands: string[] = [];
  const listener = (event: Event) => commands.push((event as CustomEvent).detail.command);
  window.addEventListener('builder-command', listener);
  try {
    act(() => {
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection(), traced: false, greyed: false }));
      rig.emit(hostMessage('syncState', { target: netTarget(), state: 'source-only' }));
    });
    const button = screen.getByRole('button', { name: 'Export graph' });
    expect(button).toBeEnabled(); fireEvent.click(button);
    expect(commands).toContain('export-graph-file');
    act(() => rig.emit(hostMessage('activeTarget', { target: { ...netTarget(), cls: 'OtherNet' } })));
    expect(screen.getByRole('button', { name: 'Export graph' })).toBeDisabled();
  } finally { window.removeEventListener('builder-command', listener); }
});


it('offers Publish in place of Export graph where VS Code can upload it', () => {
  const rig = makeTransport();
  render(<LocalBrowserApp createTransport={() => rig.transport} />);
  // The browser page has no VS Code host to upload through, so it keeps Export graph.
  expect(screen.queryByRole('button', { name: 'Publish graph' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Export graph' })).toBeInTheDocument();
  const active = jest.spyOn(bridge, 'bridgeActive').mockReturnValue(true);
  const commands: string[] = [];
  const listener = (event: Event) => commands.push((event as CustomEvent).detail.command);
  window.addEventListener('builder-command', listener);
  try {
    act(() => {
      rig.emit(hostMessage('activeTarget', { target: netTarget() }));
      rig.emit(hostMessage('projection', { target: netTarget(), projection: netProjection(), traced: false, greyed: false }));
      rig.emit(hostMessage('syncState', { target: netTarget(), state: 'source-only' }));
    });
    const publish = screen.getByRole('button', { name: 'Publish graph' });
    expect(publish).toHaveTextContent('Publish');
    expect(screen.queryByRole('button', { name: 'Export graph' })).not.toBeInTheDocument();
    expect(publish).toBeEnabled();
    fireEvent.click(publish);
    expect(commands).toContain('share-graph-file');
    expect(commands).not.toContain('export-graph-file');
  } finally {
    window.removeEventListener('builder-command', listener);
    active.mockRestore();
  }
});

it('renders project indexing failures and sends canvas expansion with the host project identity', () => {
  const rig = makeTransport();
  render(<LocalBrowserApp createTransport={() => rig.transport} />);
  const project = { workspaceFolder: '/work', projectRoot: '', experimentId: 'train', indexDigest: 'a'.repeat(64), indexRevision: 1 };
  act(() => {
    rig.emit(hostMessage('projectIndex', { project, index: { experiments: [], unreached: [], roots: [], truncated: false }, truncated: false }));
    rig.emit(hostMessage('projectDocument', { project, document: { experiment_id: 'train', nodes: [], edges: [] }, spanmap: {}, expanded: [] }));
  });
  act(() => reportProjectExpand('net', true));
  expect(rig.transport.send).toHaveBeenCalledWith({ type: 'projectExpand', project, nodeId: 'net', expanded: true });
  act(() => rig.emit(hostMessage('error', { code: 'INVALID_MESSAGE', message: 'Project graph could not be refreshed. Run Open Project as Graph to retry.' })));
  expect(screen.getByText(/Project graph could not be refreshed/)).toBeInTheDocument();
  const before = (rig.transport.send as jest.Mock).mock.calls.length;
  act(() => reportProjectExpand('net', false));
  expect((rig.transport.send as jest.Mock).mock.calls).toHaveLength(before);
});


it('dismisses model actions with Escape and outside clicks without sending commands', () => {
  const rig = makeTransport();
  render(<LocalBrowserApp createTransport={() => rig.transport} />);
  const trigger = screen.getByRole('button', { name: 'Model and file actions' });
  fireEvent.click(trigger);
  expect(screen.getByRole('group', { name: 'Model actions' })).toBeInTheDocument();
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(trigger).toHaveFocus();
  expect(screen.queryByRole('group', { name: 'Model actions' })).not.toBeInTheDocument();
  fireEvent.click(trigger);
  fireEvent.pointerDown(screen.getByTestId('canvas-app'));
  expect(trigger).toHaveAttribute('aria-expanded', 'false');
  expect(rig.transport.send).not.toHaveBeenCalled();
});

describe('draft graph preview', () => {
  const draftState = (extra: Record<string, unknown> = {}) => ({ ...JSON.parse(JSON.stringify(draftFixture)), id: 'draft-review',
    beforeLabel: 'Current', afterLabel: 'Draft', changedFiles: ['vision_models.py'], stale: false,
    proposal: { id: 'proposal-1' }, liveContext: { workspaceFolder: '/ws', file: 'models/net.py', cls: 'Net' }, ...extra });

  it('previews the source draft, never the retained capture, and routes code to the proposal diff', () => {
    const rig = readyValidationPage();
    const live = netProjection({ traced: true, document: captureDocument });
    act(() => rig.emit(hostMessage('projection', { target: netTarget(), projection: live, traced: true, greyed: false })));
    act(() => rig.emit(hostMessage('graphReviewState', { state: draftState() })));
    const projection = currentProjection() as any;
    expect(projection.capturedContext).toBeUndefined();
    expect(projection.proposal).toBe(true);
    expect(projection.reviewGraph.nodes.find((node: any) => node.id === 'review:after:fc2').data.diffStatus).toBe('added');
    const toolbar = screen.getByRole('region', { name: 'Draft graph preview' });
    expect(within(toolbar).getByText('Draft · Not run')).toBeVisible();
    expect(screen.queryByRole('region', { name: 'Graph diff review' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Close graph diff' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Refresh graph diff' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Run model' })).not.toBeInTheDocument();
    (rig.transport.send as jest.Mock).mockClear();
    fireEvent.click(within(toolbar).getByRole('button', { name: 'Next graph change' }));
    expect((rig.transport.send as jest.Mock).mock.calls.some(([message]) => message.action === 'reveal')).toBe(false);
    fireEvent.click(within(toolbar).getByRole('button', { name: 'Open proposed code diff' }));
    expect(rig.transport.send).toHaveBeenLastCalledWith({ type: 'graphReviewRequest', action: 'reveal', reviewId: 'draft-review', side: 'after' });
    act(() => rig.emit(hostMessage('graphReviewState', { state: null })));
    expect((currentProjection() as any).reviewGraph).toBeUndefined();
  });

  it('opens the draft at readable changed-node context instead of fitting the entire model', () => {
    jest.useFakeTimers();
    const commands: any[] = [];
    const listener = (event: Event) => commands.push((event as CustomEvent).detail);
    window.addEventListener('builder-command', listener);
    try {
      const rig = readyValidationPage();
      act(() => rig.emit(hostMessage('graphReviewState', { state: draftState() })));
      act(() => jest.advanceTimersByTime(100));
      expect(commands.find(command => command.command === 'focus-selection')).toBeUndefined();
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: currentProjection() })));
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: currentProjection() })));
      expect(commands.filter(command => command.command === 'focus-selection')).toHaveLength(1);
      const focus = commands.find(command => command.command === 'focus-selection');
      expect(focus.payload.nodeIds).toEqual(expect.arrayContaining(['review:after:fc2', 'review:after:output']));
      expect(focus.payload.nodeIds).not.toContain('review:root');
      expect(focus.payload.target).toMatchObject({ cls: 'ResNet152', file: 'vision_models.py', backendRevision: 0 });
      expect(commands.some(command => command.command === 'fit-view')).toBe(false);
    } finally { window.removeEventListener('builder-command', listener); jest.useRealTimers(); }
  });

  it.each(['capture-first', 'review-first', 'source-first', 'greyed-first'])('shows replacements one call at a time after %s reconnect', arrival => {
    jest.useFakeTimers();
    const commands: any[] = [];
    const listener = (event: Event) => commands.push((event as CustomEvent).detail);
    window.addEventListener('builder-command', listener);
    try {
      const rig = makeTransport();
      render(<LocalBrowserApp createTransport={() => rig.transport} />);
      const capture = nestedFixture.capture as ProjectionPayload;
      const target = { workspaceFolder: '/ws', file: 'model.py', cls: 'Net', documentVersion: 3, sourceDigest: capture.sourceDigest, backendRevision: capture.revision };
      const state = { ...JSON.parse(JSON.stringify({ before: nestedFixture.before, after: nestedFixture.after })), id: 'nested-draft',
        beforeLabel: 'Current code', afterLabel: 'Proposed code', changedFiles: ['model.py'], stale: false,
        proposal: { id: 'nested-draft' }, liveContext: { workspaceFolder: '/ws', file: 'model.py', cls: 'Net' } };
      if (arrival === 'source-first' || arrival === 'greyed-first') {
        act(() => {
          rig.emit(hostMessage('activeTarget', { target }));
          rig.emit(hostMessage('projection', { target, projection: arrival === 'source-first' ? nestedFixture.before : capture,
            traced: arrival === 'greyed-first', greyed: arrival === 'greyed-first' }));
        });
        act(() => rig.emit(hostMessage('graphReviewState', { state })));
        expect((currentProjection() as any).capturedContext).toBeUndefined();
      }
      if (arrival === 'review-first') act(() => rig.emit(hostMessage('graphReviewState', { state })));
      act(() => {
        rig.setStatus({ phase: 'connected', hostLink: 'connected', error: null });
        rig.emit(hostMessage('activeTarget', { target }));
        rig.emit(hostMessage('projection', { target, projection: capture, traced: true, greyed: false }));
        rig.emit(hostMessage('syncState', { target, state: 'runtime-traced' }));
      });
      if (arrival === 'capture-first') act(() => rig.emit(hostMessage('graphReviewState', { state })));
      const projection = currentProjection() as any;
      expect(projection).toMatchObject({ proposal: true, capturedContext: true, sourceDigest: nestedFixture.after.sourceDigest });
      expect(projection.reviewGraph.nodes.some((node: any) => node.data.sourceAssignment)).toBe(false);
      const toolbar = screen.getByRole('region', { name: 'Draft graph preview' });
      expect(within(toolbar).getByText('Draft · Not run')).toBeVisible();
      expect(within(toolbar).getByText('blocks[0].relu → nn.GELU · previous call 1/3')).toBeVisible();
      act(() => jest.advanceTimersByTime(100));
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: currentProjection() })));
      const nodes = (capture.document as any).semantic.nodes as any[];
      const id = (label: string) => nodes.find(node => node.metadata.captureLabel === label).node_id;
      const focus = commands.find(command => command.command === 'focus-selection');
      const calls = [1, 2, 3].map(i => id(`blocks.0.relu · call ${i}/3`));
      expect(focus.payload.nodeIds).toEqual([calls[0]]);
      expect(focus.payload.inset).toBe(24);
      fireEvent.click(within(toolbar).getByRole('button', { name: 'Next graph change' }));
      act(() => jest.advanceTimersByTime(100));
      expect(commands.filter(command => command.command === 'focus-selection').pop().payload.nodeIds).toEqual([calls[1]]);
      // Selecting a call on the canvas selects that call, not the change's first one.
      act(() => reportNodeClick(calls[2]));
      expect(within(toolbar).getByText('blocks[0].relu → nn.GELU · previous call 3/3')).toBeVisible();
      act(() => reportNodeClick(calls[1]));
      expect(within(toolbar).getByText('blocks[0].relu → nn.GELU · previous call 2/3')).toBeVisible();
      // Ancestors open around the call without forcing the whole block to fit.
      for (const ancestor of ['Net', 'blocks', 'blocks.0', 'aten.addmm.default']) expect(focus.payload.nodeIds).not.toContain(id(ancestor));
      expect(focus.payload.target).toEqual({ file: projection.file, cls: projection.cls, sourceDigest: projection.sourceDigest, backendRevision: projection.revision });
      expect(commands.some(command => command.command === 'fit-view')).toBe(false);
      const frozen = currentProjection();
      act(() => rig.emit(hostMessage('projection', { target, projection: { ...capture, revision: 900 }, traced: true, greyed: false })));
      expect(mockCapturedLocalContext?.page.projection?.revision).toBe(900);
      expect(currentProjection()).toBe(frozen);
      fireEvent.click(within(toolbar).getByRole('button', { name: 'Accept', exact: true }));
      act(() => rig.emit(hostMessage('graphReviewState', { state: null })));
      commands.length = 0;
      // Closing the draft can briefly redraw its old capture; wait for the exact accepted source.
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: capture })));
      expect(commands.some(command => command.command === 'focus-selection')).toBe(false);
      const accepted = nestedFixture.accepted as ProjectionPayload;
      for (const wrong of [{ sourceDigest: 'other-edit' }, { file: 'other.py' }, { cls: 'Other' }, { traced: true }]) {
        act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: { ...accepted, ...wrong } })));
        expect(commands.some(command => command.command === 'focus-selection')).toBe(false);
      }
      act(() => rig.emit(hostMessage('activeTarget', { target: { ...target, workspaceFolder: '/another-workspace' } })));
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: accepted })));
      expect(commands.some(command => command.command === 'focus-selection')).toBe(false);
      act(() => rig.emit(hostMessage('activeTarget', { target })));
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: accepted })));
      act(() => window.dispatchEvent(new CustomEvent('tensorviz-projection-rendered', { detail: accepted })));
      const restored = commands.filter(command => command.command === 'focus-selection');
      expect(restored).toHaveLength(1);
      expect(restored[0].payload.nodeIds).toEqual([calls[1]]);
      expect(restored[0].payload.target.backendRevision).toBe(accepted.revision);
    } finally { window.removeEventListener('builder-command', listener); jest.useRealTimers(); }
  });

  it('shows an honest unavailable view instead of an empty graph when the draft cannot be projected', () => {
    const rig = readyValidationPage();
    act(() => rig.emit(hostMessage('graphReviewState', { state: draftState({ proposal: { id: 'proposal-1', unavailable: true } }) })));
    expect(screen.getByText('Graph preview unavailable for this draft.')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Open proposed code diff' })).toBeVisible();
    expect(screen.queryByText('1 / 1')).toBeNull();
    expect((currentProjection() as any).reviewGraph).toBeUndefined();
    expect(document.querySelector('.graph-opening-canvas')).toHaveStyle({ visibility: 'hidden' });
  });

  const agentState = (revision: number, extra: Record<string, unknown>) => hostMessage('agentState', { viewId: 'agent-view', state: { sessionId: 'chat', revision,
    provider: 'codex', agents: { codex: { status: 'ready', detail: '', models: [], defaultModel: undefined } }, status: 'idle', activity: '', messages: [], ...extra } });
  const agentPage = () => {
    const rig = readyValidationPage();
    (rig.transport as { session: unknown }).session = { viewId: 'agent-view', capabilities: {} };
    return rig;
  };
  const sentMessages = (rig: ReturnType<typeof readyValidationPage>) => (rig.transport.send as jest.Mock).mock.calls.map(([message]) => message);

  it('keeps the agent conversation and its model while a draft is pending, and Enter revises that draft', () => {
    const rig = agentPage();
    act(() => rig.emit(agentState(1, { status: 'running', activity: 'Editing model…', messages: [{ id: 'turn-1', role: 'user', text: 'Add a second classifier head' }] })));
    fireEvent.click(screen.getByRole('button', { name: /^Open agent chat/ }));
    act(() => rig.emit(hostMessage('graphReviewState', { state: draftState() })));
    expect(screen.getByText('Add a second classifier head')).toBeVisible();
    expect(screen.getByText('Editing model…')).toBeVisible();
    act(() => rig.emit(agentState(2, { messages: [{ id: 'turn-1', role: 'user', text: 'Add a second classifier head' }, { id: 'turn-2', role: 'assistant', text: 'Proposed fc2.' }] })));
    expect(screen.getByText('Proposed fc2.')).toBeVisible();
    const input = screen.getByRole('textbox', { name: 'Message to coding agent' });
    expect(input).toHaveAttribute('placeholder', 'Describe a revision to the draft…');
    fireEvent.change(input, { target: { value: 'add a sigmoid at end' } });
    expect(screen.getByRole('button', { name: 'Send request' })).toBeEnabled();
    fireEvent.keyDown(input, { key: 'Enter' });
    // The revision names the draft; the model identity stays the live target, and nothing is accepted.
    const agentActions = sentMessages(rig).filter(message => message.type === 'agentRequest');
    expect(agentActions.filter(message => message.action === 'send')).toEqual([expect.objectContaining({ prompt: 'add a sigmoid at end', draftId: 'proposal-1', target: netTarget() })]);
    expect(sentMessages(rig).some(message => message.type === 'graphReviewRequest' && message.action === 'acceptDraft')).toBe(false);
    act(() => rig.emit(hostMessage('graphReviewState', { state: null })));
    expect(screen.getByText('Proposed fc2.')).toBeVisible();
    expect(screen.getByRole('textbox')).toHaveAttribute('placeholder', 'Ask or describe a change…');
  });

  it('Accept and Discard send one exact draft decision until the host answers', () => {
    const rig = agentPage();
    act(() => rig.emit(hostMessage('graphReviewState', { state: draftState() })));
    const decision = screen.getByRole('group', { name: 'Draft decision' });
    (rig.transport.send as jest.Mock).mockClear();
    fireEvent.click(within(decision).getByRole('button', { name: 'Accept' }));
    fireEvent.click(within(decision).getByRole('button', { name: 'Accepting…' }));
    fireEvent.click(within(decision).getByRole('button', { name: 'Discard' }));
    expect(sentMessages(rig)).toEqual([{ type: 'graphReviewRequest', action: 'acceptDraft', reviewId: 'draft-review' }]);
    // A refused Accept republishes the draft; Discard is then available once.
    act(() => rig.emit(hostMessage('graphReviewState', { state: draftState() })));
    fireEvent.click(within(screen.getByRole('group', { name: 'Draft decision' })).getByRole('button', { name: 'Discard' }));
    fireEvent.click(within(screen.getByRole('group', { name: 'Draft decision' })).getByRole('button', { name: 'Discarding…' }));
    expect(sentMessages(rig).slice(1)).toEqual([{ type: 'graphReviewRequest', action: 'discardDraft', reviewId: 'draft-review' }]);
    act(() => rig.emit(hostMessage('graphReviewState', { state: null })));
    expect(screen.queryByRole('group', { name: 'Draft decision' })).toBeNull();
  });

  it('a stale draft cannot be accepted or revised, but can be discarded', () => {
    const rig = agentPage();
    act(() => rig.emit(agentState(1, {})));
    fireEvent.click(screen.getByRole('button', { name: /^Open agent chat/ }));
    act(() => rig.emit(hostMessage('graphReviewState', { state: draftState({ stale: true }) })));
    expect(screen.getByRole('button', { name: 'Accept' })).toBeDisabled();
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'add dropout' } });
    expect(screen.getByRole('button', { name: 'Send request' })).toBeDisabled();
    expect(screen.getByText('Source changed since this draft. Discard it to send a new request.')).toBeVisible();
    (rig.transport.send as jest.Mock).mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Discard' }));
    expect(sentMessages(rig)).toEqual([{ type: 'graphReviewRequest', action: 'discardDraft', reviewId: 'draft-review' }]);
  });

  it('Continue editing focuses the minimized chat with its text kept, sending nothing', () => {
    const rig = agentPage();
    act(() => rig.emit(agentState(1, {})));
    fireEvent.click(screen.getByRole('button', { name: /^Open agent chat/ }));
    act(() => rig.emit(hostMessage('graphReviewState', { state: draftState() })));
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'add a sigmoid' } });
    fireEvent.click(screen.getByRole('button', { name: 'Minimize agent chat' }));
    (rig.transport.send as jest.Mock).mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Continue editing' }));
    expect(screen.getByRole('textbox')).toHaveValue('add a sigmoid');
    expect(screen.getByRole('textbox')).toHaveFocus();
    expect(sentMessages(rig).filter(message => message.action !== 'discover')).toEqual([]);
  });

  it('typed drafts carry inspection context without offering a workspace-code revision', () => {
    const rig = agentPage();
    act(() => rig.emit(agentState(1, {})));
    act(() => rig.emit(hostMessage('graphReviewState', { state: draftState({ proposal: { id: 'proposal-1', canRevise: false } }) })));
    expect(screen.getByRole('button', { name: 'Accept' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Discard' })).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Continue editing' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /^Open agent chat/ }));
    expect(screen.getByRole('textbox')).toHaveAttribute('placeholder', 'Ask about this draft…');
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Inspect the retained candidate result.' } });
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
    expect(sentMessages(rig).filter(message => message.type === 'agentRequest' && message.action === 'send'))
      .toEqual([expect.objectContaining({ draftId: 'proposal-1', target: netTarget(), prompt: 'Inspect the retained candidate result.' })]);
    expect(sentMessages(rig).some(message => message.type === 'graphReviewRequest')).toBe(false);
  });

  it('keeps candidate controls mounted for typed drafts, with their existing lifecycle', () => {
    const panel = jest.spyOn(require('../../ide/CandidateReviewPanel'), 'CandidateReviewPanel')
      .mockImplementation(() => <div data-testid="candidate-controls" />);
    try {
      const rig = agentPage();
      act(() => rig.emit(hostMessage('graphReviewState', { state: draftState({ proposal: { id: 'typed', canRevise: false } }) })));
      expect(screen.getByTestId('candidate-controls')).toBeInTheDocument();
      act(() => rig.emit(hostMessage('graphReviewState', { state: draftState() })));
      expect(screen.queryByTestId('candidate-controls')).toBeNull();
    } finally { panel.mockRestore(); }
  });

  it('a Git review offers no draft decisions and leaves the chat without a draft', () => {
    const rig = agentPage();
    act(() => rig.emit(hostMessage('graphReviewState', { state: draftState({ proposal: undefined }) })));
    expect(screen.getByRole('region', { name: 'Graph diff review' })).toBeVisible();
    expect(screen.queryByRole('group', { name: 'Draft decision' })).toBeNull();
    for (const name of ['Accept', 'Discard', 'Continue editing']) expect(screen.queryByRole('button', { name })).toBeNull();
  });
});
