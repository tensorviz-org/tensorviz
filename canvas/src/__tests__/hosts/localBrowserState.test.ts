/** Step-5 page state: the rules that keep a live graph honest.
 *
 * What is pinned here, in product language:
 *  - the page is only ever the external browser page when it is NOT framed and
 *    carries a launch fragment or a remembered session id — a bare visit to the
 *    backend origin must keep behaving as before;
 *  - a refreshed view renders exactly the snapshot the host answered with;
 *  - no late or cross-target message paints another class's graph under the
 *    active class's name;
 *  - a broken buffer dims the graph with its reason, a fix un-dims it, and the
 *    sync label always answers "where did these shapes come from".
 */

import {
  EMPTY_LOCAL_BROWSER_PAGE_STATE,
  LocalBrowserPageState,
  reduceLocalBrowserViewMessage,
} from '../../ide/localBrowserState';
import {
  LOCAL_BROWSER_PROTOCOL_VERSION,
  LocalBrowserTargetIdentity,
  ProjectionPayload,
} from '../../ide/protocol.generated';
import { LocalBrowserViewMessage } from '../../ide/localBrowserViewTransport';

const FOLDER = '/ws';

/** The Net identity at a backend revision; digest/version roll with it. */
function net(revision: number): LocalBrowserTargetIdentity {
  return {
    workspaceFolder: FOLDER,
    file: 'models/net.py',
    cls: 'Net',
    documentVersion: revision + 10,
    sourceDigest: `digest-${revision}`,
    backendRevision: revision,
  };
}

/** A different class in the same file: the retarget the reducer must not mix. */
function gate(revision: number): LocalBrowserTargetIdentity {
  return { ...net(revision), cls: 'Gate' };
}

function projectionPayload(
  revision: number,
  cls: string,
  extra: Partial<ProjectionPayload> = {},
): ProjectionPayload {
  return {
    graphId: `g-${cls}-${revision}`,
    revision,
    cls,
    file: 'models/net.py',
    sourceDigest: `digest-${revision}`,
    traced: false,
    greyed: false,
    document: { view: {}, semantic: { nodes: [] } },
    viewRestored: false,
    spanmap: {},
    ...extra,
  };
}

let sequence = 0;

/** A host message with the envelope frozen; the wire fields are constant here
 *  because ordering/dedup is the transport's pinned job, not the reducer's. */
function hostMessage<T extends string, Rest extends Record<string, unknown>>(
  type: T,
  rest: Rest,
): LocalBrowserViewMessage {
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

/** Apply a list of messages in order, the way the transport delivers them. */
function applyAll(
  messages: LocalBrowserViewMessage[],
  start: LocalBrowserPageState = EMPTY_LOCAL_BROWSER_PAGE_STATE,
): LocalBrowserPageState {
  return messages.reduce(reduceLocalBrowserViewMessage, start);
}

describe('reduceLocalBrowserViewMessage', () => {
  it.each(['resolving', 'awaiting_execution', 'source_only', 'ready'])(
    'keeps selection and code-focus requests on the projected revision during %s setup updates', phase => {
      const target = net(2);
      const setupTarget = { ...target, backendRevision: 0 };
      const projection = projectionPayload(2, 'Net');
      const opening = { id: 'opening', target: setupTarget, phase, hasCapture: false, detail: 'Setup update' };
      const state = applyAll([
        hostMessage('activeTarget', { target: setupTarget }),
        hostMessage('projection', { target, projection, traced: false, greyed: false }),
        hostMessage('syncState', { target, state: 'source-only' }),
        hostMessage('graphOpening', { state: opening }),
      ]);

      // Code focus and node reveal use this exact identity; rolling it back
      // makes the host reject an otherwise current source selection.
      expect(state.activeTarget).toEqual(target);
      expect(state.projection).toBe(projection);
      expect(state.opening).toBe(opening);
      expect(state.opening?.target).toEqual(setupTarget);
    },
  );

  it.each([
    { ...net(2), backendRevision: 3 },
    { ...net(2), backendRevision: 0, sourceDigest: 'new-source' },
    { ...net(2), backendRevision: 0, documentVersion: 13 },
    gate(0),
    { ...net(0), file: 'other.py' },
    { ...net(0), workspaceFolder: '/other' },
  ])('accepts a new setup target without borrowing the previous graph identity: %j', target => {
    const state = applyAll([
      hostMessage('activeTarget', { target: net(2) }),
      hostMessage('graphOpening', { state: {
        id: 'opening', target, phase: 'resolving', hasCapture: false, detail: 'Opening model',
      } }),
    ]);
    expect(state.activeTarget).toEqual(target);
  });

  it('restores a first-load failure without inventing a projection and clears it on recovery', () => {
    const failed = applyAll([hostMessage('snapshot', { state: {
      activeTarget: net(0), projection: null, traced: false, greyed: true,
      syncState: 'stale', reason: 'Graph refresh timed out. Reopen from VS Code.', connectionState: 'connected',
    } })]);
    expect(failed.projection).toBeNull();
    expect(failed.syncState).toBe('stale');
    expect(failed.greyed).toBe(true);
    const fresh = applyAll([
      hostMessage('projection', { target: net(1), projection: projectionPayload(1, 'Net'), traced: false, greyed: false }),
      hostMessage('syncState', { target: net(1), state: 'source-only' }),
    ], failed);
    expect(fresh.greyed).toBe(false);
    expect(fresh.reason).toBeNull();
    expect(fresh.syncState).toBe('source-only');
  });

  it('dims a retained canvas when a new target fails before its first projection', () => {
    const failed = applyAll([
      hostMessage('activeTarget', { target: net(1) }),
      hostMessage('projection', { target: net(1), projection: projectionPayload(1, 'Net'), traced: false, greyed: false }),
      hostMessage('activeTarget', { target: gate(0) }),
      hostMessage('syncState', { target: gate(0), state: 'stale', reason: 'Server unavailable' }),
    ]);
    expect(failed.greyed).toBe(true);
    expect(failed.projection?.greyed).toBe(true);
    expect(failed.syncState).toBe('stale');
  });

  it('a snapshot replaces everything — a refresh never merges with a dead epoch', () => {
    const dirty = applyAll([
      hostMessage('activeTarget', { target: net(1) }),
      hostMessage('projection', {
        target: net(1),
        projection: projectionPayload(1, 'Net'),
        traced: false,
        greyed: false,
      }),
      hostMessage('syncState', { target: net(1), state: 'source-only' }),
    ]);
    expect(dirty.projection).not.toBeNull();

    const restored = reduceLocalBrowserViewMessage(
      dirty,
      hostMessage('snapshot', {
        role: 'host',
        state: {
          activeTarget: null,
          projection: null,
          traced: false,
          greyed: false,
          syncState: null,
          connectionState: 'connected',
        },
      }),
    );

    expect(restored).toEqual(EMPTY_LOCAL_BROWSER_PAGE_STATE);
  });

  it('a stale snapshot restores the greyed graph WITH its reason, atomically', () => {
    const state = applyAll([
      hostMessage('snapshot', {
        role: 'host',
        state: {
          activeTarget: net(2),
          projection: projectionPayload(2, 'Net', { greyed: true }),
          traced: false,
          greyed: true,
          syncState: 'stale',
          reason: 'models/net.py:4 does not parse (expected colon)',
          connectionState: 'connected',
        },
      }),
    ]);

    expect(state.syncState).toBe('stale');
    expect(state.greyed).toBe(true);
    expect(state.reason).toContain('models/net.py:4');
    expect(state.projection?.greyed).toBe(true);
    expect(state.activeTarget?.backendRevision).toBe(2);
  });

  it('a retarget keeps the graph on screen but drops every claim about it', () => {
    const state = applyAll([
      hostMessage('activeTarget', { target: net(1) }),
      hostMessage('projection', {
        target: net(1),
        projection: projectionPayload(1, 'Net', { traced: true }),
        traced: true,
        greyed: false,
      }),
      hostMessage('syncState', { target: net(1), state: 'runtime-traced' }),
    ]);

    const retargeted = reduceLocalBrowserViewMessage(
      state,
      hostMessage('activeTarget', { target: gate(0) }),
    );

    expect(retargeted.activeTarget?.cls).toBe('Gate');
    expect(retargeted.projection?.cls).toBe('Net', 'no canvas flash: the old graph stays until the new one lands');
    expect(retargeted.syncState).toBeNull();
    expect(retargeted.reason).toBeNull();
  });

  it('the same target re-anchored keeps its claims; identity rolls forward', () => {
    const state = applyAll([
      hostMessage('activeTarget', { target: net(0) }),
      hostMessage('projection', {
        target: net(1),
        projection: projectionPayload(1, 'Net'),
        traced: false,
        greyed: false,
      }),
      hostMessage('syncState', { target: net(1), state: 'source-only' }),
    ]);

    const reanchored = reduceLocalBrowserViewMessage(
      state,
      hostMessage('activeTarget', { target: net(1) }),
    );

    expect(reanchored.syncState).toBe('source-only');
    expect(reanchored.projection).not.toBeNull();
  });

  it('a projection for a class the user already left is dropped, name and all', () => {
    const state = applyAll([
      hostMessage('activeTarget', { target: gate(0) }),
    ]);

    const late = reduceLocalBrowserViewMessage(
      state,
      hostMessage('projection', {
        target: net(1),
        projection: projectionPayload(1, 'Net'),
        traced: false,
        greyed: false,
      }),
    );

    expect(late).toBe(state);
  });

  it('a fresh projection un-dims and clears the reason; the label re-arms', () => {
    const stale = applyAll([
      hostMessage('activeTarget', { target: net(1) }),
      hostMessage('staleState', {
        target: net(1),
        lastValidProjection: projectionPayload(1, 'Net', { greyed: true }),
        reason: 'models/net.py:4 does not parse (expected colon)',
      }),
      hostMessage('syncState', {
        target: net(1),
        state: 'stale',
        reason: 'models/net.py:4 does not parse (expected colon)',
      }),
    ]);
    expect(stale.syncState).toBe('stale');
    expect(stale.greyed).toBe(true);

    const fixed = applyAll(
      [
        hostMessage('projection', {
          target: net(2),
          projection: projectionPayload(2, 'Net'),
          traced: false,
          greyed: false,
        }),
        hostMessage('syncState', { target: net(2), state: 'source-only' }),
      ],
      stale,
    );

    expect(fixed.greyed).toBe(false);
    expect(fixed.reason).toBeNull();
    expect(fixed.syncState).toBe('source-only');
    expect(fixed.projection?.revision).toBe(2);
  });

  it('a stale message swaps in the last-valid projection, greyed, with the line', () => {
    const state = applyAll([
      hostMessage('activeTarget', { target: net(1) }),
      hostMessage('projection', {
        target: net(1),
        projection: projectionPayload(1, 'Net', { traced: true }),
        traced: true,
        greyed: false,
      }),
      hostMessage('syncState', { target: net(1), state: 'runtime-traced' }),
    ]);

    const stale = applyAll(
      [
        hostMessage('staleState', {
          target: net(1),
          lastValidProjection: projectionPayload(1, 'Net', { traced: true, greyed: true }),
          reason: 'models/net.py:4 does not parse (expected colon)',
        }),
      ],
      state,
    );

    expect(stale.syncState).toBe('stale');
    expect(stale.greyed).toBe(true);
    expect(stale.traced).toBe(true, 'the greyed graph keeps its measured tier');
    expect(stale.reason).toContain('net.py:4');
  });

  it('syncState and staleState for another class are refused identically', () => {
    const state = applyAll([hostMessage('activeTarget', { target: net(1) })]);

    const wrongSync = reduceLocalBrowserViewMessage(
      state,
      hostMessage('syncState', { target: gate(0), state: 'runtime-traced' }),
    );
    const wrongStale = reduceLocalBrowserViewMessage(
      state,
      hostMessage('staleState', {
        target: gate(0),
        lastValidProjection: projectionPayload(0, 'Gate'),
        reason: 'nope',
      }),
    );

    expect(wrongSync).toBe(state);
    expect(wrongStale).toBe(state);
  });

  it('navigation cues, acks and errors carry no graph state', () => {
    const state = applyAll([
      hostMessage('activeTarget', { target: net(1) }),
      hostMessage('syncState', { target: net(1), state: 'source-only' }),
    ]);

    for (const noise of [
      hostMessage('focusNode', { target: net(1), nodeId: 'n1' }),
      hostMessage('ack', { target: net(1), ackId: 'a1', ok: true }),
      hostMessage('error', { code: 'STALE_TARGET', message: 'drift', target: net(1) }),
    ]) {
      expect(reduceLocalBrowserViewMessage(state, noise)).toBe(state);
    }
  });
});

describe('reduceLocalBrowserViewMessage — the project plane', () => {
  const project = {
    workspaceFolder: '/w',
    projectRoot: '',
    experimentId: 'exp:1',
    indexDigest: 'a'.repeat(64),
    indexRevision: 1,
  };
  const summary = { experiments: [], unreached: [], roots: [''], truncated: false };

  function reduce(messages: any[]) {
    return messages.reduce(
      (state, message) => reduceLocalBrowserViewMessage(state, message),
      EMPTY_LOCAL_BROWSER_PAGE_STATE,
    );
  }

  it('an index fills the rail without clearing the canvas', () => {
    // A re-index must not blank the graph for the gap before the document lands,
    // the same rule a class retarget follows.
    const state = reduce([
      { type: 'projectIndex', project, index: summary, truncated: false },
      { type: 'projectDocument', project, document: { nodes: [] }, spanmap: {}, expanded: ['net'] },
      { type: 'projectIndex', project, index: summary, truncated: false },
    ]);
    expect(state.projectIndex).toEqual(summary);
    expect(state.projection).not.toBeNull();
  });

  it('a document for a project the page has moved on from is dropped', () => {
    const stale = { ...project, indexDigest: 'b'.repeat(64) };
    const state = reduce([
      { type: 'projectIndex', project, index: summary, truncated: false },
      { type: 'projectDocument', project: stale, document: { nodes: ['x'] }, spanmap: {}, expanded: [] },
    ]);
    expect(state.projection).toBeNull();
  });

  it('carries the expansion set the host echoed, so the two cannot disagree', () => {
    const state = reduce([
      { type: 'projectIndex', project, index: summary, truncated: false },
      { type: 'projectDocument', project, document: { nodes: [] }, spanmap: {}, expanded: ['a', 'b'] },
    ]);
    expect(state.expanded).toEqual(['a', 'b']);
  });

  it('a snapshot clears the rail rather than keeping a dead epoch', () => {
    const state = reduce([
      { type: 'projectIndex', project, index: summary, truncated: false },
      {
        type: 'snapshot',
        state: {
          activeTarget: null,
          projection: null,
          traced: false,
          greyed: false,
          syncState: null,
          connectionState: 'connected',
        },
      },
    ]);
    expect(state.projectIndex).toBeNull();
    expect(state.project).toBeNull();
    expect(state.expanded).toEqual([]);
  });
});

describe('project identity and failure state', () => {
  const project = { workspaceFolder: '/work', projectRoot: '', experimentId: 'train', indexDigest: 'a'.repeat(64), indexRevision: 2 };
  const index = { experiments: [], unreached: [], roots: [], truncated: false };
  it('rejects documents from another workspace or revision even with an identical digest', () => {
    const state = reduceLocalBrowserViewMessage(EMPTY_LOCAL_BROWSER_PAGE_STATE,
      hostMessage('projectIndex', { project, index, truncated: false }));
    for (const other of [{ ...project, workspaceFolder: '/other' }, { ...project, indexRevision: 1 }]) {
      expect(reduceLocalBrowserViewMessage(state, hostMessage('projectDocument', {
        project: other, document: {}, spanmap: {}, expanded: [],
      }))).toBe(state);
    }
  });
  it('shows a first-load project error after replay and clears it on recovery', () => {
    const failed = reduceLocalBrowserViewMessage(EMPTY_LOCAL_BROWSER_PAGE_STATE,
      hostMessage('error', { code: 'INVALID_MESSAGE', message: 'Run Open Project as Graph to retry.' }));
    expect(failed).toMatchObject({ syncState: 'stale', greyed: true, reason: 'Run Open Project as Graph to retry.' });
    const recovered = reduceLocalBrowserViewMessage(failed, hostMessage('projectIndex', { project, index, truncated: false }));
    expect(recovered.reason).toBeNull();
    expect(recovered.greyed).toBe(false);
  });
});
