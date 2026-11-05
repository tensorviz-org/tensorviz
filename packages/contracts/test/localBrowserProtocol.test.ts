import * as assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  LOCAL_BROWSER_ERROR_CODES,
  LOCAL_BROWSER_MESSAGE_TYPES,
  LOCAL_BROWSER_PROTOCOL_VERSION,
  LOCAL_BROWSER_ROLES,
  isLocalBrowserMessage,
  isGraphEnvironmentRecovery,
  isGraphSetupAction,
  isGraphFailure,
  isModelImportDiagnostic,
  isCanvasAgentState,
  type ModelImportDiagnostic,
} from '@tensorviz/contracts/protocol';

const fixturePath = join(__dirname, '../../../tests/fixtures/contracts/local-browser-protocol.json');
const corpus = JSON.parse(readFileSync(fixturePath, 'utf8')) as {
  expectedWorkspaceSessionId: string;
  valid: unknown[];
  invalid: unknown[];
};

suite('local browser protocol', () => {
  const importMessage = "cannot import name 'ALL_ATTENTION_FUNCTIONS' from 'transformers.modeling_utils'";
  const importDiagnostic: ModelImportDiagnostic = {
    exceptionType: 'ImportError', message: importMessage,
    traceback: `Traceback (most recent call last):\n  File "/project/model.py", line 4, in <module>\nImportError: ${importMessage}`,
    causes: [{ exceptionType: 'ImportError', message: importMessage, module: 'transformers.modeling_utils' }],
    runtime: { platform: 'darwin', architecture: 'arm64', python: '3.12.13', torch: null, cuda: null, cudaAvailable: null, mpsAvailable: null },
    truncated: false,
  };
  const importFailure = { kind: 'environment', code: 'MODEL_IMPORT_FAILED', importFailure: true, diagnostic: importDiagnostic };
  const failedAgent = { sessionId: 'chat', revision: 1, provider: 'codex', status: 'error', activity: importMessage, messages: [], failure: importFailure };
  const agentWire = (state: unknown) => ({ type: 'agentState', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'evidence', viewId: 'view', state });

  test('retained import evidence survives the wire exactly, including unknown runtime observations', () => {
    const received = JSON.parse(JSON.stringify(agentWire(failedAgent)));
    assert.ok(isModelImportDiagnostic(received.state.failure.diagnostic));
    assert.ok(isGraphFailure(received.state.failure)); assert.ok(isCanvasAgentState(received.state));
    assert.ok(isLocalBrowserMessage(received)); assert.deepEqual(received, agentWire(failedAgent));
    assert.equal(received.state.failure.diagnostic.message, importMessage);
    assert.equal(received.state.failure.diagnostic.traceback, importDiagnostic.traceback);
    assert.deepEqual(received.state.failure.diagnostic.runtime, importDiagnostic.runtime);
    const observed = { ...importDiagnostic, runtime: { ...importDiagnostic.runtime, torch: '2.6.0', cuda: null, cudaAvailable: false, mpsAvailable: true } };
    assert.ok(isModelImportDiagnostic(observed));
    assert.ok(isLocalBrowserMessage(agentWire({ ...failedAgent, failure: { ...importFailure, diagnostic: observed } })));
  });

  test('import diagnostic text and cause bounds accept the limit and reject larger or malformed evidence', () => {
    const maximum = { ...importDiagnostic, exceptionType: 'E'.repeat(128), message: 'm'.repeat(2048), traceback: 't'.repeat(12000),
      causes: Array.from({ length: 8 }, () => ({ exceptionType: 'E'.repeat(128), message: 'm'.repeat(2048), module: 'p'.repeat(256) })) };
    assert.ok(isModelImportDiagnostic(maximum));
    const malformed: Array<[string, unknown]> = [
      ['absent', undefined], ['null', null], ['array', []],
      ['missing field', { ...importDiagnostic, traceback: undefined }], ['extra field', { ...importDiagnostic, explanation: 'unobserved' }],
      ['empty exception', { ...importDiagnostic, exceptionType: '' }], ['long exception', { ...importDiagnostic, exceptionType: 'E'.repeat(129) }],
      ['long message', { ...importDiagnostic, message: 'm'.repeat(2049) }], ['non-text message', { ...importDiagnostic, message: {} }],
      ['long traceback', { ...importDiagnostic, traceback: 't'.repeat(12001) }], ['non-text traceback', { ...importDiagnostic, traceback: [] }],
      ['non-boolean truncation', { ...importDiagnostic, truncated: 'false' }], ['too many causes', { ...maximum, causes: [...maximum.causes, maximum.causes[0]] }],
      ['non-array causes', { ...importDiagnostic, causes: {} }], ['null cause', { ...importDiagnostic, causes: [null] }],
      ['cause missing type', { ...importDiagnostic, causes: [{ message: importMessage }] }],
      ['cause long type', { ...importDiagnostic, causes: [{ exceptionType: 'E'.repeat(129), message: '' }] }],
      ['cause long message', { ...importDiagnostic, causes: [{ exceptionType: 'ImportError', message: 'm'.repeat(2049) }] }],
      ['cause invalid module', { ...importDiagnostic, causes: [{ exceptionType: 'ImportError', message: '', module: 7 }] }],
      ['cause long module', { ...importDiagnostic, causes: [{ exceptionType: 'ImportError', message: '', module: 'p'.repeat(257) }] }],
      ['cause extra field', { ...importDiagnostic, causes: [{ exceptionType: 'ImportError', message: '', command: 'install something' }] }],
    ];
    for (const [name, diagnostic] of malformed) {
      assert.equal(isModelImportDiagnostic(diagnostic), false, name);
      // A malformed nested value must not be admitted through the enclosing wire message.
      if (diagnostic !== undefined) assert.equal(isLocalBrowserMessage(agentWire({ ...failedAgent, failure: { ...importFailure, diagnostic } })), false, name);
    }
  });

  test('runtime observations require the exact typed fields and retain explicit unknowns', () => {
    const runtime = importDiagnostic.runtime;
    assert.ok(isModelImportDiagnostic({ ...importDiagnostic, runtime: { platform: 'p'.repeat(128), architecture: 'a'.repeat(128), python: 'v'.repeat(128),
      torch: 't'.repeat(128), cuda: 'c'.repeat(128), cudaAvailable: true, mpsAvailable: false } }));
    const malformed = [null, [], { ...runtime, python: undefined }, { ...runtime, extra: 'unsupported inference' },
      { ...runtime, platform: '' }, { ...runtime, architecture: 'a'.repeat(129) }, { ...runtime, python: 3.12 },
      { ...runtime, torch: '' }, { ...runtime, torch: 't'.repeat(129) }, { ...runtime, cuda: 12 },
      { ...runtime, cudaAvailable: 'false' }, { ...runtime, mpsAvailable: 0 }];
    for (const observed of malformed) {
      assert.equal(isModelImportDiagnostic({ ...importDiagnostic, runtime: observed }), false, JSON.stringify(observed));
      assert.equal(isLocalBrowserMessage(agentWire({ ...failedAgent, failure: { ...importFailure,
        diagnostic: { ...importDiagnostic, runtime: observed } } })), false, JSON.stringify(observed));
    }
  });

  test('structured import diagnosis belongs only to environment failures', () => {
    assert.ok(isGraphFailure(importFailure));
    const opening = { type: 'graphOpening', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'opening', hostSequence: 1,
      state: { id: 'opening', target: { workspaceFolder: '/project', file: 'model.py', cls: 'Net', documentVersion: 1,
        sourceDigest: 'a'.repeat(64), backendRevision: 1 }, phase: 'failed', detail: importMessage, hasCapture: false, failure: importFailure } };
    assert.equal(isLocalBrowserMessage(opening), true);
    for (const kind of ['setup', 'source', 'execution', 'capture']) {
      assert.ok(isGraphFailure({ kind }));
      assert.equal(isGraphFailure({ kind, diagnostic: importDiagnostic }), false, kind);
      assert.equal(isLocalBrowserMessage({ ...opening, state: { ...opening.state, failure: { kind, diagnostic: importDiagnostic } } }), false, kind);
    }
    for (const failure of [{ ...importFailure, diagnostic: null }, { ...importFailure, importFailure: false },
      { ...importFailure, code: 'MODEL IMPORT FAILED' }, { ...importFailure, retryCommand: 'pip install arbitrary' }]) {
      assert.equal(isGraphFailure(failure), false); assert.equal(isLocalBrowserMessage(agentWire({ ...failedAgent, failure })), false);
    }
  });

  test('retained agent failures are valid only in terminal error state and remain optional for older hosts', () => {
    for (const status of ['idle', 'running', 'stopped', 'error']) {
      const state = { ...failedAgent, status };
      assert.equal(isCanvasAgentState(state), status === 'error', status);
      assert.equal(isLocalBrowserMessage(agentWire(state)), status === 'error', status);
      const { failure: _failure, ...legacy } = state;
      assert.ok(isCanvasAgentState(legacy), `legacy ${status}`); assert.ok(isLocalBrowserMessage(agentWire(legacy)), `legacy ${status}`);
    }
    for (const failure of [null, [], 'ImportError', { ...importFailure, diagnostic: { ...importDiagnostic, traceback: 42 } }]) {
      assert.equal(isCanvasAgentState({ ...failedAgent, failure }), false);
      assert.equal(isLocalBrowserMessage(agentWire({ ...failedAgent, failure })), false);
    }
  });

  test('draft revision capability is optional for legacy code drafts and strictly boolean when present', () => {
    const message = { type: 'graphReviewState', protocolVersion: 1, workspaceSessionId: 'session', messageId: 'draft',
      state: { id: 'draft', proposal: { id: 'proposal' }, before: {}, after: {}, beforeLabel: 'Current', afterLabel: 'Proposed', stale: false, changedFiles: [] } };
    assert.equal(isLocalBrowserMessage(message), true);
    for (const canRevise of [true, false]) assert.equal(isLocalBrowserMessage({ ...message,
      state: { ...message.state, proposal: { id: 'proposal', canRevise } } }), true);
    assert.equal(isLocalBrowserMessage({ ...message, state: { ...message.state, proposal: { id: 'proposal', canRevise: 'true' } } }), false);
  });
  test('setup help carries only a validated draft and source links require a positive line', () => {
    assert.ok(isGraphSetupAction({ kind: 'agent_help', draft: { profile: {} } }));
    assert.equal(isGraphSetupAction({ kind: 'agent_help', draft: [] }), false);
    assert.equal(isGraphSetupAction({ kind: 'agent_help', instructions: 'arbitrary' }), false);
    assert.ok(isGraphSetupAction({ kind: 'reveal_setup_source', file: 'model.py', line: 7 }));
    for (const line of [0, -1, 1.5, '7']) assert.equal(isGraphSetupAction({ kind: 'reveal_setup_source', file: 'model.py', line }), false);
  });
  test('agent help accepts a bounded pending draft without accepting execution fields', () => {
    assert.ok(isGraphSetupAction({ kind: 'agent_help', draft: { profile: { kind: 'model', inputs: { args: [], kwargs: {} } } } }));
    assert.ok(isGraphSetupAction({ kind: 'agent_help' }));
    assert.equal(isGraphSetupAction({ kind: 'agent_help', draft: { profile: 'invalid' } }), false);
    assert.equal(isGraphSetupAction({ kind: 'agent_help', draft: { profile: {} }, execute: true }), false);
  });
  test('environment recovery carries bounded candidate identities and refuses incoherent or unbounded state', () => {
    const candidate = { id: 'project', interpreter: '/workspace/.venv/bin/python', source: 'Saved', selected: true,
      status: 'libraries_present', python: '3.12.13', torch: '2.6.0+cu124', fingerprint: 'a'.repeat(64) };
    const environment = { reason: 'ready', detail: 'Project Python is ready.', candidates: [candidate], selectedId: candidate.id };
    assert.ok(isGraphEnvironmentRecovery(environment));
    for (const changed of [
      { ...environment, selectedId: 'unknown' },
      { ...environment, selectedId: undefined },
      { ...environment, candidates: [candidate, candidate] },
      { ...environment, candidates: Array.from({ length: 17 }, (_, i) => ({ ...candidate, id: String(i) })) },
      { ...environment, candidates: [{ ...candidate, fingerprint: 'not-an-environment-fingerprint' }] },
      { ...environment, candidates: [{ ...candidate, privateIndexCredential: 'forbidden' }] },
      { ...environment, requirements: Array.from({ length: 33 }, () => 'known-library') },
      { ...environment, declaredPins: [42] },
      { ...environment, declaredPins: Array.from({ length: 33 }, () => 'torch==2.0.0') },
      // Replacement is only meaningful for pins the host checked.
      { ...environment, reason: 'incompatible', replaceable: true },
      { ...environment, reason: 'incompatible', declaredPins: ['torch==2.0.0'], replaceable: false },
    ]) assert.equal(isGraphEnvironmentRecovery(changed), false);
    assert.ok(isGraphEnvironmentRecovery({ ...environment, reason: 'incompatible', declaredPins: ['torch==2.0.0'] }));
    assert.ok(isGraphEnvironmentRecovery({ ...environment, reason: 'incompatible', declaredPins: ['torch==2.0.0'], replaceable: true }));
  });

  test('candidate selection accepts an ID rather than a caller-supplied Python path or install command', () => {
    assert.ok(isGraphSetupAction({ kind: 'select_environment', candidateId: 'project' }));
    assert.ok(isGraphSetupAction({ kind: 'install_dependencies' }));
    assert.ok(isGraphSetupAction({ kind: 'refresh_environments' }));
    // Trying installed versions names no pin or version; the host uses its own checked recovery.
    assert.ok(isGraphSetupAction({ kind: 'use_installed_versions' }));
    assert.equal(isGraphSetupAction({ kind: 'use_installed_versions', declaredPins: ['torch==1.0'] }), false);
    assert.ok(isGraphSetupAction({ kind: 'replace_versions' }));
    assert.equal(isGraphSetupAction({ kind: 'replace_versions', packages: ['torch==1.0'] }), false);
    assert.equal(isGraphSetupAction({ kind: 'select_environment', interpreter: '/other/python' }), false);
    assert.equal(isGraphSetupAction({ kind: 'select_environment', candidateId: 'project', interpreter: '/other/python' }), false);
    assert.equal(isGraphSetupAction({ kind: 'install_dependencies', command: 'arbitrary install' }), false);
  });

  test('exposes the frozen version, roles, messages, and error codes', () => {
    assert.equal(LOCAL_BROWSER_PROTOCOL_VERSION, 1);
    assert.deepEqual(LOCAL_BROWSER_ROLES, ['host', 'view']);
    assert.deepEqual(LOCAL_BROWSER_MESSAGE_TYPES, [
      'codeFocusRequest', 'codeFocusState',
      'agentRequest', 'agentState', 'nodeExplanation', 'nodeExplanationRequest', 'modelExplanation', 'modelExplanationRequest',
      'graphReviewRequest', 'graphReviewState', 'graphReviewFocus', 'graphReviewApplied',
      'graphOpening', 'graphSetupRequest',
      'captureExportRequest', 'captureExportCancel', 'captureExportResult',
      'captureInspectRequest', 'captureInspectCancel', 'captureInspectResult',
      'viewState', 'viewSelection', 'focusView', 'viewFocusResult',
      'connected',
      'hostHandshake',
      'viewHandshake',
      'snapshotRequest',
      'snapshot',
      'activeTarget',
      'projection',
      // The project plane, added alongside the class plane. These four never
      // carry a class target: the two identities have different freshness rules.
      'projectIndex',
      'projectDocument',
      'projectExpand',
      'projectFocus',
      'projectDeclare',
      'connectionState',
      'syncState',
      'staleState',
      'revealNode',
      'focusNode',
      'sourceEditRequest',
      'sourceEditCancel',
      'sourceEditResult',
      'sourceEditReview', 'candidateRequest', 'candidateResult',
      'validateRequest',
      'validateCancel',
      'validateResult',
      'validationState',
      'ack',
      'error',
    ]);
    assert.deepEqual(LOCAL_BROWSER_ERROR_CODES, [
      'PROTOCOL_VERSION_MISMATCH',
      'SESSION_MISMATCH',
      'INVALID_ORIGIN',
      'INVALID_CREDENTIALS',
      'CREDENTIAL_EXPIRED',
      'HOST_ALREADY_CONNECTED',
      'STALE_TARGET',
      'UNKNOWN_MESSAGE',
      'INVALID_MESSAGE',
    ]);
  });

  test('accepts every canonical valid message', () => {
    assert.ok(corpus.valid.length > 0);
    for (const message of corpus.valid) {
      assert.equal(
        isLocalBrowserMessage(message, corpus.expectedWorkspaceSessionId),
        true,
        JSON.stringify(message),
      );
    }
  });

  test('refuses every canonical malformed, unknown, and drifted message', () => {
    assert.ok(corpus.invalid.length > 0);
    for (const message of corpus.invalid) {
      assert.equal(
        isLocalBrowserMessage(message, corpus.expectedWorkspaceSessionId),
        false,
        JSON.stringify(message),
      );
    }
  });

  test('requires strict target identity and role-aware ordering', () => {
    const validTarget = {
      workspaceFolder: 'workspace-a',
      file: 'model.py',
      cls: 'Net',
      documentVersion: 3,
      sourceDigest: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
      backendRevision: 7,
    };
    const projection = {
      type: 'projection',
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: 'session-1',
      messageId: 'message-1',
      hostSequence: 4,
      target: validTarget,
      projection: {},
      traced: false,
      greyed: false,
    };
    const reveal = {
      type: 'revealNode',
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: 'session-1',
      messageId: 'message-2',
      target: validTarget,
      nodeId: 'node-1',
    };

    assert.equal(isLocalBrowserMessage(projection, 'session-1'), true);
    assert.equal(isLocalBrowserMessage(reveal, 'session-1'), true);
    assert.equal(
      isLocalBrowserMessage({ ...projection, hostSequence: undefined }, 'session-1'),
      false,
    );
    assert.equal(
      isLocalBrowserMessage({ ...reveal, hostSequence: 1 }, 'session-1'),
      false,
    );
    assert.equal(isLocalBrowserMessage(projection, 'session-2'), false);
  });

  test('accepts Code focus as an optional boolean on view selections only', () => {
    const validTarget = {
      workspaceFolder: 'workspace-a',
      file: 'model.py',
      cls: 'Net',
      documentVersion: 3,
      sourceDigest: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
      backendRevision: 7,
    };
    const selection = {
      type: 'viewSelection',
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: 'session-1',
      messageId: 'message-1',
      viewId: 'view-1',
      target: validTarget,
      nodeIds: ['fc'],
      edgeIds: [],
    };
    assert.equal(isLocalBrowserMessage(selection, 'session-1'), true, 'older views omit the mode');
    assert.equal(isLocalBrowserMessage({ ...selection, follow: true }, 'session-1'), true);
    assert.equal(isLocalBrowserMessage({ ...selection, follow: false }, 'session-1'), true);
    assert.equal(isLocalBrowserMessage({ ...selection, follow: 'edit' }, 'session-1'), false);
    assert.equal(isLocalBrowserMessage({ ...selection, follow: undefined }, 'session-1'), false, 'a present key must carry a value');
    assert.equal(isLocalBrowserMessage({ ...selection, nodeIds: [], tooLarge: true, follow: false }, 'session-1'), true);
    assert.equal(isLocalBrowserMessage({ ...selection, type: 'focusView', requestId: 'focus-1', follow: true }, 'session-1'), false);
  });

  test('enforces forbidden-field key presence and optional value semantics', () => {
    const hostHandshake = corpus.valid.find((message) => {
      return (message as { type?: string }).type === 'hostHandshake';
    });
    assert.ok(hostHandshake);
    assert.equal(isLocalBrowserMessage({ ...hostHandshake, session: null }, 'session-1'), false);

    const error = {
      type: 'error',
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: 'session-1',
      messageId: 'message-1',
      code: 'SESSION_MISMATCH',
      message: 'wrong session',
    };
    assert.equal(isLocalBrowserMessage(error, 'session-1'), true);
    assert.equal(isLocalBrowserMessage({ ...error, target: null }, 'session-1'), false);
    assert.equal(isLocalBrowserMessage({ ...error, target: {} }, 'session-1'), false);

    const ack = {
      type: 'ack',
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: 'session-1',
      messageId: 'message-2',
      target: {
        workspaceFolder: 'workspace-a',
        file: 'model.py',
        cls: 'Net',
        documentVersion: 3,
        sourceDigest: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
        backendRevision: 7,
      },
      ackId: 'message-3',
      ok: false,
    };
    assert.equal(isLocalBrowserMessage({ ...ack, errorCode: 'STALE_TARGET' }, 'session-1'), true);
    assert.equal(isLocalBrowserMessage({ ...ack, errorCode: null }, 'session-1'), false);
  });

  test('keeps atomic snapshot state internally consistent', () => {
    const target = {
      workspaceFolder: 'workspace-a',
      file: 'model.py',
      cls: 'Net',
      documentVersion: 3,
      sourceDigest: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
      backendRevision: 7,
    };
    const snapshot = (state: unknown) => ({
      type: 'snapshot',
      protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
      workspaceSessionId: 'session-1',
      messageId: 'message-1',
      role: 'host',
      hostSequence: 5,
      state,
    });
    const emptyState = {
      activeTarget: null,
      projection: null,
      traced: false,
      greyed: false,
      syncState: null,
      connectionState: 'connected',
    };
    const staleState = {
      activeTarget: target,
      projection: {},
      traced: false,
      greyed: true,
      syncState: 'stale',
      reason: 'source changed',
      connectionState: 'connected',
    };

    assert.equal(isLocalBrowserMessage(snapshot(emptyState), 'session-1'), true);
    assert.equal(isLocalBrowserMessage(snapshot(staleState), 'session-1'), true);
    assert.equal(
      isLocalBrowserMessage(snapshot({ ...emptyState, syncState: 'source-only' }), 'session-1'),
      false,
    );
    assert.equal(isLocalBrowserMessage(snapshot({ ...emptyState, projection: {} }), 'session-1'), false);
    assert.equal(isLocalBrowserMessage(snapshot({ ...staleState, reason: null }), 'session-1'), false);
    assert.equal(
      isLocalBrowserMessage(snapshot({ ...staleState, syncState: 'source-only' }), 'session-1'),
      false,
    );
  });

  test('declares the browser surface read-only', () => {
    const capabilities = {
      readOnly: true,
      canNavigateSource: true,
      canWriteSource: false,
      canRunAgent: false,
      canEditGraph: false,
    };
    const connected = corpus.valid.find((message) => {
      return (message as { type?: string }).type === 'connected';
    }) as { session: { capabilities: typeof capabilities } };
    assert.deepEqual(connected.session.capabilities, capabilities);
  });
});

suite('project protocol bounds', () => {
  const project = { workspaceFolder: '/w', projectRoot: '', experimentId: '', indexDigest: 'a'.repeat(64), indexRevision: 0 };
  const base = { type: 'projectDocument', protocolVersion: 1, workspaceSessionId: 'session-1', messageId: 'bounds',
    project, hostSequence: 0, document: {}, spanmap: {} };
  test('matches Python at the expansion ceiling and rejects duplicates and role metadata', () => {
    const expanded = Array.from({ length: 4096 }, (_, i) => `node-${i}`);
    assert.ok(isLocalBrowserMessage({ ...base, expanded }));
    assert.ok(!isLocalBrowserMessage({ ...base, expanded: [...expanded, 'overflow'] }));
    assert.ok(!isLocalBrowserMessage({ ...base, expanded: ['node', 'node'] }));
    assert.ok(!isLocalBrowserMessage({ ...base, expanded, target: null }));
    assert.ok(!isLocalBrowserMessage({ ...base, expanded, project: { ...project, indexDigest: project.indexDigest + '\n' } }));
  });
});
