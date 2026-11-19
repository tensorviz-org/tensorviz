/** The extension's side of the local-browser broker (extension-webpage step 3).
 *
 * Two responsibilities, both behind this module's interface so that step 4's
 * `Open as Graph` flow never touches HTTP or socket plumbing:
 *
 *  1. `mintLocalBrowserSession` — the one authenticated `POST /local/sessions`
 *     that trades the per-launch token for session material: the view
 *     credential destined for the URL fragment, and the host cookie key that
 *     must never leave extension-host memory.
 *  2. `createLocalBrowserHostTransport` — the host-role WebSocket adapter:
 *     handshake, ordered publish with transport-owned `hostSequence`,
 *     reconnect with backoff, terminal-refusal mapping, and idempotent
 *     disposal.
 *
 * The contract and its frozen message shapes live in `./protocol.ts`; the
 * broker that enforces them is `backend/services/ide/local_browser_broker.py`;
 * behavior is documented in `docs/local-browser-transports.md`.
 *
 * Secret hygiene: the launch token, view credential, and cookie key are never
 * logged, never included in error text, and never written anywhere but the
 * wire. `log` lines below describe events, never payloads.
 */

import { decodeLocalBrowserMessage, encodeLocalBrowserMessage } from './localBrowserWire';
import {
  LOCAL_BROWSER_CLOSE_CODES,
  LOCAL_BROWSER_COOKIE_NAME,
  LOCAL_BROWSER_PROTOCOL_VERSION,
  LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES,
  LOCAL_BROWSER_SESSIONS_PATH,
  LOCAL_BROWSER_WEBSOCKET_PATH,
  LocalBrowserAcknowledgementMessage,
  LocalBrowserActiveTargetMessage,
  LocalBrowserSourceEditResultMessage,
  LocalBrowserSourceEditReviewMessage,
  LocalBrowserCandidateResultMessage,
  LocalBrowserValidateResultMessage,
  LocalBrowserCaptureInspectResultMessage,
  LocalBrowserCaptureExportResultMessage,
  LocalBrowserGraphPublishStateMessage,
  LocalBrowserGraphPublishResultMessage,
  LocalBrowserFeedbackResultMessage,
  LocalBrowserValidationStateMessage,
  LocalBrowserGraphOpeningMessage,
  LocalBrowserErrorCode,
  LocalBrowserErrorMessage,
  LocalBrowserFocusNodeMessage,
  LocalBrowserFocusViewMessage,
  LocalBrowserMessage,
  LocalBrowserProjectIndexMessage,
  LocalBrowserProjectDocumentMessage,
  LocalBrowserProjectionMessage,
  LocalBrowserSnapshotMessage,
  LocalBrowserStaleStateMessage,
  LocalBrowserSyncStateMessage,
  encodeLocalBrowserFragment,
  isLocalBrowserMessage,
  newLocalBrowserMessageId,
} from '@tensorviz/contracts/protocol';
import {
  LocalWebSocket,
  LocalWebSocketCloseEvent,
  LocalWebSocketHandlers,
  LocalWebSocketOptions,
  connectLocalWebSocket,
} from './localBrowserSocket';

/** Same spelling as middleware.py's LOCAL_TOKEN_HEADER and projection.ts's
 *  TOKEN_HEADER; declared here so this module stays free of their imports. */
const LOCAL_TOKEN_HEADER = 'X-tensorViz-Token';

const DEFAULT_RECONNECT_DELAYS_MS = [100, 250, 500, 1000, 2000];
const HANDSHAKE_TIMEOUT_MS = 10_000;
const DEFAULT_MINT_TIMEOUT_MS = 10_000;

/** Post-handshake close codes in the broker's 44xx band are terminal protocol
 *  refusals, not link loss — reconnecting would mint the same refusal forever. */
const TERMINAL_CLOSE_CODES = new Set(Object.values(LOCAL_BROWSER_CLOSE_CODES));

const CLOSE_CODE_TO_ERROR = new Map<number, LocalBrowserErrorCode>(
  Object.entries(LOCAL_BROWSER_CLOSE_CODES).map(([code, closeCode]) => [
    closeCode,
    code as LocalBrowserErrorCode,
  ]),
);

const SEQUENCED_TYPES: ReadonlySet<string> = new Set(LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES);

// ---------------------------------------------------------------------------
// Session mint
// ---------------------------------------------------------------------------

/** Everything the extension needs to arm both roles. `viewCredential` and
 *  `pageUrl` are the only values meant to reach a browser, in the fragment. */
export interface LocalBrowserSessionMaterial {
  workspaceSessionId: string;
  launchOrigin: string;
  expiresAt: string;
  hostCookieKey: string;
  viewCredential: string;
  /** The external URL step 4 opens: launch origin + fragment, nothing else. */
  pageUrl: string;
}

export class LocalBrowserSessionError extends Error {
  constructor(message: string, readonly status: number | null) {
    super(message);
    this.name = 'LocalBrowserSessionError';
  }
}

export interface MintLocalBrowserSessionOptions {
  /** Backend loopback origin the extension already started, e.g. http://127.0.0.1:port. */
  backendBaseUrl: string;
  /** The per-launch secret the backend was started with. Proves host identity. */
  launchToken: string;
  /** The exact Origin the browser page will present — the shipped flow serves
   *  the page from the backend itself, so this is the same origin. */
  launchOrigin: string;
  ttlSeconds?: number;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

export async function mintLocalBrowserSession(
  options: MintLocalBrowserSessionOptions,
): Promise<LocalBrowserSessionMaterial> {
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  const base = options.backendBaseUrl.replace(/\/+$/, '');
  let response;
  try {
    response = await fetchImpl(`${base}${LOCAL_BROWSER_SESSIONS_PATH}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        [LOCAL_TOKEN_HEADER]: options.launchToken,
      },
      body: JSON.stringify({
        launchOrigin: options.launchOrigin,
        ...(options.ttlSeconds === undefined ? {} : { ttlSeconds: options.ttlSeconds }),
      }),
      signal: AbortSignal.timeout(options.timeoutMs ?? DEFAULT_MINT_TIMEOUT_MS),
    });
  } catch {
    throw new LocalBrowserSessionError('The backend did not answer the session mint.', null);
  }
  if (!response.ok) {
    const message =
      response.status === 401
        ? 'The backend rejected the per-launch token.'
        : response.status === 404
          ? 'The backend is not running in extension mode.'
          : `Session mint failed with HTTP ${response.status}.`;
    throw new LocalBrowserSessionError(message, response.status);
  }
  const body: unknown = await response.json();
  const record = body as Record<string, unknown> | null;
  const strings = ['workspaceSessionId', 'launchOrigin', 'expiresAt', 'viewCredential', 'hostCookieKey'];
  if (
    record === null ||
    typeof record !== 'object' ||
    record.protocolVersion !== LOCAL_BROWSER_PROTOCOL_VERSION ||
    record.cookieName !== LOCAL_BROWSER_COOKIE_NAME ||
    strings.some((field) => typeof record[field] !== 'string' || (record[field] as string) === '')
  ) {
    throw new LocalBrowserSessionError(
      'The session mint response does not match the protocol contract.',
      response.status,
    );
  }
  const workspaceSessionId = record.workspaceSessionId as string;
  const viewCredential = record.viewCredential as string;
  const launchOrigin = record.launchOrigin as string;
  return {
    workspaceSessionId,
    launchOrigin,
    expiresAt: record.expiresAt as string,
    hostCookieKey: record.hostCookieKey as string,
    viewCredential,
    pageUrl: `${launchOrigin}/${encodeLocalBrowserFragment(workspaceSessionId, viewCredential)}`,
  };
}

export interface LocalBrowserViewLink {
  workspaceSessionId: string;
  handoffId: string;
  expiresAt: string;
  pageUrl: string;
}

/** Short-lived view access only; never expose the session's reusable credential. */
export async function mintLocalBrowserViewLink(options: {
  backendBaseUrl: string; launchToken: string; session: LocalBrowserSessionMaterial;
  signal: AbortSignal; fetchImpl?: typeof fetch;
}): Promise<LocalBrowserViewLink> {
  const { session } = options;
  try {
    const response = await (options.fetchImpl ?? globalThis.fetch)(
      `${options.backendBaseUrl.replace(/\/+$/, '')}/local/sessions/${encodeURIComponent(session.workspaceSessionId)}/view-links`, {
        method: 'POST', redirect: 'error',
        headers: { [LOCAL_TOKEN_HEADER]: options.launchToken },
        signal: AbortSignal.any([options.signal, AbortSignal.timeout(DEFAULT_MINT_TIMEOUT_MS)]),
      });
    if (!response.ok) throw new Error();
    const body = await response.json() as Record<string, unknown>;
    const expires = Date.parse(String(body.expiresAt));
    if (body.workspaceSessionId !== session.workspaceSessionId || body.launchOrigin !== session.launchOrigin ||
      !['handoffId', 'viewCredential'].every(key => typeof body[key] === 'string' && /^[A-Za-z0-9_-]{16,128}$/.test(body[key] as string)) ||
      !Number.isFinite(expires) || expires <= Date.now() || expires > Date.now() + 125_000) throw new Error();
    return { workspaceSessionId: session.workspaceSessionId, handoffId: body.handoffId as string,
      expiresAt: body.expiresAt as string,
      pageUrl: `${session.launchOrigin}/${encodeLocalBrowserFragment(session.workspaceSessionId, body.viewCredential as string)}` };
  } catch {
    throw new LocalBrowserSessionError('The browser view link could not be created. Retry from the current graph.', null);
  }
}

// ---------------------------------------------------------------------------
// Host transport
// ---------------------------------------------------------------------------

export type LocalBrowserHostPhase = 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'ended';

/** The broker's word on whether any browser view holds a socket on this session.
 *
 *  `unknown` is the launch window and any backend that predates `viewState`, and it
 *  must NEVER be read as "the tab is gone": the broker only ever states `detached`,
 *  so treating silence as absence would reopen a page that is already on screen. */
export type LocalBrowserViewPresence = 'unknown' | 'attached' | 'detached';

/** Local connection failures stay host-side; they are not broker protocol messages. */
export type LocalBrowserHostError = LocalBrowserErrorCode | 'WEBSOCKET_UNAVAILABLE';

export interface LocalBrowserHostStatus {
  phase: LocalBrowserHostPhase;
  /** Increments per successful broker handshake; each epoch restarts hostSequence at 1. */
  epoch: number;
  /** The terminal protocol or connection error; null when ended by local disposal. */
  error: LocalBrowserHostError | null;
}

type WithoutWire<T> = Omit<
  T,
  'protocolVersion' | 'workspaceSessionId' | 'messageId' | 'hostSequence' | 'role'
>;

/** Every host-originated message, minus the fields the transport owns. The
 *  publish caller supplies semantics; the transport supplies the envelope, the
 *  snapshot role marker, and the epoch-scoped ordering. */
export type LocalBrowserHostPublish =
  | WithoutWire<import('@tensorviz/contracts/protocol').CodeFocusState>
  | WithoutWire<import('@tensorviz/contracts/protocol').LocalBrowserAgentState>
  | WithoutWire<import('@tensorviz/contracts/protocol').LocalBrowserNodeExplanation>
  | WithoutWire<import('@tensorviz/contracts/protocol').LocalBrowserModelExplanation>
  | WithoutWire<import('@tensorviz/contracts/protocol').LocalBrowserModelCatalog>
  | WithoutWire<import('@tensorviz/contracts/protocol').GraphReviewMessage>
  | WithoutWire<import('@tensorviz/contracts/protocol').GraphReviewFocus>
  | WithoutWire<LocalBrowserGraphOpeningMessage>
  | WithoutWire<LocalBrowserProjectIndexMessage>
  | WithoutWire<LocalBrowserProjectDocumentMessage>
  | WithoutWire<LocalBrowserFocusViewMessage>
  | WithoutWire<LocalBrowserSnapshotMessage>
  | WithoutWire<LocalBrowserActiveTargetMessage>
  | WithoutWire<LocalBrowserProjectionMessage>
  | WithoutWire<LocalBrowserSyncStateMessage>
  | WithoutWire<LocalBrowserStaleStateMessage>
  | WithoutWire<LocalBrowserFocusNodeMessage>
  | WithoutWire<LocalBrowserSourceEditResultMessage>
  | WithoutWire<LocalBrowserSourceEditReviewMessage>
  | WithoutWire<LocalBrowserCandidateResultMessage>
  | WithoutWire<LocalBrowserValidateResultMessage>
  | WithoutWire<LocalBrowserCaptureInspectResultMessage>
  | WithoutWire<LocalBrowserCaptureExportResultMessage>
  | WithoutWire<LocalBrowserGraphPublishStateMessage>
  | WithoutWire<LocalBrowserGraphPublishResultMessage>
  | WithoutWire<LocalBrowserFeedbackResultMessage>
  | WithoutWire<LocalBrowserValidationStateMessage>
  | WithoutWire<LocalBrowserAcknowledgementMessage>
  | WithoutWire<LocalBrowserErrorMessage>;

export interface LocalBrowserHostTransportOptions {
  onPublish?: (message: LocalBrowserHostPublish) => void;
  backendBaseUrl: string;
  session: LocalBrowserSessionMaterial;
  connectSocket?: (
    url: string,
    handlers: LocalWebSocketHandlers,
    options: LocalWebSocketOptions,
  ) => LocalWebSocket;
  messageId?: () => string;
  reconnectDelaysMs?: number[];
  /** Events worth surfacing (transitions, dropped protocol violations). Never secrets. */
  log?: (line: string) => void;
}

export interface LocalBrowserHostTransport {
  readonly status: LocalBrowserHostStatus;
  /** Inbound messages that failed contract validation — always zero against the
   *  broker, so a nonzero count names a contract bug, and it must show up in tests. */
  readonly protocolViolations: number;
  /** Whether the broker has said this session's last view went away. `unknown` until
   *  it says so — see LocalBrowserViewPresence. */
  readonly viewPresence: LocalBrowserViewPresence;
  /** Resolves on the first successful epoch; rejects on terminal refusal or disposal. */
  connect(): Promise<void>;
  /** Builds, orders, and sends one host message. Returns the wire message, or
   *  null when the transport is not connected — callers resend on 'connected'
   *  rather than queueing, because the new epoch's state supersedes the old. */
  publish(message: LocalBrowserHostPublish): LocalBrowserMessage | null;
  /** Validated view-originated messages: snapshotRequest, revealNode, ack, error. */
  onMessage(listener: (message: LocalBrowserMessage) => void): () => void;
  onStatusChange(listener: (status: LocalBrowserHostStatus) => void): () => void;
  /** Idempotent. Closes the live socket politely and stops all reconnect work. */
  dispose(): void;
}

export function createLocalBrowserHostTransport(
  options: LocalBrowserHostTransportOptions,
): LocalBrowserHostTransport {
  const session = options.session;
  const connectSocket = options.connectSocket ?? connectLocalWebSocket;
  const messageId = options.messageId ?? newLocalBrowserMessageId;
  const delays = options.reconnectDelaysMs ?? DEFAULT_RECONNECT_DELAYS_MS;
  const log = options.log ?? ((line: string) => console.warn(`[tensorviz local-browser host] ${line}`));

  const wsUrl = `${options.backendBaseUrl.replace(/^http/, 'ws').replace(/\/+$/, '')}${LOCAL_BROWSER_WEBSOCKET_PATH}`;
  const socketHeaders = {
    Origin: session.launchOrigin,
    Cookie: `${LOCAL_BROWSER_COOKIE_NAME}=${session.hostCookieKey}`,
  };

  let phase: LocalBrowserHostPhase = 'idle';
  // Reset per epoch: a level belongs to the host connection that was told it, and a
  // stale `detached` carried across a reconnect would reopen a live page.
  let viewPresence: LocalBrowserViewPresence = 'unknown';
  let epoch = 0;
  let endedError: LocalBrowserHostError | null = null;
  let disposed = false;
  let everConnected = false;
  let hostSequence = 0;
  let attempt = 0;
  let violations = 0;
  let socket: LocalWebSocket | null = null;
  let handshakeTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let connectPromise: Promise<void> | null = null;
  let pendingResolve: (() => void) | null = null;
  let pendingReject: ((error: Error) => void) | null = null;
  const messageListeners = new Set<(message: LocalBrowserMessage) => void>();
  const statusListeners = new Set<(status: LocalBrowserHostStatus) => void>();

  function status(): LocalBrowserHostStatus {
    return { phase, epoch, error: endedError };
  }

  function setPhase(next: LocalBrowserHostPhase): void {
    if (phase === next) return;
    phase = next;
    const snapshot = status();
    for (const listener of statusListeners) listener(snapshot);
  }

  function clearTimers(): void {
    if (handshakeTimer !== null) {
      clearTimeout(handshakeTimer);
      handshakeTimer = null;
    }
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function settlePending(error: Error | null): void {
    const resolve = pendingResolve;
    const reject = pendingReject;
    pendingResolve = null;
    pendingReject = null;
    if (error === null) {
      resolve?.();
    } else {
      reject?.(error);
    }
  }

  function end(errorCode: LocalBrowserHostError | null, detail?: string): void {
    if (phase === 'ended') return;
    disposed = true;
    endedError = errorCode;
    clearTimers();
    socket?.close(1000, 'host transport ended');
    socket = null;
    if (errorCode !== null) {
      settlePending(new Error(detail ?? `local browser host transport ended: ${errorCode}`));
    } else {
      settlePending(new Error('local browser host transport disposed before connecting'));
    }
    setPhase('ended');
    log(errorCode === null ? 'disposed' : `ended: ${errorCode}`);
  }

  function handshook(): void {
    clearTimers();
    epoch += 1;
    hostSequence = 0;
    attempt = 0;
    everConnected = true;
    viewPresence = 'unknown';  // a new epoch is levelled by the broker, not by memory
    setPhase('connected');
    settlePending(null);
  }

  function scheduleReconnect(): void {
    const delay = delays[Math.min(attempt, delays.length - 1)];
    attempt += 1;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (!disposed) startAttempt();
    }, delay);
    reconnectTimer.unref?.();
  }

  function onSocketClose(event: LocalWebSocketCloseEvent): void {
    if (disposed) return;
    clearTimers();
    socket = null;
    if (event.kind === 'closed' && TERMINAL_CLOSE_CODES.has(event.code)) {
      end(CLOSE_CODE_TO_ERROR.get(event.code) ?? 'INVALID_MESSAGE');
      return;
    }
    if (event.kind === 'refused') {
      if (event.status === 401 || event.status === 403) {
        end('INVALID_CREDENTIALS');
      } else {
        const status = event.status === null ? '' : ` (HTTP ${event.status})`;
        end('WEBSOCKET_UNAVAILABLE',
          `The backend could not establish the WebSocket connection${status}. ` +
          'Run tensorViz: Check the Python environment or tensorViz: Repair runtime, then Open as Graph again.');
      }
      return;
    }
    log(`socket ${event.kind === 'error' ? `error (${event.message})` : `closed (${event.code})`}; reconnecting`);
    setPhase(everConnected ? 'reconnecting' : 'connecting');
    scheduleReconnect();
  }

  function onSocketText(text: string): void {
    let parsed: unknown;
    try {
      parsed = decodeLocalBrowserMessage(text);
    } catch {
      violations += 1;
      log('dropped a message that is not JSON');
      return;
    }
    if (!everConnected || phase === 'connecting' || phase === 'reconnecting') {
      // Pre-handshake: the only lawful message is the broker's `connected`,
      // or a typed `error` refusal immediately before it closes the socket.
      if (isLocalBrowserMessage(parsed, session.workspaceSessionId) && parsed.type === 'error') {
        end(parsed.code);
      } else if (
        isLocalBrowserMessage(parsed, session.workspaceSessionId) &&
        parsed.type === 'connected' &&
        parsed.role === 'host'
      ) {
        handshook();
      } else {
        violations += 1;
        log('dropped a pre-handshake message that is not the broker greeting');
      }
      return;
    }
    if (!isLocalBrowserMessage(parsed, session.workspaceSessionId)) {
      violations += 1;
      log('dropped a message that failed contract validation');
      return;
    }
    if (parsed.type === 'viewState' && 'state' in parsed) {
      // Broker-owned and host-bound: consumed here, never forwarded to listeners,
      // and deliberately ahead of the misdirected-type check below — it is not a
      // view-originated message and must not be counted as one.
      viewPresence = parsed.state === 'detached' ? 'detached' : 'attached';
      log(`broker says the view roster is ${parsed.state}`);
      return;
    }
    if (parsed.type === 'viewState' && 'connected' in parsed && parsed.connected) {
      viewPresence = 'attached';
    }
    if (
      parsed.type !== 'viewState' &&
      parsed.type !== 'viewSelection' &&
      parsed.type !== 'viewFocusResult' &&
      parsed.type !== 'codeFocusRequest' &&
      parsed.type !== 'graphReviewApplied' &&
      parsed.type !== 'graphReviewRequest' &&
      parsed.type !== 'projectFocus' &&
      parsed.type !== 'projectExpand' &&
      parsed.type !== 'projectDeclare' &&
      parsed.type !== 'snapshotRequest' &&
      parsed.type !== 'revealNode' &&
      parsed.type !== 'sourceEditRequest' &&
      parsed.type !== 'agentRequest' &&
      parsed.type !== 'nodeExplanationRequest' &&
      parsed.type !== 'modelExplanationRequest' &&
      parsed.type !== 'modelCatalogRequest' &&
      parsed.type !== 'sourceEditCancel' &&
      parsed.type !== 'candidateRequest' &&
      parsed.type !== 'captureInspectRequest' &&
      parsed.type !== 'captureInspectCancel' &&
      parsed.type !== 'captureExportRequest' &&
      parsed.type !== 'captureExportCancel' &&
      parsed.type !== 'graphPublishRequest' &&
      parsed.type !== 'graphPublishCancel' &&
      parsed.type !== 'feedbackRequest' &&
      parsed.type !== 'validateRequest' &&
      parsed.type !== 'graphSetupRequest' &&
      parsed.type !== 'validateCancel' &&
      parsed.type !== 'ack' &&
      parsed.type !== 'error'
    ) {
      violations += 1;
      log(`dropped a misdirected message of type ${parsed.type}`);
      return;
    }
    for (const listener of messageListeners) listener(parsed);
  }

  function startAttempt(): void {
    socket = connectSocket(
      wsUrl,
      {
        onOpen: () => {
          socket?.sendText(
            JSON.stringify({
              type: 'hostHandshake',
              viewControl: true,
              protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
              workspaceSessionId: session.workspaceSessionId,
              messageId: messageId(),
              role: 'host',
            }),
          );
          handshakeTimer = setTimeout(() => {
            log('broker handshake timed out; retrying');
            socket?.close(1001, 'handshake timeout');
          }, HANDSHAKE_TIMEOUT_MS);
          handshakeTimer.unref?.();
        },
        onText: onSocketText,
        onClose: onSocketClose,
      },
      { headers: socketHeaders },
    );
  }

  return {
    get status() {
      return status();
    },
    get protocolViolations() {
      return violations;
    },
    get viewPresence() {
      return viewPresence;
    },
    connect(): Promise<void> {
      if (connectPromise !== null) return connectPromise;
      if (phase === 'ended') {
        return Promise.reject(
          new Error(`local browser host transport already ended${endedError ? `: ${endedError}` : ''}`),
        );
      }
      setPhase('connecting');
      connectPromise = new Promise((resolve, reject) => {
        pendingResolve = resolve;
        pendingReject = reject;
      });
      startAttempt();
      return connectPromise;
    },
    publish(message: LocalBrowserHostPublish): LocalBrowserMessage | null {
      if (phase !== 'connected' || socket === null) return null;
      const full: Record<string, unknown> = {
        ...message,
        protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
        workspaceSessionId: session.workspaceSessionId,
        messageId: messageId(),
      };
      if (SEQUENCED_TYPES.has(message.type)) {
        hostSequence += 1;
        full.hostSequence = hostSequence;
      }
      if (message.type === 'snapshot') {
        full.role = 'host';
      }
      if (!isLocalBrowserMessage(full, session.workspaceSessionId)) {
        // The contract and the publish union disagree — a bug, not a runtime
        // condition, and it must surface at the call site rather than as a
        // broker refusal attributed to the wire.
        throw new Error(`localBrowserHostTransport built an invalid ${message.type} message`);
      }
      if (!socket.sendText(encodeLocalBrowserMessage(full))) return null;
      try { options.onPublish?.(message); } catch { /* Optional observation never affects delivery. */ }
      return full;
    },
    onMessage(listener: (message: LocalBrowserMessage) => void): () => void {
      messageListeners.add(listener);
      return () => messageListeners.delete(listener);
    },
    onStatusChange(listener: (status: LocalBrowserHostStatus) => void): () => void {
      statusListeners.add(listener);
      return () => statusListeners.delete(listener);
    },
    dispose(): void {
      end(null);
    },
  };
}
