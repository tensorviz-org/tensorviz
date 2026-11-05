/** The browser's side of the local-browser broker (extension-webpage step 3).
 *
 * Everything the local webpage needs from the wire behind one interface, so no
 * React component ever owns a socket, a cookie, or a reconnect timer:
 *
 *  - fragment bootstrap: reads `#session=…&credential=…` from the page URL,
 *    erases it with history.replaceState before anything else, trades it once
 *    against `POST /local/bootstrap` for the HttpOnly cookie;
 *  - refresh restore: the credential exists nowhere after bootstrap, and the
 *    cookie alone cannot name the workspace session the handshake envelope
 *    requires, so the transport persists the session id — identity only, never
 *    a secret — in sessionStorage (the contract forbids persisting the
 *    credential, not the routing id). A refresh reuses cookie + stored id;
 *  - ordering: sequenced host state is held back until the requested snapshot
 *    restores one atomic state, and anything at or below the watermark is
 *    dropped, so a view never renders an update older than what it shows;
 *  - reconnect with backoff, terminal broker close-code mapping, and
 *    idempotent disposal.
 *
 * The broker (`backend/services/ide/local_browser_broker.py`) enforces the
 * contract again server-side; the validation here exists so drift and
 * cross-session traffic die at the page boundary. Behavior is documented in
 * docs/local-browser-transports.md.
 */

import { decodeLocalBrowserMessage, MAX_DECODED_BYTES } from './viewWire';
import {
  LOCAL_BROWSER_BOOTSTRAP_PATH,
  LOCAL_BROWSER_CLOSE_CODES,
  LOCAL_BROWSER_PROTOCOL_VERSION,
  LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES,
  LOCAL_BROWSER_WEBSOCKET_PATH,
  LocalBrowserAcknowledgementMessage,
  LocalBrowserActiveTargetMessage,
  LocalBrowserSourceEditRequestMessage,
  LocalBrowserSourceEditCancelMessage,
  LocalBrowserSourceEditResultMessage,
  LocalBrowserSourceEditReviewMessage,
  LocalBrowserCandidateRequestMessage,
  LocalBrowserCandidateResultMessage,
  LocalBrowserValidateRequestMessage,
  LocalBrowserValidateCancelMessage,
  LocalBrowserValidateResultMessage,
  LocalBrowserCaptureInspectRequestMessage,
  LocalBrowserCaptureInspectCancelMessage,
  LocalBrowserCaptureInspectResultMessage,
  LocalBrowserCaptureExportRequestMessage,
  LocalBrowserCaptureExportCancelMessage,
  LocalBrowserCaptureExportResultMessage,
  LocalBrowserValidationStateMessage,
  LocalBrowserGraphOpeningMessage,
  LocalBrowserGraphSetupRequestMessage,
  LocalBrowserErrorCode,
  LocalBrowserErrorMessage,
  LocalBrowserFocusNodeMessage,
  LocalBrowserFocusViewMessage,
  LocalBrowserViewSelectionMessage,
  LocalBrowserViewFocusResultMessage,
  LocalBrowserMessage,
  LocalBrowserProjectDeclareMessage,
  LocalBrowserProjectDocumentMessage,
  LocalBrowserProjectExpandMessage,
  LocalBrowserProjectFocusMessage,
  LocalBrowserProjectIndexMessage,
  LocalBrowserProjectionMessage,
  LocalBrowserRevealNodeMessage,
  LocalBrowserSessionInfo,
  LocalBrowserSnapshotMessage,
  LocalBrowserStaleStateMessage,
  LocalBrowserSyncStateMessage,
  decodeLocalBrowserFragment,
  isLocalBrowserMessage,
  newLocalBrowserMessageId,
} from '@tensorviz/contracts/protocol';

/** Per-origin+tab storage for the workspace session id — the only field of the
 *  session material a page may keep, because the handshake envelope requires
 *  naming it and a refresh erases every other trace. */
export const LOCAL_BROWSER_SESSION_STORAGE_KEY = 'tensorviz.localBrowser.workspaceSessionId';

const DEFAULT_RECONNECT_DELAYS_MS = [250, 500, 1000, 2000, 5000];
const DEFAULT_HANDSHAKE_TIMEOUT_MS = 10_000;
/** Consecutive failed attach attempts before the page gives up and asks the
 *  user to reopen from VS Code — the honest answer for an expired cookie,
 *  which a browser sandbox hides behind an undifferentiated WS failure. */
const DEFAULT_MAX_CONNECT_ATTEMPTS = 12;

const TERMINAL_CLOSE_CODES = new Set<number>(Object.values(LOCAL_BROWSER_CLOSE_CODES));
const CLOSE_CODE_TO_ERROR = new Map<number, LocalBrowserErrorCode>(
  Object.entries(LOCAL_BROWSER_CLOSE_CODES).map(([code, closeCode]) => [
    closeCode,
    code as LocalBrowserErrorCode,
  ]),
);
const SEQUENCED_TYPES: ReadonlySet<string> = new Set(LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES);

export type LocalBrowserViewPhase = 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'ended';
/** The broker's word on whether the extension host is attached. */
export type LocalBrowserHostLink = 'unknown' | 'connected' | 'disconnected';

export interface LocalBrowserViewStatus {
  phase: LocalBrowserViewPhase;
  hostLink: LocalBrowserHostLink;
  /** The terminal protocol error when the transport ended on a refusal/expiry; null for local disposal. */
  error: LocalBrowserErrorCode | null;
}

type WithoutWire<T> = Omit<
  T,
  'protocolVersion' | 'workspaceSessionId' | 'messageId' | 'hostSequence' | 'role'
>;

/** What a view may originate (minus the envelope fields the transport owns).
 *  snapshotRequest is transport-owned too: it fires on (re)connect with the
 *  current watermark, which is exactly the only moment it is correct. The ask
 *  pair is per-request traffic: it carries no hostSequence, and its replies
 *  arrive matched by requestId, so ordering says nothing about them. */
export type LocalBrowserViewSend =
  | WithoutWire<import('@tensorviz/contracts/protocol').CodeFocusRequest>
  | ({ type: 'agentRequest'; viewId: string } & import('@tensorviz/contracts/protocol').CanvasAgentAction)
  | WithoutWire<Extract<import('@tensorviz/contracts/protocol').LocalBrowserNodeExplanationRequest, { action: 'retry' }>>
  | WithoutWire<Extract<import('@tensorviz/contracts/protocol').LocalBrowserNodeExplanationRequest, { action: 'reveal' }>>
  | WithoutWire<import('@tensorviz/contracts/protocol').GraphReviewRequest>
  | WithoutWire<import('@tensorviz/contracts/protocol').GraphReviewApplied>
  | WithoutWire<LocalBrowserGraphSetupRequestMessage>
  | WithoutWire<LocalBrowserViewSelectionMessage>
  | WithoutWire<LocalBrowserViewFocusResultMessage>
  | WithoutWire<LocalBrowserRevealNodeMessage>
  | WithoutWire<LocalBrowserSourceEditRequestMessage>
  | WithoutWire<LocalBrowserSourceEditCancelMessage>
  | WithoutWire<LocalBrowserCandidateRequestMessage>
  // The project plane's view traffic. Like the ask pair it carries no
  // hostSequence: only ordered HOST state is sequenced, and these are requests.
  | WithoutWire<LocalBrowserProjectFocusMessage>
  | WithoutWire<LocalBrowserProjectExpandMessage>
  | WithoutWire<LocalBrowserProjectDeclareMessage>
  | WithoutWire<LocalBrowserCaptureInspectRequestMessage>
  | WithoutWire<LocalBrowserCaptureInspectCancelMessage>
  | WithoutWire<LocalBrowserCaptureExportRequestMessage>
  | WithoutWire<LocalBrowserCaptureExportCancelMessage>
  | WithoutWire<LocalBrowserValidateRequestMessage>
  | WithoutWire<LocalBrowserValidateCancelMessage>
  | WithoutWire<LocalBrowserAcknowledgementMessage>
  | WithoutWire<LocalBrowserErrorMessage>;

/** Validated, ordered, deduplicated host traffic ready to render. */
export type LocalBrowserViewMessage =
  | import('@tensorviz/contracts/protocol').CodeFocusState
  | import('@tensorviz/contracts/protocol').LocalBrowserAgentState
  | import('@tensorviz/contracts/protocol').LocalBrowserNodeExplanation
  | import('@tensorviz/contracts/protocol').GraphReviewMessage
  | import('@tensorviz/contracts/protocol').GraphReviewFocus
  | LocalBrowserGraphOpeningMessage
  | LocalBrowserProjectIndexMessage
  | LocalBrowserProjectDocumentMessage
  | LocalBrowserFocusViewMessage
  | LocalBrowserSnapshotMessage
  | LocalBrowserActiveTargetMessage
  | LocalBrowserProjectionMessage
  | LocalBrowserSyncStateMessage
  | LocalBrowserStaleStateMessage
  | LocalBrowserFocusNodeMessage
  | LocalBrowserSourceEditResultMessage
  | LocalBrowserSourceEditReviewMessage
  | LocalBrowserCandidateResultMessage
  | LocalBrowserCaptureInspectResultMessage
  | LocalBrowserCaptureExportResultMessage
  | LocalBrowserValidateResultMessage
  | LocalBrowserValidationStateMessage
  | LocalBrowserAcknowledgementMessage
  | LocalBrowserErrorMessage;

/** The slice of the DOM WebSocket this transport uses; the production default
 *  is the real thing, tests inject a scripted fake. */
export interface LocalBrowserSocketLike {
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onclose: ((event: { code: number; reason: string }) => void) | null;
  onerror: (() => void) | null;
  send(data: string): void;
  close(code?: number, reason?: string): void;
}

export interface LocalBrowserViewLocation {
  hash: string;
  origin: string;
  pathname: string;
  search: string;
}

export interface LocalBrowserViewTransportOptions {
  location?: () => LocalBrowserViewLocation;
  /** Erases the fragment immediately after it is read. */
  replaceState?: (urlWithoutFragment: string) => void;
  fetchImpl?: (input: string, init?: RequestInit) => Promise<Response>;
  connectSocket?: (url: string) => LocalBrowserSocketLike;
  storage?: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'> | null;
  websocketUrl?: string;
  messageId?: () => string;
  reconnectDelaysMs?: number[];
  handshakeTimeoutMs?: number;
  maxConnectAttempts?: number;
}

export interface LocalBrowserViewTransport {
  readonly status: LocalBrowserViewStatus;
  readonly session: LocalBrowserSessionInfo | null;
  /** Inbound messages refused by contract validation — always zero against the
   *  broker, so a nonzero count names contract drift, and tests assert it. */
  readonly protocolViolations: number;
  /** Resolves on the first established epoch; rejects on terminal refusal. */
  connect(): Promise<void>;
  /** Builds and sends one view message, or returns null when not connected. */
  send(message: LocalBrowserViewSend): LocalBrowserMessage | null;
  onMessage(listener: (message: LocalBrowserViewMessage) => void): () => void;
  onStatusChange(listener: (status: LocalBrowserViewStatus) => void): () => void;
  /** Idempotent; closes the socket politely and stops all reconnect work. */
  dispose(): void;
}

/** Rejection carriers so the connect loop can tell "wait and retry" from
 *  "report to the user": a transient is a network/timeout/restart shape, a
 *  terminal is a protocol refusal with its frozen error code. */
class TransientFailure extends Error {}
class TerminalFailure extends Error {
  constructor(readonly code: LocalBrowserErrorCode) {
    super(code);
  }
}

export function createLocalBrowserViewTransport(
  options: LocalBrowserViewTransportOptions = {},
): LocalBrowserViewTransport {
  const readLocation =
    options.location ??
    (() => ({
      hash: window.location.hash,
      origin: window.location.origin,
      pathname: window.location.pathname,
      search: window.location.search,
    }));
  const replaceState =
    options.replaceState ??
    ((urlWithoutFragment: string) =>
      window.history.replaceState(window.history.state, '', urlWithoutFragment));
  const fetchImpl = options.fetchImpl ?? ((input: string, init?: RequestInit) => window.fetch(input, init));
  // The DOM WebSocket's handler properties are this-typed; the transport only
  // ever uses the subset LocalBrowserSocketLike declares, so the cast lives
  // exactly here at the browser boundary and nowhere else.
  const connectSocket =
    options.connectSocket ??
    ((url: string): LocalBrowserSocketLike => new WebSocket(url) as unknown as LocalBrowserSocketLike);
  const storage =
    options.storage !== undefined
      ? options.storage
      : safeSessionStorage();
  const messageId = options.messageId ?? newLocalBrowserMessageId;
  const delays = options.reconnectDelaysMs ?? DEFAULT_RECONNECT_DELAYS_MS;
  const handshakeTimeoutMs = options.handshakeTimeoutMs ?? DEFAULT_HANDSHAKE_TIMEOUT_MS;
  const maxConnectAttempts = options.maxConnectAttempts ?? DEFAULT_MAX_CONNECT_ATTEMPTS;

  let phase: LocalBrowserViewPhase = 'idle';
  let hostLink: LocalBrowserHostLink = 'unknown';
  let terminalError: LocalBrowserErrorCode | null = null;
  let session: LocalBrowserSessionInfo | null = null;
  let expectedSessionId: string | null = null;
  let handoffId: string | null = null;
  const handoffStorageKey = `${LOCAL_BROWSER_SESSION_STORAGE_KEY}.handoff`;
  let socket: LocalBrowserSocketLike | null = null;
  let watermark = 0;
  let awaitingSnapshot = true;
  let snapshotOutstanding = false;
  let attempts = 0;
  let violations = 0;
  let everConnected = false;
  let disposed = false;
  let connectPromise: Promise<void> | null = null;
  let firstResolve: (() => void) | null = null;
  let firstReject: ((error: Error) => void) | null = null;
  let handshakeTimer: ReturnType<typeof setTimeout> | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  const messageListeners = new Set<(message: LocalBrowserViewMessage) => void>();
  const statusListeners = new Set<(status: LocalBrowserViewStatus) => void>();

  function status(): LocalBrowserViewStatus {
    return { phase, hostLink, error: terminalError };
  }

  function setPhase(next: LocalBrowserViewPhase): void {
    if (phase === next) return;
    phase = next;
    emit();
  }

  function emit(): void {
    const current = status();
    statusListeners.forEach((listener) => listener(current));
  }

  function clearTimers(): void {
    if (handshakeTimer !== null) {
      clearTimeout(handshakeTimer);
      handshakeTimer = null;
    }
    if (retryTimer !== null) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
  }

  function end(error: LocalBrowserErrorCode | null): void {
    if (phase === 'ended') return;
    disposed = true;
    terminalError = error;
    clearTimers();
    if (error === 'CREDENTIAL_EXPIRED' || error === 'INVALID_CREDENTIALS') {
      // A dead credential's routing id is itself useless next launch.
      storage?.removeItem(LOCAL_BROWSER_SESSION_STORAGE_KEY);
      storage?.removeItem(handoffStorageKey);
    }
    const ws = socket;
    socket = null;
    if (ws) {
      ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null;
      try {
        ws.close(1000, error === null ? 'page done' : error);
      } catch {
        // A dead socket raises on close in some jsdom-ish fakes; irrelevant here.
      }
    }
    setPhase('ended');
    if (firstReject !== null) {
      firstReject(
        new Error(`local browser view transport ended${error ? `: ${error}` : ''}`),
      );
      firstResolve = null;
      firstReject = null;
    }
  }

  function connected(sessionInfo: LocalBrowserSessionInfo): void {
    clearTimers();
    session = sessionInfo;
    attempts = 0;
    everConnected = true;
    setPhase('connected');
    if (firstResolve !== null) {
      firstResolve();
      firstResolve = null;
      firstReject = null;
    }
    requestSnapshot(watermark);
  }

  async function ensureBootstrapped(): Promise<string> {
    if (expectedSessionId !== null) return expectedSessionId;
    const where = readLocation();
    const fragment = decodeLocalBrowserFragment(where.hash);
    if (fragment !== null) {
      // Erase before any await: no code path past this line may still see the
      // credential in the address bar.
      replaceState(`${where.pathname}${where.search}`);
      let response: Response;
      try {
        response = await fetchImpl(LOCAL_BROWSER_BOOTSTRAP_PATH, {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            workspaceSessionId: fragment.workspaceSessionId,
            credential: fragment.credential,
          }),
        });
      } catch {
        throw new TransientFailure('bootstrap request failed');
      }
      if (!response.ok) {
        const code: LocalBrowserErrorCode =
          response.status === 410
            ? 'CREDENTIAL_EXPIRED'
            : response.status === 403
              ? 'INVALID_ORIGIN'
              : 'INVALID_CREDENTIALS';
        throw new TerminalFailure(code);
      }
      const info = (await response.json()) as LocalBrowserSessionInfo;
      if (
        info.protocolVersion !== LOCAL_BROWSER_PROTOCOL_VERSION ||
        typeof info.workspaceSessionId !== 'string' ||
        info.workspaceSessionId !== fragment.workspaceSessionId ||
        (info.handoffId !== undefined && (typeof info.handoffId !== 'string' || !info.handoffId || info.handoffId.length > 128)) ||
        info.role !== 'view'
      ) {
        throw new TerminalFailure('PROTOCOL_VERSION_MISMATCH');
      }
      expectedSessionId = info.workspaceSessionId;
      session = info;
      handoffId = info.handoffId ?? null;
      storage?.removeItem(handoffStorageKey);
      if (handoffId) storage?.setItem(handoffStorageKey, JSON.stringify([info.workspaceSessionId, handoffId]));
      storage?.setItem(LOCAL_BROWSER_SESSION_STORAGE_KEY, info.workspaceSessionId);
      return expectedSessionId;
    }
    const stored = storage?.getItem(LOCAL_BROWSER_SESSION_STORAGE_KEY) ?? null;
    if (stored === null || stored === '') {
      // Nothing lawful to handshake with: no fragment and no remembered id.
      throw new TerminalFailure('INVALID_CREDENTIALS');
    }
    expectedSessionId = stored;
    try {
      const saved = JSON.parse(storage?.getItem(handoffStorageKey) ?? 'null');
      if (Array.isArray(saved) && saved[0] === stored && typeof saved[1] === 'string' && saved[1].length > 0 && saved[1].length <= 128) handoffId = saved[1];
    } catch { /* A malformed routing hint cannot grant access. */ }
    return expectedSessionId;
  }

  function websocketUrl(): string {
    if (options.websocketUrl) return options.websocketUrl;
    const origin = readLocation().origin;
    return `${origin.replace(/^http/, 'ws')}${LOCAL_BROWSER_WEBSOCKET_PATH}`;
  }

  function requestSnapshot(lastHostSequence: number): void {
    if (socket === null || expectedSessionId === null) return;
    socket.send(
      JSON.stringify({
        type: 'snapshotRequest',
        protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
        workspaceSessionId: expectedSessionId,
        messageId: messageId(),
        role: 'view',
        lastHostSequence,
      }),
    );
    snapshotOutstanding = true;
    awaitingSnapshot = true;
  }

  function openSocketThenHandshake(sessionId: string): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const ws = connectSocket(websocketUrl());
      socket = ws;
      let opened = false;
      ws.onopen = () => {
        opened = true;
        ws.send(
          JSON.stringify({
            type: 'viewHandshake',
            ...(handoffId ? { handoffId } : {}),
            protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
            workspaceSessionId: sessionId,
            messageId: messageId(),
            role: 'view',
          }),
        );
        handshakeTimer = setTimeout(() => {
          try {
            ws.close(1001, 'handshake timeout');
          } catch {
            /* close of a half-open socket is best-effort */
          }
          reject(new TransientFailure('handshake timed out'));
        }, handshakeTimeoutMs);
      };
      function acceptMessage(parsed: unknown): void {
        if (socket !== ws) return;
        if (!isLocalBrowserMessage(parsed, sessionId)) {
          violations += 1;
          return;
        }
        if (parsed.type === 'connected') {
          if (phase === 'connected') {
            // A second acceptance inside one epoch is broker drift, not a
            // re-handshake; it must not mint another snapshot request.
            violations += 1;
            return;
          }
          if (parsed.role !== 'view') {
            reject(new TerminalFailure('SESSION_MISMATCH'));
            ws.close(4000, 'session mismatch');
            return;
          }
          resolveOnceConnected(parsed.session);
          return;
        }
        if (parsed.type === 'error' && phase !== 'connected') {
          // Handshake-stage refusal; the broker closes with the paired code next.
          reject(new TerminalFailure(parsed.code));
          try {
            ws.close(4000, 'handshake refused');
          } catch {
            /* best-effort */
          }
          return;
        }
        dispatch(parsed);
      }
      // Inflation is asynchronous. Later plain messages must wait too, or a
      // ready/stale update can overtake the graph it describes. A closed socket
      // never delivers work still inflating from its old connection epoch.
      let pending: Promise<void> | undefined;
      let queuedBytes = 0;
      ws.onmessage = (event: { data: unknown }) => {
        if (socket !== ws || typeof event.data !== 'string') return;
        const text = event.data;
        const bytes = new TextEncoder().encode(text).length;
        if (queuedBytes + bytes > MAX_DECODED_BYTES) {
          violations += 1;
          end('INVALID_MESSAGE');
          return;
        }
        try {
          const decoded = pending
            ? pending.then(() => socket === ws ? decodeLocalBrowserMessage(text) : undefined)
            : decodeLocalBrowserMessage(text);
          if (!(decoded instanceof Promise)) { acceptMessage(decoded); return; }
          queuedBytes += bytes;
          const current = decoded.then(acceptMessage).catch(() => {
            if (socket === ws) { violations += 1; end('INVALID_MESSAGE'); }
          }).finally(() => {
            queuedBytes -= bytes;
            if (pending === current) pending = undefined;
          });
          pending = current;
        } catch { violations += 1; }
      };
      ws.onclose = (event: { code: number; reason: string }) => {
        clearHandshakeTimer();
        if (socket === ws) socket = null;
        if (event.code >= 4000 && TERMINAL_CLOSE_CODES.has(event.code)) {
          reject(new TerminalFailure(CLOSE_CODE_TO_ERROR.get(event.code) ?? 'INVALID_MESSAGE'));
          return;
        }
        reject(new TransientFailure(opened ? 'socket closed' : 'upgrade refused or unreachable'));
      };
      ws.onerror = () => {
        // Browsers intentionally reveal nothing here; the close event follows.
      };

      function resolveOnceConnected(info: LocalBrowserSessionInfo): void {
        clearHandshakeTimer();
        connected(info);
        resolve();
      }

      function clearHandshakeTimer(): void {
        if (handshakeTimer !== null) {
          clearTimeout(handshakeTimer);
          handshakeTimer = null;
        }
      }
    });
  }

  function dispatch(message: LocalBrowserMessage): void {
    switch (message.type) {
      case 'connectionState': {
        const next: LocalBrowserHostLink = message.state === 'connected' ? 'connected' : 'disconnected';
        if (next === 'disconnected' && hostLink !== 'disconnected') {
          // A host leaves: its sequence epoch dies with it, so the watermark it
          // produced is meaningless to whoever attaches next.
          watermark = 0;
          awaitingSnapshot = true;
        }
        if (next === 'connected' && hostLink !== 'connected' && !snapshotOutstanding) {
          requestSnapshot(watermark);
        }
        hostLink = next;
        emit();
        return;
      }
      case 'snapshot':
        // The atomic restore point supersedes anything the watermark says and
        // is accepted across epochs by construction.
        awaitingSnapshot = false;
        snapshotOutstanding = false;
        watermark = message.hostSequence;
        break;
      // View-originated traffic landing back at a view is a broker routing
      // bug: a healthy reply is a source-edit or validation result for our requestId, never
      // an echo of the request.
      case 'connected':
      case 'hostHandshake':
      case 'viewHandshake':
      case 'viewState':
      case 'viewSelection':
      case 'viewFocusResult':
      case 'codeFocusRequest':
      case 'projectFocus':
      case 'projectExpand':
      case 'projectDeclare':
      case 'snapshotRequest':
      case 'revealNode':
      case 'sourceEditRequest':
      case 'agentRequest':
      case 'nodeExplanationRequest':
      case 'sourceEditCancel':
      case 'candidateRequest':
      case 'captureInspectRequest':
      case 'captureInspectCancel':
      case 'captureExportRequest':
      case 'captureExportCancel':
      case 'validateRequest':
      case 'graphReviewRequest':
      case 'graphReviewApplied':
      case 'graphSetupRequest':
      case 'validateCancel':
        violations += 1;
        return;
      default: {
        if (SEQUENCED_TYPES.has(message.type)) {
          if (!('hostSequence' in message)) return;
          if (awaitingSnapshot || message.hostSequence <= watermark) {
            return; // held until the snapshot, or a replayed duplicate
          }
          watermark = message.hostSequence;
        }
        break;
      }
    }
    messageListeners.forEach((listener) => listener(message));
  }

  /** One backoff tick, assigned to the shared timer so dispose() can cancel it.
   *  A named factory-scope function keeps the sleep's closure out of the retry
   *  loop (a function declared in a loop capturing mutable state is exactly
   *  the hazard eslint's no-loop-func guards). */
  function retryDelay(ms: number): Promise<void> {
    return new Promise<void>((resolve) => {
      retryTimer = setTimeout(resolve, ms);
    });
  }

  async function connectLoop(): Promise<void> {
    for (;;) {
      if (disposed) return;
      try {
        await openSocketThenHandshake(await ensureBootstrapped());
        socketLifecycleWatch();
        return;
      } catch (failure) {
        if (failure instanceof TerminalFailure) {
          end(failure.code);
          return;
        }
        attempts += 1;
        if (attempts >= maxConnectAttempts) {
          // An expired cookie presents exactly like an unreachable broker from
          // inside a browser sandbox; this is the honest claim after enough tries.
          end('INVALID_CREDENTIALS');
          return;
        }
        setPhase(everConnected ? 'reconnecting' : 'connecting');
        await retryDelay(delays[Math.min(attempts - 1, delays.length - 1)]);
      }
    }
  }

  function socketLifecycleWatch(): void {
    const ws = socket;
    if (ws === null) return;
    ws.onclose = (event: { code: number; reason: string }) => {
      socket = null;
      snapshotOutstanding = false;
      hostLink = 'unknown';
      emit();
      if (disposed) return;
      if (TERMINAL_CLOSE_CODES.has(event.code)) {
        end(CLOSE_CODE_TO_ERROR.get(event.code) ?? 'INVALID_MESSAGE');
        return;
      }
      setPhase('reconnecting');
      awaitingSnapshot = true;
      void connectLoop();
    };
    ws.onerror = () => {
      // Detail-free by browser design; close follows.
    };
  }

  return {
    get status() {
      return status();
    },
    get session() {
      return session;
    },
    get protocolViolations() {
      return violations;
    },
    connect(): Promise<void> {
      if (connectPromise !== null) return connectPromise;
      if (disposed) {
        return Promise.reject(new Error('local browser view transport already disposed'));
      }
      setPhase('connecting');
      connectPromise = new Promise<void>((resolve, reject) => {
        firstResolve = resolve;
        firstReject = reject;
      });
      void connectLoop().catch(() => {
        // connectLoop routes every failure through end(); a throw escaping it is
        // a bug, and the connect promise must still settle rather than hang.
        end('INVALID_MESSAGE');
      });
      return connectPromise;
    },
    send(message: LocalBrowserViewSend): LocalBrowserMessage | null {
      if (phase !== 'connected' || socket === null || expectedSessionId === null) return null;
      const full: Record<string, unknown> = {
        ...message,
        protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION,
        workspaceSessionId: expectedSessionId,
        messageId: messageId(),
      };
      if (!isLocalBrowserMessage(full, expectedSessionId)) {
        throw new Error(`localBrowserViewTransport built an invalid ${message.type} message`);
      }
      socket.send(JSON.stringify(full));
      return full;
    },
    onMessage(listener: (message: LocalBrowserViewMessage) => void): () => void {
      messageListeners.add(listener);
      return () => messageListeners.delete(listener);
    },
    onStatusChange(listener: (status: LocalBrowserViewStatus) => void): () => void {
      statusListeners.add(listener);
      return () => statusListeners.delete(listener);
    },
    dispose(): void {
      end(null);
    },
  };
}

/** sessionStorage throws SecurityError in some privacy modes; absence is
 *  survivable (a refresh then simply cannot re-attach), a thrown read is not. */
function safeSessionStorage(): Pick<Storage, 'getItem' | 'setItem' | 'removeItem'> | null {
  try {
    if (typeof window === 'undefined' || !window.sessionStorage) return null;
    return window.sessionStorage;
  } catch {
    return null;
  }
}
