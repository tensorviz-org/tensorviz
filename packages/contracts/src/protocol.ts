/** Shared host/view wire contract. Browser consumers receive checked copies through
 * tooling/contracts/sync.mjs so the frontend and Node compiler versions stay separate.
 * Keep this source compatible with the frontend compiler and free of host APIs.
 * Python owns the generated semantic and validation schemas imported below.
 */

/** Python-owned validation v1 contracts, ported from the validation redesign so the
 * explicit-profile engine can run behind this build's existing canvas. Raw RunSpec stays
 * host/worker-only: it can contain private argv and literals. */
import type { components as ValidationComponents } from './validationContracts.generated';
import type { components as SemanticComponents } from './semanticContracts.generated';
import { isNodeExplanationState, isModelExplanationState, type NodeExplanationState, type ModelExplanationState } from './nodeExplanation';
import { isModelCatalog, type ModelCatalog } from './modelCatalog';
export type LiteralEditCapability = SemanticComponents['schemas']['LiteralEditCapability'];
export type SetLiteralEdit = SemanticComponents['schemas']['SetLiteralEdit'];
export type ActivationEditCapability = SemanticComponents['schemas']['ActivationEditCapability'];
export type ReplaceActivationEdit = SemanticComponents['schemas']['ReplaceActivationEdit'];
export type ConstructorLiteralCapability = SemanticComponents['schemas']['ConstructorLiteralCapability'];
export type SetConstructorLiteralEdit = SemanticComponents['schemas']['SetConstructorLiteralEdit'];
export type ConfigurationLiteralCapability = SemanticComponents['schemas']['ConfigurationLiteralCapability'];
export type SetConfigurationLiteralEdit = SemanticComponents['schemas']['SetConfigurationLiteralEdit'];
export type ReconnectArgumentCapability = SemanticComponents['schemas']['ReconnectArgumentCapability'];
export type ReconnectArgumentEdit = SemanticComponents['schemas']['ReconnectArgumentEdit'];
export type InsertUnaryCapability = SemanticComponents['schemas']['InsertUnaryCapability'];
export type InsertUnaryEdit = SemanticComponents['schemas']['InsertUnaryEdit'];
export type DeleteUnaryCapability = SemanticComponents['schemas']['DeleteUnaryCapability'];
export type DeleteUnaryEdit = SemanticComponents['schemas']['DeleteUnaryEdit'];
export type CandidatePostconditions = SemanticComponents['schemas']['CandidatePostconditions'];
export type CaptureReconciliation = SemanticComponents['schemas']['CaptureReconciliation'];
export type SourceEditCapability = LiteralEditCapability | ActivationEditCapability | ConstructorLiteralCapability | ConfigurationLiteralCapability | ReconnectArgumentCapability | InsertUnaryCapability | DeleteUnaryCapability;
export type CapturedSourceEdit = SetLiteralEdit | ReplaceActivationEdit | SetConstructorLiteralEdit | SetConfigurationLiteralEdit | ReconnectArgumentEdit | InsertUnaryEdit | DeleteUnaryEdit;
export type SourceNumber = SetLiteralEdit['value'];
export const ACTIVATION_NAMES = ['relu', 'gelu', 'silu', 'sigmoid', 'tanh'] as const;
export function isActivationName(value: unknown): value is ReplaceActivationEdit['value'] {
  return typeof value === 'string' && (ACTIVATION_NAMES as readonly string[]).includes(value);
}
export type ValidationConfig = ValidationComponents['schemas']['ValidationConfig'];
export type RunSpec = ValidationComponents['schemas']['RunSpec'];
export type ValidationRequirement = ValidationComponents['schemas']['Requirement'];
export type ValidationResolution = ValidationComponents['schemas']['ValidationContracts']['resolution'];
export type ExecutionRecord = ValidationComponents['schemas']['ExecutionRecord'];
export type ValidationExecutionUpdate = ValidationComponents['schemas']['ExecutionUpdate'];
export type ValidationWorkerBootstrap = ValidationComponents['schemas']['WorkerBootstrap'];
export type ValidationWorkerMessage = ValidationComponents['schemas']['ValidationContracts']['worker'];
export type ValidationState = ValidationComponents['schemas']['ValidationContracts']['state'];

/** Bumped when a message's shape changes incompatibly; both sides refuse a mismatch. */
export const PROTOCOL_VERSION = 1;

/** The external-browser broker contract. Separate from the webview protocol so
 *  the local browser surface can version independently of the embedded panel. */
export const LOCAL_BROWSER_PROTOCOL_VERSION = 1;

export type LocalBrowserRole = 'host' | 'view';
export const LOCAL_BROWSER_ROLES = ['host', 'view'] as const;
export type LocalBrowserErrorCode =
  | 'PROTOCOL_VERSION_MISMATCH'
  | 'SESSION_MISMATCH'
  | 'INVALID_ORIGIN'
  | 'INVALID_CREDENTIALS'
  | 'CREDENTIAL_EXPIRED'
  | 'HOST_ALREADY_CONNECTED'
  | 'STALE_TARGET'
  | 'UNKNOWN_MESSAGE'
  | 'INVALID_MESSAGE';

export interface LocalBrowserTargetIdentity {
  workspaceFolder: string;
  file: string;
  cls: string;
  documentVersion: number;
  sourceDigest: string;
  backendRevision: number;
}

/** What the page is looking at when the subject is a PROJECT rather than one class.
 *
 * Deliberately a second type, not a widening of the six fields above. `file`,
 * `documentVersion` and `sourceDigest` are inherently per-file — there is no project
 * analogue for "the version of the document" — so making them optional would turn an
 * all-fields equality check into a partial one, and that check is the staleness
 * refusal protecting `revealNode`, `focusNode` and every Ask request. The class
 * identity is therefore untouched and survives as the per-frame drill-down target.
 *
 * `indexDigest` plays the role `sourceDigest` plays for a class: it changes whenever
 * anything in the indexed source set changes, so a page holding a stale project is
 * refused rather than allowed to act on it. `indexRevision` orders two indexes of one
 * workspace the way `backendRevision` orders two projections of one class. */
export interface LocalBrowserProjectIdentity {
  workspaceFolder: string;
  /** The derived `sys.path` root this project resolves against: empty at the
   *  workspace folder itself, `packages/alpha` for a project nested inside one. */
  projectRoot: string;
  /** The focused experiment, or '' when the canvas is showing none. */
  experimentId: string;
  indexDigest: string;
  indexRevision: number;
}

export interface LocalBrowserCapabilities {
  readOnly: true;
  canNavigateSource: true;
  canWriteSource: false;
  canRunAgent: false;
  canEditGraph: false;
  /** Proposals only; native host review remains the sole write authority. */
  canProposeSourceEdits?: true;
}

/** The security metadata a view needs after its fragment credential has been
 *  consumed and erased. The credential itself never appears in a message. */
export interface LocalBrowserSessionInfo {
  workspaceSessionId: string;
  role: LocalBrowserRole;
  protocolVersion: number;
  capabilities: LocalBrowserCapabilities;
  expiresAt: string;
  /** Broker-assigned identity of this view connection, never a credential. */
  viewId?: string;
  /** Non-secret correlation for an agent-opened browser tab. */
  handoffId?: string;
}

export interface LocalBrowserEnvelope<Type extends string> {
  type: Type;
  protocolVersion: number;
  workspaceSessionId: string;
  messageId: string;
}

export type LocalBrowserHostHandshakeMessage = LocalBrowserEnvelope<'hostHandshake'> & {
  role: 'host';
  viewControl?: true;
};

export type LocalBrowserViewHandshakeMessage = LocalBrowserEnvelope<'viewHandshake'> & {
  role: 'view';
  handoffId?: string;
};

export type LocalBrowserConnectionMessage = LocalBrowserEnvelope<'connected'> & {
  role: LocalBrowserRole;
  session: LocalBrowserSessionInfo;
};

export type LocalBrowserSnapshotRequestMessage = LocalBrowserEnvelope<'snapshotRequest'> & {
  role: 'view';
  lastHostSequence: number;
};

interface LocalBrowserSnapshotStateBase {
  projection: unknown;
  traced: boolean;
  greyed: boolean;
  connectionState: 'connected' | 'reconnecting' | 'disconnected';
  /** Latest host-owned attempt summary, independent of the displayed projection. */
  validation?: LocalBrowserValidationSummary;
  opening?: GraphOpeningState;
}

export type LocalBrowserEmptySnapshotState = LocalBrowserSnapshotStateBase & {
  activeTarget: null;
  projection: null;
  traced: false;
  greyed: false;
  syncState: null;
  reason?: never;
};

export type LocalBrowserActiveSnapshotState = LocalBrowserSnapshotStateBase & {
  activeTarget: LocalBrowserTargetIdentity;
  syncState: 'source-only' | 'runtime-traced';
  reason?: never;
};

export type LocalBrowserStaleSnapshotState = LocalBrowserSnapshotStateBase & {
  activeTarget: LocalBrowserTargetIdentity;
  syncState: 'stale';
  reason: string;
};

export type LocalBrowserSnapshotState =
  | LocalBrowserEmptySnapshotState
  | LocalBrowserActiveSnapshotState
  | LocalBrowserStaleSnapshotState;

export type LocalBrowserSnapshotMessage = LocalBrowserEnvelope<'snapshot'> & {
  role: 'host';
  hostSequence: number;
  state: LocalBrowserSnapshotState;
};

export type LocalBrowserActiveTargetMessage = LocalBrowserEnvelope<'activeTarget'> & {
  hostSequence: number;
  target: LocalBrowserTargetIdentity;
};

export type LocalBrowserProjectionMessage = LocalBrowserEnvelope<'projection'> & {
  hostSequence: number;
  target: LocalBrowserTargetIdentity;
  projection: unknown;
  traced: boolean;
  greyed: boolean;
};

/** What the project contains: experiments, their status, and the files nothing
 *  reaches. The rail is built from this; it carries no graph, so it stays small
 *  even for a large workspace. */
export type LocalBrowserProjectIndexMessage = LocalBrowserEnvelope<'projectIndex'> & {
  hostSequence: number;
  project: LocalBrowserProjectIdentity;
  index: unknown;
  /** The caller's admission that its file set was incomplete, so a partial project
   *  never looks whole. */
  truncated: boolean;
  /** Files actually indexed, and files enumeration saw before any budget applied.
   *
   *  Two numbers rather than one flag, because "part of the project" is not an
   *  answer a reader can act on: 592 of 596 is a rounding error and 200 of 4,000 is
   *  a different tool. Only the host can supply `discovered` — the backend sees
   *  what was uploaded, which is the capped set, so it cannot know what was left
   *  out. */
  indexed?: number;
  discovered?: number;
};

/** One experiment, drawn. Sent when the rail selection changes or a frame opens —
 *  not on every index, because only expanded frames project their contents and a
 *  collapsed project is cheap to hold but pointless to redraw. */
export type LocalBrowserProjectDocumentMessage = LocalBrowserEnvelope<'projectDocument'> & {
  hostSequence: number;
  project: LocalBrowserProjectIdentity;
  document: unknown;
  spanmap: unknown;
  /** Frame ids currently open, echoed so page and host cannot disagree. */
  expanded: string[];
};

/** The page asking to open or close a frame.
 *
 *  Named `projectExpand` rather than reusing `expandNode`: that message already
 *  exists on the webview union and means "re-run the worker one level deeper",
 *  which EXECUTES the model. Static expansion and trace-deepening are different
 *  operations wearing one gesture, and conflating them in the protocol would make
 *  a click that merely opens a card indistinguishable from one that runs code. */
export type LocalBrowserProjectExpandMessage = LocalBrowserEnvelope<'projectExpand'> & {
  project: LocalBrowserProjectIdentity;
  nodeId: string;
  expanded: boolean;
};

/** The page confirming a guess: write this declaration into the source.
 *
 *  A proposal, not a write. The host validates the identity, calls the backend for
 *  a patch, and puts it in front of the user as a native diff they accept or
 *  discard — `canProposeSourceEdits`, never `canWriteSource`.
 *
 *  The page names a declaration in the graph's own vocabulary (a kind and labels)
 *  and never a file, a line or a line of Python. Which file it lands in is the
 *  host's decision, from its own index. */
export type LocalBrowserProjectDeclareMessage = LocalBrowserEnvelope<'projectDeclare'> & {
  project: LocalBrowserProjectIdentity;
  declaration: 'node' | 'edge' | 'experiment';
  /** node: [kind, label] · edge: [from, to, relationship] · experiment: [name] */
  args: string[];
};

/** The page asking to focus a different experiment. */
export type LocalBrowserProjectFocusMessage = LocalBrowserEnvelope<'projectFocus'> & {
  project: LocalBrowserProjectIdentity;
  experimentId: string;
};

export type LocalBrowserConnectionStateMessage = LocalBrowserEnvelope<'connectionState'> & {
  state: 'connected' | 'reconnecting' | 'disconnected';
};

/** Broker -> host: whether ANY browser view holds a socket on this session.
 *
 *  The exact mirror of `connectionState`, which is broker -> views about the one
 *  host. Without it the host cannot tell a closed tab from a live one: closing the
 *  browser kills only the VIEW socket, the host's own socket stays connected, and
 *  "Open as Graph" took its retarget branch and published into an empty roster —
 *  the command appeared to do nothing.
 *
 *  Edge-triggered: `detached` means the LAST view left, not that a view left. The
 *  level is re-stated at every host handshake, because a new host epoch knows
 *  nothing about what came before. Broker-owned: neither peer may send it. */
export type LocalBrowserViewPresenceMessage = LocalBrowserEnvelope<'viewState'> & {
  state: 'attached' | 'detached';
};

export type LocalBrowserSyncStateMessage = LocalBrowserEnvelope<'syncState'> & {
  hostSequence: number;
  target: LocalBrowserTargetIdentity;
  state: 'source-only' | 'runtime-traced' | 'stale';
  reason?: string;
};

export type LocalBrowserStaleStateMessage = LocalBrowserEnvelope<'staleState'> & {
  hostSequence: number;
  target: LocalBrowserTargetIdentity;
  lastValidProjection: unknown;
  reason: string;
};

export type LocalBrowserRevealNodeMessage = LocalBrowserEnvelope<'revealNode'> & {
  target: LocalBrowserTargetIdentity;
  nodeId: string;
};

export type LocalBrowserFocusNodeMessage = LocalBrowserEnvelope<'focusNode'> & {
  hostSequence: number;
  target: LocalBrowserTargetIdentity;
  nodeId: string;
  /** Every instance whose span covers the cursor's line, sorted, innermost
   *  span width only. Present since the multi-file projection: one definition's
   *  line can be several nodes at once (`enc1.fc` and `enc2.fc` both point at
   *  `Block`'s declaration), and picking one silently would be a guess. `nodeId`
   *  stays the deterministic primary — `nodeIds[0]` — so older pages keep working. */
  nodeIds?: string[];
};

/** Unsequenced, view-specific presentation traffic. The broker owns viewState. */
export type LocalBrowserViewStateMessage = LocalBrowserEnvelope<'viewState'> & {
  viewId: string;
  handoffId?: string;
  connected: boolean;
};
export type LocalBrowserViewSelectionMessage = LocalBrowserEnvelope<'viewSelection'> & {
  /** An oversized selection is explicitly unavailable, never silently narrowed. */
  tooLarge?: true;
  /** Code focus: true opens the selection's source beside the Canvas, false leaves
   *  the editor alone. Absent keeps the code focus setting's own behaviour. */
  follow?: boolean;
  viewId: string;
  target: LocalBrowserTargetIdentity;
  nodeIds: string[];
  edgeIds: string[];
};
export type LocalBrowserFocusViewMessage = LocalBrowserEnvelope<'focusView'> & {
  viewId: string;
  target: LocalBrowserTargetIdentity;
  requestId: string;
  nodeIds: string[];
  edgeIds: string[];
};
export type LocalBrowserViewFocusResultMessage = LocalBrowserEnvelope<'viewFocusResult'> & {
  viewId: string;
  target: LocalBrowserTargetIdentity;
  requestId: string;
  ok: boolean;
  detail: string;
};

export interface CodeFocusOptions {
  enabled: boolean;
  scope: 'class' | 'function' | 'node';
  references: 'declaration' | 'uses' | 'both';
  opacity: number;
}
export const DEFAULT_CODE_FOCUS: CodeFocusOptions = { enabled: false, scope: 'node', references: 'both', opacity: 25 };
export function isCodeFocusOptions(value: unknown): value is CodeFocusOptions {
  if (!isRecord(value)) return false;
  return Object.keys(value).length === 4 && typeof value.enabled === 'boolean'
    && typeof value.scope === 'string' && ['class', 'function', 'node'].includes(value.scope)
    && typeof value.references === 'string' && ['declaration', 'uses', 'both'].includes(value.references)
    && typeof value.opacity === 'number' && Number.isInteger(value.opacity) && value.opacity >= 10 && value.opacity <= 60;
}
export type CodeFocusRequest = LocalBrowserEnvelope<'codeFocusRequest'> & {
  viewId: string;
  target: LocalBrowserTargetIdentity;
  options: CodeFocusOptions;
};
export type CodeFocusState = LocalBrowserEnvelope<'codeFocusState'> & {
  viewId: string;
  target: LocalBrowserTargetIdentity;
  options: CodeFocusOptions;
  parameter: boolean;
  detail: string;
};

export const LOCAL_BROWSER_REQUEST_ID_MAX_LENGTH = 128;

/** Read-only paths into one immutable retained capture. Containers are paged at
 * each level, so a large argument tree never requires an oversized record. */
export interface CaptureInspectionQuery {
  captureId: string;
  sourceId: string;
  entity?: { kind: 'operation' | 'invocation' | 'value' | 'input' | 'output'; id: string };
  path: Array<string | number>;
  offset: number;
  limit: number;
}
export interface CaptureInspectionEntry {
  key: string | number | null;
  kind: 'object' | 'array' | 'string' | 'number' | 'boolean' | 'null';
  count?: number;
  value?: string | number | boolean | null;
  label?: string;
}
export interface CaptureInspectionPage {
  captureId: string;
  sourceId: string;
  entity?: CaptureInspectionQuery['entity'];
  path: Array<string | number>;
  node: CaptureInspectionEntry;
  items: CaptureInspectionEntry[];
  total: number;
  nextOffset: number | null;
}
export type LocalBrowserCaptureInspectRequestMessage = LocalBrowserEnvelope<'captureInspectRequest'> & {
  target: LocalBrowserTargetIdentity; requestId: string; query: CaptureInspectionQuery;
};
export type LocalBrowserCaptureInspectCancelMessage = LocalBrowserEnvelope<'captureInspectCancel'> & {
  target: LocalBrowserTargetIdentity; requestId: string;
};
export type LocalBrowserCaptureInspectResultMessage = LocalBrowserEnvelope<'captureInspectResult'> & {
  target: LocalBrowserTargetIdentity; requestId: string; page: CaptureInspectionPage | null; detail: string;
};

export const LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES = 16 * 1024 * 1024;
export interface CaptureExportReference { captureId: string; sourceId: string }
export type LocalBrowserCaptureExportRequestMessage = LocalBrowserEnvelope<'captureExportRequest'> & {
  target: LocalBrowserTargetIdentity; requestId: string; reference: CaptureExportReference;
};
export type LocalBrowserCaptureExportCancelMessage = LocalBrowserEnvelope<'captureExportCancel'> & {
  target: LocalBrowserTargetIdentity; requestId: string;
};
export type LocalBrowserCaptureExportResultMessage = LocalBrowserEnvelope<'captureExportResult'> & {
  target: LocalBrowserTargetIdentity; requestId: string; reference: CaptureExportReference;
  /** Validated with the Python-generated portable schema by the final consumer. */
  capture: unknown | null; detail: string;
};

export type SourceEditValue = string | number | boolean | null | SourceEditValue[];
export type LocalBrowserSourceEdit =
  | (CapturedSourceEdit & { nodeId: string })
  | { kind: 'set_param'; nodeId: string; param: string; value: SourceEditValue }
  | { kind: 'add_layer'; afterNodeId: string; nodeId: string; layerType: string; params: Record<string, SourceEditValue> }
  | { kind: 'remove_layer'; nodeId: string }
  | { kind: 'group_layers'; nodeIds: string[]; name: string };

export type LocalBrowserSourceEditRequestMessage = LocalBrowserEnvelope<'sourceEditRequest'> & {
  /** Broker verifies this connection identity; omitted by older static-only pages. */
  viewId?: string;
  target: LocalBrowserTargetIdentity;
  requestId: string;
  edit: LocalBrowserSourceEdit;
};
export type LocalBrowserSourceEditCancelMessage = LocalBrowserEnvelope<'sourceEditCancel'> & {
  target: LocalBrowserTargetIdentity;
  requestId: string;
};
export type LocalBrowserSourceEditResultMessage = LocalBrowserEnvelope<'sourceEditResult'> & {
  target: LocalBrowserTargetIdentity;
  requestId: string;
  status: 'applied' | 'cancelled' | 'refused';
  detail: string;
};

export interface CandidateCheckpointSummary {
  loadStatus: 'succeeded' | 'failed'; complete: boolean; entries: number;
  shapeDifferences: number; dtypeDifferences: number; registeredOnly: number; checkpointOnly: number; unknownPresence: number;
}
/** Read-only human-readable metadata page; structured records remain available through MCP. */
export interface CandidateCheckpointPage {
  runId: string; offset: number; count: number; nextOffset: number | null; text: string;
}
export interface CandidateReviewSnapshot {
  runId?: string;
  staticCheck: 'passed';
  runtime: 'not_run' | 'running' | 'succeeded' | 'failed' | 'refused' | 'cancelled' | 'timed_out';
  freshness: 'unverified' | 'current' | 'stale';
  capture: 'unavailable' | 'complete' | 'partial' | 'failed';
  inputs: 'same' | 'different' | 'unavailable';
  state: 'same' | 'different' | 'unavailable';
  semanticPostconditions: 'not_checked' | CandidatePostconditions['status'];
  postconditions?: CandidatePostconditions;
  checkpoint?: CandidateCheckpointSummary;
  cleanup: 'not_run' | 'pending' | 'complete' | 'failed' | 'unavailable';
  diagnostics: string[];
}
export interface CandidateRecord {
  runId: string; startedAt: number; finishedAt: number | null;
  review: 'open' | 'historical'; summary: CandidateReviewSnapshot;
  executionId: string | null;
  capture: { captureId: string; sourceId: string } | null;
  unavailable: 'not_captured' | 'invalid_capture' | 'too_large' | 'evicted' | null;
}
export interface AppliedSourceFile { file: string; beforeDigest: string; afterDigest: string }
export interface AppliedSourceReceipt extends AppliedSourceFile {
  appliedAt: number;
  /** Complete source set; the legacy single-file fields identify its first entry. */
  files?: AppliedSourceFile[];
  unavailable: 'too_large' | 'evicted' | null;
}
export interface AppliedSourceSetComparison {
  baseCaptureId: string; captureId: string; sourceId: string;
  reason: CaptureReconciliation['reason']; dependencyScope: 'complete' | 'partial';
  inputs: 'same' | 'different' | 'unavailable'; state: 'same' | 'different' | 'unavailable';
  semanticPostconditions: 'not_checked';
  comparison: { equivalent: boolean | null; coverageComplete: boolean;
    changedFieldCount: number; observedFieldCount: number; unresolvedFieldCount: number;
    unmatchedRecordCount: number; uncertainMatchCount: number;
    connectionChangeCount: number; unresolvedConnectionCount: number } | null;
}
export interface AppliedSourceObservation {
  receipt: AppliedSourceReceipt; checkedAt: number;
  source: 'reviewed_unsaved' | 'reviewed_saved' | 'reverted' | 'changed' | 'unavailable';
  captureFreshness: 'current' | 'stale' | 'unavailable';
  reconciliation: CaptureReconciliation | null;
  sourceSetComparison?: AppliedSourceSetComparison;
  unavailable: 'source_not_saved' | 'source_changed' | 'source_unavailable' | 'evidence_unavailable'
    | 'capture_unavailable' | 'check_stale' | 'check_failed' | null;
}
export type CandidateAction = { kind: 'run'; retryId: string } | { kind: 'cancel'; runId: string }
  | { kind: 'checkpoint'; runId: string; offset: number; limit: number }
  | { kind: 'list' } | { kind: 'reconcile' } | { kind: 'inspect'; runId: string; query?: CaptureInspectionQuery };
export type LocalBrowserSourceEditReviewMessage = LocalBrowserEnvelope<'sourceEditReview'> & {
  requestId: string; target: LocalBrowserTargetIdentity; candidate: CandidateReviewSnapshot | null;
};
export type LocalBrowserCandidateRequestMessage = LocalBrowserEnvelope<'candidateRequest'> & {
  viewId: string; requestId: string; proposalId: string; target: LocalBrowserTargetIdentity; action: CandidateAction;
};
export type LocalBrowserCandidateResultMessage = LocalBrowserEnvelope<'candidateResult'> & {
  viewId: string; requestId: string; proposalId: string; target: LocalBrowserTargetIdentity;
  result: { candidates: CandidateRecord[]; page: CaptureInspectionPage | null; applied?: AppliedSourceObservation;
    checkpoint?: CandidateCheckpointPage } | null; detail: string;
};

export interface GraphSetupField {
  /** Verified declaration location, not the origin of an edited value. */
  source?: { file: string; line: number };
  name: string;
  kind: 'positional' | 'keyword' | 'keyword_only' | 'varargs' | 'kwargs';
  required: boolean;
  annotation?: string;
  defaultText?: string;
  defaultValue?: unknown;
}
export interface GraphSetupDiscovery {
  model: string;
  sourceDigest: string;
  init: GraphSetupField[] | null;
  inputs: GraphSetupField[] | null;
  examples: Array<{ file: string; line: number; init: Record<string, unknown>; inputs?: Record<string, unknown> }>;
  factories: Array<{ file: string; callable: string; line: number; fields: GraphSetupField[] }>;
  configs: Array<{ file: string; cls: string; parameter: string | null; fields: GraphSetupField[] | null; construction: 'call' | 'attributes' }>;
  parents: Array<{ file: string; cls: string; attribute: string; line: number }>;
  constraints: Array<{ input: string; axis: number; file: string; line: number; size?: number; initParameter?: string }>;
  truncated: boolean;
  resolution?: {
    status: 'complete' | 'partial';
    runtimeInputs?: boolean;
    draft: GraphSetupDraft;
    evidence: GraphSetupEvidence[];
    unresolved: string[];
  };
}
export interface GraphSetupEvidence {
  field: string;
  kind: 'repository' | 'source' | 'example';
  detail: string;
  file?: string;
  line?: number;
}
export interface GraphSetupDraft {
  profile: Record<string, unknown>;
  profileName?: string;
  configObject?: { index: number; parameter: string; values: Record<string, unknown> };
  configDefaults?: true;
}
/** Metadata presence is not proof that the model imports or its forward succeeds. */
export interface GraphEnvironmentCandidate {
  id: string;
  interpreter: string;
  source: string;
  selected?: true;
  pythonSelected?: true;
  managed?: true;
  status: 'libraries_present' | 'missing_libraries' | 'incompatible' | 'unavailable';
  python?: string;
  torch?: string | null;
  prefix?: string;
  implementation?: string;
  abi?: string | null;
  architecture?: string;
  fingerprint?: string;
  missing?: string[];
  incompatible?: string[];
  detail?: string;
  /** The opening's static model check ran for this environment as well as the capture check. */
  modelChecked?: true;
  /** Required model libraries that are not installed: the requirement when known, otherwise the module. */
  modelMissing?: string[];
  /** Installed versions that do not satisfy the project's declared requirements. */
  modelIncompatible?: string[];
}
export interface GraphEnvironmentRecovery {
  reason: 'ready' | 'selection_required' | 'unavailable' | 'missing_libraries' | 'incompatible' | 'worker_failed';
  detail: string;
  candidates: GraphEnvironmentCandidate[];
  selectedId?: string;
  missingModules?: string[];
  /** Exact known package requirements, never guesses from import names. */
  requirements?: string[];
  compatibilityPackages?: string[];
  /** Project-declared requirements the chosen environment's installed versions do not satisfy.
   * Present only when the host can run with the installed versions instead. */
  declaredPins?: string[];
  /** The host can replace the installed versions with these pins in the saved environment itself,
   * after native review. Never offered for the tensorViz-managed environment. */
  replaceable?: true;
  /** Native host can create an isolated environment and resume this opening. */
  canCreate?: true;
  /** The candidate needing the least to run this model, preselected unless the saved choice is the user's own. */
  recommendedId?: string;
}
/** Observations from the extension's execution host, never a device reservation. */
export interface ExecutionResources {
  sampledAt: number;
  host: string;
  remote: boolean;
  cpu: string;
  cores: number;
  cpuPercent: number | null;
  memoryFree: number;
  memoryTotal: number;
  devices: Array<{ id: string; name: string; utilization: number | null; memoryFree: number | null; memoryTotal: number | null }>;
  detail: string;
}
export interface ExecutionReviewState {
  id: string;
  phase: 'review' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled';
  kind: 'prepare' | 'capture' | 'trace';
  configuredDevice: string;
  device: string;
  detail: string;
  stages: Array<{ label: string; calls: number }>;
  inputs: string;
  mode: string;
  gradients: boolean;
  interpreter: string;
  profileName?: string;
  resources?: ExecutionResources;
  /** Set on a finished invocation whose result was reused instead of run (the model was not run again). */
  reused?: ExecutionReuse;
  /** This invocation bypasses reused results (Run fresh). */
  fresh?: true;
}
export type ExecutionReviewAction = { kind: 'execution'; reviewId: string; device: string; action: 'run' | 'cancel' | 'refresh' | 'select' };

export interface GraphOpeningState {
  id: string;
  target: LocalBrowserTargetIdentity;
  execution?: ExecutionReviewState;
  phase: 'awaiting_execution' | 'resolving' | 'needs_setup' | 'saving' | 'capturing' | 'ready' | 'failed' | 'cancelled' | 'source_only';
  detail: string;
  hasCapture: boolean;
  /** Source invalidation can be refreshed; an unmarked cancellation must stop. */
  cancellationReason?: 'source_changed';
  /** Undefined means presence could not be checked; it does not mean absent. */
  configFileExists?: boolean;
  discovery?: GraphSetupDiscovery;
  draft?: GraphSetupDraft;
  requirements?: string[];
  evidence?: GraphSetupEvidence[];
  failure?: GraphFailure;
  environment?: GraphEnvironmentRecovery;
  /** Name of the current native environment creation, retained across reconnects. */
  environmentName?: string;
}
/** Recovery is chosen by the host from structured diagnostic codes and phases. */
export interface GraphFailure {
  kind: 'setup' | 'source' | 'environment' | 'execution' | 'capture';
  /** Host-owned environment diagnostic; renderers never infer recovery from prose. */
  code?: string;
  fields?: string[];
  missingModules?: string[];
  importFailure?: true;
  diagnostic?: ModelImportDiagnostic;
}
/** Bounded observed import evidence, retained even when a failed temporary Python is removed. */
export interface ModelImportDiagnostic {
  exceptionType: string;
  message: string;
  traceback: string;
  causes: Array<{ exceptionType: string; message: string; module?: string }>;
  runtime: { platform: string; architecture: string; python: string; torch: string | null; cuda: string | null;
    cudaAvailable: boolean | null; mpsAvailable: boolean | null };
  truncated: boolean;
}
export type GraphSetupAction =
  | ExecutionReviewAction
  | { kind: 'edit_execution'; reviewId: string; field: (string | number)[] }
  | { kind: 'submit'; draft: GraphSetupDraft }
  | { kind: 'parent'; index: number }
  | { kind: 'factory_template' | 'agent_help'; draft?: GraphSetupDraft }
  | { kind: 'reveal_setup_source'; file: string; line: number }
  | { kind: 'select_environment'; candidateId: string }
  | { kind: 'create_environment'; name?: string }
  | { kind: 'install_dependencies' | 'refresh_environments' | 'use_installed_versions' | 'replace_versions' }
  | { kind: 'retry' | 'change_inputs' | 'cancel' | 'source_only' | 'save_sources' | 'environment' | 'edit_config' };
export type LocalBrowserGraphOpeningMessage = LocalBrowserEnvelope<'graphOpening'> & {
  hostSequence: number;
  state: GraphOpeningState;
};
export type LocalBrowserGraphSetupRequestMessage = LocalBrowserEnvelope<'graphSetupRequest'> & {
  target: LocalBrowserTargetIdentity;
  openingId: string;
  requestId: string;
  action: GraphSetupAction;
};

/** A manual runtime check; all execution details are resolved by the host. */
export type LocalBrowserValidateRequestMessage = LocalBrowserEnvelope<'validateRequest'> & {
  target: LocalBrowserTargetIdentity;
  requestId: string;
  /** Run fresh: never return a reused identical result (execution cache) for this run. */
  fresh?: true;
};
export type LocalBrowserValidateCancelMessage = LocalBrowserEnvelope<'validateCancel'> & {
  target: LocalBrowserTargetIdentity;
  requestId: string;
};
export interface LocalBrowserValidationOutcome {
  /** Worker deadline expiry is failed; cancelled is an explicit/lifecycle stop.
   *  The host owns timeout policy, so views must not impose a shorter deadline. */
  status: 'passed' | 'failed' | 'refused' | 'cancelled';
  detail: string;
  failure?: GraphFailure;
  /** No Python ran: the graph is an earlier identical run's result (execution cache), and
   *  `capturedAt` (epoch ms) is when that run finished. */
  reused?: ExecutionReuse;
}
/** An identical earlier result shown instead of running (execution cache, `on` mode). */
export interface ExecutionReuse { capturedAt: number }
export function isExecutionReuse(value: unknown): value is ExecutionReuse {
  return isRecord(value) && Object.keys(value).length === 1 && typeof value.capturedAt === 'number'
    && Number.isSafeInteger(value.capturedAt) && value.capturedAt >= 0;
}
export type LocalBrowserValidateResultMessage = LocalBrowserEnvelope<'validateResult'> &
  LocalBrowserValidationOutcome & { target: LocalBrowserTargetIdentity; requestId: string };

/** Host-owned manual/MCP progress; completion names only its measured projection. */
export interface LocalBrowserValidationEvidence {
  origin: 'manual' | 'mcp' | 'automatic';
  attemptStatus: 'requested' | 'running' | 'needs_setup' | 'succeeded' | 'failed' | 'cancelled' | 'timed_out';
  sourceId: string | null;
  planDigest: string | null;
  lastCompleteCaptureId: string | null;
  lastPartialCaptureId: string | null;
  displayedCapture: {
    id: string;
    attemptId: string;
    format: 'legacy' | 'semantic';
    status: 'complete' | 'partial' | 'failed';
    forwardStatus: 'passed' | 'failed';
    sourceId: string | null;
    planDigest: string | null;
    unavailable: 'retention_limit' | null;
  } | null;
  /** Only non-sensitive setup labels; recipes, inputs and environment paths stay in the host. */
  setup: { origin: 'profile' | 'explicit'; profileName: string | null; device: string; mode: 'eval' | 'train' | 'preserve' } | null;
}

export interface LocalBrowserValidationSummary {
  runId: string;
  target: LocalBrowserTargetIdentity;
  status: 'running' | LocalBrowserValidationOutcome['status'];
  detail: string;
  projectionRevision?: number;
  evidence?: LocalBrowserValidationEvidence;
}
export type LocalBrowserValidationStateMessage = LocalBrowserEnvelope<'validationState'> & LocalBrowserValidationSummary;

export type LocalBrowserAcknowledgementMessage =
  LocalBrowserEnvelope<'ack'> & {
    target: LocalBrowserTargetIdentity;
    ackId: string;
    ok: boolean;
    errorCode?: LocalBrowserErrorCode;
    detail?: string;
  };

export type LocalBrowserErrorMessage = LocalBrowserEnvelope<'error'> & {
  target?: LocalBrowserTargetIdentity;
  code: LocalBrowserErrorCode;
  message: string;
};

/** Native Git review: immutable source projections, never an execution result. */
export interface GraphReviewState {
  id: string;
  /** Pending native source proposal; never a capture or an applied edit. */
  proposal?: { id: string; unavailable?: boolean; canRevise?: boolean };
  /** Standalone source-copy root; immutable Git snapshots still label each side. */
  workspaceRoot?: string;
  beforeLabel: string;
  afterLabel: string;
  before: ProjectionPayload;
  after: ProjectionPayload;
  changedFiles: string[];
  stale: boolean;
  /** The retained local model may provide visual context, never revision evidence. */
  liveContext?: Pick<LocalBrowserTargetIdentity, 'workspaceFolder' | 'file' | 'cls'>;
  title?: string;
  models?: { id: string; label: string; file: string; status: 'modified' | 'added' | 'removed' | 'renamed' | 'unresolved' }[];
  selectedModel?: string;
  modelStatus?: 'modified' | 'added' | 'removed' | 'renamed' | 'unresolved';
  notes?: string[];
  /** Discovery limits outside the changed files and selected model's dependencies. */
  repositoryNotes?: string[];
  updated?: boolean;
  /** Proven root-file rename, used only for identity matching, never navigation. */
  renamedFiles?: Record<string, string>;
}
export type GraphReviewRequest = LocalBrowserEnvelope<'graphReviewRequest'> & {
  action: 'open' | 'openPR' | 'refresh' | 'close' | 'reveal' | 'selectModel' | 'acceptDraft' | 'discardDraft';
  reviewId?: string;
  side?: 'before' | 'after';
  nodeId?: string;
  modelId?: string;
};
export type GraphReviewMessage = LocalBrowserEnvelope<'graphReviewState'> & { state: GraphReviewState | null };
export type GraphReviewFocus = LocalBrowserEnvelope<'graphReviewFocus'> & {
  reviewId: string; side: 'before' | 'after'; nodeIds: string[];
};
/** A view acknowledges the exact review only after its canvas renders. */
export type GraphReviewApplied = LocalBrowserEnvelope<'graphReviewApplied'> & {
  reviewId: string; viewId: string; mode: 'before' | 'diff' | 'after';
};

export type CanvasAgentProvider = 'codex' | 'claude';
export interface CanvasAgentSettings { model?: string; effort?: string; }
export interface CanvasAgentModel {
  id: string;
  label: string;
  description: string;
  efforts: string[];
  defaultEffort?: string;
}
export interface CanvasAgentAvailability {
  status: 'unchecked' | 'checking' | 'ready' | 'notInstalled' | 'signInRequired' | 'error';
  detail: string;
  models: CanvasAgentModel[];
  defaultModel?: string;
  checkedAt?: number;
}
export type CanvasAgentSelections = Partial<Record<CanvasAgentProvider, CanvasAgentSettings>>;
export interface CanvasAgentTurn {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  settings?: CanvasAgentSettings;
}
export type CanvasAgentPermissionMode = 'ask' | 'auto_setup' | 'full_setup';
/** Effects describe authority, independently of operation names or generated wording. */
export type CanvasAgentPermissionEffect = 'create' | 'install' | 'replace' | 'execute_code';
export type CanvasAgentPermissionFact = null | boolean | number | string | CanvasAgentPermissionFact[] | { [key: string]: CanvasAgentPermissionFact };
/** Host-owned facts from the actual operation. No generated copy or executable agent input. */
export interface CanvasAgentPermissionRequest {
  operation: string;
  scope: 'isolated_environment' | 'existing_environment' | 'machine';
  effects: CanvasAgentPermissionEffect[];
  facts: Record<string, CanvasAgentPermissionFact>;
}
/** The active agent explains an existing host request; this never grants authority. */
export interface CanvasAgentPermissionPresentation {
  title: string;
  summary: string;
  allowLabel: string;
  denyLabel: string;
}
export interface CanvasAgentPermission extends CanvasAgentPermissionRequest {
  id: string;
  requestId: string;
  status: 'pending' | 'approved' | 'denied' | 'cancelled';
  presentation?: CanvasAgentPermissionPresentation;
  approval?: 'user' | 'policy';
  /** Recorded only when a chat policy approved this exact setup request. */
  mode?: CanvasAgentPermissionMode;
}
export const CANVAS_AGENT_PERMISSION_LIMIT = 20;
export interface CanvasAgentState {
  sessionId: string;
  revision: number;
  provider: CanvasAgentProvider;
  ack?: string;
  selections?: CanvasAgentSelections;
  agents?: Record<CanvasAgentProvider, CanvasAgentAvailability>;
  status: 'idle' | 'running' | 'stopped' | 'error';
  messages: CanvasAgentTurn[];
  activity: string;
  /** Host-confirmed setup evidence, available for inspection after a failed task. */
  failure?: GraphFailure;
  requestId?: string;
  context?: { target: LocalBrowserTargetIdentity; nodeIds: string[] };
  permissions?: CanvasAgentPermission[];
  /** This conversation only. Missing on older hosts means ask. */
  permissionMode?: CanvasAgentPermissionMode;
}
export type CanvasAgentAction =
  | { action: 'sync' }
  | { action: 'discover'; refresh?: boolean; /** Inspect the selected provider without preparing a chat or changing providers. */ catalogOnly?: boolean }
  | { action: 'configure'; provider: CanvasAgentProvider; settings: CanvasAgentSettings; requestId?: string }
  | { action: 'clear' }
  | { action: 'stop'; requestId: string }
  | { action: 'permission'; permissionId: string; decision: 'allow' | 'deny' }
  | { action: 'permission_mode'; mode: CanvasAgentPermissionMode }
  | { action: 'send'; requestId: string; provider: CanvasAgentProvider; prompt: string; settings?: CanvasAgentSettings;
      target: LocalBrowserTargetIdentity; nodeIds: string[]; draftId?: string };
export type LocalBrowserAgentRequest = LocalBrowserEnvelope<'agentRequest'> & { viewId: string } & CanvasAgentAction;
export type LocalBrowserAgentState = LocalBrowserEnvelope<'agentState'> & { viewId: string; state: CanvasAgentState };
export type LocalBrowserNodeExplanation = LocalBrowserEnvelope<'nodeExplanation'> & {
  viewId: string; target: LocalBrowserTargetIdentity; state: NodeExplanationState;
};
export type LocalBrowserNodeExplanationRequest = LocalBrowserEnvelope<'nodeExplanationRequest'> & {
  viewId: string; target: LocalBrowserTargetIdentity; nodeId: string; requestId: string;
} & ({ action: 'retry' } | { action: 'reveal'; citationId: string });

export type LocalBrowserModelExplanation = LocalBrowserEnvelope<'modelExplanation'> & {
  viewId: string; target: LocalBrowserTargetIdentity; state: ModelExplanationState;
};
export type LocalBrowserModelExplanationRequest = LocalBrowserEnvelope<'modelExplanationRequest'> & {
  viewId: string; target: LocalBrowserTargetIdentity; requestId: string;
  action: 'start' | 'cancel'; nodeIds: string[];
  /** A per-pass choice. Missing retains the chat selection; an empty object uses provider defaults. */
  settings?: CanvasAgentSettings;
};

export type LocalBrowserModelCatalogRequest = LocalBrowserEnvelope<'modelCatalogRequest'> & {
  viewId: string; target: LocalBrowserTargetIdentity; requestId: string;
} & ({ action: 'list' } | { action: 'select'; modelId: string });
export type LocalBrowserModelCatalog = LocalBrowserEnvelope<'modelCatalog'> & {
  viewId: string; target: LocalBrowserTargetIdentity; requestId: string; catalog: ModelCatalog;
};

export function isCanvasAgentSettings(value: unknown): value is CanvasAgentSettings {
  return isRecord(value) && Object.keys(value).every(key => ['model', 'effort'].includes(key))
    && (value.model === undefined || isBoundedString(value.model, 256))
    && (value.effort === undefined || isBoundedString(value.effort, 32));
}
export function isCanvasAgentAvailability(value: unknown): value is CanvasAgentAvailability {
  return isRecord(value) && ['unchecked', 'checking', 'ready', 'notInstalled', 'signInRequired', 'error'].includes(String(value.status))
    && typeof value.detail === 'string' && value.detail.length <= 2048
    && (value.defaultModel === undefined || isBoundedString(value.defaultModel, 256))
    && (value.checkedAt === undefined || (Number.isSafeInteger(value.checkedAt) && Number(value.checkedAt) >= 0))
    && Array.isArray(value.models) && value.models.length <= 100 && value.models.every(model => isRecord(model)
      && isBoundedString(model.id, 256) && isBoundedString(model.label, 256)
      && typeof model.description === 'string' && model.description.length <= 1024
      && Array.isArray(model.efforts) && model.efforts.length <= 16 && model.efforts.every(effort => isBoundedString(effort, 32))
      && (model.defaultEffort === undefined || isBoundedString(model.defaultEffort, 32)));
}

export function isCanvasAgentPermissionMode(value: unknown): value is CanvasAgentPermissionMode {
  return typeof value === 'string' && ['ask', 'auto_setup', 'full_setup'].includes(value);
}

export function isCanvasAgentPermissionRequest(value: unknown): value is CanvasAgentPermissionRequest {
  return isRecord(value) && Object.keys(value).length === 4
    && Object.keys(value).every(key => ['operation', 'scope', 'effects', 'facts'].includes(key))
    && typeof value.operation === 'string' && /^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$/.test(value.operation) && value.operation.length <= 128
    && typeof value.scope === 'string' && ['isolated_environment', 'existing_environment', 'machine'].includes(value.scope)
    && Array.isArray(value.effects) && value.effects.length > 0 && value.effects.length <= 4
    && new Set(value.effects).size === value.effects.length
    && value.effects.every(effect => typeof effect === 'string' && ['create', 'install', 'replace', 'execute_code'].includes(effect))
    && isRecord(value.facts) && Object.keys(value.facts).length > 0 && Object.keys(value.facts).length <= 64
    && isSetupJson(value.facts, 12000, 8);
}

export function isCanvasAgentPermissionPresentation(value: unknown): value is CanvasAgentPermissionPresentation {
  const text = (item: unknown, max: number): boolean => isBoundedString(item, max)
    && typeof item === 'string' && item === item.trim()
    && Array.from(item).every(character => character.charCodeAt(0) >= 32 && character.charCodeAt(0) !== 127);
  return isRecord(value) && Object.keys(value).length === 4
    && Object.keys(value).every(key => ['title', 'summary', 'allowLabel', 'denyLabel'].includes(key))
    && text(value.title, 160) && text(value.summary, 512) && text(value.allowLabel, 64) && text(value.denyLabel, 64)
    && value.allowLabel !== value.denyLabel;
}

/** Policies depend on host effects and resource scope, never agent text or operation IDs. */
export function canvasAgentPermissionPolicyAllows(request: CanvasAgentPermissionRequest, mode: CanvasAgentPermissionMode): boolean {
  return (mode === 'full_setup' && request.scope !== 'machine') || (mode === 'auto_setup' && request.scope === 'isolated_environment'
    && request.effects.every(effect => effect === 'create' || effect === 'install'));
}

export function isCanvasAgentPermission(value: unknown): value is CanvasAgentPermission {
  if (!isRecord(value) || Object.keys(value).length !== 7 + Number(value.presentation !== undefined) + Number(value.approval !== undefined) + Number(value.mode !== undefined)
    || !Object.keys(value).every(key => ['operation', 'scope', 'effects', 'facts', 'presentation', 'id', 'requestId', 'status', 'approval', 'mode'].includes(key))) return false;
  const { id, requestId, status, presentation, approval, mode, ...request } = value;
  return isCanvasAgentPermissionRequest(request) && isBoundedString(id, 128) && isBoundedString(requestId, 128)
    && typeof status === 'string' && ['pending', 'approved', 'denied', 'cancelled'].includes(status)
    && (presentation === undefined ? status !== 'approved' : isCanvasAgentPermissionPresentation(presentation))
    && (approval === undefined || (status === 'approved' && typeof approval === 'string' && ['user', 'policy'].includes(approval)))
    && (approval === 'policy' ? status === 'approved' && isCanvasAgentPermissionMode(mode) && mode !== 'ask'
      && canvasAgentPermissionPolicyAllows(request, mode) : mode === undefined);
}

export function isCanvasAgentState(value: unknown): value is CanvasAgentState {
  if (!isRecord(value) || !Object.keys(value).every(key => ['sessionId', 'revision', 'provider', 'ack', 'selections', 'agents',
    'status', 'messages', 'activity', 'failure', 'requestId', 'context', 'permissions', 'permissionMode'].includes(key))) return false;
  return isBoundedString(value.sessionId, 128) && Number.isSafeInteger(value.revision) && Number(value.revision) >= 0
    && ['codex', 'claude'].includes(String(value.provider))
    && (value.ack === undefined || isBoundedString(value.ack, 128))
    && (value.selections === undefined || (isRecord(value.selections) && Object.entries(value.selections)
      .every(([key, settings]) => ['codex', 'claude'].includes(key) && isCanvasAgentSettings(settings))))
    && (value.agents === undefined || (isRecord(value.agents) && Object.keys(value.agents).length === 2
      && isCanvasAgentAvailability(value.agents.codex) && isCanvasAgentAvailability(value.agents.claude)))
    && ['idle', 'running', 'stopped', 'error'].includes(String(value.status))
    && typeof value.activity === 'string' && value.activity.length <= 2048
    && (value.failure === undefined || (value.status === 'error' && isGraphFailure(value.failure)))
    && (value.requestId === undefined || isBoundedString(value.requestId, 128))
    && (value.permissionMode === undefined || isCanvasAgentPermissionMode(value.permissionMode))
    && (value.permissions === undefined || (Array.isArray(value.permissions) && value.permissions.length <= CANVAS_AGENT_PERMISSION_LIMIT
      && value.permissions.every(isCanvasAgentPermission)
      && new Set(value.permissions.map(permission => permission.id)).size === value.permissions.length
      && value.permissions.filter(permission => permission.status === 'pending').length <= 1
      && value.permissions.every(permission => permission.status !== 'pending'
        || (value.status === 'running' && permission.requestId === value.requestId && value.context !== undefined))))
    && (value.context === undefined || (isRecord(value.context) && isLocalBrowserTarget(value.context.target)
      && Array.isArray(value.context.nodeIds) && value.context.nodeIds.length <= 100
      && value.context.nodeIds.every(id => isBoundedString(id, 1024))))
    && Array.isArray(value.messages) && value.messages.length <= 40
    && value.messages.every(turn => isRecord(turn) && isBoundedString(turn.id, 128)
      && (turn.settings === undefined || isCanvasAgentSettings(turn.settings))
      && ['user', 'assistant'].includes(String(turn.role)) && typeof turn.text === 'string' && turn.text.length <= 32000);
}

export type LocalBrowserMessage =
  | CodeFocusRequest | CodeFocusState
  | LocalBrowserAgentRequest | LocalBrowserAgentState
  | LocalBrowserNodeExplanation | LocalBrowserNodeExplanationRequest
  | LocalBrowserModelExplanation | LocalBrowserModelExplanationRequest
  | LocalBrowserModelCatalog | LocalBrowserModelCatalogRequest
  | GraphReviewRequest | GraphReviewMessage | GraphReviewFocus | GraphReviewApplied
  | LocalBrowserGraphOpeningMessage
  | LocalBrowserGraphSetupRequestMessage
  | LocalBrowserCaptureExportRequestMessage
  | LocalBrowserCaptureExportCancelMessage
  | LocalBrowserCaptureExportResultMessage
  | LocalBrowserCaptureInspectRequestMessage
  | LocalBrowserCaptureInspectCancelMessage
  | LocalBrowserCaptureInspectResultMessage
  | LocalBrowserViewStateMessage
  | LocalBrowserViewSelectionMessage
  | LocalBrowserFocusViewMessage
  | LocalBrowserViewFocusResultMessage
  | LocalBrowserConnectionMessage
  | LocalBrowserHostHandshakeMessage
  | LocalBrowserViewHandshakeMessage
  | LocalBrowserSnapshotRequestMessage
  | LocalBrowserSnapshotMessage
  | LocalBrowserActiveTargetMessage
  | LocalBrowserProjectionMessage
  | LocalBrowserProjectIndexMessage
  | LocalBrowserProjectDocumentMessage
  | LocalBrowserProjectExpandMessage
  | LocalBrowserProjectFocusMessage
  | LocalBrowserProjectDeclareMessage
  | LocalBrowserConnectionStateMessage
  | LocalBrowserViewPresenceMessage
  | LocalBrowserSyncStateMessage
  | LocalBrowserStaleStateMessage
  | LocalBrowserRevealNodeMessage
  | LocalBrowserFocusNodeMessage
  | LocalBrowserSourceEditRequestMessage
  | LocalBrowserSourceEditCancelMessage
  | LocalBrowserSourceEditResultMessage
  | LocalBrowserSourceEditReviewMessage
  | LocalBrowserCandidateRequestMessage
  | LocalBrowserCandidateResultMessage
  | LocalBrowserValidateRequestMessage
  | LocalBrowserValidateCancelMessage
  | LocalBrowserValidateResultMessage
  | LocalBrowserValidationStateMessage
  | LocalBrowserAcknowledgementMessage
  | LocalBrowserErrorMessage;

export const LOCAL_BROWSER_MESSAGE_TYPES = [
  'codeFocusRequest', 'codeFocusState',
  'agentRequest', 'agentState', 'nodeExplanation', 'nodeExplanationRequest', 'modelExplanation', 'modelExplanationRequest', 'modelCatalog', 'modelCatalogRequest',
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
] as const;

const SOURCE_DIGEST_PATTERN = /^[0-9a-f]{64}$/;

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0;
}

/** A non-empty string inside the protocol field ceiling. */
function isBoundedString(value: unknown, maxLength: number): value is string {
  return isNonEmptyString(value) && value.length <= maxLength;
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

export function isLocalBrowserSourceEdit(value: unknown): value is LocalBrowserSourceEdit {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const edit = value as Record<string, unknown>;
  const name = (v: unknown) => typeof v === 'string' && /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(v);
  const node = (v: unknown) => isBoundedString(v, 256);
  const tensorValue = (v: unknown) => !!v && typeof v === 'object' && !Array.isArray(v) &&
    Object.keys(v).length === 2 && (v as Record<string, unknown>).kind === 'value' &&
    isBoundedString((v as Record<string, unknown>).id, 128);
  const literal = (v: unknown, depth = 0): boolean => depth <= 8 && (
    v === null || typeof v === 'boolean' || (typeof v === 'number' && Number.isFinite(v)) ||
    (typeof v === 'string' && v.length <= 1024) ||
    (Array.isArray(v) && v.length <= 64 && v.every(item => literal(item, depth + 1)))
  );
  const keys = (...fields: string[]) => Object.keys(edit).length === fields.length && fields.every(field => field in edit);
  switch (edit.kind) {
    case 'set_literal':
    case 'set_constructor_literal':
    case 'set_configuration_literal':
    case 'reconnect_argument':
    case 'insert_unary_op':
    case 'delete_unary_op':
    case 'replace_activation': {
      const target = edit.target as Record<string, unknown> | null;
      const constructor = edit.kind === 'set_constructor_literal' || edit.kind === 'set_configuration_literal';
      const structural = edit.kind === 'reconnect_argument' || edit.kind === 'insert_unary_op' || edit.kind === 'delete_unary_op';
      return keys('kind', 'nodeId', 'capabilityId', 'target', 'expected', 'value', 'scope',
        ...(edit.kind === 'set_literal' || structural ? ['argumentPath'] : constructor ? ['sourceArgumentPath'] : [])) && node(edit.nodeId) &&
        isBoundedString(edit.capabilityId, 128) && !!target && typeof target === 'object' && !Array.isArray(target) &&
        Object.keys(target).length === 5 && target.version === 1 &&
        (constructor ? target.kind === 'invocation'
          : target.kind === 'operation' || (edit.kind === 'replace_activation' && target.kind === 'invocation')) &&
        isBoundedString(target.captureId, 128) && isBoundedString(target.sourceId, 128) && isBoundedString(target.semanticId, 128) &&
        edit.scope === (edit.kind === 'set_configuration_literal' ? 'configuration_definition' : 'source_expression') && (edit.kind === 'replace_activation'
          ? isActivationName(edit.expected) && isActivationName(edit.value)
          : structural
            ? Array.isArray(edit.argumentPath) && edit.argumentPath.length === 2 && edit.argumentPath[0] === 'args' &&
              (edit.argumentPath[1] === 0 || (edit.kind !== 'delete_unary_op' && edit.argumentPath[1] === 1)) &&
              tensorValue(edit.expected) && (edit.kind === 'insert_unary_op' ? isActivationName(edit.value) : tensorValue(edit.value))
          : (constructor ? isConstructorArgumentPath(edit.sourceArgumentPath)
            : Array.isArray(edit.argumentPath) && edit.argumentPath.length === 2 && edit.argumentPath[0] === 'args' && edit.argumentPath[1] === 1) &&
            isSourceNumber(edit.expected) && isSourceNumber(edit.value));
    }
    case 'set_param':
      return keys('kind', 'nodeId', 'param', 'value') && node(edit.nodeId) && name(edit.param) && literal(edit.value);
    case 'remove_layer':
      return keys('kind', 'nodeId') && node(edit.nodeId);
    case 'add_layer': {
      const params = edit.params;
      return keys('kind', 'afterNodeId', 'nodeId', 'layerType', 'params') && node(edit.afterNodeId) &&
        name(edit.nodeId) && name(edit.layerType) && !!params && typeof params === 'object' && !Array.isArray(params) &&
        Object.keys(params).length <= 32 && Object.entries(params).every(([key, val]) => name(key) && literal(val));
    }
    case 'group_layers':
      return keys('kind', 'nodeIds', 'name') && name(edit.name) && Array.isArray(edit.nodeIds) &&
        edit.nodeIds.length >= 2 && edit.nodeIds.length <= 128 && edit.nodeIds.every(node) &&
        new Set(edit.nodeIds).size === edit.nodeIds.length;
    default: return false;
  }
}

export function isConstructorArgumentPath(value: unknown): value is SetConstructorLiteralEdit['sourceArgumentPath'] {
  return Array.isArray(value) && value.length === 2 &&
    ((value[0] === 'args' && Number.isInteger(value[1]) && value[1] >= 0 && value[1] < 256) ||
      (value[0] === 'kwargs' && typeof value[1] === 'string' && Array.from(value[1]).length <= 128 &&
        /^[_\p{XID_Start}][_\p{XID_Continue}]*(?![\s\S])/u.test(value[1])));
}

export function isSourceNumber(value: unknown): value is SourceNumber {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  if (Object.keys(record).length !== 2 || !('kind' in record) || !('value' in record)) return false;
  return record.kind === 'int'
    ? typeof record.value === 'string' && record.value.length <= 4096 && /^(?:0|-?[1-9][0-9]*)(?![\s\S])/.test(record.value)
    : record.kind === 'float' && (record.value === '-0' || (typeof record.value === 'number' && Number.isFinite(record.value) && !Object.is(record.value, -0)));
}

function isLocalBrowserTarget(value: unknown): value is LocalBrowserTargetIdentity {
  const record = value as Partial<LocalBrowserTargetIdentity> | null;
  return (
    typeof record === 'object' &&
    record !== null &&
    isNonEmptyString(record.workspaceFolder) &&
    isNonEmptyString(record.file) &&
    isNonEmptyString(record.cls) &&
    isNonNegativeInteger(record.documentVersion) &&
    isNonEmptyString(record.sourceDigest) &&
    SOURCE_DIGEST_PATTERN.test(record.sourceDigest) &&
    isNonNegativeInteger(record.backendRevision)
  );
}

/** The project identity's validator, held to the same standard as the class one:
 *  every field required and typed, so a partially-formed project target is refused
 *  rather than compared field-by-field at the call site. `experimentId` may be empty
 *  — "no experiment focused" is a real state — but must still be a string. */
export function isLocalBrowserProjectIdentity(
  value: unknown,
): value is LocalBrowserProjectIdentity {
  const record = value as Partial<LocalBrowserProjectIdentity> | null;
  return (
    typeof record === 'object' &&
    record !== null &&
    Object.keys(record).length === 5 &&
    isBoundedString(record.workspaceFolder, 4096) &&
    typeof record.projectRoot === 'string' && record.projectRoot.length <= 4096 &&
    typeof record.experimentId === 'string' && record.experimentId.length <= 512 &&
    isNonEmptyString(record.indexDigest) && record.indexDigest.length === 64 &&
    SOURCE_DIGEST_PATTERN.test(record.indexDigest) &&
    isProjectCount(record.indexRevision)
  );
}

function isProjectCount(value: unknown): value is number {
  return isNonNegativeInteger(value) && Number.isSafeInteger(value);
}

/** Shared with Python's project wire gate and the native declaration review. */
export function isProjectDeclaration(declaration: unknown, args: unknown): args is string[] {
  return Array.isArray(args) &&
    (declaration === 'node' ? args.length === 2 : declaration === 'edge' ?
      args.length === 2 || args.length === 3 : declaration === 'experiment' && args.length === 1) &&
    args.every(arg => isBoundedString(arg, 512) && arg.trim().length > 0);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isSetupJson(value: unknown, maxBytes = 131072, maxDepth = 16): boolean {
  let entries = 0;
  const visit = (item: unknown, depth: number): boolean => {
    if (++entries > 16384 || depth > maxDepth) return false;
    if (item === null || typeof item === 'boolean' || typeof item === 'string') return true;
    if (typeof item === 'number') return Number.isFinite(item);
    if (Array.isArray(item)) return item.every(child => visit(child, depth + 1));
    return isRecord(item) && Object.entries(item).every(([key, child]) => key.length <= 1024 && visit(child, depth + 1));
  };
  return visit(value, 0) && new TextEncoder().encode(JSON.stringify(value)).length <= maxBytes;
}

export function isGraphSetupDiscovery(value: unknown): value is GraphSetupDiscovery {
  if (!isRecord(value) || !isSetupJson(value, 262144)) return false;
  const fields = (items: unknown): boolean => items === null || (Array.isArray(items) && items.length <= 64 && items.every(item =>
    isRecord(item) && isBoundedString(item.name, 128) && typeof item.required === 'boolean'
    && ['positional', 'keyword', 'keyword_only', 'varargs', 'kwargs'].includes(item.kind as string)
    && (item.annotation === undefined || typeof item.annotation === 'string')
    && (item.defaultText === undefined || typeof item.defaultText === 'string')
    && (item.source === undefined || (isRecord(item.source) && isBoundedString(item.source.file, 4096)
      && isNonNegativeInteger(item.source.line) && item.source.line > 0))));
  const list = (items: unknown, check: (item: Record<string, unknown>) => boolean): boolean =>
    Array.isArray(items) && items.length <= 64 && items.every(item => isRecord(item) && check(item));
  const source = (item: Record<string, unknown>) => isBoundedString(item.file, 4096);
  return isBoundedString(value.model, 512) && typeof value.sourceDigest === 'string' && SOURCE_DIGEST_PATTERN.test(value.sourceDigest)
    && typeof value.truncated === 'boolean' && fields(value.init) && fields(value.inputs)
    && list(value.examples, item => source(item) && isNonNegativeInteger(item.line) && isRecord(item.init)
      && (item.inputs === undefined || isRecord(item.inputs)))
    && list(value.factories, item => source(item) && isBoundedString(item.callable, 128) && Array.isArray(item.fields) && fields(item.fields))
    && list(value.configs, item => source(item) && isBoundedString(item.cls, 128) && fields(item.fields)
      && (item.parameter === null || isBoundedString(item.parameter, 128)) && ['call', 'attributes'].includes(item.construction as string))
    && list(value.parents, item => source(item) && isBoundedString(item.cls, 128) && isBoundedString(item.attribute, 128))
    && list(value.constraints, item => source(item) && isBoundedString(item.input, 128) && Number.isInteger(item.axis)
      && (item.size === undefined || isNonNegativeInteger(item.size)) && (item.initParameter === undefined || isBoundedString(item.initParameter, 128)))
    && (value.resolution === undefined || (isRecord(value.resolution)
      && ['complete', 'partial'].includes(value.resolution.status as string) && isGraphSetupDraft(value.resolution.draft)
      && (value.resolution.runtimeInputs === undefined || typeof value.resolution.runtimeInputs === 'boolean')
      && isGraphSetupEvidence(value.resolution.evidence) && Array.isArray(value.resolution.unresolved)
      && value.resolution.unresolved.length <= 64 && value.resolution.unresolved.every(item => isBoundedString(item, 1024))
      && (value.resolution.status !== 'complete' || value.resolution.unresolved.length === 0)));
}

function isGraphSetupEvidence(value: unknown): value is GraphSetupEvidence[] {
  return Array.isArray(value) && value.length <= 64 && value.every(item => isRecord(item)
    && isBoundedString(item.field, 256) && ['repository', 'source', 'example'].includes(item.kind as string)
    && isBoundedString(item.detail, 512) && (item.file === undefined || isBoundedString(item.file, 4096))
    && (item.line === undefined || isNonNegativeInteger(item.line)));
}

export function isGraphSetupDraft(value: unknown): value is GraphSetupDraft {
  if (!isRecord(value) || !isSetupJson(value) || !isRecord(value.profile)
    || Object.keys(value).some(key => !['profile', 'profileName', 'configObject', 'configDefaults'].includes(key))
    || (value.profileName !== undefined && !isBoundedString(value.profileName, 64))) return false;
  if (value.configDefaults !== undefined && (value.configDefaults !== true
    || value.configObject !== undefined || value.profile.factory !== undefined)) return false;
  const config = value.configObject;
  return config === undefined || (isRecord(config) && Object.keys(config).length === 3
    && isNonNegativeInteger(config.index) && isBoundedString(config.parameter, 128) && isRecord(config.values));
}

/** A portable directory name, never a path or shell argument. */
export function isModelEnvironmentName(value: unknown): value is string {
  return typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$/.test(value)
    && !/^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i.test(value);
}

export function isGraphSetupAction(value: unknown): value is GraphSetupAction {
  if (!isRecord(value)) return false;
  if (value.kind === 'execution') return isBoundedString(value.reviewId, 128) && isExecutionDevice(value.device)
    && Object.keys(value).length === 4 && ['run', 'cancel', 'refresh', 'select'].includes(value.action as string);
  if (value.kind === 'edit_execution') return Object.keys(value).length === 3 && isBoundedString(value.reviewId, 128)
    && Array.isArray(value.field) && value.field.length > 0 && value.field.length <= 12
    && ['inputs', 'mode', 'grad_enabled', 'device'].includes(value.field[0])
    && (value.field[0] === 'inputs' || value.field.length === 1)
    && value.field.every(part => isBoundedString(part, 128) || (isNonNegativeInteger(part) && part < 64));
  if (value.kind === 'submit') return Object.keys(value).length === 2 && isGraphSetupDraft(value.draft);
  if (value.kind === 'reveal_setup_source') return Object.keys(value).length === 3 && isBoundedString(value.file, 4096)
    && isNonNegativeInteger(value.line) && value.line > 0;
  if (value.kind === 'factory_template' || value.kind === 'agent_help') return Object.keys(value).every(key => ['kind', 'draft'].includes(key)) && (value.draft === undefined || isGraphSetupDraft(value.draft));
  if (value.kind === 'parent') return Object.keys(value).length === 2 && isNonNegativeInteger(value.index) && Number(value.index) < 64;
  if (value.kind === 'select_environment') return Object.keys(value).length === 2 && isBoundedString(value.candidateId, 64);
  if (value.kind === 'create_environment') return Object.keys(value).every(key => ['kind', 'name'].includes(key))
    && (value.name === undefined || isModelEnvironmentName(value.name));
  return Object.keys(value).length === 1 && ['retry', 'change_inputs', 'cancel', 'source_only', 'save_sources', 'environment', 'edit_config', 'agent_help', 'factory_template', 'install_dependencies', 'refresh_environments', 'use_installed_versions', 'replace_versions', 'create_environment'].includes(value.kind as string);
}

export function isGraphEnvironmentCandidate(value: unknown): value is GraphEnvironmentCandidate {
  return isRecord(value) && isBoundedString(value.id, 64) && isBoundedString(value.interpreter, 4096)
    && isBoundedString(value.source, 256) && ['libraries_present', 'missing_libraries', 'incompatible', 'unavailable'].includes(value.status as string)
    && ['selected', 'pythonSelected', 'managed'].every(key => value[key] === undefined || value[key] === true)
    && ['python', 'implementation', 'architecture'].every(key => value[key] === undefined || isBoundedString(value[key], 256))
    && (value.prefix === undefined || isBoundedString(value.prefix, 4096))
    && ['torch', 'abi'].every(key => value[key] === undefined || value[key] === null || isBoundedString(value[key], 256))
    && (value.fingerprint === undefined || (typeof value.fingerprint === 'string' && /^[a-f0-9]{64}$/.test(value.fingerprint)))
    && ['missing', 'incompatible', 'modelMissing', 'modelIncompatible'].every(key => { const items = value[key]; return items === undefined || (Array.isArray(items)
      && items.length <= 32 && items.every((item: unknown) => isBoundedString(item, 1024))); })
    && (value.detail === undefined || isBoundedString(value.detail, 1024))
    && (value.modelChecked === undefined || value.modelChecked === true)
    && (value.modelChecked === true || (value.modelMissing === undefined && value.modelIncompatible === undefined))
    && Object.keys(value).every(key => ['id', 'interpreter', 'source', 'status', 'selected', 'pythonSelected', 'managed', 'python', 'torch',
      'prefix', 'implementation', 'abi', 'architecture', 'fingerprint', 'missing', 'incompatible', 'detail',
      'modelChecked', 'modelMissing', 'modelIncompatible'].includes(key));
}

export function isGraphEnvironmentRecovery(value: unknown): value is GraphEnvironmentRecovery {
  return isRecord(value) && ['ready', 'selection_required', 'unavailable', 'missing_libraries', 'incompatible', 'worker_failed'].includes(value.reason as string)
    && isBoundedString(value.detail, 1024) && Array.isArray(value.candidates) && value.candidates.length <= 16
    && value.candidates.every(isGraphEnvironmentCandidate) && new Set(value.candidates.map(candidate => candidate.id)).size === value.candidates.length
    && (value.selectedId === undefined || value.candidates.some(candidate => candidate.id === value.selectedId))
    && (value.reason !== 'ready' || value.selectedId !== undefined)
    && ['requirements', 'compatibilityPackages', 'missingModules', 'declaredPins'].every(key => { const items = value[key]; return items === undefined || (Array.isArray(items)
      && items.length <= 32 && items.every((item: unknown) => isBoundedString(item, 1024))); })
    && (value.replaceable === undefined || (value.replaceable === true && Array.isArray(value.declaredPins) && value.declaredPins.length > 0))
    && (value.canCreate === undefined || value.canCreate === true)
    && (value.recommendedId === undefined || value.candidates.some(candidate => candidate.id === value.recommendedId))
    && Object.keys(value).every(key => ['reason', 'detail', 'candidates', 'selectedId', 'requirements', 'compatibilityPackages', 'missingModules', 'declaredPins', 'replaceable', 'canCreate', 'recommendedId'].includes(key));
}

export function isExecutionDevice(value: unknown): value is string {
  return typeof value === 'string' && /^(cpu|mps|cuda(?::[0-9]{1,3})?)$/.test(value);
}
export function isExecutionReviewState(value: unknown): value is ExecutionReviewState {
  const numeric = (n: unknown) => typeof n === 'number' && Number.isFinite(n) && n >= 0;
  const optionalNumber = (n: unknown) => n === null || numeric(n);
  if (!isRecord(value) || !isBoundedString(value.id, 128) || !isExecutionDevice(value.device) || !isExecutionDevice(value.configuredDevice)
    || !['review', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled'].includes(value.phase as string)
    || !['prepare', 'capture', 'trace'].includes(value.kind as string)
    || !isBoundedString(value.detail, 1024) || !isBoundedString(value.inputs, 4096)
    || !isBoundedString(value.mode, 128) || !isBoundedString(value.interpreter, 4096) || typeof value.gradients !== 'boolean'
    || !Array.isArray(value.stages) || value.stages.length > 8
    || !value.stages.every(stage => isRecord(stage) && Object.keys(stage).length === 2
      && isBoundedString(stage.label, 128) && isNonNegativeInteger(stage.calls))) return false;
  if (value.profileName !== undefined && !isBoundedString(value.profileName, 64)) return false;
  if ((value.reused !== undefined && (!isExecutionReuse(value.reused) || value.phase !== 'succeeded'))
    || (value.fresh !== undefined && (value.fresh !== true || value.reused !== undefined))) return false;
  if (!Object.keys(value).every(key => ['id', 'phase', 'kind', 'device', 'configuredDevice', 'detail', 'stages', 'inputs', 'mode', 'gradients', 'interpreter', 'profileName', 'resources', 'reused', 'fresh'].includes(key))) return false;
  const r = value.resources;
  return r === undefined || (isRecord(r) && Object.keys(r).length === 10 && numeric(r.sampledAt)
    && isBoundedString(r.host, 512) && typeof r.remote === 'boolean' && isBoundedString(r.cpu, 512) && numeric(r.cores)
    && optionalNumber(r.cpuPercent) && numeric(r.memoryFree) && numeric(r.memoryTotal) && isBoundedString(r.detail, 1024)
    && Array.isArray(r.devices) && r.devices.length <= 128 && r.devices.every(d => isRecord(d) && Object.keys(d).length === 5
      && isExecutionDevice(d.id) && isBoundedString(d.name, 512) && optionalNumber(d.utilization)
      && optionalNumber(d.memoryFree) && optionalNumber(d.memoryTotal)));
}

export function isGraphOpeningState(value: unknown): value is GraphOpeningState {
  return isRecord(value) && isBoundedString(value.id, 128) && isLocalBrowserTarget(value.target)
    && ['awaiting_execution', 'resolving', 'needs_setup', 'saving', 'capturing', 'ready', 'failed', 'cancelled', 'source_only'].includes(value.phase as string)
    && (value.execution === undefined || isExecutionReviewState(value.execution))
    && isBoundedString(value.detail, 1024) && typeof value.hasCapture === 'boolean'
    && (value.cancellationReason === undefined || (value.phase === 'cancelled' && value.cancellationReason === 'source_changed'))
    && (value.configFileExists === undefined || typeof value.configFileExists === 'boolean')
    && (value.discovery === undefined || isGraphSetupDiscovery(value.discovery))
    && (value.draft === undefined || isGraphSetupDraft(value.draft))
    && (value.requirements === undefined || (Array.isArray(value.requirements) && value.requirements.length <= 64
      && value.requirements.every(item => isBoundedString(item, 1024))))
    && (value.evidence === undefined || isGraphSetupEvidence(value.evidence))
    && (value.failure === undefined || isGraphFailure(value.failure))
    && (value.environment === undefined || isGraphEnvironmentRecovery(value.environment))
    && (value.environmentName === undefined || isModelEnvironmentName(value.environmentName))
    && Object.keys(value).every(key => ['id', 'target', 'phase', 'detail', 'hasCapture', 'execution', 'cancellationReason', 'configFileExists', 'discovery', 'draft', 'requirements', 'evidence', 'failure', 'environment', 'environmentName'].includes(key));
}

export function isGraphFailure(value: unknown): value is GraphFailure {
  return isRecord(value) && ['setup', 'source', 'environment', 'execution', 'capture'].includes(value.kind as string)
    && (value.code === undefined || (value.kind === 'environment' && typeof value.code === 'string' && /^[A-Z][A-Z0-9_]{0,63}$/.test(value.code)))
    && (value.fields === undefined || (Array.isArray(value.fields) && value.fields.length <= 64
      && value.fields.every(field => isBoundedString(field, 256))))
    && (value.missingModules === undefined || (value.kind === 'environment' && Array.isArray(value.missingModules)
      && value.missingModules.length <= 32 && value.missingModules.every(name => typeof name === 'string' && /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name))))
    && (value.importFailure === undefined || (value.kind === 'environment' && value.importFailure === true))
    && (value.diagnostic === undefined || (value.kind === 'environment' && isModelImportDiagnostic(value.diagnostic)))
    && Object.keys(value).every(key => ['kind', 'code', 'fields', 'missingModules', 'importFailure', 'diagnostic'].includes(key));
}

export function isModelImportDiagnostic(value: unknown): value is ModelImportDiagnostic {
  const text = (item: unknown, max: number) => typeof item === 'string' && item.length <= max;
  const nullableText = (item: unknown) => item === null || isBoundedString(item, 128);
  const nullableBoolean = (item: unknown) => item === null || typeof item === 'boolean';
  if (!isRecord(value) || Object.keys(value).length !== 6
    || !Object.keys(value).every(key => ['exceptionType', 'message', 'traceback', 'causes', 'runtime', 'truncated'].includes(key))) return false;
  const runtime = value.runtime;
  return isBoundedString(value.exceptionType, 128) && text(value.message, 2048) && text(value.traceback, 12000)
    && typeof value.truncated === 'boolean' && Array.isArray(value.causes) && value.causes.length <= 8
    && value.causes.every(cause => isRecord(cause) && Object.keys(cause).every(key => ['exceptionType', 'message', 'module'].includes(key))
      && isBoundedString(cause.exceptionType, 128) && text(cause.message, 2048) && (cause.module === undefined || isBoundedString(cause.module, 256)))
    && isRecord(runtime) && Object.keys(runtime).length === 7
    && Object.keys(runtime).every(key => ['platform', 'architecture', 'python', 'torch', 'cuda', 'cudaAvailable', 'mpsAvailable'].includes(key))
    && isBoundedString(runtime.platform, 128) && isBoundedString(runtime.architecture, 128) && isBoundedString(runtime.python, 128)
    && nullableText(runtime.torch) && nullableText(runtime.cuda) && nullableBoolean(runtime.cudaAvailable) && nullableBoolean(runtime.mpsAvailable);
}

export function isCaptureExportReference(value: unknown): value is CaptureExportReference {
  return isRecord(value) && Object.keys(value).length === 2
    && isBoundedString(value.captureId, 128) && isBoundedString(value.sourceId, 128);
}

function isPortableCaptureEnvelope(value: unknown): boolean {
  return isRecord(value) && Object.keys(value).length === 3 && value.version === 1
    && value.sourceNames === 'anonymized' && isRecord(value.capture)
    && new Blob([JSON.stringify(value)]).size <= LOCAL_BROWSER_CAPTURE_EXPORT_MAX_BYTES;
}

export function isCaptureInspectionQuery(value: unknown): value is CaptureInspectionQuery {
  if (!isRecord(value)) return false;
  const entity = value.entity;
  return Object.keys(value).length === (entity === undefined ? 5 : 6)
    && isBoundedString(value.captureId, 128) && isBoundedString(value.sourceId, 128)
    && (entity === undefined || (isRecord(entity) && Object.keys(entity).length === 2
      && ['operation', 'invocation', 'value', 'input', 'output'].includes(entity.kind as string) && isBoundedString(entity.id, 128)))
    && Array.isArray(value.path) && value.path.length <= 128 && JSON.stringify(value.path).length <= 8192
    && value.path.every(key => (typeof key === 'string' && key.length <= 128) || (isProjectCount(key) && key <= 1_000_000))
    && isProjectCount(value.offset) && value.offset <= 1_000_000
    && isProjectCount(value.limit) && value.limit >= 1 && value.limit <= 200;
}

export function isCandidatePostconditions(value: unknown): value is CandidatePostconditions {
  if (!isRecord(value) || Object.keys(value).length !== 9 + Number(value.observedInitializers !== undefined)
      + Number(value.configurationUses !== undefined) || value.version !== 1
    || !isBoundedString(value.executionId, 128) || !(value.captureId === null || isBoundedString(value.captureId, 128))
    || !(value.bindingId === null || isBoundedString(value.bindingId, 128))
    || ![null, 'binary_scalar_v1', 'functional_activation_v1', 'module_activation_v1', 'straight_line_tensor_v1',
      'straight_line_unary_insert_v1', 'straight_line_unary_delete_v1', 'constructor_scalar_v1', 'configuration_scalar_v1'].includes(value.recipe as string | null)
    || value.scope !== (value.recipe === 'configuration_scalar_v1' ? 'observed_configuration_arguments'
      : value.recipe === 'constructor_scalar_v1' ? 'observed_initializer_entries' : 'observed_expression')
    || !['verified', 'failed', 'unavailable'].includes(value.status as string)
    || !isProjectCount(value.observedOperations) || value.observedOperations > 100000
    || (value.observedInitializers !== undefined && (!isProjectCount(value.observedInitializers) || value.observedInitializers > 100000))
    || !['observed_requested_literal', 'observed_literal_mismatch', 'observed_requested_activation', 'observed_activation_mismatch',
      'observed_requested_argument', 'observed_argument_mismatch', 'observed_requested_insertion', 'observed_insertion_mismatch',
      'observed_requested_deletion', 'observed_deletion_mismatch',
      'observed_requested_constructor', 'observed_constructor_mismatch', 'initializer_arguments_unavailable',
      'observed_requested_configuration', 'observed_configuration_mismatch', 'configuration_reference_unavailable',
      'unsupported_recipe', 'capture_unavailable',
      'capture_incomplete', 'capture_identity_mismatch', 'source_mapping_unavailable', 'expression_not_observed',
      'ambiguous_observation'].includes(value.reason as string)) return false;
  const checked = value.status !== 'unavailable';
  const configuration = value.recipe === 'configuration_scalar_v1';
  const constructor = value.recipe === 'constructor_scalar_v1', initializers = value.observedInitializers ?? 0;
  const uses = value.configurationUses === undefined ? [] : value.configurationUses;
  if (!Array.isArray(uses) || uses.length > 256
    || uses.some(use => !isRecord(use) || Object.keys(use).length !== 2 || !isBoundedString(use.bindingId, 128)
      || /[\uD800-\uDFFF]/u.test(use.bindingId)
      || !isConstructorArgumentPath(use.sourceArgumentPath))
    || new TextEncoder().encode(JSON.stringify(uses)).length > 32768
    || Boolean(uses.length) !== (checked && configuration)
    || new Set(uses.map(use => JSON.stringify([use.bindingId, use.sourceArgumentPath]))).size !== uses.length
    || (uses.length > 0 && !uses.some(use => use.bindingId === value.bindingId))) return false;
  if (((constructor || configuration) ? value.observedOperations : initializers) !== 0) return false;
  const count = constructor || configuration ? initializers as number : value.observedOperations;
  const name = value.recipe === 'binary_scalar_v1' ? 'literal' : value.recipe === 'straight_line_tensor_v1' ? 'argument'
    : configuration ? 'configuration' : constructor ? 'constructor'
    : value.recipe === 'straight_line_unary_insert_v1' ? 'insertion' : value.recipe === 'straight_line_unary_delete_v1' ? 'deletion' : 'activation';
  return checked === ['observed_requested_literal', 'observed_literal_mismatch', 'observed_requested_activation', 'observed_activation_mismatch',
    'observed_requested_argument', 'observed_argument_mismatch', 'observed_requested_insertion', 'observed_insertion_mismatch',
    'observed_requested_deletion', 'observed_deletion_mismatch', 'observed_requested_constructor', 'observed_constructor_mismatch',
    'observed_requested_configuration', 'observed_configuration_mismatch'].includes(value.reason as string)
    && (value.status !== 'verified' || value.reason === `observed_requested_${name}`)
    && (value.status !== 'failed' || value.reason === `observed_${name}_mismatch`)
    && (checked ? value.recipe !== null && value.captureId !== null && value.bindingId !== null && count > 0
      : value.bindingId === null && count === 0)
    && (value.recipe === null) === (value.reason === 'unsupported_recipe');
}

export function isCandidateCheckpointSummary(value: unknown): value is CandidateCheckpointSummary {
  if (!isRecord(value) || Object.keys(value).length !== 8 || !['succeeded', 'failed'].includes(value.loadStatus as string)
    || typeof value.complete !== 'boolean' || !isProjectCount(value.entries) || value.entries > 8192) return false;
  const counts = ['shapeDifferences', 'dtypeDifferences', 'registeredOnly', 'checkpointOnly', 'unknownPresence'];
  if (!counts.every(key => isProjectCount(value[key]) && (value[key] as number) <= (value.entries as number))) return false;
  const both = value.entries - (value.registeredOnly as number) - (value.checkpointOnly as number) - (value.unknownPresence as number);
  return both >= 0 && (value.shapeDifferences as number) <= both && (value.dtypeDifferences as number) <= both
    && (!value.complete || value.unknownPresence === 0);
}
export function isCandidateCheckpointPage(value: unknown): value is CandidateCheckpointPage {
  return isRecord(value) && Object.keys(value).length === 5 && isBoundedString(value.runId, 128)
    && isProjectCount(value.offset) && value.offset <= 8192 && isProjectCount(value.count) && value.count <= 8192
    && (value.nextOffset === null || (isProjectCount(value.nextOffset) && value.nextOffset > value.offset && value.nextOffset < value.count))
    && isBoundedString(value.text, 131072);
}
export function isCandidateReviewSnapshot(value: unknown): value is CandidateReviewSnapshot {
  return isRecord(value) && Object.keys(value).length === 9 + Number(value.runId !== undefined) + Number(value.postconditions !== undefined) + Number(value.checkpoint !== undefined)
    && (value.runId === undefined || isBoundedString(value.runId, 128)) && value.staticCheck === 'passed'
    && ['not_run', 'running', 'succeeded', 'failed', 'refused', 'cancelled', 'timed_out'].includes(value.runtime as string)
    && ['unverified', 'current', 'stale'].includes(value.freshness as string)
    && ['unavailable', 'complete', 'partial', 'failed'].includes(value.capture as string)
    && ['same', 'different', 'unavailable'].includes(value.inputs as string)
    && ['same', 'different', 'unavailable'].includes(value.state as string)
    && (value.checkpoint === undefined || (isCandidateCheckpointSummary(value.checkpoint)
      && ['succeeded', 'failed'].includes(value.runtime as string) && value.cleanup === 'complete'
      && (value.checkpoint.loadStatus !== 'failed' || (value.runtime === 'failed' && value.capture === 'unavailable'))))
    && (value.postconditions === undefined ? value.semanticPostconditions === 'not_checked'
      : isCandidatePostconditions(value.postconditions) && value.semanticPostconditions === value.postconditions.status
        && ['succeeded', 'failed'].includes(value.runtime as string) && value.cleanup === 'complete'
        && (value.postconditions.status === 'unavailable' || value.capture === 'complete'))
    && ['not_run', 'pending', 'complete', 'failed', 'unavailable'].includes(value.cleanup as string)
    && Array.isArray(value.diagnostics) && value.diagnostics.length <= 16
    && value.diagnostics.every(code => typeof code === 'string' && /^[a-z_]{1,80}(?![\s\S])/.test(code));
}
export function isCandidateRecord(value: unknown): value is CandidateRecord {
  return isRecord(value) && Object.keys(value).length === 8 && isBoundedString(value.runId, 128)
    && isProjectCount(value.startedAt) && (value.finishedAt === null || (isProjectCount(value.finishedAt) && value.finishedAt >= value.startedAt))
    && ['open', 'historical'].includes(value.review as string) && isCandidateReviewSnapshot(value.summary)
    && value.summary.runId === value.runId && (value.executionId === null || isBoundedString(value.executionId, 128))
    && (value.capture === null || isCaptureExportReference(value.capture))
    && (value.summary.checkpoint?.loadStatus !== 'failed' || value.capture === null)
    && (!value.summary.postconditions || (value.summary.postconditions.executionId === value.executionId
      && (value.summary.postconditions.captureId === null || value.capture === null
        || (isRecord(value.capture) && value.summary.postconditions.captureId === value.capture.captureId))))
    && (value.unavailable === null || ['not_captured', 'invalid_capture', 'too_large', 'evicted'].includes(value.unavailable as string));
}

export function isCaptureReconciliation(value: unknown): value is CaptureReconciliation {
  if (!isRecord(value) || Object.keys(value).length !== 11 || value.version !== 1
    || ![value.baseCaptureId, value.captureId, value.sourceId].every(id => isBoundedString(id, 128))
    || typeof value.reviewedDigest !== 'string' || value.reviewedDigest.length !== 64 || !SOURCE_DIGEST_PATTERN.test(value.reviewedDigest)
    || !['complete', 'partial'].includes(value.dependencyScope as string)
    || !['checked', 'unavailable'].includes(value.status as string)
    || !['reviewed_source', 'source_reverted', 'source_changed', 'dependencies_changed', 'environment_changed',
      'different_model', 'capture_not_new', 'source_unavailable'].includes(value.reason as string)
    || !['same', 'different', 'unavailable'].includes(value.inputs as string)
    || !['same', 'different', 'unavailable'].includes(value.state as string)) return false;
  const checked = value.status === 'checked';
  return checked === (value.reason === 'reviewed_source') && (checked
    ? isCandidatePostconditions(value.postconditions) && value.postconditions.captureId === value.captureId
    : value.postconditions === null && value.inputs === 'unavailable' && value.state === 'unavailable');
}

export function isAppliedSourceFiles(value: unknown): value is AppliedSourceFile[] {
  return Array.isArray(value) && value.length > 0 && value.length <= 16 && value.every(item => isRecord(item)
    && Object.keys(item).length === 3 && isBoundedString(item.file, 1024)
    && /^(?!\/|[A-Za-z]:)(?!.*\\).+\.py$/.test(item.file as string)
    && !(item.file as string).split('').some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)
    && !(item.file as string).split('/').some(part => part === '' || part === '.' || part === '..')
    && [item.beforeDigest, item.afterDigest].every(digest => typeof digest === 'string' && digest.length === 64 && SOURCE_DIGEST_PATTERN.test(digest))
    && item.beforeDigest !== item.afterDigest) && new Set(value.map(item => item.file)).size === value.length;
}

export function isAppliedSourceSetComparison(value: unknown): value is AppliedSourceSetComparison {
  if (!isRecord(value) || Object.keys(value).length !== 9
    || ![value.baseCaptureId, value.captureId, value.sourceId].every(id => isBoundedString(id, 128))
    || !['complete', 'partial'].includes(value.dependencyScope as string) || value.semanticPostconditions !== 'not_checked'
    || !['reviewed_source', 'source_reverted', 'source_changed', 'dependencies_changed', 'environment_changed',
      'different_model', 'capture_not_new', 'source_unavailable'].includes(value.reason as string)
    || !['same', 'different', 'unavailable'].includes(value.inputs as string)
    || !['same', 'different', 'unavailable'].includes(value.state as string)) return false;
  if (value.reason !== 'reviewed_source') return value.comparison === null && value.inputs === 'unavailable' && value.state === 'unavailable';
  const result = value.comparison;
  return value.baseCaptureId !== value.captureId && isRecord(result) && Object.keys(result).length === 9
    && (result.equivalent === null || typeof result.equivalent === 'boolean') && typeof result.coverageComplete === 'boolean'
    && ['changedFieldCount', 'observedFieldCount', 'unresolvedFieldCount', 'unmatchedRecordCount', 'uncertainMatchCount',
      'connectionChangeCount', 'unresolvedConnectionCount'].every(key => isProjectCount(result[key]))
    && (result.equivalent !== false || Number(result.changedFieldCount) > 0 || Number(result.connectionChangeCount) > 0)
    && (result.equivalent !== true || (result.coverageComplete === true && result.changedFieldCount === 0
      && result.unresolvedFieldCount === 0 && result.unmatchedRecordCount === 0 && result.uncertainMatchCount === 0
      && result.connectionChangeCount === 0 && result.unresolvedConnectionCount === 0));
}

export function isAppliedSourceObservation(value: unknown): value is AppliedSourceObservation {
  if (!isRecord(value) || Object.keys(value).length !== 6 + Number(value.sourceSetComparison !== undefined) || !isRecord(value.receipt)) return false;
  const receipt = value.receipt;
  if (Object.keys(receipt).length !== 5 + Number(receipt.files !== undefined) || !isProjectCount(receipt.appliedAt) || !isBoundedString(receipt.file, 4096)
    || ![receipt.beforeDigest, receipt.afterDigest].every(digest => typeof digest === 'string' && digest.length === 64 && SOURCE_DIGEST_PATTERN.test(digest))
    || receipt.beforeDigest === receipt.afterDigest || ![null, 'too_large', 'evicted'].includes(receipt.unavailable as null | string)
    || !isProjectCount(value.checkedAt) || value.checkedAt < receipt.appliedAt
    || !['reviewed_unsaved', 'reviewed_saved', 'reverted', 'changed', 'unavailable'].includes(value.source as string)
    || !['current', 'stale', 'unavailable'].includes(value.captureFreshness as string)
    || ![null, 'source_not_saved', 'source_changed', 'source_unavailable', 'evidence_unavailable', 'capture_unavailable',
      'check_stale', 'check_failed'].includes(value.unavailable as null | string)) return false;
  if (receipt.files !== undefined && (!isAppliedSourceFiles(receipt.files) || receipt.files[0].file !== receipt.file
    || receipt.files[0].beforeDigest !== receipt.beforeDigest || receipt.files[0].afterDigest !== receipt.afterDigest)) return false;
  if (value.sourceSetComparison !== undefined) return receipt.files !== undefined && value.reconciliation === null
    && value.unavailable === null && receipt.unavailable === null && value.source === 'reviewed_saved'
    && value.captureFreshness !== 'stale' && isAppliedSourceSetComparison(value.sourceSetComparison);
  if (value.reconciliation === null) return value.unavailable !== null && value.captureFreshness !== 'current';
  return receipt.files === undefined && value.unavailable === null && receipt.unavailable === null && value.source === 'reviewed_saved'
    && value.captureFreshness !== 'stale' && isCaptureReconciliation(value.reconciliation)
    && value.reconciliation.reviewedDigest === receipt.afterDigest;
}

/** One scoped explanation across native review, browser and agent metadata. */
export function candidatePostconditionsText(summary: Pick<CandidateReviewSnapshot, 'postconditions'>): string {
  const evidence = summary.postconditions;
  if (!evidence) return 'not checked';
  if ((evidence.recipe === 'constructor_scalar_v1' || evidence.recipe === 'configuration_scalar_v1') && evidence.status !== 'unavailable') {
    const count = `${evidence.observedInitializers} retained initializer entr${evidence.observedInitializers === 1 ? 'y' : 'ies'}`;
    if (evidence.recipe === 'configuration_scalar_v1') return evidence.status === 'verified'
      ? `Requested configuration value observed at ${evidence.configurationUses?.length} direct constructor arguments across ${count}. Other references and later behavior are unverified.`
      : `Recorded configuration value differs from the request; ${count} checked.`;
    return evidence.status === 'verified'
      ? `Requested constructor value observed in ${count}. Other constructors and later behavior are unverified.`
      : `Recorded constructor value differs from the request; ${count} checked.`;
  }
  const count = `${evidence.observedOperations} captured occurrence${evidence.observedOperations === 1 ? '' : 's'}`;
  const name = evidence.recipe === 'binary_scalar_v1' ? 'literal' : evidence.recipe === 'straight_line_tensor_v1' ? 'argument'
    : evidence.recipe === 'straight_line_unary_insert_v1' ? 'insertion' : evidence.recipe === 'straight_line_unary_delete_v1' ? 'deletion' : 'activation';
  if (evidence.status === 'verified') return `Requested ${name} observed in ${count}. Other executions are unverified.`;
  if (evidence.status === 'failed') return `Recorded ${name} differs from the request; ${count} checked.`;
  const reasons: Record<CandidatePostconditions['reason'], string> = {
    unsupported_recipe: 'This edit recipe has no semantic check yet.',
    capture_unavailable: 'No usable capture is available.', capture_incomplete: 'Capture is incomplete.',
    capture_identity_mismatch: 'Capture belongs to a different execution.',
    source_mapping_unavailable: 'The edited expression could not be mapped exactly.',
    expression_not_observed: 'The edited expression was not observed.',
    ambiguous_observation: 'Recorded operations do not establish the requested change.',
    initializer_arguments_unavailable: 'The required bound initializer parameter was not recorded.',
    configuration_reference_unavailable: 'A configuration reference could not be checked as a direct constructor argument.',
    observed_requested_configuration: '', observed_configuration_mismatch: '',
    observed_requested_constructor: '', observed_constructor_mismatch: '',
    observed_requested_literal: '', observed_literal_mismatch: '',
    observed_requested_activation: '', observed_activation_mismatch: '',
    observed_requested_argument: '', observed_argument_mismatch: '',
    observed_requested_insertion: '', observed_insertion_mismatch: '',
    observed_requested_deletion: '', observed_deletion_mismatch: '',
  };
  return `Unverified. ${reasons[evidence.reason]}`;
}
export function isCandidateAction(value: unknown): value is CandidateAction {
  if (!isRecord(value)) return false;
  if (value.kind === 'list' || value.kind === 'reconcile') return Object.keys(value).length === 1;
  if (value.kind === 'run') return Object.keys(value).length === 2 && isBoundedString(value.retryId, 128);
  if (value.kind === 'cancel') return Object.keys(value).length === 2 && isBoundedString(value.runId, 128);
  if (value.kind === 'checkpoint') return Object.keys(value).length === 4 && isBoundedString(value.runId, 128)
    && isProjectCount(value.offset) && value.offset <= 8192 && isProjectCount(value.limit) && value.limit >= 1 && value.limit <= 200;
  return value.kind === 'inspect' && isBoundedString(value.runId, 128)
    && Object.keys(value).length === (value.query === undefined ? 2 : 3)
    && (value.query === undefined || isCaptureInspectionQuery(value.query));
}

export function isCaptureInspectionPage(value: unknown): value is CaptureInspectionPage {
  if (!isRecord(value) || Object.keys(value).length !== (value.entity === undefined ? 7 : 8)
    || !isCaptureInspectionQuery({ captureId: value.captureId, sourceId: value.sourceId,
      ...(value.entity === undefined ? {} : { entity: value.entity }), path: value.path, offset: 0, limit: 1 })
    || !isProjectCount(value.total) || value.total > 1_000_000
    || !(value.nextOffset === null || (isProjectCount(value.nextOffset) && value.nextOffset < value.total))
    || !Array.isArray(value.items) || value.items.length > 200 || value.items.length > value.total
    || JSON.stringify(value).length > 65536) return false;
  const entry = (item: unknown, root: boolean): boolean => {
    if (!isRecord(item) || !Object.keys(item).every(key => ['key', 'kind', 'value', 'count', 'label'].includes(key))
      || (root ? item.key !== null : !((typeof item.key === 'string' && item.key.length <= 128) || (isProjectCount(item.key) && item.key <= 1_000_000)))
      || (item.label !== undefined && !(typeof item.label === 'string' && Array.from(item.label).length <= 160))) return false;
    if (item.kind === 'object' || item.kind === 'array') return !('value' in item) && isProjectCount(item.count) && item.count <= 1_000_000;
    return !('count' in item) && !('label' in item) && (item.kind === 'null' ? item.value === null
      : item.kind === 'string' ? typeof item.value === 'string' && Array.from(item.value).length <= 4096
      : item.kind === 'number' ? typeof item.value === 'number' && Number.isFinite(item.value)
      : item.kind === 'boolean' && typeof item.value === 'boolean');
  };
  if (!entry(value.node, true) || !value.items.every(item => entry(item, false))) return false;
  const node = value.node as unknown as CaptureInspectionEntry;
  return (node.kind === 'object' || node.kind === 'array' ? node.count === value.total : value.total === 0 && value.nextOffset === null)
    && new Set(value.items.map(item => JSON.stringify(item.key))).size === value.items.length;
}

/** Exact equality, like `sameTargetIdentity`. A project one revision behind is
 *  refused rather than acted on, for the same reason a stale class target is. */
export function sameProjectIdentity(
  asked: LocalBrowserProjectIdentity | undefined,
  current: LocalBrowserProjectIdentity | undefined,
): boolean {
  if (!asked || !current) {
    return false;
  }
  return (
    asked.workspaceFolder === current.workspaceFolder &&
    asked.projectRoot === current.projectRoot &&
    asked.experimentId === current.experimentId &&
    asked.indexDigest === current.indexDigest &&
    asked.indexRevision === current.indexRevision
  );
}

function isLocalBrowserSession(value: unknown): value is LocalBrowserSessionInfo {
  const record = value as Partial<LocalBrowserSessionInfo> | null;
  return (
    typeof record === 'object' &&
    record !== null &&
    isNonEmptyString(record.workspaceSessionId) &&
    (record.role === 'host' || record.role === 'view') &&
    record.protocolVersion === LOCAL_BROWSER_PROTOCOL_VERSION &&
    isLocalBrowserCapabilities(record.capabilities) &&
    isNonEmptyString(record.expiresAt) &&
    (record.handoffId === undefined || isBoundedString(record.handoffId, 128)) &&
    (record.viewId === undefined || (typeof record.viewId === 'string' && record.viewId.length > 0 && record.viewId.length <= 128))
  );
}

function hasRequiredString(value: object, key: string): boolean {
  return isNonEmptyString((value as Record<string, unknown>)[key]);
}

function hasForbiddenField(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function hasOptionalString(value: object, key: string): boolean {
  const property = (value as Record<string, unknown>)[key];
  return !Object.prototype.hasOwnProperty.call(value, key) || isNonEmptyString(property);
}

function hasOptionalTarget(value: object): boolean {
  return (
    !hasForbiddenField(value, 'target') ||
    isLocalBrowserTarget((value as Record<string, unknown>).target)
  );
}

function hasOptionalErrorCode(value: object): boolean {
  if (!hasForbiddenField(value, 'errorCode')) {
    return true;
  }
  const code = (value as Record<string, unknown>).errorCode;
  return (
    (typeof code === 'string' &&
      (LOCAL_BROWSER_ERROR_CODES as readonly string[]).includes(code))
  );
}
function isLocalBrowserViewState(value: unknown): value is 'attached' | 'detached' {
  return value === 'attached' || value === 'detached';
}


function isLocalBrowserConnectionState(value: unknown): boolean {
  return value === 'connected' || value === 'reconnecting' || value === 'disconnected';
}

function isLocalBrowserSyncState(value: unknown): boolean {
  return value === 'source-only' || value === 'runtime-traced' || value === 'stale';
}

function isLocalBrowserCapabilities(value: unknown): value is LocalBrowserCapabilities {
  const record = value as Partial<LocalBrowserCapabilities> | null;
  return (
    typeof record === 'object' &&
    record !== null &&
    Object.keys(record).length === (record.canProposeSourceEdits === true ? 6 : 5) &&
    (!hasForbiddenField(record, 'canProposeSourceEdits') || record.canProposeSourceEdits === true) &&
    record.readOnly === true &&
    record.canNavigateSource === true &&
    record.canWriteSource === false &&
    record.canRunAgent === false &&
    record.canEditGraph === false
  );
}

function isLocalBrowserSnapshotState(value: unknown): value is LocalBrowserSnapshotState {
  const record = value as Partial<LocalBrowserSnapshotState> | null;
  if (typeof record !== 'object' || record === null) {
    return false;
  }
  if (record.opening !== undefined && !isGraphOpeningState(record.opening)) return false;
  if (record.validation !== undefined && (!isLocalBrowserValidationSummary(record.validation)
    || !record.activeTarget || !['workspaceFolder', 'file', 'cls'].every(key =>
      record.activeTarget![key as keyof LocalBrowserTargetIdentity] === record.validation!.target[key as keyof LocalBrowserTargetIdentity]))) return false;
  if (record.activeTarget === null) {
    return (
      record.projection === null &&
      record.traced === false &&
      record.greyed === false &&
      record.syncState === null &&
      !hasForbiddenField(record, 'reason') &&
      isLocalBrowserConnectionState(record.connectionState)
    );
  }
  return (
    isLocalBrowserTarget(record.activeTarget) &&
    'projection' in record &&
    typeof record.traced === 'boolean' &&
    typeof record.greyed === 'boolean' &&
    isLocalBrowserSyncState(record.syncState) &&
    (record.syncState !== 'stale' || hasRequiredString(record, 'reason')) &&
    (record.syncState === 'stale' || !hasForbiddenField(record, 'reason')) &&
    isLocalBrowserConnectionState(record.connectionState)
  );
}

function isLocalBrowserValidationSummary(value: unknown): value is LocalBrowserValidationSummary {
  if (!isRecord(value)) return false;
  return Object.keys(value).length === 4 + Number(value.projectionRevision !== undefined) + Number(value.evidence !== undefined)
    && isLocalBrowserTarget(value.target) && isBoundedString(value.runId, 128)
    && ['running', 'passed', 'failed', 'refused', 'cancelled'].includes(value.status as string)
    && isBoundedString(value.detail, 512)
    && (value.evidence === undefined || isLocalBrowserValidationEvidence(value.evidence, value.status as string))
    && (value.projectionRevision === undefined || ((value.status === 'passed' || value.status === 'failed')
      && Number.isSafeInteger(value.projectionRevision) && (value.projectionRevision as number) >= 0));
}

function isLocalBrowserValidationEvidence(value: unknown, status: string): value is LocalBrowserValidationEvidence {
  if (!isRecord(value) || Object.keys(value).length !== 8) return false;
  const id = (item: unknown) => item === null || isBoundedString(item, 256);
  const identity = (record: Record<string, unknown>) => id(record.sourceId)
    && (record.planDigest === null || (typeof record.planDigest === 'string' && /^[0-9a-f]{64}$/.test(record.planDigest)))
    && (record.planDigest === null || record.sourceId !== null);
  const outcomes: Record<string, string[]> = { running: ['requested', 'running'], passed: ['succeeded'],
    failed: ['failed', 'timed_out'], refused: ['needs_setup', 'failed', 'cancelled', 'timed_out'], cancelled: ['cancelled'] };
  if (!['manual', 'mcp', 'automatic'].includes(value.origin as string)
    || !outcomes[status]?.includes(value.attemptStatus as string) || !identity(value)
    || !id(value.lastCompleteCaptureId) || !id(value.lastPartialCaptureId)) return false;
  const capture = value.displayedCapture;
  if (capture !== null && (!isRecord(capture) || Object.keys(capture).length !== 8
    || !isBoundedString(capture.id, 256) || !isBoundedString(capture.attemptId, 256)
    || !['legacy', 'semantic'].includes(capture.format as string)
    || !['complete', 'partial', 'failed'].includes(capture.status as string)
    || (capture.format === 'legacy' && capture.status !== 'partial')
    || !['passed', 'failed'].includes(capture.forwardStatus as string) || !identity(capture)
    || ![null, 'retention_limit'].includes(capture.unavailable as null | string))) return false;
  const setup = value.setup;
  return setup === null || (isRecord(setup) && Object.keys(setup).length === 4 && value.planDigest !== null
    && ['profile', 'explicit'].includes(setup.origin as string)
    && (setup.origin === 'profile' ? isBoundedString(setup.profileName, 128) : setup.profileName === null)
    && typeof setup.device === 'string' && /^(cpu|mps|cuda(?::[0-9]{1,3})?)$/.test(setup.device)
    && ['eval', 'train', 'preserve'].includes(setup.mode as string));
}

export const LOCAL_BROWSER_ERROR_CODES = [
  'PROTOCOL_VERSION_MISMATCH',
  'SESSION_MISMATCH',
  'INVALID_ORIGIN',
  'INVALID_CREDENTIALS',
  'CREDENTIAL_EXPIRED',
  'HOST_ALREADY_CONNECTED',
  'STALE_TARGET',
  'UNKNOWN_MESSAGE',
  'INVALID_MESSAGE',
] as const;

export function isLocalBrowserMessage(
  value: unknown,
  expectedWorkspaceSessionId?: string,
): value is LocalBrowserMessage {
  const record = value as Record<string, unknown> | null;
  if (
    typeof record !== 'object' ||
    record === null ||
    record.protocolVersion !== LOCAL_BROWSER_PROTOCOL_VERSION ||
    !isNonEmptyString(record.workspaceSessionId) ||
    typeof record.type !== 'string' ||
    !(LOCAL_BROWSER_MESSAGE_TYPES as readonly string[]).includes(record.type) ||
    !isNonEmptyString(record.messageId)
  ) {
    return false;
  }
  if (
    typeof expectedWorkspaceSessionId !== 'undefined' &&
    (expectedWorkspaceSessionId.length === 0 ||
      record.workspaceSessionId !== expectedWorkspaceSessionId)
  ) {
    return false;
  }

  switch (record.type) {
    case 'agentRequest': {
      if (!isBoundedString(record.viewId, 128)) return false;
      if (record.action === 'sync' || record.action === 'clear') return Object.keys(record).length === 6;
      if (record.action === 'discover') return Object.keys(record).every(key => ['type', 'protocolVersion', 'workspaceSessionId', 'messageId', 'viewId', 'action', 'refresh', 'catalogOnly'].includes(key))
        && (record.refresh === undefined || typeof record.refresh === 'boolean')
        && (record.catalogOnly === undefined || typeof record.catalogOnly === 'boolean');
      if (record.action === 'configure') return Object.keys(record).length === (record.requestId === undefined ? 8 : 9)
        && (record.requestId === undefined || isBoundedString(record.requestId, 128))
        && ['codex', 'claude'].includes(String(record.provider)) && isCanvasAgentSettings(record.settings);
      if (record.action === 'permission') return Object.keys(record).length === 8
        && Object.keys(record).every(key => ['type', 'protocolVersion', 'workspaceSessionId', 'messageId', 'viewId', 'action', 'permissionId', 'decision'].includes(key))
        && isBoundedString(record.permissionId, 128) && typeof record.decision === 'string' && ['allow', 'deny'].includes(record.decision);
      if (record.action === 'permission_mode') return Object.keys(record).length === 7
        && Object.keys(record).every(key => ['type', 'protocolVersion', 'workspaceSessionId', 'messageId', 'viewId', 'action', 'mode'].includes(key))
        && isCanvasAgentPermissionMode(record.mode);
      if (!isBoundedString(record.requestId, 128)) return false;
      if (record.action === 'stop') return Object.keys(record).length === 7;
      return record.action === 'send' && Object.keys(record).length === (record.settings === undefined ? 11 : 12) + (record.draftId === undefined ? 0 : 1)
        && (record.draftId === undefined || isBoundedString(record.draftId, 128))
        && (record.settings === undefined || isCanvasAgentSettings(record.settings))
        && ['codex', 'claude'].includes(String(record.provider)) && isBoundedString(record.prompt, 16000)
        && isLocalBrowserTarget(record.target) && Array.isArray(record.nodeIds) && record.nodeIds.length <= 100
        && record.nodeIds.every(id => isBoundedString(id, 1024));
    }
    case 'agentState':
      return Object.keys(record).length === 6 && isBoundedString(record.viewId, 128) && isCanvasAgentState(record.state);
    case 'nodeExplanation':
      return Object.keys(record).length === 7 && isBoundedString(record.viewId, 128)
        && isLocalBrowserTarget(record.target) && isNodeExplanationState(record.state);
    case 'modelCatalog':
      return Object.keys(record).length === 8 && isBoundedString(record.viewId, 128)
        && isLocalBrowserTarget(record.target) && isBoundedString(record.requestId, 128) && isModelCatalog(record.catalog);
    case 'modelCatalogRequest':
      return Object.keys(record).length === (record.action === 'select' ? 9 : 8)
        && Object.keys(record).every(key => ['type', 'protocolVersion', 'workspaceSessionId', 'messageId', 'viewId', 'target', 'requestId', 'action', 'modelId'].includes(key))
        && isBoundedString(record.viewId, 128) && isLocalBrowserTarget(record.target) && isBoundedString(record.requestId, 128)
        && (record.action === 'list' || (record.action === 'select' && isBoundedString(record.modelId, 128)));
    case 'modelExplanation':
      return Object.keys(record).length === 7 && isBoundedString(record.viewId, 128)
        && isLocalBrowserTarget(record.target) && isModelExplanationState(record.state);
    case 'modelExplanationRequest':
      return Object.keys(record).length === (record.settings === undefined ? 9 : 10)
        && Object.keys(record).every(key => ['type', 'protocolVersion', 'workspaceSessionId', 'messageId', 'viewId', 'target', 'requestId', 'action', 'nodeIds', 'settings'].includes(key))
        && isBoundedString(record.viewId, 128) && isLocalBrowserTarget(record.target)
        && isBoundedString(record.requestId, 128) && Array.isArray(record.nodeIds)
        && record.nodeIds.length <= 128 && record.nodeIds.every(id => isBoundedString(id, 1024))
        && new Set(record.nodeIds).size === record.nodeIds.length
        && (record.action === 'start' ? record.nodeIds.length > 0 && (record.settings === undefined || isCanvasAgentSettings(record.settings))
          : record.action === 'cancel' && record.nodeIds.length === 0 && record.settings === undefined);
    case 'nodeExplanationRequest':
      return Object.keys(record).length === (record.action === 'reveal' ? 10 : 9)
        && Object.keys(record).every(key => ['type', 'protocolVersion', 'workspaceSessionId', 'messageId', 'viewId', 'target', 'nodeId', 'requestId', 'action', 'citationId'].includes(key))
        && isBoundedString(record.viewId, 128) && isLocalBrowserTarget(record.target)
        && isBoundedString(record.nodeId, 1024) && isBoundedString(record.requestId, 128)
        && (record.action === 'retry' || (record.action === 'reveal' && isBoundedString(record.citationId, 128)));
    case 'graphReviewApplied':
      return Object.keys(record).length === 7 && isBoundedString(record.reviewId, 128)
        && isBoundedString(record.viewId, 128) && ['before', 'diff', 'after'].includes(String(record.mode));
    case 'graphReviewRequest':
      return ['open', 'openPR', 'refresh', 'close', 'reveal', 'selectModel', 'acceptDraft', 'discardDraft'].includes(String(record.action))
        && Object.keys(record).every(key => ['type', 'protocolVersion', 'workspaceSessionId', 'messageId', 'action', 'reviewId', 'side', 'nodeId', 'modelId'].includes(key))
        && (record.action === 'open' || record.action === 'openPR' || isBoundedString(record.reviewId, 128))
        && (record.action !== 'selectModel' || isBoundedString(record.modelId, 128))
        && (record.action !== 'reveal' || ((record.side === 'before' || record.side === 'after') && (record.nodeId === undefined || isBoundedString(record.nodeId, 1024))));
    case 'graphReviewState':
      return record.state === null || (isRecord(record.state) && isBoundedString(record.state.id, 128)
        && (record.state.proposal === undefined || (isRecord(record.state.proposal) && isBoundedString(record.state.proposal.id, 128)
          && (record.state.proposal.unavailable === undefined || typeof record.state.proposal.unavailable === 'boolean')
          && (record.state.proposal.canRevise === undefined || typeof record.state.proposal.canRevise === 'boolean')))
        && isRecord(record.state.before) && isRecord(record.state.after)
        && typeof record.state.beforeLabel === 'string' && typeof record.state.afterLabel === 'string'
        && typeof record.state.stale === 'boolean' && Array.isArray(record.state.changedFiles)
        && record.state.changedFiles.every(file => typeof file === 'string'));
    case 'graphReviewFocus':
      return isBoundedString(record.reviewId, 128) && (record.side === 'before' || record.side === 'after')
        && Array.isArray(record.nodeIds) && record.nodeIds.length <= 100 && record.nodeIds.every(id => isBoundedString(id, 1024));
    case 'projectIndex':
    case 'projectDocument':
    case 'projectFocus':
    case 'projectExpand':
    case 'projectDeclare': {
      const fields = ['type', 'protocolVersion', 'workspaceSessionId', 'messageId', 'project'];
      const keys = (required: string[], optional: string[] = []) =>
        required.every(key => key in record) &&
        Object.keys(record).every(key => [...fields, ...required, ...optional].includes(key));
      if (!isLocalBrowserProjectIdentity(record.project)) return false;
      switch (record.type) {
        case 'projectIndex':
          return keys(['hostSequence', 'index', 'truncated'], ['indexed', 'discovered']) &&
            isProjectCount(record.hostSequence) && isRecord(record.index) &&
            typeof record.truncated === 'boolean' &&
            (!('indexed' in record) || isProjectCount(record.indexed)) &&
            (!('discovered' in record) || isProjectCount(record.discovered));
        case 'projectDocument':
          return keys(['hostSequence', 'document', 'spanmap', 'expanded']) &&
            isProjectCount(record.hostSequence) && isRecord(record.document) && isRecord(record.spanmap) &&
            Array.isArray(record.expanded) && record.expanded.length <= 4096 &&
            record.expanded.every(id => isBoundedString(id, 512)) && new Set(record.expanded).size === record.expanded.length;
        case 'projectFocus':
          return keys(['experimentId']) && typeof record.experimentId === 'string' && record.experimentId.length <= 512;
        case 'projectExpand':
          return keys(['nodeId', 'expanded']) && isBoundedString(record.nodeId, 512) && typeof record.expanded === 'boolean';
        case 'projectDeclare':
          return keys(['declaration', 'args']) && isProjectDeclaration(record.declaration, record.args);
        default: return false;
      }
    }
    case 'codeFocusRequest':
    case 'codeFocusState':
      return Object.keys(record).length === (record.type === 'codeFocusRequest' ? 7 : 9)
        && isBoundedString(record.viewId, 128) && isLocalBrowserTarget(record.target) && isCodeFocusOptions(record.options)
        && (record.type === 'codeFocusRequest' || (typeof record.parameter === 'boolean' && isBoundedString(record.detail, 512)));
    case 'viewState':
      if (!('viewId' in record) && !('connected' in record)) {
        return !hasForbiddenField(record, 'hostSequence') && !hasForbiddenField(record, 'target') && isLocalBrowserViewState(record.state);
      }
      return Object.keys(record).length === (record.handoffId === undefined ? 6 : 7) &&
        (record.handoffId === undefined || isBoundedString(record.handoffId, 128)) && typeof record.viewId === 'string' && record.viewId.length > 0 && record.viewId.length <= 128 && typeof record.connected === 'boolean';
    case 'viewSelection':
    case 'focusView': {
      const ids = (value: unknown) => Array.isArray(value) && value.length <= 100 && value.every(id => typeof id === 'string' && id.length > 0 && id.length <= 512) && new Set(value).size === value.length;
      const optional = record.type === 'focusView' ? 1 : Number(record.tooLarge !== undefined) + Number(record.follow !== undefined);
      return Object.keys(record).length === 8 + optional &&
        (record.tooLarge === undefined || (record.type === 'viewSelection' && record.tooLarge === true && Array.isArray(record.nodeIds) && record.nodeIds.length === 0 && Array.isArray(record.edgeIds) && record.edgeIds.length === 0)) &&
        (record.follow === undefined || (record.type === 'viewSelection' && typeof record.follow === 'boolean')) &&
        typeof record.viewId === 'string' && record.viewId.length > 0 && record.viewId.length <= 128 &&
        isLocalBrowserTarget(record.target) && ids(record.nodeIds) && ids(record.edgeIds) &&
        (record.type !== 'focusView' || (typeof record.requestId === 'string' && record.requestId.length > 0 && record.requestId.length <= 128));
    }
    case 'viewFocusResult':
      return Object.keys(record).length === 9 && typeof record.viewId === 'string' && record.viewId.length > 0 && record.viewId.length <= 128 &&
        isLocalBrowserTarget(record.target) && typeof record.requestId === 'string' && record.requestId.length > 0 && record.requestId.length <= 128 &&
        typeof record.ok === 'boolean' && typeof record.detail === 'string' && record.detail.length > 0 && record.detail.length <= 512;

    case 'connected':
      return (
        (record.role === 'host' || record.role === 'view') &&
        isLocalBrowserSession(record.session) &&
        record.session.workspaceSessionId === record.workspaceSessionId &&
        record.session.role === record.role &&
        !hasForbiddenField(record, 'hostSequence') &&
        !hasForbiddenField(record, 'target')
      );
    case 'hostHandshake':
      return (
        record.role === 'host' &&
        (record.viewControl === undefined || record.viewControl === true) &&
        !hasForbiddenField(record, 'hostSequence') &&
        !hasForbiddenField(record, 'target') &&
        !hasForbiddenField(record, 'session')
      );
    case 'viewHandshake':
      return (
        record.role === 'view' &&
        (record.handoffId === undefined || isBoundedString(record.handoffId, 128)) &&
        !hasForbiddenField(record, 'hostSequence') &&
        !hasForbiddenField(record, 'target') &&
        !hasForbiddenField(record, 'session')
      );
    case 'snapshotRequest':
      return (
        record.role === 'view' &&
        isNonNegativeInteger(record.lastHostSequence) &&
        !hasForbiddenField(record, 'hostSequence') &&
        !hasForbiddenField(record, 'target')
      );
    case 'snapshot':
      return (
        record.role === 'host' &&
        isNonNegativeInteger(record.hostSequence) &&
        isLocalBrowserSnapshotState(record.state)
      );
    case 'activeTarget':
      return (
        isNonNegativeInteger(record.hostSequence) &&
        isLocalBrowserTarget(record.target) &&
        !('lastValidProjection' in record) &&
        !('nodeId' in record)
      );
    case 'projection':
      return (
        isNonNegativeInteger(record.hostSequence) &&
        isLocalBrowserTarget(record.target) &&
        'projection' in record &&
        typeof record.traced === 'boolean' &&
        typeof record.greyed === 'boolean'
      );
    case 'connectionState':
      return (
        !hasForbiddenField(record, 'hostSequence') &&
        !hasForbiddenField(record, 'target') &&
        isLocalBrowserConnectionState(record.state)
      );
    case 'syncState':
      return (
        isNonNegativeInteger(record.hostSequence) &&
        isLocalBrowserTarget(record.target) &&
        isLocalBrowserSyncState(record.state) &&
        hasOptionalString(record, 'reason')
      );
    case 'staleState':
      return (
        isNonNegativeInteger(record.hostSequence) &&
        isLocalBrowserTarget(record.target) &&
        'lastValidProjection' in record &&
        hasRequiredString(record, 'reason')
      );
    case 'revealNode':
      return (
        !hasForbiddenField(record, 'hostSequence') &&
        isLocalBrowserTarget(record.target) &&
        hasRequiredString(record, 'nodeId')
      );
    case 'focusNode':
      return (
        isNonNegativeInteger(record.hostSequence) &&
        isLocalBrowserTarget(record.target) &&
        hasRequiredString(record, 'nodeId') &&
        (record.nodeIds === undefined ||
          (Array.isArray(record.nodeIds) && record.nodeIds.every(isNonEmptyString)))
      );
    case 'captureExportRequest':
      return Object.keys(record).length === 7 && isLocalBrowserTarget(record.target)
        && isBoundedString(record.requestId, 128) && isCaptureExportReference(record.reference);
    case 'captureExportResult':
      return Object.keys(record).length === 9 && isLocalBrowserTarget(record.target)
        && isBoundedString(record.requestId, 128) && isBoundedString(record.detail, 512)
        && isCaptureExportReference(record.reference) && (record.capture === null || isPortableCaptureEnvelope(record.capture));
    case 'captureInspectRequest':
      return Object.keys(record).length === 7 && isLocalBrowserTarget(record.target)
        && isBoundedString(record.requestId, 128) && isCaptureInspectionQuery(record.query);
    case 'captureInspectResult':
      return Object.keys(record).length === 8 && isLocalBrowserTarget(record.target)
        && isBoundedString(record.requestId, 128) && isBoundedString(record.detail, 512)
        && (record.page === null || isCaptureInspectionPage(record.page));
    case 'graphOpening':
      return Object.keys(record).length === 6 && isNonNegativeInteger(record.hostSequence) && isGraphOpeningState(record.state);
    case 'graphSetupRequest':
      return Object.keys(record).length === 8 && isLocalBrowserTarget(record.target) && isBoundedString(record.openingId, 128)
        && isBoundedString(record.requestId, 128) && isGraphSetupAction(record.action);
    case 'captureExportCancel':
    case 'captureInspectCancel':
    case 'validateCancel':
      return Object.keys(record).length === 6 && isLocalBrowserTarget(record.target) &&
        isBoundedString(record.requestId, LOCAL_BROWSER_REQUEST_ID_MAX_LENGTH);
    case 'validateRequest':
      return Object.keys(record).length === 6 + Number(record.fresh !== undefined)
        && (record.fresh === undefined || record.fresh === true)
        && isLocalBrowserTarget(record.target) && isBoundedString(record.requestId, LOCAL_BROWSER_REQUEST_ID_MAX_LENGTH);
    case 'validationState':
      return isLocalBrowserValidationSummary(Object.fromEntries(Object.entries(record)
        .filter(([key]) => !['type', 'protocolVersion', 'workspaceSessionId', 'messageId'].includes(key))));
    case 'validateResult':
      return Object.keys(record).length === 8 + Number(record.failure !== undefined) + Number(record.reused !== undefined)
        && isLocalBrowserTarget(record.target) &&
        (record.reused === undefined || (isExecutionReuse(record.reused) && record.status === 'passed')) &&
        (record.failure === undefined || isGraphFailure(record.failure)) &&
        isBoundedString(record.requestId, LOCAL_BROWSER_REQUEST_ID_MAX_LENGTH) &&
        ['passed', 'failed', 'refused', 'cancelled'].includes(record.status as string) &&
        isBoundedString(record.detail, 512);
    case 'sourceEditRequest':
      return Object.keys(record).length === (record.viewId === undefined ? 7 : 8) && !hasForbiddenField(record, 'hostSequence') &&
        (record.viewId === undefined || isBoundedString(record.viewId, 128)) &&
        isLocalBrowserTarget(record.target) &&
        isBoundedString(record.requestId, LOCAL_BROWSER_REQUEST_ID_MAX_LENGTH) &&
        isLocalBrowserSourceEdit(record.edit) && JSON.stringify(record.edit).length <= 16384;
    case 'sourceEditCancel':
      return Object.keys(record).length === 6 && isLocalBrowserTarget(record.target) &&
        isBoundedString(record.requestId, LOCAL_BROWSER_REQUEST_ID_MAX_LENGTH);
    case 'sourceEditResult':
      return Object.keys(record).length === 8 && isLocalBrowserTarget(record.target) &&
        isBoundedString(record.requestId, LOCAL_BROWSER_REQUEST_ID_MAX_LENGTH) &&
        ['applied', 'cancelled', 'refused'].includes(record.status as string) &&
        isBoundedString(record.detail, 512);
    case 'sourceEditReview':
      return Object.keys(record).length === 7 && isLocalBrowserTarget(record.target) &&
        isBoundedString(record.requestId, 128) && (record.candidate === null || isCandidateReviewSnapshot(record.candidate));
    case 'candidateRequest':
    case 'candidateResult':
      if (!isLocalBrowserTarget(record.target) || !isBoundedString(record.viewId, 128)
        || !isBoundedString(record.requestId, 128) || !isBoundedString(record.proposalId, 128)) return false;
      if (record.type === 'candidateRequest') return Object.keys(record).length === 9 && isCandidateAction(record.action);
      return Object.keys(record).length === 10 && isBoundedString(record.detail, 512)
        && (record.result === null || (isRecord(record.result) && Object.keys(record.result).length === 2 + Number(record.result.applied !== undefined) + Number(record.result.checkpoint !== undefined)
          && Array.isArray(record.result.candidates) && record.result.candidates.length <= 32
          && record.result.candidates.every(isCandidateRecord)
          && (record.result.applied === undefined || isAppliedSourceObservation(record.result.applied))
          && (record.result.checkpoint === undefined || (isCandidateCheckpointPage(record.result.checkpoint)
            && record.result.page === null && record.result.applied === undefined && record.result.candidates.length === 1
            && record.result.candidates[0].runId === record.result.checkpoint.runId
            && record.result.candidates[0].summary.checkpoint?.entries === record.result.checkpoint.count))
          && (record.result.page === null || isCaptureInspectionPage(record.result.page))));
    case 'ack':
      return (
        !hasForbiddenField(record, 'hostSequence') &&
        isLocalBrowserTarget(record.target) &&
        hasRequiredString(record, 'ackId') &&
        typeof record.ok === 'boolean' &&
        hasOptionalErrorCode(record) &&
        hasOptionalString(record, 'detail')
      );
    case 'error':
      return (
        !hasForbiddenField(record, 'hostSequence') &&
        hasOptionalTarget(record) &&
        typeof record.code === 'string' &&
        (LOCAL_BROWSER_ERROR_CODES as readonly string[]).includes(record.code) &&
        hasRequiredString(record, 'message')
      );
    default:
      return false;
  }
}

/* ── Wire surface: the HTTP/WS routes and fragment grammar the transports speak ──
 *
 * These are not protocol messages, so the frozen contract above does not carry
 * them as message fields, but the extension host transport and the browser view
 * transport must agree on them byte-for-byte. The broker keeps its own copies
 * (backend/api/routers/local_browser.py and services/ide/local_browser_broker.py);
 * docs/local-browser-transports.md documents the pairing. */

/** Session mint: the extension only, behind the per-launch token. */
export const LOCAL_BROWSER_SESSIONS_PATH = '/local/sessions';
/** Fragment credential trade: the browser page, once, same-origin. */
export const LOCAL_BROWSER_BOOTSTRAP_PATH = '/local/bootstrap';
/** The broker socket both roles connect to. */
export const LOCAL_BROWSER_WEBSOCKET_PATH = '/local/ws';
/** The HttpOnly cookie bootstrap sets and the socket authenticates on. */
export const LOCAL_BROWSER_COOKIE_NAME = 'tensorviz_local_browser';

/** Host-originated ordered state: these carry `hostSequence`, which must
 *  strictly increase within one host epoch (broker `_SEQUENCED_HOST_TYPES`). */
export const LOCAL_BROWSER_SEQUENCED_MESSAGE_TYPES = [
  'graphOpening',
  'snapshot',
  'activeTarget',
  'projection',
  // Ordered host state: a project index and the document drawn from it must
  // arrive in the order they were produced, or the page renders a document
  // against an index it has not seen.
  'projectIndex',
  'projectDocument',
  'syncState',
  'staleState',
  'focusNode',
] as const;

/** Close codes 4400-4499 the broker pairs 1:1 with contract error codes; a close
 *  in this band after the handshake is a terminal refusal, not a dropped link. */
export const LOCAL_BROWSER_CLOSE_CODES: Readonly<Record<string, number>> = {
  INVALID_CREDENTIALS: 4401,
  CREDENTIAL_EXPIRED: 4402,
  INVALID_ORIGIN: 4403,
  PROTOCOL_VERSION_MISMATCH: 4404,
  SESSION_MISMATCH: 4405,
  INVALID_MESSAGE: 4406,
  UNKNOWN_MESSAGE: 4407,
  HOST_ALREADY_CONNECTED: 4408,
};

/** The fragment grammar of the page URL the extension opens. The fragment is the
 *  only place the view credential may appear: fragments never reach access logs,
 *  referrer headers, or browser history once the page replaces state. */
export function encodeLocalBrowserFragment(
  workspaceSessionId: string,
  credential: string,
): string {
  const params = new URLSearchParams({
    session: workspaceSessionId,
    credential: credential,
  });
  return `#${params.toString()}`;
}

/** Inverse of `encodeLocalBrowserFragment`. Returns null for any fragment that
 *  is not exactly a session+credential pair (a router path, say), so callers
 *  never bootstrap against an unrelated anchor. */
export function decodeLocalBrowserFragment(
  hash: string,
): { workspaceSessionId: string; credential: string } | null {
  const raw = hash.startsWith('#!') ? hash.slice(2) : hash.startsWith('#') ? hash.slice(1) : hash;
  if (raw === '' || raw.startsWith('/')) {
    return null;
  }
  const params = new URLSearchParams(raw);
  const workspaceSessionId = params.get('session');
  const credential = params.get('credential');
  if (workspaceSessionId === null || credential === null) {
    return null;
  }
  return { workspaceSessionId, credential };
}

/** A fresh message id. `crypto.randomUUID` exists in every supported runtime
 *  (Node >= 18 extension host, browsers on the loopback secure context); the
 *  fallback exists for the jsdom test environment, which predates it. The id is
 *  correlation only, never authentication, so the fallback needs no entropy
 *  guarantee beyond uniqueness within a run. */
export function newLocalBrowserMessageId(): string {
  const cryptoApi = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (cryptoApi && typeof cryptoApi.randomUUID === 'function') {
    return cryptoApi.randomUUID();
  }
  return `msg-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
}

/** One file the host should write to disk. */
export interface HostFile {
  /** Base name only. The host rejects anything with a path separator in it. */
  name: string;
  /** base64 of the file's bytes. */
  bytes: string;
}

/**
 * One run, as much of it as a comparison reads (task.md P6).
 *
 * A comparison record, not the whole traced graph. A full record carries every
 * span, which is
 * megabytes for a real model — and a comparison reads the node ids, the ops,
 * the parameters and the loss. Sending the rest would put a model's entire
 * trace through `postMessage` twice to render a table of what changed.
 *
 * `spans` in particular is never sent: it is the largest field on a record and
 * the one thing a comparison has no use for at all.
 */
export interface ComparedRun {
  run_id: string;
  /** `model`, `data` or `train_step`. */
  kind: string;
  /** `<file>::<class or callable>`. */
  target: string;
  /** ISO-8601 UTC. The only thing that decides which run is "before". */
  started_at: string;
  graph: { nodes?: Array<{ node_id: string; op?: string; params?: Record<string, unknown> }> } | null;
  diagnostics?: { loss?: number | null } | null;
}

/** App → host. */
export type AppMessage =
  /** The app booted and the bridge is listening. `t` is Date.now() at that moment. */
  | { type: 'ready'; protocolVersion: number; t: number }
  /**
   * Whether tldraw is still mounted, posted on ready and every 2 s afterwards.
   * This is the in-VS-Code proof that the licence gate did not unmount the
   * editor five seconds in (task.md D2): a `mounted: false` at t = ready + 8000
   * is exactly the failure Stage A exists to avoid.
   */
  | { type: 'canvasState'; mounted: boolean; t: number }
  /** A download. The host writes it under the workspace and reveals it. */
  | { type: 'saveFile'; file: HostFile }
  /** A multi-file export. `dir` is one path segment under tensorviz-exports/. */
  | { type: 'saveFiles'; dir: string; files: HostFile[] }
  /**
   * Add to tensorViz: the graph file Export graph would save, which the host
   * uploads to the member's tensorViz account after an explicit click. Nothing
   * else leaves the machine, and the host decides where it goes.
   */
  | { type: 'shareGraph'; file: HostFile }
  | { type: 'openExternal'; url: string }
  | { type: 'clipboard'; text: string; requestId?: string }
  /** Explicit paste in the native chat prompt; never requested on mount or focus. */
  | { type: 'readClipboard'; requestId: string }
  /** A fixed native action; the app cannot supply a command or arguments. */
  | { type: 'showCommandPalette' }
  /** The native empty Canvas opens the existing model picker; no webview-supplied target. */
  | { type: 'openAsGraph' }
  /**
   * The canvas finished drawing a `projection`. This is the only report of what
   * the user can actually see: the host knows what it sent, not what survived
   * the merge, and `nodeCount` is counted off the canvas after applyProjection
   * has merged by node id. `sourceDigest` echoes the buffer the projection was
   * computed from, so a report that arrives after the file moved on can be
   * recognised as stale rather than believed.
   */
  | {
      type: 'projectionApplied';
      nodeCount: number;
      traced: boolean;
      greyed: boolean;
      sourceDigest: string;
    }
  /**
   * Which folder is open (task.md M6).
   *
   * The app cannot work this out for itself: it is served over http by the
   * backend and framed by the shell, and nothing in that URL says which
   * workspace it belongs to. The Workspace palette section needs an absolute
   * root to scan, so it asks. Asking rather than listening for an announcement
   * is what makes it reliable — the shell answers `hostReady` within
   * milliseconds of startup, long before any palette exists.
   */
  | { type: 'workspaceRoot' }
  | { type: 'toast'; level: 'info' | 'warn' | 'error'; message: string };

/** One class's graph, as `/api/ide/parse` returned it. */
export interface GraphDebugContext {
  /** Display context only. It is never the current source or export authority. */
  previousDocument?: unknown;
  nodePairs: { beforeId: string; afterId: string }[];
  changed: { nodeId: string; details: string[] }[];
}
export interface ProjectionPayload {
  /** `ide:<sha256(rel file)[:12]>#<Class>` — the AI harness's CAS anchor (D7). */
  graphId: string;
  revision: number;
  cls: string;
  /** Workspace-relative, forward slashes. */
  file: string;
  sourceDigest: string;
  /** False while the graph is the AST tier only, so shapes badge as stale.
   *  True once a trace from the user's own interpreter has been merged in, which
   *  is the difference between "what the source says" and "what Python did". */
  traced: boolean;
  /** The forward raised while this projection was being traced. The graph is real
   *  up to that point and every node after it says `unreached`; the message is
   *  also filed as a diagnostic on the raising line. */
  traceError?: { message: string; nodeId?: string; file?: string; line?: number };
  /**
   * The buffer no longer parses and this is the last projection that did. The
   * canvas keeps drawing it, dimmed, rather than blanking: a graph that vanishes
   * on every half-typed line is worse than a stale one that says it is stale.
   */
  greyed: boolean;
  /** GraphDocumentV2. Opaque here: the extension never models the app's schema. */
  document: unknown;
  debug?: GraphDebugContext;
  /**
   * A layout the user arranged was merged into `document.view.node_positions`
   * from `.tensorviz/views/`.
   *
   * The canvas needs telling because it cannot see the difference: every
   * projection carries positions, and the ones the tracer emits are not a layout
   * anybody chose. False means opening this graph may arrange it; true means the
   * positions are the user's work and must be drawn as they are.
   */
  viewRestored: boolean;
  /** node id -> {file, start, end, kind, digest} (task.md §6.2). */
  spanmap: Record<string, unknown>;
  /** Multi-file projection fields (P0 step 4): present only on the external
   *  local page's tier, which projects through `/api/ide/multifile/project`.
   *  `projectionDigest` identifies the whole source set the graph was computed
   *  from (an edit to any dependency moves it); `dependencies` are the files the
   *  projection consulted, with the digests used; `diagnostics` explain every
   *  node that deliberately stayed opaque or partial; `truncated` is the host's
   *  admission that the source set it could supply was incomplete. */
  projectionDigest?: string;
  dependencies?: IdeSourceFileDigest[];
  diagnostics?: IdeProjectionDiagnostic[];
  truncated?: boolean;
}

/**
 * The versioned multi-file source-projection HTTP contract
 * (POST /api/ide/multifile/project; CURRENT_TASK.md "P0 Reliable Multi-File
 * Model Graphs", docs/ide-multifile-projection-contract.md).
 *
 * snake_case throughout, on purpose: this is backend wire JSON, mirrored by
 * pydantic in backend/services/ide/multifile.py — the same discipline
 * ActiveTargetPayload already follows. The page reads `diagnostics` and the
 * enriched spanmap through the projection the host relays, so both sides of
 * `sync-protocol.mjs` parse this block verbatim.
 *
 * Digests:
 *  - `source_digest` identifies the ROOT buffer only. It is the same digest
 *    /api/ide/apply compares before editing the root file, and nothing else
 *    may reuse it.
 *  - `projection_digest` identifies the projection over the ENTIRE validated
 *    source set. An edit to any supplied dependency changes it, so the host
 *    can notice a dirty dependency before resolution even reads the file.
 */
export const IDE_PROJECTION_PROTOCOL_VERSION = 1;

/** The closed set of diagnostic codes a projection may report. */
export type IdeProjectionDiagnosticCode =
  | 'UNRESOLVED_IMPORT'
  | 'UNRESOLVED_DYNAMIC'
  | 'AMBIGUOUS_BINDING'
  | 'CYCLE_BOUNDARY'
  | 'EXTERNAL_PACKAGE'
  | 'DEPENDENCY_SYNTAX_ERROR'
  | 'SOURCE_SET_TRUNCATED'
  | 'UNSUPPORTED_IMPORT_FORM'
  | 'UNSUPPORTED_INHERITANCE'
  | 'RESOURCE_LIMIT';

export interface IdeProjectionDiagnostic {
  code: IdeProjectionDiagnosticCode;
  severity: 'info' | 'warning' | 'error';
  /** Normalised workspace-relative POSIX path, '' when the diagnostic is not file-scoped. */
  file: string;
  /** 1-based line; 0 when there is no anchor line. */
  line: number;
  /** The import/call text that produced this, '' when unknown. */
  import_text: string;
  /** One plain sentence a user can act on. */
  detail: string;
  /** The canvas node this diagnosis belongs to (a cycle boundary card, an
   *  unexpandable instance); absent when the diagnosis belongs to a file or
   *  the whole request. Present since step 3 of the multi-file task. */
  node?: string;
}

export interface IdeSourceFileDigest {
  /** Normalised workspace-relative POSIX path. */
  file: string;
  digest: string;
}

export interface IdeMultifileSourceEntry {
  file: string;
  source: string;
  digest: string;
  /** VS Code document version when the file is open (the dirty buffer's version). */
  document_version?: number;
}

export interface IdeMultifileRoot {
  file: string;
  cls: string;
  source_digest: string;
  document_version?: number;
}

export interface IdeMultifileProjectionRequest {
  protocol_version: number;
  root: IdeMultifileRoot;
  /** The complete candidate Python source set; dirty buffers preferred over disk. */
  sources: IdeMultifileSourceEntry[];
  /** The host's own admission that it could not enumerate the full candidate set. */
  truncated: boolean;
  /** Stateless source review; never overwrites the live projection or accepts a trace. */
  review_only?: boolean;
  /** The executed-tier worker envelope, exactly as /api/ide/project's trace_result. */
  trace_result?: unknown;
}

/** spanmap entry: `file_digest` is the buffer's digest, null when the span's file
 *  was not in the supplied set (a trace frame in torch's own source, say). */
export interface IdeMultifileSpan {
  file: string;
  file_digest: string | null;
  start: [number, number];
  end: [number, number];
  kind: string;
  digest: string;
}

export interface IdeMultifileProjectionResponse {
  protocol_version: number;
  cls: string;
  document: unknown;
  spanmap: Record<string, IdeMultifileSpan>;
  source_digest: string;
  projection_digest: string;
  graph_id: string;
  revision: number;
  traced: boolean;
  trace_error?: unknown;
  project_id: string;
  /** The files the projection actually depended on, with the digests used. */
  dependencies: IdeSourceFileDigest[];
  diagnostics: IdeProjectionDiagnostic[];
  truncated: boolean;
}

/** Host → app. */
export type HostMessage =
  /** Sent once when the shell has a port to talk to; carries nothing the app needs yet. */
  | { type: 'hostReady'; protocolVersion: number }
  /** The absolute path of the open workspace folder, or '' when none is open.
   *  Answered by the shell itself, which already holds it. */
  | { type: 'workspaceRoot'; root: string }
  | { type: 'clipboardText'; requestId: string; text: string }
  | { type: 'clipboardText'; requestId: string; error: true }
  | { type: 'clipboardWritten'; requestId: string; error?: true }
  | { type: 'compareRuns'; runs: [ComparedRun, ComparedRun] };

export type BridgeMessage = AppMessage | HostMessage;

export function isAppMessage(value: unknown): value is AppMessage {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as { type?: unknown }).type === 'string'
  );
}
