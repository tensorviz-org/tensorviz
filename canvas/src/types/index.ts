import { Node, Edge } from 'reactflow';
import { GraphDocumentV2 } from './graphDocument';
import type { components } from './semanticContracts.generated';

export type CaptureReference = components['schemas']['CaptureReference'];
export type CapturePort = components['schemas']['DisplayPort'];
export type CaptureEdge = components['schemas']['ProjectionEdge'];
export interface GraphIssue {
  id: string;
  nodeId: string;
  label: string;
  severity: 'error' | 'warning';
  message: string;
}

export * from './graphDocument';

export interface LayerType {
  id: string;
  name: string;
  displayName: string;
  description: string;
  icon: string;
  category: 'io' | 'module' | 'conv' | 'linear' | 'activation' | 'pool' | 'norm' | 'misc' | 'recurrent' | 'embedding' | 'loss' | 'tensor' | 'dropout';
  parameters: Parameter[];
  inputShape?: number[];
  outputShape?: number[];
}

export interface Parameter {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'select';
  defaultValue: any;
  required: boolean;
  description: string;
  options?: string[]; // for select type
  min?: number; // for number type
  max?: number; // for number type
  /**
   * Render this parameter only when the predicate holds. Presentational ONLY —
   * the backend still owns the real constraint (OpSpec.param_schema and the
   * required-arg guard in code_generation), so a hand-edited or AI-built graph is
   * still checked. Needed because some torch signatures are mutually exclusive:
   * F.interpolate takes `size` XOR `scale_factor`, and `align_corners` is valid
   * only for the linear family. Without this the generic form renders every field
   * unconditionally and the default outcome is a graph that cannot run.
   *
   * ParameterEditor CLEARS a hidden field's value to '', which is what keeps the
   * predicate out of Python: codegen already skips empty params.
   */
  showIf?: { param: string; equals?: unknown; oneOf?: unknown[] };
}

export interface ModelNodeData {
  layerType: LayerType;
  parameters: Record<string, any>;
  /** Ordered semantic port identities, including ports with no tensor edge. */
  inputPorts?: string[];
  outputPorts?: string[];
  captureRef?: CaptureReference;
  /** Verified presentation hint; excluded from semantic serialization. */
  capturePassThrough?: { valueId: string; inputPortId: string; sourceNodeId: string; sourcePortId: string };
  passThroughShown?: boolean;
  /** Presentation preference on the root invocation; never a capture fact. */
  captureDetailLevel?: 'layers' | 'operations';
  operationDetailsShown?: boolean;
  /** Derived display group. Original operation records and parents stay intact. */
  operationGroup?: { kind?: 'slice'; members: string[]; labels: string[]; inputShape?: number[]; outputShape?: number[] };
  operationGroupHiddenBy?: string;
  /** Exact saved-source expression binding, verified by the host for slice presentation. */
  captureSourceExpression?: string;
  captureSliceAxis?: number;
  onRevealOperations?: (nodeId: string) => void;
  capturePorts?: CapturePort[];
  captureLabel?: string;
  capturedOperationCount?: number;
  sourceEditCapabilities?: (components['schemas']['LiteralEditCapability'] | components['schemas']['ActivationEditCapability'] | components['schemas']['ConstructorLiteralCapability'] | components['schemas']['ConfigurationLiteralCapability'] | components['schemas']['ReconnectArgumentCapability'] | components['schemas']['InsertUnaryCapability'] | components['schemas']['DeleteUnaryCapability'])[];
  sourceEditLimitation?: string;
  /** Derived collapsed interfaces; never persisted as semantic ports. */
  captureDisplayPorts?: CapturePort[];
  inputShape?: number[];
  outputShape?: number[];
  error?: string;
  sourceLocation?: { file: string; line: number };
  captureDiagnostic?: components['schemas']['Diagnostic'];
  shapeEvidence?: 'measured' | 'inferred' | 'stale' | 'unknown';
  /** Local debugging presentation; excluded from graph-file semantics. */
  debugEvidence?: 'current' | 'previous' | 'source';
  debugPreviousId?: string;
  debugChanges?: string[];
  /** Display-only literal source assignment card; excluded from graph-file semantics. */
  sourceAssignment?: { target: string; conditional: boolean };
  /** A captured call whose module a literal source assignment replaced since that
   * run. Display-only: `key` is the assignment's span id, `before` the label it replaces. */
  sourceReplacement?: { key: string; target: string; before?: string; capturedType?: string; coverage?: 'complete' | 'partial' };
  /** Exact introspected count, including known zero; never rounded millions. */
  parameterCount?: number;
  /** One complete observed tensor output. Logical footprint, not peak/storage memory. */
  outputTensor?: { shape: number[]; dtype: string; logicalBytes: number };
  isSelected?: boolean;
  // For sub-flows and grouping
  parentId?: string;
  /** Ephemeral focus projection only; never applied to the source graph. */
  focusDirection?: 'LR' | 'TB';
  focusBoundary?: { direction: 'in' | 'out'; externalNodeId: string };
  extent?: 'parent' | [number, number, number, number];
  type?: 'default' | 'input' | 'output' | 'group' | 'module';
  expandable?: boolean;
  expanded?: boolean;
  isGroup?: boolean;
  collapsed?: boolean;
  moduleInstanceId?: string;
  moduleTemplateId?: string;
  moduleEntryNodeId?: string;
  moduleExitNodeId?: string;
  moduleLayerCount?: number;
  /** Trace boundary captured, implementation omitted by the tree depth bound. */
  moduleCapture?: { status: 'depth-limited'; maxDepth: number };
  moduleStatus?: 'ok' | 'warning' | 'error';
  /** Derived from descendants for presentation; recomputed on every reconcile. */
  moduleDiagnosticStatus?: 'warning' | 'error';
  moduleDiagnosticCount?: number;
  moduleDiagnostics?: GraphIssue[];
  moduleLayoutState?: 'deterministic' | 'custom';
  moduleMembers?: string[];
  modulePreviewNodes?: Array<{ id: string; x: number; y: number }>;
  modulePreviewEdges?: Array<{ source: string; target: string }>;
  /** Static multi-file resolution state. It explains why this instance remains
   * opaque without pretending that a runtime trace is required to expand the
   * modules the parser did resolve. */
  resolutionCode?: string;
  resolutionSeverity?: 'info' | 'warning' | 'error';
  resolutionDetail?: string;
  // Repeated-stack metadata (e.g. "Bottleneck ×3", "EncoderLayer ×12").
  // Additive: the module still contains its full materialised internals.
  repeatCount?: number;
  repeatBlockLabel?: string;
  // A child that belongs to a repeated module but is NOT the canonical (first)
  // instance — kept in the graph for codegen/verify but never rendered, so an
  // expanded ×N module shows ONE representative block instead of all N copies.
  repeatHidden?: boolean;
  onToggleExpand?: (moduleInstanceId: string) => void;
  onResetLayout?: (moduleInstanceId: string) => void;
  onSetRepeat?: (moduleInstanceId: string, count: number) => void;
  review?: import('../ide/graphReview').ReviewNode;
  diffStatus?: 'added' | 'removed' | 'modified' | 'rewired';
}

export interface ModelNode extends Node {
  data: ModelNodeData;
}

export interface ModelEdgeData {
  debugPrevious?: boolean;
  captureEdge?: CaptureEdge;
  _relationship?: string;
  sourceHandle?: string;
  targetHandle?: string;
  // Display-only edge synthesized by the reconciler to connect a collapsed
  // module box to its neighbours (the real child-targeted edges are hidden, not
  // rerouted). Never sent to the backend — stripped in buildInferablePayload.
  synthetic?: boolean;
  role?: 'residual' | 'mixed';
  roleEvidence?: 'identity-bypass' | 'projection-shortcut' | 'declared';
  originalEdgeIds?: string[];
  diffStatus?: 'added' | 'removed' | 'modified';
}

export type ModelEdge = Edge<ModelEdgeData>;

export interface ModelGraph {
  nodes: ModelNode[];
  edges: ModelEdge[];
  /**
   * Graph-level semantic metadata, carried through unchanged.
   *
   * Holds `template_instances`, which records which nodes belong to which
   * instantiated block — real structure, not a hint. Dropping it on load left the
   * canvas guessing a 107-node AI build's shape with repeat detection (3 wrong
   * groups instead of the 12 declared blocks), and rebuilding it on save erased it.
   *
   * The backend resolver also reads it, but has a topology fallback, so the loss
   * was the canvas's alone.
   */
  metadata?: Record<string, unknown>;
}

export interface PendingGraphCommit {
  document: GraphDocumentV2;
  author: 'user' | 'agent';
  opSummary?: string;
  /** Stable across retries of the first POST so creation is idempotent. */
  createKey?: string;
}

export interface SavedGraph {
  id: string;
  name: string;
  description?: string;
  /** Authoritative saved representation returned by the project API. */
  document: GraphDocumentV2;
  currentRevision: number;
  /** Derived canvas projection; never written directly to project storage. */
  graph: ModelGraph;
  createdAt: string;
  updatedAt: string;
  /** True only while a revision-tagged local draft awaits a confirmed CAS commit. */
  recoveredDraft?: boolean;
  /** Ordered commit boundaries restored from the crash journal. */
  recoveredCommits?: PendingGraphCommit[];
  /** A pre-revision local draft was preserved but not attached to this authority. */
  legacyRecoveryAvailable?: boolean;
  /** A local draft could not be rebased onto this graph; it was archived, not applied. */
  conflictedDraftAvailable?: boolean;
  thumbnail?: string; // base64 encoded image
  tags?: string[];
}

export interface ModelSummary {
  totalNodes: number;
  totalParameters: number;
  inputShape?: number[];
  outputShape?: number[];
  memoryUsage?: string;
} 
