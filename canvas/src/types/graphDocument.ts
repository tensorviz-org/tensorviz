export type ArchitectureNodeKind = 'input' | 'output' | 'layer' | 'module' | 'op';

export interface ArchitectureNode {
  node_id: string;
  kind: ArchitectureNodeKind;
  op: string;
  params: Record<string, unknown>;
  in_ports: string[];
  out_ports: string[];
  metadata: Record<string, unknown>;
}

export interface ArchitectureEdge {
  edge_id: string;
  source_node_id: string;
  source_port: string;
  target_node_id: string;
  target_port: string;
  metadata: Record<string, unknown>;
}

export interface ArchitectureModule {
  module_id: string;
  name: string;
  input_ports: string[];
  output_ports: string[];
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
  params: Record<string, unknown>;
}

export interface ArchitectureGraph {
  graph_id: string;
  name: string;
  revision: number;
  inputs: Array<Record<string, unknown>>;
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
  modules: ArchitectureModule[];
  operations: Array<Record<string, unknown>>;
  metadata: Record<string, unknown>;
}

export interface GraphLayout {
  /** Presentation axis; omitted on legacy graphs. */
  direction?: 'horizontal' | 'vertical';
  strategy?: string;
  canvas_view_mode?: 'overview' | 'expanded' | 'focus';
  focused_module_id?: string | null;
  expanded_module_ids?: string[];
  custom_module_ids?: string[];
  node_canvas?: Record<string, Record<string, unknown>>;
  node_data?: Record<string, Record<string, unknown>>;
  edge_canvas?: Record<string, Record<string, unknown>>;
  [key: string]: unknown;
}

export interface GraphView {
  graph_id: string;
  node_positions: Record<string, { x: number; y: number }>;
  collapsed_modules: string[];
  selection: string[];
  layout: GraphLayout;
}

export interface GraphDocumentV2 {
  schemaVersion: 2;
  semantic: ArchitectureGraph;
  view: GraphView;
}
