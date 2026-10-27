import { ModelGraph, ModelNode, ModelEdge } from '../types';
import { API_BASE_URL, localApiHeaders } from '../config/apiBase';

export interface ShapeInferenceRequest {
  id: string;
  name: string;
  description?: string;
  graph: {
    nodes: ModelNode[];
    edges: ModelEdge[];
  };
  createdAt: string;
  updatedAt: string;
  tags: string[];
  input_shape: number[];
}

export interface ShapeInferenceResponse {
  status?: 'valid' | 'error';
  shapes: Record<string, number[]>;
  errors?: Array<{
    nodeId: string;
    message: string;
    error_type: string;
    input_shape?: number[];
    expected_shape?: string;
    layer_type?: string;
    suggestion?: string;
    suggested_repairs?: Array<{
      op: string;
      node_id?: string;
      param?: string;
      value?: any;
      edge_id?: string;
      source_node_id?: string;
      source_port?: string;
      target_node_id?: string;
      target_port?: string;
    }>;
  }>;
}

export interface CodeGenerationRequest {
  graph: ModelGraph;
  input_shape: number[];
  class_name?: string;
}

export interface UnsupportedModule {
  node_id: string;
  class_name: string;
  op: string;
  source_class?: string | null;
  module_path?: string | null;
  imported: boolean;
  reason: string;
}

export interface CodeGenerationResponse {
  code: string;
  summary?: {
    total_parameters: number;
    trainable_parameters: number;
    model_size_mb: number;
  };
  /** Modules whose bodies are not in the graph, so the export cannot reproduce them.
   *  Imported models are full of these: their real implementations live in a library. */
  unsupported?: UnsupportedModule[];
  complete?: boolean;
}

export interface CodeParseRequest {
  code: string;
  model_name?: string;
  input_shape?: number[];
}

export interface CodeParseResponse {
  nodes: any[];
  edges: any[];
  input_shape: number[];
  class_name: string;
  /** Every nn.Module class the file defined, so callers can report what was not imported. */
  available_classes?: string[];
}

export interface SessionHistoryState {
  undo_count: number;
  redo_count: number;
  event_count: number;
}

export interface HistoryEventRecord {
  sequence: number;
  id: string;
  workspace_id: string;
  project_id: string;
  session_id: string;
  user_id: string;
  event_type: string;
  event_version: number;
  payload: Record<string, any>;
  idempotency_key: string;
  created_at: string;
}

export interface HistoryEventsResponse {
  items: HistoryEventRecord[];
  next_cursor?: string | null;
  count: number;
}

// The paper-to-canvas pipeline is not mounted (backend/server_v2.py), and the
// eleven /api/papers/* client methods that drove it had no caller anywhere in src/.
// Removed with their response types; see docs/tensorviz-plan.md §6. The one live
// paper endpoint is template-inventory below, which BuilderPaletteV2 loads on mount
// through a compatibility route server_v2.py serves directly.
export interface PaperTemplateInventorySummary {
  template_total: number;
  template_builtin: number;
  template_dynamic: number;
  mapping_total: number;
  mapping_dynamic: number;
  staging_pending: number;
  staging_approved: number;
  staging_rejected: number;
}

export interface PaperTemplateInventoryTemplate {
  template_id: string;
  name: string;
  source: 'builtin' | 'dynamic' | string;
  required_nodes: number;
  required_edges: number;
  training_minimums: number;
  input_expectation: string;
}

export interface PaperTemplateInventoryResponse {
  summary: PaperTemplateInventorySummary;
  templates: PaperTemplateInventoryTemplate[];
  dynamic_template_ids: string[];
  dynamic_mappings: Record<string, string>;
}

export interface VerifyModelResponse {
  ok: boolean;
  stage: 'shape_inference' | 'build' | 'forward' | 'complete';
  input_shape: number[];
  output_shape: number[] | null;
  shapes: Record<string, any>;
  errors: Array<{ node_id?: string; message: string }>;
  trace?: string | null;
  summary?: {
    layer_count: number;
    total_parameters: number;
    trainable_parameters: number;
    model_size_mb: number;
  } | null;
}

export interface ApiRequestOptions extends RequestInit {
  /** Abort this JSON request if the backend never responds. Omit for no deadline. */
  timeoutMs?: number;
}

export class ApiService {
  private static accessTokenProvider?: () => Promise<string | undefined>;

  static setAccessTokenProvider(provider: () => Promise<string | undefined>): void {
    this.accessTokenProvider = provider;
  }

  /**
   * Host launch headers and caller headers work without cloud authentication.
   * The hosted entry supplies its current access token at request time.
   */
  private static async buildHeaders(
    extra?: Record<string, string>
  ): Promise<Record<string, string>> {
    const adminApiKey = process.env.REACT_APP_ADMIN_API_KEY;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(adminApiKey ? { 'X-Admin-Api-Key': adminApiKey } : {}),
      // The IDE's per-launch secret (task.md D12). Empty in the hosted product,
      // where the middleware that checks it is not registered.
      ...localApiHeaders(),
      ...extra,
    };

    if (this.accessTokenProvider) {
      const token = await this.accessTokenProvider();
      if (token !== undefined) headers['Authorization'] = `Bearer ${token}`;
    }

    return headers;
  }

  private static async makeRequest<T>(
    endpoint: string,
    options: ApiRequestOptions = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;

    const { timeoutMs, ...requestOptions } = options;
    const timeoutController = timeoutMs ? new AbortController() : null;
    let timedOut = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const abortFromCaller = () => timeoutController?.abort();
    if (timeoutController) {
      if (options.signal?.aborted) {
        timeoutController.abort();
      } else {
        options.signal?.addEventListener('abort', abortFromCaller, { once: true });
      }
      timer = setTimeout(() => {
        timedOut = true;
        timeoutController.abort();
      }, timeoutMs);
    }

    const defaultOptions: RequestInit = {
      ...requestOptions,
      ...(timeoutController ? { signal: timeoutController.signal } : {}),
      headers: await this.buildHeaders(requestOptions.headers as Record<string, string> | undefined),
    };

    try {
      const response = await fetch(url, defaultOptions);
      
      if (!response.ok) {
        const raw = await response.text();
        let parsed: any = null;
        try {
          parsed = raw ? JSON.parse(raw) : null;
        } catch {
          parsed = null;
        }
        const detail = parsed?.detail;
        if (detail && typeof detail === 'object') {
          const parts = [`API Error (${response.status})`];
          if (detail.step) parts.push(`[${detail.step}]`);
          if (detail.message) parts.push(detail.message);
          if (detail.hint) parts.push(`Hint: ${detail.hint}`);
          if (typeof detail.duration_ms === 'number') {
            parts.push(`Duration: ${detail.duration_ms}ms`);
          }
          throw new Error(parts.join(' '));
        }
        const detailText =
          typeof detail === 'string'
            ? detail
            : raw || response.statusText || 'Request failed';
        throw new Error(`API Error (${response.status}): ${detailText}`);
      }
      
      return await response.json();
    } catch (error) {
      if (timedOut) {
        throw new Error(`API request timed out after ${timeoutMs}ms`);
      }
      if (options.signal?.aborted) throw error;
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      options.signal?.removeEventListener('abort', abortFromCaller);
    }
  }

  static async inferShapes(request: ShapeInferenceRequest): Promise<ShapeInferenceResponse> {
    return this.makeRequest<ShapeInferenceResponse>('/api/validate_shapes', {
      method: 'POST',
      body: JSON.stringify({
        nodes: request.graph.nodes,
        edges: request.graph.edges,
        input_shape: request.input_shape
      }),
    });
  }

  static async generateCode(request: CodeGenerationRequest): Promise<CodeGenerationResponse> {
    return this.makeRequest<CodeGenerationResponse>('/api/models/generate-code', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Throws when the request itself fails, rather than reporting the graph as
   * invalid. The previous blanket catch turned an unreachable backend, a 401 and
   * a 500 into the same "Failed to validate model" the caller could not tell
   * from a genuinely broken graph — which would send a user to fix a model that
   * was never checked. A resolved promise now means the backend answered.
   */
  static async validateModel(graph: ModelGraph, inputShape: number[]): Promise<{
    isValid: boolean;
    errors: Array<{ nodeId: string; message: string }>;
  }> {
    const response = await this.makeRequest<{
      valid: boolean;
      errors: Array<{ nodeId: string; message: string }>;
    }>('/api/models/validate', {
      method: 'POST',
      body: JSON.stringify({ graph, input_shape: inputShape }),
    });

    const errors = response.errors ?? [];
    return {
      isValid: response.valid ?? errors.length === 0,
      errors,
    };
  }

  static async verifyModel(graph: ModelGraph, inputShape: number[], runForward: boolean = true): Promise<VerifyModelResponse> {
    return this.makeRequest<VerifyModelResponse>('/api/models/verify', {
      method: 'POST',
      body: JSON.stringify({ graph, input_shape: inputShape, run_forward: runForward }),
    });
  }

  /**
   * Instantiate a paper-inventory / canonical template (e.g. 'vit_block') into
   * its real layer subgraph. Returns flat nodes/edges the canvas places so
   * template blocks drop real, validatable layers instead of an opaque shell.
   */
  static async instantiateCanonicalTemplate(
    templateId: string,
    config: Record<string, any> = {}
  ): Promise<{
    nodes: Array<{ id: string; op: string; kind: string; params: Record<string, any> }>;
    edges: Array<{ id: string; source: string; target: string }>;
    entry_node_id: string;
    exit_node_id: string;
  }> {
    return this.makeRequest(`/api/canonical-templates/${encodeURIComponent(templateId)}/instantiate`, {
      method: 'POST',
      body: JSON.stringify({ config }),
    });
  }

  /**
   * Download a zip with model.py / sample_run.py / README / requirements.
   * Returns the blob; caller is responsible for triggering the browser
   * download (we keep this generic so the dialog can pick a filename).
   */
  static async exportPackage(graph: ModelGraph, inputShape: number[], modelName: string): Promise<Blob> {
    const url = `${API_BASE_URL}/api/models/export-package`;
    const res = await fetch(url, {
      method: 'POST',
      headers: await this.buildHeaders(),
      body: JSON.stringify({ graph, input_shape: inputShape, class_name: modelName }),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`Export package failed (${res.status}): ${text}`);
    }
    return await res.blob();
  }

  static async checkBackendHealth(): Promise<boolean> {
    try {
      await this.makeRequest<{ status: string }>('/health');
      return true;
    } catch {
      return false;
    }
  }

  static async parseCode(request: CodeParseRequest): Promise<CodeParseResponse> {
    const response = await this.makeRequest<any>('/api/code/parse', {
      method: 'POST',
      body: JSON.stringify(request),
    });

    // Canonical parser response: { success, graphs: [{ name, graph, ... }], ... }
    if (response?.success === true && Array.isArray(response.graphs) && response.graphs.length > 0) {
      // A file with several nn.Module classes used to import whichever one came
      // first in the source, silently dropping the rest — so moving a helper class
      // above the model changed what you got. Prefer the class no other class
      // instantiates: that is the model, wherever it sits in the file.
      const usable = response.graphs.filter((g: any) => g?.graph);
      const chosen = usable.find((g: any) => g.is_top_level) || usable[0] || response.graphs[0];
      const graph = chosen?.graph;
      // The parser's discovered/inferred shape lives on the input node. Ignoring it
      // in favour of the image default is what made a blank Input Shape on an MLP
      // fail validation with a mat1/mat2 mismatch.
      const inputNode = (graph?.nodes || []).find((n: any) => n?.data?.layerType?.id === 'input');
      const parsedShape = inputNode?.data?.parameters?.shape;
      return {
        nodes: graph?.nodes || [],
        edges: graph?.edges || [],
        input_shape: request.input_shape
          || (Array.isArray(parsedShape) ? parsedShape : undefined)
          || [1, 3, 224, 224],
        class_name: chosen?.name || request.model_name || 'ImportedModel',
        // Everything the file defined, so the caller can say what it did not import.
        available_classes: response.graphs.map((g: any) => g?.name).filter(Boolean),
      };
    }

    if (response?.success === false) {
      const message = response?.error || 'Code parser returned success=false';
      throw new Error(message);
    }

    throw new Error('Unexpected /api/code/parse response format');
  }

  static async getHistoryEvents(payload: {
    workspace_id: string;
    project_id?: string;
    session_id?: string;
    event_type_prefix?: string;
    cursor?: string;
    limit?: number;
  }): Promise<HistoryEventsResponse> {
    const params = new URLSearchParams();
    params.set('workspace_id', payload.workspace_id);
    if (payload.project_id) params.set('project_id', payload.project_id);
    if (payload.session_id) params.set('session_id', payload.session_id);
    if (payload.event_type_prefix) params.set('event_type_prefix', payload.event_type_prefix);
    if (payload.cursor) params.set('cursor', payload.cursor);
    if (typeof payload.limit === 'number') params.set('limit', String(payload.limit));
    return this.makeRequest<HistoryEventsResponse>(`/api/history/events?${params.toString()}`, {
      method: 'GET',
    });
  }

  static async getPaperTemplateInventory(): Promise<PaperTemplateInventoryResponse> {
    return this.makeRequest<PaperTemplateInventoryResponse>('/api/papers/template-inventory', {
      method: 'GET',
    });
  }

  // Generic request method for direct API calls
  static async request<T = any>(endpoint: string, options: ApiRequestOptions = {}): Promise<T> {
    return this.makeRequest<T>(endpoint, options);
  }

  /**
   * Open an authenticated streaming response without consuming its body.
   * Error bodies are intentionally not logged: AI endpoints may carry private
   * prompt or graph context in validation details.
   */
  static async openStream(endpoint: string, options: RequestInit = {}): Promise<Response> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: await this.buildHeaders(options.headers as Record<string, string> | undefined),
    });
    if (!response.ok) {
      throw new Error(`API stream error (${response.status}): ${response.statusText || 'Request failed'}`);
    }
    return response;
  }
} 
