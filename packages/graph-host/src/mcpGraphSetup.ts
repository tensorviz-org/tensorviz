import { randomUUID } from 'node:crypto';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { graphToolResult, GraphToolError } from './mcpGraph';
import { INPUT_CONTRACT } from './graphInputContract';
import { isGraphSetupDraft, type GraphOpeningState } from '@tensorviz/contracts/protocol';
import type { GraphOpeningController, GraphOpeningDependencies } from './graphOpening';

export function registerMcpGraphSetup(server: McpServer, opening: GraphOpeningController, setup: Pick<GraphOpeningDependencies, 'discover'>): void {
  const requireState = (): GraphOpeningState => {
    if (!opening.state) throw new GraphToolError('NO_MODEL', 'Call open_graph with the model file and class to begin setup.');
    return opening.state;
  };
  server.registerTool('get_graph_setup', {
    description: 'Read opening progress and discover constructor, input, configuration and enclosing-model choices from source without importing or executing Python. Examples require explicit selection.',
    inputSchema: {}, annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
  }, () => graphToolResult(async () => {
    const state = requireState();
    if (['saving', 'capturing', 'resolving'].includes(state.phase)) return { opening: state };
    const discovered = await setup.discover(state.target, new AbortController().signal, state.draft);
    if (opening.state !== state) throw new GraphToolError('STALE_TARGET', 'The opening or draft changed. Read graph setup again.');
    const draft = state.draft?.configObject ? state.draft : discovered.restoredDraft ?? state.draft
      ?? discovered.draft ?? discovered.discovery.resolution?.draft;
    return { opening: state, ...discovered, ...(draft ? { draft } : {}) };
  }));
  server.registerTool('suggest_graph_inputs', {
    description: 'Package the pending setup for an LLM: the model file/class, required constructor and input fields with their annotations, defaults, examples, constraints, the unresolved values, the exact draft skeleton to fill, and `inputContract` (the full tensor/generator/dtype grammar). Reads source only; it neither executes Python nor saves files. Fill the draft, then call configure_graph_setup.',
    inputSchema: {}, annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
  }, () => graphToolResult(async () => {
    const state = requireState();
    if (['saving', 'capturing', 'resolving'].includes(state.phase)) return { opening: state };
    const prepared = await setup.discover(state.target, new AbortController().signal, state.draft);
    if (opening.state !== state) throw new GraphToolError('STALE_TARGET', 'The opening or draft changed. Read graph setup again.');
    const { discovery } = prepared;
    const draft = state.draft?.configObject ? state.draft : prepared.restoredDraft ?? state.draft
      ?? prepared.draft ?? discovery.resolution?.draft ?? null;
    return { opening: opening.state,
      model: { file: state.target.file, cls: state.target.cls, module: discovery.model },
      resolution: discovery.resolution ? { status: discovery.resolution.status, unresolved: discovery.resolution.unresolved } : null,
      fields: { init: discovery.init, inputs: discovery.inputs },
      examples: discovery.examples, constraints: discovery.constraints,
      factories: discovery.factories, configs: discovery.configs, draft,
      inputContract: INPUT_CONTRACT,
      ...(prepared.requirements?.length ? { requirements: prepared.requirements } : {}),
      guidance: `Read the ${state.target.cls}.forward (including inherited or imported definitions) to choose each missing input name, shape and dtype. Preserve the returned draft's configured values and profileName; source resolution describes discovery, not a replacement for this selected invocation. Fill missing draft.profile values (init, inputs.args, inputs.kwargs, device, mode) using inputContract for the exact tensor grammar; no tensorViz source is needed. Then call configure_graph_setup with this openingId and sourceDigest. Use examples and constraints as evidence; change configured values only when the requested correction requires it.` };
  }));
  server.registerTool('configure_graph_setup', {
    description: 'Save explicit model setup in tensorviz.toml and continue the opening, which captures the saved call once on its saved device. No shape search. Poll get_graph_setup for progress until the opening is ready, failed or cancelled; awaiting_execution means the user chooses Run in Canvas. Saving alone is not a captured graph. Requires the current opening id and source digest.',
    inputSchema: { openingId: z.string().min(1).max(128), sourceDigest: z.string().regex(/^[a-f0-9]{64}$/),
      draft: z.record(z.string(), z.json()) },
    annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: false },
  }, args => graphToolResult(() => {
    const state = requireState();
    if (state.id !== args.openingId || state.target.sourceDigest !== args.sourceDigest || state.phase !== 'needs_setup') {
      throw new GraphToolError('STALE_TARGET', 'Setup is no longer waiting on this source. Read graph setup again.');
    }
    if (!isGraphSetupDraft(args.draft)) throw new GraphToolError('INVALID_SETUP',
      'Supply a bounded draft whose profile matches the inputContract from suggest_graph_inputs: { profile: { kind: "model", model, inputs: { args, kwargs } } } with tensor descriptions { kind: "tensor", shape, dtype, requires_grad, generator }.');
    void opening.action(state.id, randomUUID(), state.target, { kind: 'submit', draft: args.draft });
    return { opening: opening.state };
  }));
}
