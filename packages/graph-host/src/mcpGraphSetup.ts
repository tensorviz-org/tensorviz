import { randomUUID } from 'node:crypto';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { graphToolResult, GraphToolError } from './mcpGraph';
import { INPUT_CONTRACT } from './graphInputContract';
import { isGraphSetupDraft, isModelEnvironmentName, type GraphOpeningState, type GraphSetupAction, type CanvasAgentPermission } from '@tensorviz/contracts/protocol';
import { graphSetupWorkflow } from '@tensorviz/contracts/graphSetupWorkflow';
import { waitForGraphOpeningUpdate, type GraphOpeningController, type GraphOpeningDependencies } from './graphOpening';

const busyPhases = ['saving', 'capturing', 'resolving'];

export function registerMcpGraphSetup(server: McpServer, opening: GraphOpeningController, setup: Pick<GraphOpeningDependencies, 'discover'>,
  pendingPermission?: () => CanvasAgentPermission | undefined): void {
  const progress = (state: GraphOpeningState) => ({ opening: state, workflow: graphSetupWorkflow(state),
    ...(pendingPermission ? { permission: pendingPermission() ?? null } : {}) });
  const requireState = (): GraphOpeningState => {
    if (!opening.state) throw new GraphToolError('NO_MODEL', 'Call open_graph with the model file and class to begin setup.');
    return opening.state;
  };
  server.registerTool('get_graph_setup', {
    description: 'Read host-confirmed opening progress and workflow.nextStep, then discover constructor, input, configuration and enclosing-model choices from source without importing or executing Python. Continue the same opening according to workflow.nextStep: environment, inputs, wait, execution_approval, complete or blocked. Python readiness alone is not completion. Canvas agent results also include a pending permission: if it lacks presentation, use present_permission before waiting. Optional waitMs (integer 0–10000, default 0) waits while an opening is busy, returning when its state changes or the timeout expires. Use waitMs:10000 while waiting for a user decision or setup instead of polling rapidly. Examples require explicit selection.',
    inputSchema: { waitMs: z.number().int().min(0).max(10000).optional() }, annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
  }, (args, extra) => graphToolResult(async () => {
    const signal = extra?.signal ?? new AbortController().signal;
    if (signal.aborted) throw new GraphToolError('CANCELLED', 'Graph setup read was cancelled.');
    let state = requireState();
    const permission = pendingPermission?.();
    if (args.waitMs && busyPhases.includes(state.phase) && (!permission || permission.presentation)) {
      try { await waitForGraphOpeningUpdate(opening, state, args.waitMs, signal); }
      catch (error) {
        if (signal.aborted) throw new GraphToolError('CANCELLED', 'Graph setup read was cancelled.');
        throw error;
      }
      state = requireState();
    }
    if (busyPhases.includes(state.phase)) return progress(state);
    const discovered = await setup.discover(state.target, signal, state.draft);
    if (signal.aborted) throw new GraphToolError('CANCELLED', 'Graph setup read was cancelled.');
    if (opening.state !== state) throw new GraphToolError('STALE_TARGET', 'The opening or draft changed. Read graph setup again.');
    const draft = state.draft?.configObject ? state.draft : discovered.restoredDraft ?? state.draft
      ?? discovered.draft ?? discovered.discovery.resolution?.draft;
    return { ...progress(state), ...discovered, ...(draft ? { draft } : {}) };
  }));
  server.registerTool('suggest_graph_inputs', {
    description: 'Package the pending setup for an LLM: the model file/class, required constructor and input fields with their annotations, defaults, examples, constraints, the unresolved values, the exact draft skeleton to fill, and `inputContract` (the full tensor/generator/dtype grammar). Reads source only; it neither executes Python nor saves files. Fill the draft, then call configure_graph_setup.',
    inputSchema: {}, annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
  }, () => graphToolResult(async () => {
    const state = requireState();
    if (busyPhases.includes(state.phase)) return progress(state);
    const prepared = await setup.discover(state.target, new AbortController().signal, state.draft);
    if (opening.state !== state) throw new GraphToolError('STALE_TARGET', 'The opening or draft changed. Read graph setup again.');
    const { discovery } = prepared;
    const draft = state.draft?.configObject ? state.draft : prepared.restoredDraft ?? state.draft
      ?? prepared.draft ?? discovery.resolution?.draft ?? null;
    return { ...progress(state),
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
    if (state.environment && state.environment.reason !== 'ready') throw new GraphToolError('ENVIRONMENT_REQUIRED',
      'Python setup is not ready. Follow workflow.nextStep from get_graph_setup before configuring inputs.');
    if (!isGraphSetupDraft(args.draft)) throw new GraphToolError('INVALID_SETUP',
      'Supply a bounded draft whose profile matches the inputContract from suggest_graph_inputs: { profile: { kind: "model", model, inputs: { args, kwargs } } } with tensor descriptions { kind: "tensor", shape, dtype, requires_grad, generator }.');
    void opening.action(state.id, randomUUID(), state.target, { kind: 'submit', draft: args.draft });
    return progress(opening.state!);
  }));
  server.registerTool('configure_graph_environment', {
    description: 'Prepare Python for the current graph opening without opening an environment picker. Select only a candidateId returned by get_graph_setup, create a separate model environment only when canCreate is true, or request installation of the host-declared missing/compatibility libraries in the saved environment. The host owns approval: an active Canvas agent task uses chat permission cards, and manual/external operations retain their existing review. No arbitrary packages or interpreter paths. Returns requested progress, not installation approval or a completed capture. Poll get_graph_setup with waitMs:10000 before continuing inputs; do not repeat declined or unchanged failed setup.',
    inputSchema: { openingId: z.string().min(1).max(128), sourceDigest: z.string().regex(/^[a-f0-9]{64}$/),
      action: z.enum(['select_environment', 'create_environment', 'install_dependencies']),
      candidateId: z.string().min(1).max(64).optional(), name: z.string().min(1).max(48).optional() },
    annotations: { readOnlyHint: false, destructiveHint: true, openWorldHint: true },
  }, args => graphToolResult(async () => {
    const state = requireState();
    if (state.id !== args.openingId || state.target.sourceDigest !== args.sourceDigest
      || !['needs_setup', 'failed'].includes(state.phase) || !state.environment || state.environment.reason === 'ready') {
      throw new GraphToolError('STALE_TARGET', 'Environment setup is no longer waiting on this source. Read graph setup again.');
    }
    const environment = state.environment;
    let action: GraphSetupAction;
    if (args.action === 'select_environment') {
      if (args.name || !args.candidateId || !environment.candidates.some(candidate => candidate.id === args.candidateId && candidate.status !== 'unavailable')) {
        throw new GraphToolError('INVALID_ENVIRONMENT', 'Select an available candidateId from the current opening; never supply an interpreter path.');
      }
      action = { kind: 'select_environment', candidateId: args.candidateId };
    } else if (args.action === 'create_environment') {
      if (args.candidateId || !environment.canCreate || !args.name || !isModelEnvironmentName(args.name)) {
        throw new GraphToolError('INVALID_ENVIRONMENT', 'This host must support model environment creation. Supply a portable name of 1–48 letters, numbers, hyphens or underscores, starting with a letter/number and avoiding reserved device names.');
      }
      action = { kind: 'create_environment', name: args.name };
    } else {
      if (args.candidateId || args.name || !environment.selectedId
        || !environment.candidates.some(candidate => candidate.id === environment.selectedId && candidate.status !== 'unavailable')
        || !(environment.reason === 'missing_libraries' && environment.requirements?.length)
          && !(environment.reason === 'incompatible' && environment.compatibilityPackages?.length)) {
        throw new GraphToolError('INVALID_ENVIRONMENT', 'Only the host-declared missing or compatibility libraries in the saved environment can be installed. Read graph setup and select Python first.');
      }
      action = { kind: 'install_dependencies' };
    }
    if (state.phase === 'failed' && (args.action !== 'select_environment' || args.candidateId === environment.selectedId
      || ['INSTALL_DECLINED', 'SOURCE_BUILD_DECLINED'].includes(state.failure?.code ?? ''))) {
      throw new GraphToolError('SETUP_BLOCKED', `The previous setup failed or was declined. Do not repeat it; inspect the reported cause and ask for the required next step. ${state.detail}`);
    }
    const discovered = await setup.discover(state.target, new AbortController().signal, state.draft);
    if (opening.state !== state || discovered.discovery.sourceDigest !== args.sourceDigest) {
      throw new GraphToolError('STALE_TARGET', 'The model, source or opening changed while checking Python setup. Read graph setup again.');
    }
    let refused = false, failed = false;
    // The controller checks source/trust and admits the action synchronously before
    // awaiting native review or installation. Keep that work owned by the opening.
    void opening.action(state.id, randomUUID(), state.target, action).then(
      accepted => { refused = !accepted; },
      () => { failed = true; },
    );
    await Promise.resolve();
    if (refused || failed || opening.state?.phase === 'cancelled') {
      throw new GraphToolError('ACTION_REFUSED', 'The opening did not accept this environment action. Read graph setup before continuing.');
    }
    return { requested: true, ...progress(opening.state!),
      nextStep: 'Poll get_graph_setup with waitMs:10000, then follow workflow.nextStep for this same opening. Approval or installation may still be in progress; requested does not mean approved, installed or captured.' };
  }));
}
