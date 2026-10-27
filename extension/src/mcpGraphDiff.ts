import * as path from 'node:path';
import * as vscode from 'vscode';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { reviewFileChanges } from './editorGraphReview';
import { GraphToolError, graphToolResult } from '@tensorviz/graph-host/mcpGraph';
import type { GraphReviewState, LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import type { ReviewSession } from './reviewSnapshots';
import { isHostFile } from './workspaceFiles';

interface GraphDiffHost {
  target(): LocalBrowserTargetIdentity | undefined;
  busy(): boolean;
  show(): Promise<boolean>;
  review(session: ReviewSession, signal: AbortSignal): Promise<GraphReviewState | undefined>;
  current(): GraphReviewState | undefined;
}

export async function openGraphDiff(args: { file?: string; cls?: string }, host: GraphDiffHost, signal: AbortSignal) {
  const cancelled = () => {
    if (signal.aborted) throw new GraphToolError('CANCELLED', 'Graph diff was cancelled.');
  };
  cancelled();
  if (host.busy()) throw new GraphToolError('BUSY', 'Finish or discard the pending source proposal, or wait for capture to finish, then open the graph diff.');
  const live = host.target();
  if (args.file && !path.isAbsolute(args.file)) throw new GraphToolError('ABSOLUTE_PATH_REQUIRED', 'Supply an absolute Python file path.');
  const uri = args.file ? vscode.Uri.file(args.file) : live ? vscode.Uri.file(path.join(live.workspaceFolder, live.file))
    : vscode.window.activeTextEditor?.document.uri;
  if (!uri || !isHostFile(uri) || !/\.pyw?$/i.test(uri.fsPath)) {
    throw new GraphToolError('MODEL_REQUIRED', 'Supply an absolute Python file, open a model in Canvas, or focus its Python editor. Do not recapture just to compare source changes.');
  }
  if (!vscode.workspace.getWorkspaceFolder(uri) && uri.fsPath !== vscode.window.activeTextEditor?.document.uri.fsPath
    && uri.fsPath !== (live && path.join(live.workspaceFolder, live.file))) {
    throw new GraphToolError('WORKSPACE_REQUIRED', 'Open this model folder or Python file in the intended VS Code window first.');
  }
  let session: ReviewSession;
  try { session = await reviewFileChanges(uri, 'head'); }
  catch {
    throw new GraphToolError('GIT_COMPARISON_UNAVAILABLE', 'Could not read HEAD and the current Python files. This model needs a local Git repository with at least one commit; no GitHub push is required. Check the TensorViz output if Git is unavailable.');
  }
  session.preferredClass = args.cls ?? (live && uri.fsPath === path.join(live.workspaceFolder, live.file) ? live.cls : undefined);
  cancelled();
  if (!await host.show()) throw new GraphToolError('VIEW_UNAVAILABLE', 'TensorViz could not open Canvas. Select a working backend environment in VS Code and retry.');
  cancelled();
  const state = await host.review(session, signal);
  cancelled();
  const current = host.current();
  if (!state || current?.id !== state.id || current.stale) throw new GraphToolError('STALE_REVIEW', 'The source or selected comparison changed while opening. Call open_graph_diff again.');
  const selected = state.models?.find(model => model.id === state.selectedModel);
  return { status: 'opened', reviewId: state.id,
    comparison: { before: state.beforeLabel, after: state.afterLabel, executed: false },
    selectedModel: selected ?? null, models: state.models ?? [], changedFiles: state.changedFiles,
    coverage: { incomplete: state.modelStatus === 'unresolved' || !!state.before.truncated || !!state.after.truncated
        || !!state.before.diagnostics?.length || !!state.after.diagnostics?.length || !!state.notes?.length,
      notes: state.notes ?? [], repositoryNotes: state.repositoryNotes ?? [] },
    detail: selected
      ? 'Graph diff opened in TensorViz Canvas beside the native code diff. Green means added, red removed, amber modified. This compares source changes; retained capture shapes are not a new execution.'
      : 'Comparison opened in TensorViz Canvas, but no affected model was discovered. Check changedFiles and coverage notes before concluding there are no graph changes.' };
}

export function registerMcpGraphDiffTool(server: McpServer, host: GraphDiffHost): void {
  server.registerTool('open_graph_diff', {
    description: 'Show local code changes as a visual graph diff in TensorViz Canvas. Compares Git HEAD with current files, including staged and unsaved edits. Use this when asked to show/highlight graph changes; do not call open_graph to recapture, inspect SVG exports, or use compare_runs for a Git diff. Optional absolute file and class select an affected model; otherwise uses the retained Canvas model or active Python editor. Works without a live capture or GitHub push. Keeps an existing matching capture as context and highlights mapped changes. Opens the native code diff too and returns the selected model, changed files and source coverage notes. Status opened means the comparison is prepared and opened; it does not wait for Canvas rendering confirmation. Reads source only: never executes the model, edits/saves files, installs model dependencies or applies proposals. For an edit not yet made, use propose_edit and wait for native Apply/Discard first.',
    inputSchema: { file: z.string().min(1).max(4096).optional(), cls: z.string().regex(/^[A-Za-z_][A-Za-z0-9_]*$/).max(256).optional() },
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, (args, extra) => graphToolResult(() => openGraphDiff(args, host, extra.signal)));
}
