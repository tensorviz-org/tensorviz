/** send_feedback: lets the member tell the tensorViz team about a problem or an idea by
 * asking their coding agent. The member sees the exact report and approves it before it
 * is sent: in the client's own confirmation form when it has one, otherwise by telling the
 * agent to go ahead after it shows them the preview. See feedback.ts. */
import { randomUUID } from 'node:crypto';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { GraphToolError, graphToolResult, object, type McpModel } from './mcpGraph';
import { FEEDBACK_KINDS, FEEDBACK_LIMITS, type FeedbackGraph, type FeedbackInput, type FeedbackReport } from './feedback';
import { WebsiteError } from './websiteAccount';

export interface FeedbackToolHost {
  /** The open model's class name and block count, when a graph is open. */
  graph(): FeedbackGraph | undefined;
  /** Builds the report with this host's version, operating system and folders to hide. */
  report(input: FeedbackInput): FeedbackReport;
  /** Sends it to the tensorViz team and returns the id the team's endpoint gave it. */
  send(report: FeedbackReport): Promise<string>;
}

/** A host's open model as a report names it: its class and how many blocks it has. */
export function feedbackGraph(model: McpModel | undefined): FeedbackGraph | undefined {
  if (!model) return undefined;
  const nodes = object(object(model.projection.document).semantic).nodes;
  return { model: model.target.cls, nodes: Array.isArray(nodes) ? nodes.length : 0 };
}

const PENDING_MS = 15 * 60_000;
const ELICITATION_MS = 300_000;

/** The report as the member reads it before approving. */
export function feedbackPreview(report: FeedbackReport): string {
  const client = [`tensorViz ${report.client.version}`, report.client.editor, report.client.os].filter(Boolean).join(' · ');
  return [`Send this feedback to the tensorViz team?`, '', `Kind: ${report.kind}`, `Message: ${report.message}`,
    ...(report.graph ? [`Graph: ${report.graph.model}, ${report.graph.nodes} blocks`] : []), `Also sent: ${client}`].join('\n');
}

export function registerMcpFeedbackTool(server: McpServer, host: FeedbackToolHost, now: () => number = Date.now): void {
  let pending: { id: string; report: FeedbackReport; expires: number } | undefined;
  const send = async (report: FeedbackReport) => {
    try {
      const id = await host.send(report);
      return { status: 'sent', feedbackId: id, detail: 'Sent to the tensorViz team.' };
    } catch (error) {
      throw new GraphToolError(error instanceof WebsiteError ? `FEEDBACK_${error.code.toUpperCase().replace(/-/g, '_')}` : 'FEEDBACK_UNAVAILABLE',
        error instanceof WebsiteError ? error.message : 'The feedback could not be sent.');
    }
  };

  server.registerTool('send_feedback', {
    description: 'Send the user\'s feedback about tensorViz (a bug, an idea, anything else) to the tensorViz team. Use only when the user asks to send or report feedback; never on your own initiative and never to suggest it. Write the message in the user\'s words, adding what you observed (tool errors, what they expected). Local paths are replaced and no source code is sent. includeGraph adds only the open model\'s class name and block count. The user always approves the exact report first: if the result is needs_confirmation, show them the preview verbatim and call again with only confirmationId after they explicitly agree.',
    inputSchema: {
      message: z.string().min(1).max(FEEDBACK_LIMITS.message).optional(),
      kind: z.enum(FEEDBACK_KINDS).default('bug'),
      includeGraph: z.boolean().default(false).describe('Add the open model\'s class name and block count.'),
      confirmationId: z.string().uuid().optional().describe('From a needs_confirmation result, after the user agreed to send that exact report.'),
    },
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true },
  }, (args, extra) => graphToolResult(async () => {
    if (args.confirmationId) {
      const confirmed = pending;
      if (!confirmed || confirmed.id !== args.confirmationId || confirmed.expires < now()) {
        throw new GraphToolError('UNKNOWN_CONFIRMATION', 'This feedback preview expired or was replaced. Prepare it again.');
      }
      pending = undefined;
      return send(confirmed.report);
    }
    if (!args.message) throw new GraphToolError('INVALID_ARGUMENTS', 'Write the feedback message.');
    let report: FeedbackReport;
    try { report = host.report({ kind: args.kind, message: args.message, graph: args.includeGraph ? host.graph() : undefined }); }
    catch (error) { throw new GraphToolError('INVALID_FEEDBACK', error instanceof Error ? error.message : 'This feedback cannot be sent.'); }
    const preview = feedbackPreview(report);
    const lowLevel = (server as Partial<McpServer>).server;
    if (lowLevel?.getClientCapabilities()?.elicitation?.form) {
      const answer = await lowLevel.elicitInput({ mode: 'form', message: preview, requestedSchema: { type: 'object',
        properties: { send: { type: 'boolean', title: 'Send this feedback', default: false } }, required: ['send'] } },
      { signal: extra.signal, timeout: ELICITATION_MS });
      if (answer.action !== 'accept' || answer.content?.send !== true) return { status: 'not_sent', detail: 'The user chose not to send this feedback.' };
      pending = undefined;
      return send(report);
    }
    pending = { id: randomUUID(), report, expires: now() + PENDING_MS };
    return { status: 'needs_confirmation', confirmationId: pending.id, preview, report,
      detail: 'Nothing was sent. Show the user this preview and send it only if they explicitly agree.' };
  }));
}
