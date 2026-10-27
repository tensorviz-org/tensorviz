import { matchesHostFile } from './workspaceFiles';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';
import { randomUUID } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import * as vscode from 'vscode';
import { parse, modify, applyEdits, type ParseError } from 'jsonc-parser';

export type AgentKind = 'Codex' | 'Claude Code' | 'Cursor';
export interface AgentConfigPlan { file: string; before: string | null; after: string; snippet: string; previousUrl?: string }

export function agentConfigFile(agent: AgentKind, home = os.homedir(), env = process.env): string {
  if (agent === 'Codex') {
    if (env.CODEX_HOME && !path.isAbsolute(env.CODEX_HOME)) throw new Error('Edit this relative CODEX_HOME configuration through Codex settings.');
    return path.join(env.CODEX_HOME || path.join(home, '.codex'), 'config.toml');
  }
  if (agent === 'Claude Code') {
    if (env.CLAUDE_CONFIG_DIR) throw new Error('Edit this custom Claude Code configuration through its own MCP settings.');
    return path.join(home, '.claude.json');
  }
  return path.join(home, '.cursor', 'mcp.json');
}

const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const localEndpoint = (value: unknown): value is string => typeof value === 'string'
  && /^http:\/\/(?:127\.0\.0\.1|localhost):[0-9]{1,5}\/mcp$/.test(value)
  && Number(new URL(value).port) > 0 && Number(new URL(value).port) <= 65535;

/** Add or review a changed local endpoint while preserving unrelated settings and comments. */
export async function prepareAgentConfig(agent: AgentKind, file: string, before: string | null, url: string): Promise<AgentConfigPlan> {
  if (!/^http:\/\/127\.0\.0\.1:\d{1,5}\/mcp$/.test(url)) throw new Error('Expected the local TensorViz MCP endpoint.');
  const text = before ?? (agent === 'Codex' ? '' : '{}\n');
  const entry = agent === 'Claude Code' ? { type: 'http', url } : { url };
  let existing: unknown, after: string, snippet: string;
  if (agent === 'Codex') {
    const { parse: parseToml } = await import('smol-toml');
    const parsed = parseToml(text);
    if (parsed.mcp_servers !== undefined && !record(parsed.mcp_servers)) throw new Error('mcp_servers must be a TOML table.');
    existing = record(parsed.mcp_servers) ? parsed.mcp_servers.tensorviz : undefined;
    snippet = `[mcp_servers.tensorviz]\nurl = ${JSON.stringify(url)}\n`;
    after = existing === undefined ? `${text}${text.endsWith('\n') || !text ? '' : '\n'}\n${snippet}` : text;
    // An inline table cannot be extended; fail without rewriting the user's TOML.
    parseToml(after);
  } else {
    const errors: ParseError[] = [];
    const parsed = parse(text, errors, { allowTrailingComma: agent === 'Cursor', disallowComments: agent !== 'Cursor' });
    if (errors.length || !record(parsed) || parsed.mcpServers !== undefined && !record(parsed.mcpServers)) throw new Error('The agent configuration is not a valid MCP settings object.');
    existing = record(parsed.mcpServers) ? parsed.mcpServers.tensorviz : undefined;
    snippet = JSON.stringify({ mcpServers: { tensorviz: entry } }, null, 2);
    after = existing === undefined ? applyEdits(text, modify(text, ['mcpServers', 'tensorviz'], entry,
      { formattingOptions: { insertSpaces: true, tabSize: 2, eol: text.includes('\r\n') ? '\r\n' : '\n' } })) : text;
  }
  if (existing !== undefined && (!record(existing) || !localEndpoint(existing.url) || existing.command !== undefined
    || existing.enabled === false || agent === 'Claude Code' && !['http', 'streamable-http'].includes(String(existing.type)))) {
    throw new Error('A different or disabled tensorviz server already exists. Edit that entry in your agent settings; TensorViz will not overwrite it.');
  }
  if (record(existing) && existing.url !== url) {
    if (agent === 'Codex') {
      const { parse: parseToml } = await import('smol-toml');
      const expected = parseToml(text);
      (expected.mcp_servers as Record<string, Record<string, unknown>>).tensorviz.url = url;
      // Change only the existing URL scalar. Parsing the candidate proves comments,
      // quoted/inline tables and matching strings elsewhere cannot redirect the edit.
      const candidates: string[] = [];
      for (const quote of ['"', "'"]) {
        const literal = `${quote}${existing.url}${quote}`;
        let index = text.indexOf(literal);
        while (index !== -1) {
          const candidate = text.slice(0, index) + JSON.stringify(url) + text.slice(index + literal.length);
          try { if (isDeepStrictEqual(parseToml(candidate), expected)) candidates.push(candidate); } catch { /* Not the URL scalar. */ }
          index = text.indexOf(literal, index + literal.length);
        }
      }
      if (candidates.length !== 1) throw new Error('Edit this TensorViz URL manually; its TOML spelling cannot be updated safely.');
      after = candidates[0];
    } else {
      after = applyEdits(text, modify(text, ['mcpServers', 'tensorviz', 'url'], url,
        { formattingOptions: { insertSpaces: true, tabSize: 2, eol: text.includes('\r\n') ? '\r\n' : '\n' } }));
    }
    return { file, before, after, snippet, previousUrl: String(existing.url) };
  }
  return { file, before, after, snippet };
}

export async function readAgentConfig(file: string): Promise<string | null> {
  try {
    const stat = await fs.lstat(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 2 * 1024 * 1024) throw new Error('Edit this non-regular or oversized agent configuration manually.');
    return await fs.readFile(file, 'utf8');
  } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null; throw error; }
}

export async function applyAgentConfig(plan: AgentConfigPlan): Promise<void> {
  if (await readAgentConfig(plan.file) !== plan.before) throw new Error('Agent settings changed during review. Run setup again.');
  if (plan.before === plan.after) return;
  await fs.mkdir(path.dirname(plan.file), { recursive: true, mode: 0o700 });
  const temporary = `${plan.file}.tensorviz-${randomUUID()}.tmp`;
  try {
    const mode = plan.before === null ? 0o600 : (await fs.stat(plan.file)).mode & 0o777;
    await fs.writeFile(temporary, plan.after, { flag: 'wx', mode });
    if (await readAgentConfig(plan.file) !== plan.before) throw new Error('Agent settings changed during review. Run setup again.');
    await fs.rename(temporary, plan.file);
  } finally { await fs.rm(temporary, { force: true }); }
}

export async function setUpMcpAgent(url: string, showGuide: (configuredAgent?: AgentKind) => void,
  fileForAgent: (agent: AgentKind) => string = agentConfigFile): Promise<void> {
  const picked = await vscode.window.showQuickPick(['Codex', 'Claude Code', 'Cursor', 'VS Code Copilot / another client'].map(label => ({ label })),
    { title: 'Connect TensorViz to your coding agent', placeHolder: 'Set up once for all your local projects' });
  const agent = picked?.label;
  if (!agent) return;
  if (agent === 'VS Code Copilot / another client') { showGuide(); return; }
  try {
    const file = fileForAgent(agent as AgentKind);
    if (vscode.workspace.textDocuments.some(document => matchesHostFile(document.uri, file) && document.isDirty)) throw new Error('Save or discard pending edits to your agent settings before setup.');
    const plan = await prepareAgentConfig(agent as AgentKind, file, await readAgentConfig(file), url);
    if (plan.before !== plan.after) {
      const action = plan.previousUrl ? 'Update TensorViz' : 'Add TensorViz';
      const approved = await vscode.window.showInformationMessage(plan.previousUrl
        ? `Update TensorViz's connection in ${agent}?` : `Add TensorViz to ${agent} for all projects?`,
        { modal: true, detail: `${file}\n\n${plan.previousUrl ? `Previous endpoint: ${plan.previousUrl}\n\n` : ''}${plan.snippet}\n\nThis connects your agent to the VS Code extension. Keep VS Code open to use its graph tools. If you use the tensorViz plugin, cancel this setup to avoid a duplicate connection. Local processes can also reach the endpoint. Your agent may ask for its own tool approval.` },
        action);
      if (approved !== action) return;
      if (vscode.workspace.textDocuments.some(document => matchesHostFile(document.uri, file) && document.isDirty)) throw new Error('Agent settings now have unsaved edits. Save them and retry.');
    }
    await applyAgentConfig(plan);
    void vscode.window.showInformationMessage(`TensorViz is configured for ${agent}. Reconnect MCP or restart the agent, then ask it to open your model graph. The status bar shows Connected after it connects.`);
    showGuide(agent as AgentKind);
  } catch (error) {
    // Parser messages can include surrounding configuration and credentials.
    const known = error instanceof Error && /^(Agent settings|A different|Save or discard|Edit this|The agent configuration|mcp_servers)/.test(error.message);
    void vscode.window.showErrorMessage(known ? (error as Error).message : 'Could not finish agent setup safely. Check your agent settings and run Connect coding agent (MCP) again.');
  }
}
