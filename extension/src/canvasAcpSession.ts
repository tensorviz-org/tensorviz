import { spawn } from 'node:child_process';
import { Readable, Writable } from 'node:stream';
import { copyFile, mkdtemp, mkdir, rm, symlink } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import * as path from 'node:path';
import type { ClientConnection, NewSessionRequest } from '@agentclientprotocol/sdk' with { 'resolution-mode': 'import' };
import type { AgentRunEvent, AgentRunInput } from './canvasAgent';

export interface CanvasAcpOptions {
  adapter: string;
  executable: string;
  provider: AgentRunInput['provider'];
  settings: AgentRunInput['settings'];
  instructions: string;
  url: string;
  token: string;
  /** Provider routing only, obtained from the native CLI; never a config file or browser input. */
  codexConfig?: Record<string, unknown>;
  signal: AbortSignal;
}

/** Standard ACP sessions behind the existing private, authenticated TensorViz tool boundary. */
export class CanvasAcpSession {
  private emit?: (event: AgentRunEvent) => void;
  private active?: { interrupted: boolean; lastMessageId?: string };
  private constructor(private readonly connection: ClientConnection, private readonly sessionId: string,
    private readonly shutdown: () => Promise<void>) {}

  static async start(options: CanvasAcpOptions): Promise<CanvasAcpSession> {
    const acp = await import('@agentclientprotocol/sdk');
    const directory = await mkdtemp(path.join(tmpdir(), 'tensorviz-acp-'));
    let connection: ClientConnection | undefined;
    let child: ReturnType<typeof spawn> | undefined;
    let ended = false;
    let session: CanvasAcpSession | undefined;
    let stage = 'launch';
    let cleanupPromise: Promise<void> | undefined;
    const cleanup = (): Promise<void> => cleanupPromise ??= (async () => {
      ended = true; connection?.close();
      const pid = child?.pid;
      const kill = (signal: NodeJS.Signals) => {
        if (!pid) return;
        try { process.platform === 'win32' ? child?.kill(signal) : process.kill(-pid, signal); } catch { /* Already exited. */ }
      };
      kill('SIGTERM');
      if (child && child.exitCode === null) {
        let timer: NodeJS.Timeout | undefined;
        await Promise.race([new Promise<void>(resolve => child!.once('close', resolve)),
          new Promise<void>(resolve => { timer = setTimeout(resolve, 1500); })]);
        clearTimeout(timer);
      }
      kill('SIGKILL');
      await rm(directory, { recursive: true, force: true });
    })();
    const abort = () => { void cleanup(); };
    options.signal.addEventListener('abort', abort, { once: true });
    const deadline = setTimeout(abort, 45_000);
    try {
      const env = { ...process.env };
      // Agents need no direct project access: source, setup and proposals are MCP tools.
      // A private cwd also prevents project hooks/config from widening the tool boundary.
      if (options.provider === 'codex') {
        const home = path.join(directory, 'codex'); await mkdir(home);
        const auth = path.join(process.env.CODEX_HOME || path.join(homedir(), '.codex'), 'auth.json');
        try {
          if (process.platform === 'win32') await copyFile(auth, path.join(home, 'auth.json'));
          else await symlink(auth, path.join(home, 'auth.json'));
        } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
        env.CODEX_HOME = home; env.CODEX_PATH = options.executable; env.INITIAL_AGENT_MODE = 'read-only';
        env.TENSORVIZ_CANVAS_AGENT_TOKEN = options.token;
        env.CODEX_CONFIG = JSON.stringify({ ...options.codexConfig, ...(options.settings?.model ? { model: options.settings.model } : {}),
          ...(options.settings?.effort ? { model_reasoning_effort: options.settings.effort } : {}),
          features: { shell_tool: false, hooks: false, multi_agent: false, apps: false }, hooks: {}, plugins: {}, web_search: 'disabled',
          mcp_servers: { tensorviz_canvas: { url: options.url, bearer_token_env_var: 'TENSORVIZ_CANVAS_AGENT_TOKEN',
            default_tools_approval_mode: 'approve' } },
          developer_instructions: options.instructions });
        // ACP gateway overrides are intentionally outside the Canvas account/settings contract.
        delete env.DEFAULT_AUTH_REQUEST; delete env.MODEL_PROVIDER;
      } else env.CLAUDE_CODE_EXECUTABLE = options.executable;
      if (options.signal.aborted || ended) throw new Error('Stopped.');
      child = spawn(options.adapter, [], { cwd: directory, env, shell: false, windowsHide: true,
        detached: process.platform !== 'win32', stdio: ['pipe', 'pipe', 'pipe'] });
      child.stderr!.resume();
      child.on('error', () => connection?.close());
      child.on('exit', () => connection?.close());
      connection = acp.client({ name: 'tensorviz-canvas' })
        .onNotification(acp.methods.client.session.update, ({ params }) => {
          if (params.sessionId !== session?.sessionId) return;
          const update = params.update;
          if (update.sessionUpdate === 'agent_message_chunk' && update.content.type === 'text' && update.content.text && session.active) {
            const active = session.active;
            // ACP chunks share a messageId; a changed ID starts a new message, not a new token.
            if (active.lastMessageId && update.messageId && active.lastMessageId !== update.messageId) session.emit?.({ kind: 'text', text: '\n\n' });
            if (update.messageId) active.lastMessageId = update.messageId;
            session.emit?.({ kind: 'text', text: update.content.text });
          }
        })
        // Only our MCP tools are allowed. They already enforce native review and execution approval.
        .onRequest(acp.methods.client.session.requestPermission, () => ({ outcome: { outcome: 'cancelled' } }))
        .connect(acp.ndJsonStream(Writable.toWeb(child.stdin!) as WritableStream<Uint8Array>,
          Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>, { maxMessageBytes: 2 * 1024 * 1024 }));
      stage = 'initialize';
      const init = await connection.agent.request(acp.methods.agent.initialize, {
        protocolVersion: acp.PROTOCOL_VERSION,
        // Codex's negotiated session-index extension can name this private session
        // without spending a separate model turn generating an unused chat title.
        clientCapabilities: options.provider === 'codex' ? { _meta: { jetbrains: { air: { version: 1, capabilities: ['sessionIndex'] } } } } : {},
        clientInfo: { name: 'tensorviz-canvas', version: '1' },
      });
      if (!init.agentCapabilities?.mcpCapabilities?.http) throw new Error('ACP adapter does not support HTTP MCP.');
      const params: NewSessionRequest = { cwd: directory,
        // codex-acp rebuilds these entries with URL/headers only, dropping the
        // host's tool approval policy. Keep its sole server in CODEX_CONFIG.
        mcpServers: options.provider === 'codex' ? [] :
          [{ type: 'http', name: 'tensorviz_canvas', url: options.url, headers: [{ name: 'Authorization', value: `Bearer ${options.token}` }] }],
        ...(options.provider === 'claude' ? { _meta: { claudeCode: { options: {
          tools: [], allowedTools: ['mcp__tensorviz_canvas__*'], strictMcpConfig: true, settingSources: [],
          persistSession: false, promptSuggestions: false, systemPrompt: options.instructions,
          ...(options.settings?.model ? { model: options.settings.model } : {}),
          ...(options.settings?.effort ? { effort: options.settings.effort } : {}),
        } } } } : {}),
      };
      stage = 'create session';
      const created = await connection.agent.request(acp.methods.agent.session.new, params);
      if (options.provider === 'codex') {
        if (!created.modes?.availableModes.some(mode => mode.id === 'read-only')) throw new Error('ACP adapter has no read-only mode.');
        stage = 'set read-only mode';
        await connection.agent.request(acp.methods.agent.session.setMode, { sessionId: created.sessionId, modeId: 'read-only' });
        const extensions = init._meta?.jetbrains as { air?: { capabilities?: unknown } } | undefined;
        if (Array.isArray(extensions?.air?.capabilities) && extensions.air.capabilities.includes('sessionRename')) {
          const controller = new AbortController();
          let timer: NodeJS.Timeout | undefined;
          try {
            await Promise.race([
              connection.agent.request('_session/rename', { sessionId: created.sessionId, title: 'TensorViz Canvas' }, { cancellationSignal: controller.signal }),
              new Promise<void>(resolve => { timer = setTimeout(() => { controller.abort(); resolve(); }, 1000); }),
            ]);
          } catch { /* Optional metadata: unsupported adapters retain their normal title behavior. */ }
          finally { clearTimeout(timer); }
        }
      }
      session = new CanvasAcpSession(connection, created.sessionId, cleanup);
      if (options.signal.aborted || ended) throw new Error('Stopped.');
      return session;
    } catch (cause) {
      await cleanup();
      throw new Error(`Could not ${stage} with the ACP adapter. Check the configured adapter path, version and agent sign-in, then Refresh.`, { cause });
    } finally { clearTimeout(deadline); options.signal.removeEventListener('abort', abort); }
  }

  async prompt(text: string, emit: (event: AgentRunEvent) => void, signal: AbortSignal): Promise<void> {
    if (this.active) throw new Error('An ACP turn is already running.');
    const active = this.active = { interrupted: false };
    const acp = await import('@agentclientprotocol/sdk');
    this.emit = emit;
    const stop = () => {
      void this.connection.agent.notify(acp.methods.agent.session.cancel, { sessionId: this.sessionId }).catch(() => {});
      void this.shutdown();
    };
    signal.addEventListener('abort', stop, { once: true });
    try {
      if (signal.aborted) { stop(); return; }
      const result = await this.connection.agent.request(acp.methods.agent.session.prompt, {
        sessionId: this.sessionId, prompt: [{ type: 'text', text }],
      });
      if (result.stopReason !== 'end_turn' && !(active.interrupted && result.stopReason === 'cancelled') && !signal.aborted) throw new Error(`Incomplete turn: ${result.stopReason}`);
    } catch (cause) {
      await this.shutdown();
      if (!signal.aborted) throw new Error('The ACP agent could not complete this turn. Refresh its connection and try again.', { cause });
    } finally { signal.removeEventListener('abort', stop); this.emit = undefined; this.active = undefined; }
  }

  /** Only preparation uses this. The caller must await prompt's terminal response before reuse. */
  async interrupt(): Promise<void> {
    const active = this.active;
    if (!active) return;
    active.interrupted = true;
    const acp = await import('@agentclientprotocol/sdk');
    if (this.active === active) await this.connection.agent.notify(acp.methods.agent.session.cancel, { sessionId: this.sessionId });
  }

  dispose(): Promise<void> { return this.shutdown(); }
}
