import { spawn, type ChildProcess } from 'node:child_process';
import { constants } from 'node:fs';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { randomUUID } from 'node:crypto';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { graphToolResult, GraphToolError } from '@tensorviz/graph-host/mcpGraph';
import type { LocalBrowserTargetIdentity } from '@tensorviz/contracts/protocol';
import type { CanvasAgentPermissionRequester } from './canvasAgent';

export interface CanvasAgentProjectToolContext {
  target: LocalBrowserTargetIdentity;
  signal: AbortSignal;
  approve: CanvasAgentPermissionRequester;
  current(): Promise<boolean>;
}
export interface CanvasAgentProjectPort { capture(): CanvasAgentProjectToolContext; }
export interface CanvasAgentProjectToolService {
  register(server: McpServer): void;
  running(signal?: AbortSignal): boolean;
  unread(signal: AbortSignal): readonly string[];
  wait(signal: AbortSignal, waitMs: number): Promise<void>;
  dispose(): void;
}

type CommandStatus = 'awaiting_approval' | 'running' | 'completed' | 'failed' | 'denied' | 'cancelled' | 'timed_out';
interface ProjectCommand {
  id: string;
  context: CanvasAgentProjectToolContext;
  controller: AbortController;
  executable: string;
  args: string[];
  cwd: string;
  timeoutMs: number;
  status: CommandStatus;
  stdout: Buffer;
  stderr: Buffer;
  stdoutTruncated: boolean;
  stderrTruncated: boolean;
  exitCode: number | null;
  exitSignal: string | null;
  error?: string;
  child?: ChildProcess;
  consumed?: boolean;
}

const MAX_JOBS = 8;
const STREAM_BYTES = 32 * 1024;
const active = (job: ProjectCommand): boolean => job.status === 'awaiting_approval' || job.status === 'running' || !!job.child;
const sameTarget = (a: LocalBrowserTargetIdentity, b: LocalBrowserTargetIdentity): boolean => a.workspaceFolder === b.workspaceFolder
  && a.file === b.file && a.cls === b.cls && a.sourceDigest === b.sourceDigest;
const inside = (root: string, value: string): boolean => {
  const relative = path.relative(root, value);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
};

/** Do not inherit provider/service credentials or developer Python overrides. */
export function projectCommandEnvironment(environment: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const result: NodeJS.ProcessEnv = { PYTHONDONTWRITEBYTECODE: '1', PYTHONNOUSERSITE: '1' };
  for (const name of ['PATH', 'HOME', 'USERPROFILE', 'SystemRoot', 'WINDIR', 'PATHEXT', 'TMPDIR', 'TMP', 'TEMP', 'LANG', 'LC_ALL', 'LC_CTYPE']) {
    if (environment[name] !== undefined) result[name] = environment[name];
  }
  return result;
}

export function redactProjectCommandOutput(value: string, environment: NodeJS.ProcessEnv = process.env): string {
  let result = value;
  for (const [name, secret] of Object.entries(environment)) {
    if (secret && secret.length >= 8 && /(?:TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY|AUTHORIZATION)/i.test(name)) result = result.split(secret).join('[REDACTED]');
  }
  return result.replace(/\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{16,})\b/g, '[REDACTED]')
    .replace(/(Authorization\s*:\s*(?:Bearer|Basic)\s+)\S+/gi, '$1[REDACTED]');
}

function snapshot(job: ProjectCommand) {
  return { jobId: job.id, status: job.status, executable: job.executable, args: [...job.args], cwd: job.cwd, timeoutMs: job.timeoutMs,
    stdout: redactProjectCommandOutput(job.stdout.toString('utf8')), stderr: redactProjectCommandOutput(job.stderr.toString('utf8')),
    stdoutTruncated: job.stdoutTruncated, stderrTruncated: job.stderrTruncated,
    truncated: job.stdoutTruncated || job.stderrTruncated, outputWindow: 'tail',
    exitCode: job.exitCode, exitSignal: job.exitSignal, ...(job.error ? { error: redactProjectCommandOutput(job.error) } : {}) };
}

async function executablePath(value: string): Promise<string> {
  const candidates = path.isAbsolute(value) ? [value] : !value.includes('/') && !value.includes('\\')
    ? (process.env.PATH ?? '').split(path.delimiter).filter(folder => path.isAbsolute(folder)).map(folder => path.join(folder, value)) : [];
  for (const candidate of candidates) {
    try {
      if (!(await fs.stat(candidate)).isFile()) continue;
      await fs.access(candidate, process.platform === 'win32' ? constants.F_OK : constants.X_OK);
      // Preserve a venv's executable symlink: resolving it changes Python's prefix.
      return path.resolve(candidate);
    } catch { /* Try the next explicit PATH candidate, without invoking a shell. */ }
  }
  throw new GraphToolError('EXECUTABLE_UNAVAILABLE', 'Supply an available absolute executable path or a command installed on the host PATH.');
}

async function checked(context: CanvasAgentProjectToolContext, signal = context.signal): Promise<void> {
  signal.throwIfAborted(); context.signal.throwIfAborted();
  if (!await context.current()) throw new GraphToolError('STALE_TARGET', 'The task, model or source changed. Start from the current project before accessing it.');
  signal.throwIfAborted(); context.signal.throwIfAborted();
}

/** A shared host tool path: provider-native shells remain disabled. */
export function createCanvasAgentProjectTools(port: CanvasAgentProjectPort): CanvasAgentProjectToolService {
  const jobs = new Map<string, ProjectCommand>();
  const waiters = new Set<() => void>();
  let disposed = false;
  let requesting = false;
  const notify = () => { for (const wake of [...waiters]) wake(); };
  const capture = (): CanvasAgentProjectToolContext => {
    if (disposed) throw new GraphToolError('CANCELLED', 'The project tools have closed.');
    const context = port.capture();
    return { ...context, target: { ...context.target } };
  };
  const wait = async (signal: AbortSignal, waitMs: number): Promise<void> => {
    signal.throwIfAborted();
    if (!waitMs || disposed || ![...jobs.values()].some(active)) return;
    await new Promise<void>((resolve, reject) => {
      const finish = (error?: unknown) => { clearTimeout(timer); waiters.delete(wake); signal.removeEventListener('abort', abort); error !== undefined ? reject(error) : resolve(); };
      const wake = () => finish();
      const abort = () => finish(signal.reason ?? new Error('Stopped.'));
      const timer = setTimeout(wake, Math.min(10000, Math.max(0, waitMs)));
      waiters.add(wake); signal.addEventListener('abort', abort, { once: true });
      if (signal.aborted) abort();
    });
  };
  const stopChild = (job: ProjectCommand, force = false) => {
    const pid = job.child?.pid;
    if (!pid) return;
    try {
      if (process.platform === 'win32') spawn('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' }).unref();
      else process.kill(-pid, force ? 'SIGKILL' : 'SIGTERM');
    } catch { /* The process group already exited. */ }
  };
  const launch = async (job: ProjectCommand): Promise<void> => {
    const signal = job.controller.signal;
    const abort = () => job.controller.abort();
    job.context.signal.addEventListener('abort', abort, { once: true });
    try {
      if (disposed) job.controller.abort();
      await checked(job.context, signal);
      const approval = job.context.approve({ operation: 'project.command.run', scope: 'machine', effects: ['execute_code'],
        facts: { executable: job.executable, args: [...job.args], cwd: job.cwd, timeoutMs: job.timeoutMs,
          outputLimitBytes: STREAM_BYTES * 2, access: { filesystem: true, network: true }, sandboxed: false } }, signal);
      notify();
      const approved = await approval;
      if (approved !== true) { job.status = signal.aborted ? 'cancelled' : 'denied'; return; }
      await checked(job.context, signal);
      job.status = 'running'; notify();
      await new Promise<void>(resolve => {
        const child = job.child = spawn(job.executable, job.args, { cwd: job.cwd, env: projectCommandEnvironment(),
          shell: false, detached: process.platform !== 'win32', windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
        let ended = false, forceTimer: NodeJS.Timeout | undefined, checking = false;
        const cancel = () => {
          if (ended) return;
          if (job.status !== 'timed_out') job.status = 'cancelled';
          stopChild(job);
          forceTimer ??= setTimeout(() => stopChild(job, true), 1000);
        };
        const timeout = setTimeout(() => { job.status = 'timed_out'; cancel(); }, job.timeoutMs);
        const guard = setInterval(() => {
          if (checking || ended) return;
          checking = true;
          void job.context.current().then(current => { if (!current) job.controller.abort(); }, () => job.controller.abort())
            .finally(() => { checking = false; });
        }, 500);
        const finish = () => {
          if (ended) return;
          ended = true; clearTimeout(timeout); clearTimeout(forceTimer); clearInterval(guard);
          signal.removeEventListener('abort', cancel); job.child = undefined; resolve();
        };
        const append = (stream: 'stdout' | 'stderr', data: Buffer) => {
          const merged = Buffer.concat([job[stream], data]);
          if (merged.length > STREAM_BYTES) { job[`${stream}Truncated`] = true; job[stream] = merged.subarray(merged.length - STREAM_BYTES); }
          else job[stream] = merged;
        };
        child.stdout!.on('data', data => append('stdout', Buffer.from(data)));
        child.stderr!.on('data', data => append('stderr', Buffer.from(data)));
        child.once('error', error => { job.status = 'failed'; job.error = error.message.slice(0, 2048); finish(); });
        child.once('close', (code, exitSignal) => {
          job.exitCode = code; job.exitSignal = exitSignal;
          if (job.status === 'running') job.status = code === 0 ? 'completed' : 'failed';
          finish();
        });
        signal.addEventListener('abort', cancel, { once: true });
        if (signal.aborted) cancel();
      });
    } catch (error) {
      job.status = signal.aborted || job.context.signal.aborted ? 'cancelled' : 'failed';
      if (job.status === 'failed') job.error = (error instanceof Error ? error.message : String(error)).slice(0, 2048);
    } finally {
      job.context.signal.removeEventListener('abort', abort); notify();
    }
  };

  const register = (server: McpServer) => {
    server.registerTool('get_machine_info', {
      description: 'Read local host platform, architecture, OS release, CPU model/count and physical memory. No environment variables, credentials, hostname, network addresses, model import or commands are read or executed. These are machine facts; GPU availability must be checked in the actual selected model Python.',
      inputSchema: {}, annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
    }, (_args, extra) => graphToolResult(async () => {
      const context = capture(); await checked(context, extra.signal);
      const cpus = os.cpus();
      return { platform: os.platform(), architecture: os.arch(), release: os.release(),
        cpu: { model: cpus[0]?.model ?? null, count: cpus.length }, memory: { totalBytes: os.totalmem(), availableBytes: os.freemem() } };
    }));
    server.registerTool('read_project_file', {
      description: 'Read an explicit UTF-8 file in the current project, including README, requirements, setup scripts and Python files that are not graph dependencies. Paths are relative to the project; symlink escapes, credential files and private tool directories are refused. Returns a bounded preview with truncation evidence. Does not import, execute or edit the file.',
      inputSchema: { path: z.string().min(1).max(4096), maxBytes: z.number().int().min(1).max(65536).optional() },
      annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
    }, (args, extra) => graphToolResult(async () => {
      const context = capture(); await checked(context, extra.signal);
      const parts = args.path.split(/[\\/]/);
      const privatePart = (part: string) => /^(?:\.env(?:\..*)?|\.git|\.ssh|\.aws|\.gnupg|\.codex|\.netrc|\.npmrc|\.pypirc|\.tensorviz(?:-private)?|\.?credentials?(?:\..*)?|\.?secrets?(?:\..*)?|(?:client[-_]secret|service[-_]account)(?:\..*)?|id_(?:rsa|dsa|ecdsa|ed25519)(?:\..*)?)$/i.test(part)
        || /\.(?:pem|key|p12|pfx)$/i.test(part);
      if (path.isAbsolute(args.path) || parts.some(part => privatePart(part)) || args.path.includes('\0')) {
        throw new GraphToolError('PROJECT_FILE_REFUSED', 'Choose an explicit project file outside credential files and private tool directories.');
      }
      const root = await fs.realpath(context.target.workspaceFolder), requested = path.resolve(root, args.path);
      if (!inside(root, requested)) throw new GraphToolError('PROJECT_FILE_REFUSED', 'The file must remain inside the current project.');
      const file = await fs.realpath(requested);
      if (!inside(root, file) || path.relative(root, file).split(path.sep).some(privatePart)) throw new GraphToolError('PROJECT_FILE_REFUSED', 'The file resolves outside readable project files.');
      const handle = await fs.open(file, 'r');
      try {
        const stat = await handle.stat();
        if (!stat.isFile()) throw new GraphToolError('PROJECT_FILE_REFUSED', 'Choose a regular project text file.');
        const buffer = Buffer.alloc(Math.min(args.maxBytes ?? 32768, stat.size));
        const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
        const bytes = buffer.subarray(0, bytesRead);
        if (bytes.includes(0)) throw new GraphToolError('PROJECT_FILE_REFUSED', 'This project file contains binary data.');
        let content: string;
        try { content = new TextDecoder('utf-8', { fatal: true }).decode(bytes, { stream: bytesRead < stat.size }); }
        catch { throw new GraphToolError('PROJECT_FILE_REFUSED', 'Choose a UTF-8 project text file.'); }
        await checked(context, extra.signal);
        const safeContent = redactProjectCommandOutput(content);
        return { path: path.relative(root, file), content: safeContent, redacted: safeContent !== content, bytesRead, totalBytes: stat.size, truncated: bytesRead < stat.size };
      } finally { await handle.close(); }
    }));
    server.registerTool('request_project_command', {
      description: 'Request a local project command with an exact executable and argument array. Returns a job immediately; execution waits for generated permission presentation and user approval in Canvas. Commands have machine file/network access, no security sandbox, and always require explicit approval under current setup modes. Use only to investigate or perform the user-requested project work; prefer read-only machine/project inspection tools for facts. shell:false is enforced; shell syntax works only when explicitly requesting a shell executable with its script in args, all visible for review. Stdin is closed; use noninteractive commands. Never invent successful output or request another job while one is active. Poll get_project_command after presentation/approval.',
      inputSchema: { executable: z.string().min(1).max(4096), args: z.array(z.string().max(4096)).max(128),
        cwd: z.string().min(1).max(4096).optional(), timeoutMs: z.number().int().min(1).max(120000).optional() },
      annotations: { readOnlyHint: false, destructiveHint: true, openWorldHint: true },
    }, (args, extra) => graphToolResult(async () => {
      const context = capture(); await checked(context, extra.signal);
      if (requesting || [...jobs.values()].some(active)) throw new GraphToolError('COMMAND_BUSY', 'A project command is already waiting or running. Read its result before requesting another.');
      if ([args.executable, ...args.args, args.cwd ?? ''].some(value => value.includes('\0'))
        || Buffer.byteLength(JSON.stringify([args.executable, args.args, args.cwd ?? '.'])) > 8192) throw new GraphToolError('INVALID_COMMAND', 'Supply a bounded executable, argument array and project directory without NUL characters.');
      requesting = true;
      try {
        const root = await fs.realpath(context.target.workspaceFolder);
        if (args.cwd && path.isAbsolute(args.cwd)) throw new GraphToolError('INVALID_COMMAND', 'Use a project-relative working directory.');
        const cwd = await fs.realpath(path.resolve(root, args.cwd ?? '.'));
        if (!inside(root, cwd) || !(await fs.stat(cwd)).isDirectory()) throw new GraphToolError('INVALID_COMMAND', 'The working directory must remain inside the current project.');
        const executable = await executablePath(args.executable);
        await checked(context, extra.signal);
        if (disposed) throw new GraphToolError('CANCELLED', 'The project tools have closed.');
        while (jobs.size >= MAX_JOBS) jobs.delete(jobs.keys().next().value!);
        const job: ProjectCommand = { id: randomUUID(), context, controller: new AbortController(), executable, args: [...args.args], cwd,
          timeoutMs: args.timeoutMs ?? 30000, status: 'awaiting_approval', stdout: Buffer.alloc(0), stderr: Buffer.alloc(0),
          stdoutTruncated: false, stderrTruncated: false, exitCode: null, exitSignal: null };
        jobs.set(job.id, job); notify(); void launch(job);
        return { ...snapshot(job), requested: true, nextStep: 'Read get_pending_permission and present its explanation when needed; wait for the user decision, then poll get_project_command for this job. Requested does not mean approved or executed.' };
      } finally { requesting = false; }
    }));
    server.registerTool('get_project_command', {
      description: 'Read a project command owned by this same active agent task. Reports actual status, exit code, bounded stdout/stderr tails and truncation; a pending request is not execution. Optional waitMs (0–10000) waits for a status change or timeout. If permission needs presentation, use get_pending_permission and present_permission before waiting. This tool cannot approve, restart or change a command.',
      inputSchema: { jobId: z.string().min(1).max(128), waitMs: z.number().int().min(0).max(10000).optional() },
      annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
    }, (args, extra) => graphToolResult(async () => {
      const context = capture(); await checked(context, extra.signal);
      const job = jobs.get(args.jobId);
      if (!job || job.context.signal !== context.signal || !sameTarget(job.context.target, context.target)) throw new GraphToolError('STALE_COMMAND', 'This command does not belong to the current agent task.');
      if (args.waitMs && active(job)) await wait(extra.signal, args.waitMs);
      await checked(context, extra.signal);
      const result = snapshot(job);
      if (!active(job)) job.consumed = true;
      return result;
    }));
  };
  return { register, running: signal => [...jobs.values()].some(job => active(job) && (!signal || job.context.signal === signal)),
    unread: signal => [...jobs.values()].filter(job => job.context.signal === signal && !active(job) && !job.consumed).map(job => job.id), wait,
    dispose: () => { disposed = true; for (const job of jobs.values()) if (active(job)) job.controller.abort(); notify(); } };
}
