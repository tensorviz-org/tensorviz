import * as assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { z } from 'zod';
import { createCanvasAgentProjectTools, projectCommandEnvironment, redactProjectCommandOutput,
  type CanvasAgentProjectToolContext, type CanvasAgentProjectToolService } from '../../canvasAgentProjectTools';
import type { CanvasAgentPermissionRequest } from '@tensorviz/contracts/protocol';

const tick = () => new Promise<void>(resolve => setImmediate(resolve));
const until = async (condition: () => boolean) => {
  const deadline = Date.now() + 5000;
  while (!condition()) { if (Date.now() > deadline) throw new Error('Fixture did not settle.'); await tick(); }
};

suite('Canvas agent project diagnostics and reviewed commands', () => {
  let directory: string;
  const services: CanvasAgentProjectToolService[] = [];
  setup(async () => { directory = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-project-tools-')); });
  teardown(async () => { const closing = services.splice(0); for (const service of closing) service.dispose(); await until(() => closing.every(service => !service.running())); await fs.rm(directory, { recursive: true, force: true }); });

  function rig(automatic?: boolean) {
    const controller = new AbortController();
    const requests: CanvasAgentPermissionRequest[] = [];
    const decisions: Array<(decision: boolean) => void> = [];
    let current = true, absent = false;
    let context: CanvasAgentProjectToolContext = { target: { workspaceFolder: directory, file: 'model.py', cls: 'Net', sourceDigest: 'a'.repeat(64), documentVersion: 1, backendRevision: 0 },
      signal: controller.signal, current: async () => current,
      approve: async (request, signal) => {
        requests.push(request);
        if (automatic !== undefined) return automatic;
        return new Promise<boolean>(resolve => {
          const abort = () => resolve(false);
          signal.addEventListener('abort', abort, { once: true });
          decisions.push(decision => { signal.removeEventListener('abort', abort); resolve(decision); });
          if (signal.aborted) abort();
        });
      } };
    const service = createCanvasAgentProjectTools({ capture: () => {
      if (absent) throw new Error('No agent task is active.');
      return context;
    } });
    services.push(service);
    const register = () => {
      const tools: Record<string, { config: any; handler: (args: any, extra: { signal: AbortSignal }) => Promise<any> }> = {};
      service.register({ registerTool: (name: string, config: any, handler: any) => { tools[name] = { config, handler }; } } as never);
      return { tools, call: (name: string, args: unknown, signal = new AbortController().signal) => {
        const tool = tools[name]; return tool.handler(z.object(tool.config.inputSchema).strict().parse(args), { signal });
      } };
    };
    const registered = register();
    const idle = async () => { while (service.running()) await service.wait(new AbortController().signal, 1000); };
    return { ...registered, register, service, controller, requests, decisions, idle,
      setCurrent(value: boolean) { current = value; }, setAbsent(value: boolean) { absent = value; },
      set context(value: CanvasAgentProjectToolContext) { context = value; }, get context() { return context; },
      command: async (script: string, options: Record<string, unknown> = {}) => {
        const result = await registered.call('request_project_command', { executable: process.execPath, args: ['-e', script], ...options });
        assert.equal(result.isError, undefined, JSON.stringify(result.structuredContent)); return result.structuredContent;
      },
      result: async (jobId: string) => (await registered.call('get_project_command', { jobId })).structuredContent,
    };
  }

  test('read-only facts and command requests have distinct tool authority', () => {
    const r = rig();
    assert.deepEqual(Object.keys(r.tools).sort(), ['get_machine_info', 'get_project_command', 'read_project_file', 'request_project_command']);
    assert.equal(r.tools.get_machine_info.config.annotations.readOnlyHint, true);
    assert.equal(r.tools.read_project_file.config.annotations.readOnlyHint, true);
    assert.equal(r.tools.get_project_command.config.annotations.readOnlyHint, true);
    assert.equal(r.tools.request_project_command.config.annotations.readOnlyHint, false);
  });

  test('machine facts omit identifiers, credentials and unobserved GPU claims', async () => {
    const r = rig(), result = await r.call('get_machine_info', {});
    assert.equal(result.isError, undefined);
    assert.deepEqual(Object.keys(result.structuredContent).sort(), ['architecture', 'cpu', 'memory', 'platform', 'release']);
    assert.equal(result.structuredContent.platform, process.platform);
    assert.equal(result.structuredContent.architecture, os.arch());
    assert.ok(result.structuredContent.cpu.count > 0); assert.ok(result.structuredContent.memory.totalBytes > 0);
    assert.equal(r.requests.length, 0);
  });

  test('project files include README and setup scripts without importing or executing them', async () => {
    const r = rig();
    await fs.writeFile(path.join(directory, 'README.md'), 'Needs CUDA kernels and model assets.');
    await fs.mkdir(path.join(directory, 'scripts')); await fs.writeFile(path.join(directory, 'scripts', 'env.sh'), 'exit 77\n');
    for (const file of ['README.md', 'scripts/env.sh']) {
      const result = await r.call('read_project_file', { path: file });
      assert.equal(result.isError, undefined); assert.equal(result.structuredContent.path, file); assert.equal(result.structuredContent.truncated, false);
    }
    assert.equal(r.requests.length, 0);
  });

  test('text reads have a bounded UTF-8 preview and report truncation', async () => {
    const r = rig(); await fs.writeFile(path.join(directory, 'README.md'), 'x'.repeat(70000));
    const result = await r.call('read_project_file', { path: 'README.md', maxBytes: 65536 });
    assert.equal(result.structuredContent.content.length, 65536); assert.equal(result.structuredContent.bytesRead, 65536);
    assert.equal(result.structuredContent.totalBytes, 70000); assert.equal(result.structuredContent.truncated, true);
    await fs.writeFile(path.join(directory, 'unicode.txt'), 'éé');
    const unicode = await r.call('read_project_file', { path: 'unicode.txt', maxBytes: 3 });
    assert.equal(unicode.isError, undefined); assert.equal(unicode.structuredContent.content, 'é');
  });

  test('readable files redact recognizable credentials and disclose that the preview changed', async () => {
    const r = rig(); await fs.writeFile(path.join(directory, 'README.md'), 'Authorization: Bearer fixture-secret-value\nsk-proj-123456789012345678901234');
    const result = await r.call('read_project_file', { path: 'README.md' });
    assert.equal(result.isError, undefined); assert.equal(result.structuredContent.redacted, true);
    assert.equal(result.structuredContent.content, 'Authorization: Bearer [REDACTED]\n[REDACTED]');
  });

  test('reads refuse escapes, credentials, binary data and directories', async () => {
    const r = rig();
    await fs.writeFile(path.join(directory, '.env'), 'FIXTURE_SECRET=private');
    await fs.writeFile(path.join(directory, 'service-account.json'), '{}');
    await fs.writeFile(path.join(directory, 'private.key'), 'fixture');
    await fs.writeFile(path.join(directory, 'binary.bin'), Buffer.from([0, 1, 2]));
    await fs.symlink(path.join(directory, '.env'), path.join(directory, 'public.txt'));
    await fs.symlink(os.tmpdir(), path.join(directory, 'outside'));
    for (const file of ['../escape.txt', directory, '.env', 'service-account.json', 'private.key', 'binary.bin', 'public.txt', 'outside', '.']) {
      const result = await r.call('read_project_file', { path: file }); assert.equal(result.isError, true, file);
    }
  });

  test('closed, absent and stale contexts cannot read machine or project facts', async () => {
    const r = rig(); r.setAbsent(true);
    assert.equal((await r.call('get_machine_info', {})).isError, true);
    r.setAbsent(false); r.setCurrent(false);
    assert.equal((await r.call('get_machine_info', {})).isError, true);
    r.setCurrent(true); r.service.dispose(); assert.equal((await r.call('get_machine_info', {})).isError, true);
  });

  test('a command returns immediately and cannot execute before exact machine-scope approval', async () => {
    const r = rig(), marker = path.join(directory, 'ran.txt');
    const job = await r.command(`require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'ran')`);
    assert.equal(job.requested, true); assert.equal(job.status, 'awaiting_approval');
    await until(() => r.requests.length === 1);
    await assert.rejects(fs.access(marker));
    const request = r.requests[0]; assert.equal(request.operation, 'project.command.run'); assert.equal(request.scope, 'machine');
    assert.deepEqual(request.effects, ['execute_code']); assert.deepEqual(request.facts.args, ['-e', `require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'ran')`]);
    assert.equal(request.facts.cwd, await fs.realpath(directory)); assert.equal(request.facts.timeoutMs, 30000);
    assert.equal(request.facts.sandboxed, false); assert.deepEqual(request.facts.access, { filesystem: true, network: true });
    r.decisions[0](true); await r.idle();
    assert.equal(await fs.readFile(marker, 'utf8'), 'ran'); assert.equal((await r.result(job.jobId)).status, 'completed');
  });

  test('denial produces an unread factual result and never starts the executable', async () => {
    const r = rig(), marker = path.join(directory, 'denied.txt');
    const job = await r.command(`require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'bad')`);
    await until(() => r.requests.length === 1); r.decisions[0](false); await r.idle();
    await assert.rejects(fs.access(marker)); assert.deepEqual(r.service.unread(r.controller.signal), [job.jobId]);
    assert.equal((await r.result(job.jobId)).status, 'denied'); assert.deepEqual(r.service.unread(r.controller.signal), []);
  });

  test('argument arrays are literal and workspace-relative cwd is resolved before review', async () => {
    const r = rig(true); await fs.mkdir(path.join(directory, 'nested'));
    const literal = '$(touch unexpected); * && echo no';
    const job = await r.command('process.stdout.write(JSON.stringify({args:process.argv.slice(1),cwd:process.cwd()}))', { args: ['-e', 'process.stdout.write(JSON.stringify({args:process.argv.slice(1),cwd:process.cwd()}))', literal], cwd: 'nested' });
    await r.idle(); const result = await r.result(job.jobId);
    assert.deepEqual(JSON.parse(result.stdout), { args: [literal], cwd: await fs.realpath(path.join(directory, 'nested')) });
    assert.equal(result.exitCode, 0); assert.equal(result.truncated, false);
  });

  test('the displayed request and returned snapshot cannot mutate the actual argv', async () => {
    const r = rig(); const job = await r.command('process.stdout.write("original")');
    await until(() => r.requests.length === 1);
    (r.requests[0].facts.args as string[])[1] = 'process.stdout.write("changed")'; job.args[1] = 'process.stdout.write("also changed")';
    r.decisions[0](true); await r.idle(); assert.equal((await r.result(job.jobId)).stdout, 'original');
  });

  test('a context change during review prevents launch after a late approval', async () => {
    const r = rig(), marker = path.join(directory, 'late.txt');
    const job = await r.command(`require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'must not run')`);
    await until(() => r.requests.length === 1); r.setCurrent(false); r.decisions[0](true); await r.idle();
    await assert.rejects(fs.access(marker)); r.setCurrent(true);
    const result = await r.result(job.jobId); assert.equal(result.status, 'failed'); assert.match(result.error, /changed/);
  });

  test('an explicitly reviewed shell executable can run its exact script without native agent shell access', async function () {
    if (process.platform === 'win32') this.skip();
    try { await fs.access('/bin/bash'); } catch { this.skip(); }
    const r = rig(); const args = ['-c', 'printf "%s" "$1"', 'fixture', 'literal $(must not execute)'];
    const result = await r.call('request_project_command', { executable: '/bin/bash', args });
    assert.equal(result.isError, undefined); await until(() => r.requests.length === 1);
    assert.equal(r.requests[0].facts.executable, '/bin/bash'); assert.deepEqual(r.requests[0].facts.args, args);
    r.decisions[0](true); await r.idle(); assert.equal((await r.result(result.structuredContent.jobId)).stdout, 'literal $(must not execute)');
  });

  test('command requests reject arbitrary cwd, unavailable executables, oversized argv and concurrent requests', async () => {
    const r = rig();
    for (const args of [
      { executable: process.execPath, args: [], cwd: '..' }, { executable: process.execPath, args: [], cwd: directory },
      { executable: '/tensorviz-missing-executable', args: [] }, { executable: process.execPath, args: ['bad\0arg'] },
      { executable: process.execPath, args: ['x'.repeat(4000), 'x'.repeat(4000), 'x'.repeat(4000)] },
    ]) assert.equal((await r.call('request_project_command', args)).isError, true);
    const results = await Promise.all([r.call('request_project_command', { executable: process.execPath, args: ['-e', ''] }),
      r.call('request_project_command', { executable: process.execPath, args: ['-e', ''] })]);
    assert.equal(results.filter(result => result.isError).length, 1);
    await until(() => r.requests.length === 1); r.decisions[0](false); await r.idle();
  });

  test('completed commands retain bounded stdout/stderr tails and real failure exit codes', async () => {
    const r = rig(true), job = await r.command('process.stdout.write("x".repeat(70000)+"OUT_END");process.stderr.write("y".repeat(70000)+"ERR_END");process.exitCode=7');
    await r.idle(); const result = await r.result(job.jobId);
    assert.equal(result.status, 'failed'); assert.equal(result.exitCode, 7);
    assert.ok(Buffer.byteLength(result.stdout) <= 32768); assert.ok(Buffer.byteLength(result.stderr) <= 32768);
    assert.ok(result.stdout.endsWith('OUT_END')); assert.ok(result.stderr.endsWith('ERR_END'));
    assert.equal(result.stdoutTruncated, true); assert.equal(result.stderrTruncated, true); assert.equal(result.outputWindow, 'tail');
  });

  test('timeout ends the process and exposes a terminal timeout rather than successful execution', async () => {
    const r = rig(true), job = await r.command('setInterval(()=>{},1000)', { timeoutMs: 50 });
    await r.idle(); const result = await r.result(job.jobId); assert.equal(result.status, 'timed_out');
    assert.notEqual(result.exitCode, 0); assert.equal(r.service.running(), false);
  });

  test('Stop and disposal cancel pending operations without a native fallback', async () => {
    for (const action of ['stop', 'dispose'] as const) {
      const r = rig(); await r.command('process.stdout.write("must not run")'); await until(() => r.requests.length === 1);
      if (action === 'stop') r.controller.abort(); else r.service.dispose();
      await r.idle(); assert.equal(r.service.running(), false);
      assert.equal((await r.call('get_machine_info', {})).isError, true);
    }
  });

  test('a source/context change while executing cancels the running process', async () => {
    const r = rig(true), job = await r.command('setInterval(()=>{},1000)');
    await until(() => r.service.running() && r.requests.length === 1);
    r.setCurrent(false); await r.idle(); r.setCurrent(true);
    assert.equal((await r.result(job.jobId)).status, 'cancelled');
  });

  test('Stop terminates a process group, including descendants that keep pipes open', async function () {
    if (process.platform === 'win32') this.skip();
    const r = rig(true), marker = path.join(directory, 'group.json');
    const childSource = 'process.on("SIGTERM",()=>{});setInterval(()=>{},1000)';
    await r.command(`const c=require('node:child_process').spawn(process.execPath,['-e',${JSON.stringify(childSource)}],{stdio:'inherit'});require('node:fs').writeFileSync(${JSON.stringify(marker)},JSON.stringify({parent:process.pid,child:c.pid}));process.on('SIGTERM',()=>{});setInterval(()=>{},1000)`);
    let group: { parent: number; child: number } | undefined;
    const deadline = Date.now() + 5000;
    while (!group) {
      try { group = JSON.parse(await fs.readFile(marker, 'utf8')); } catch { if (Date.now() > deadline) throw new Error('Fixture group did not start.'); await tick(); }
    }
    r.controller.abort(); await r.idle();
    const exists = (pid: number) => { try { process.kill(pid, 0); return true; } catch { return false; } };
    await until(() => !exists(group!.parent) && !exists(group!.child));
  });

  test('another turn cannot retrieve command output or inherit unread results', async () => {
    const r = rig(true), job = await r.command('process.stdout.write("private fixture output")'); await r.idle();
    const other = new AbortController(); r.context = { ...r.context, signal: other.signal };
    assert.equal(r.service.running(other.signal), false); assert.deepEqual(r.service.unread(other.signal), []);
    assert.equal((await r.call('get_project_command', { jobId: job.jobId })).isError, true);
  });

  test('re-registering MCP sessions shares jobs in the owning service', async () => {
    const r = rig(true), job = await r.command('process.stdout.write("same owner")'); await r.idle();
    const next = r.register(); const result = await next.call('get_project_command', { jobId: job.jobId });
    assert.equal(result.structuredContent.stdout, 'same owner'); assert.equal(r.service.unread(r.controller.signal).length, 0);
  });

  test('history remains bounded and an evicted result cannot be retrieved', async () => {
    const r = rig(true); const ids: string[] = [];
    for (let i = 0; i < 10; i++) { const job = await r.command(''); ids.push(job.jobId); await r.idle(); }
    assert.equal(r.service.unread(r.controller.signal).length, 8);
    assert.equal((await r.call('get_project_command', { jobId: ids[0] })).isError, true);
    assert.equal((await r.result(ids.at(-1)!)).status, 'completed');
  });

  test('host waits are bounded and reject active cancellation', async () => {
    const r = rig(); await r.command(''); await until(() => r.requests.length === 1);
    const waiter = new AbortController(), waiting = r.service.wait(waiter.signal, 10000); waiter.abort();
    await assert.rejects(waiting); r.decisions[0](false); await r.idle();
  });

  test('a late approval request wakes the task keepalive before any user decision', async () => {
    const r = rig(); let checks = 0, release!: () => void;
    const delayed = new Promise<boolean>(resolve => { release = () => resolve(true); });
    r.context = { ...r.context, current: async () => ++checks === 3 ? delayed : true };
    await r.command(''); assert.equal(r.requests.length, 0);
    const waiting = r.service.wait(r.controller.signal, 10000); release(); await waiting;
    assert.equal(r.requests.length, 1); assert.equal(r.service.running(), true);
    r.decisions[0](false); await r.idle();
  });

  test('child environment excludes service/provider secrets and Python overrides', () => {
    const result = projectCommandEnvironment({ PATH: '/bin', HOME: '/fixture', LANG: 'en_US.UTF-8', OPENAI_API_KEY: 'fixture-openai-key',
      TENSORVIZ_LOCAL_TOKEN: 'fixture-local-token', REACT_APP_TLDRAW_LICENSE_KEY: 'fixture-license', PYTHONPATH: '/developer/modules', BASH_ENV: '/developer/bash' });
    assert.deepEqual(result, { PYTHONDONTWRITEBYTECODE: '1', PYTHONNOUSERSITE: '1', PATH: '/bin', HOME: '/fixture', LANG: 'en_US.UTF-8' });
    const redacted = redactProjectCommandOutput('fixture-openai-key Authorization: Bearer fixture-bearer-value sk-proj-123456789012345678901234', { OPENAI_API_KEY: 'fixture-openai-key' });
    assert.equal(redacted, '[REDACTED] Authorization: Bearer [REDACTED] [REDACTED]');
  });
});
