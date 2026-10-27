/** Real isolated Python snapshot -> lookup -> manual host lifecycle, with execution recorded. */
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { LocalBrowserValidation } from '../localBrowserValidation';
import type { LocalBrowserSync } from '../localBrowserSync';
import { hasValidationConfig, readValidationSnapshot } from '../validationHost';
import type { LocalBrowserHostPublish } from '@tensorviz/graph-host/localBrowserHostTransport';
import type { LocalBrowserValidateRequestMessage, LocalBrowserValidationOutcome } from '@tensorviz/contracts/protocol';
import type { ExecutionSpec } from '@tensorviz/graph-host/executionSpec';

export async function runModelProfileFixture(scenario: string) {
  const repo = path.resolve(__dirname, '../../..');
  const fixtures = path.join(repo, 'backend/tests/fixtures/model_profiles');
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'model-profiles-'));
  const python = process.env.TENSORVIZ_TEST_PYTHON || path.join(repo, 'venv/bin/python');
  const executions: Array<{ execution?: ExecutionSpec; timeoutMs?: number }> = [];
  const snapshots: Array<string | null> = [];
  const choices: string[][] = [];
  try {
    const file = path.join(root, 'tensorviz.toml');
    let source = await fs.readFile(path.join(fixtures, 'audit_models.py'), 'utf8');
    // A snapshot that imports this source either creates the sentinel or fails.
    source = `from pathlib import Path\nPath(${JSON.stringify(path.join(root, 'executed'))}).touch()\n` + source;
    if (scenario === 'selected_failure') source = source.replace('def __init__(self):', 'def __init__(self, width):');
    await fs.writeFile(path.join(root, 'audit_models.py'), source);
    let config = await fs.readFile(path.join(fixtures, scenario === 'invalid' ? 'invalid.toml' : 'valid.toml'), 'utf8');
    if (scenario === 'syntax') config = 'version = [private-source';
    if (scenario === 'limit') config = config.replaceAll('timeout_ms = 30000', 'timeout_ms = 99');
    if (scenario === 'unrelated') config = config.replaceAll('model = "audit_models:GoodNet"', 'model = "other:Other"\ncwd = "missing"');
    if (scenario !== 'absent') await fs.writeFile(file, config);
    if (scenario === 'unreadable') { await fs.unlink(file); await fs.mkdir(file); }
    const request: LocalBrowserValidateRequestMessage = { type: 'validateRequest', protocolVersion: 1,
      workspaceSessionId: 'session', messageId: 'message', requestId: 'request',
      target: { workspaceFolder: root, file: 'audit_models.py', cls: 'GoodNet',
        documentVersion: 1, sourceDigest: 'a'.repeat(64), backendRevision: 1 } };
    let manager: LocalBrowserValidation;
    let receive!: (reply: LocalBrowserHostPublish) => void;
    const reply = new Promise<LocalBrowserHostPublish>(resolve => { receive = resolve; });
    manager = new LocalBrowserValidation({
      validate: async (_target: unknown, _signal: AbortSignal, _inputs?: unknown, _progress?: unknown,
        _measured?: unknown, _captured?: unknown, execution?: ExecutionSpec, _job?: unknown, timeoutMs?: number) => {
        executions.push({ execution, timeoutMs });
        return { status: 'passed', detail: 'Execution seam reached.' } as LocalBrowserValidationOutcome;
      },
    } as unknown as LocalBrowserSync, receive, {
      hasConfig: hasValidationConfig,
      snapshot: async (_root, profile, signal) => {
        snapshots.push(profile ?? null);
        const prepared = await readValidationSnapshot(scenario === 'snapshot_failure' ? path.join(root, 'no-python') : python,
          path.join(repo, 'backend'), { root, documents: scenario === 'dirty' ? { 'tensorviz.toml': { version: 2, dirty: true } } : {}, executable: python, environmentRevision: 0,
            session: 'session', workspaceId: 'workspace', runId: 'run', generation: 1, profile }, signal);
        if (scenario === 'cancel_resolution' && profile) manager.cancel(request.requestId);
        if (scenario === 'cancel_listing' && !profile) manager.cancel(request.requestId);
        return prepared;
      },
      choose: async names => {
        choices.push(names);
        if (scenario === 'dismiss') return undefined;
        if (scenario === 'cancel_choice') manager.cancel(request.requestId);
        if (scenario === 'removed') await fs.unlink(file);
        return 'audit_large';
      },
    });
    manager.handle(request);
    const result = await reply;
    let sourceExecuted = false;
    try { await fs.access(path.join(root, 'executed')); sourceExecuted = true; } catch { /* absent */ }
    return { result, executions, snapshots, choices, sourceExecuted };
  } finally { await fs.rm(root, { recursive: true, force: true }); }
}
