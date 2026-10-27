import { randomBytes, timingSafeEqual } from 'node:crypto';
import { createServer, type Server } from 'node:http';
import { HostError, digest } from './workspace';
import { PluginPermissions, permissionSummary } from './permissions';
import { withResourceLock } from './locks';
import { setTimeout as delay } from 'node:timers/promises';

export interface OperationPlan {
  operation: 'inspect_environment' | 'prepare_inputs' | 'capture' | 'save_setup' | 'install_dependencies' | 'build_dependencies' | 'managed_runtime';
  workspaceRoot: string;
  interpreter: string;
  detail: string;
  facts: Record<string, unknown>;
}
interface PendingApproval {
  id: string;
  plan: OperationPlan;
  method: 'native' | 'browser';
  expiresAt: string;
  url?: string;
  scope: 'plugin';
  permissions: string;
}
type NativeApproval = (message: string, signal: AbortSignal) => Promise<boolean | undefined>;

/** Dismissing a native form is not a durable decision to disable the plugin. */
export function nativePermissionDecision(result: { action: string; content?: Record<string, unknown> }): boolean {
  if (result.action === 'cancel') throw new HostError('APPROVAL_CANCELLED', 'Permission confirmation was cancelled. Nothing was started or saved.');
  return result.action === 'accept' && result.content?.confirm === true;
}

/** One user decision covers the disclosed plugin capabilities; tool arguments cannot grant it. */
export class ApprovalGate {
  pending?: PendingApproval;
  private server?: Server;
  private cancel?: () => void;
  private blockedPlan?: OperationPlan;
  readonly permissions: PluginPermissions;
  constructor(private readonly native: NativeApproval, storage: string) {
    this.permissions = new PluginPermissions(storage);
  }

  async approve(plan: OperationPlan, signal: AbortSignal): Promise<void> {
    this.blockedPlan = plan;
    return this.decide(plan, signal, false);
  }

  /** Only an explicit user-requested recovery may reopen a declined decision. */
  async reconsider(signal: AbortSignal): Promise<void> {
    if (!this.blockedPlan) throw new HostError('APPROVAL_UNAVAILABLE', 'Open the intended model before requesting permission recovery.');
    return this.decide(this.blockedPlan, signal, true);
  }

  private async decide(plan: OperationPlan, signal: AbortSignal, reconsider: boolean): Promise<void> {
    signal.throwIfAborted();
    const status = await this.permissions.status();
    signal.throwIfAborted();
    if (status === 'granted') return;
    if (status === 'denied' && !reconsider) throw new HostError('APPROVAL_DECLINED', 'TensorViz permission was declined or revoked. Request Enable TensorViz to review the permission again.');
    // A second chat waits for the same user's decision instead of opening another prompt.
    while (true) {
      try {
        return await withResourceLock(this.permissions.file, signal, async check => {
          const current = await this.permissions.status();
          check();
          if (current === 'granted') return;
          if (current === 'denied' && !reconsider) throw new HostError('APPROVAL_DECLINED', 'TensorViz permission was declined or revoked.');
          await this.request(plan, signal, check);
        });
      } catch (error) {
        if (!(error instanceof HostError) || error.code !== 'RESOURCE_BUSY') throw error;
        await delay(100, undefined, { signal });
      }
    }
  }

  private async request(plan: OperationPlan, signal: AbortSignal, check: () => void): Promise<void> {
    if (this.pending) throw new HostError('APPROVAL_BUSY', 'Another operation is awaiting a decision.');
    const frozen = JSON.parse(JSON.stringify(plan)) as OperationPlan;
    const id = digest(JSON.stringify(frozen));
    const controller = new AbortController();
    const combined = AbortSignal.any([signal, controller.signal]);
    const timer = setTimeout(() => controller.abort(), 300_000);
    this.cancel = () => controller.abort();
    this.pending = { id, plan: frozen, scope: 'plugin', permissions: permissionSummary,
      method: 'native', expiresAt: new Date(Date.now() + 300_000).toISOString() };
    try {
      const before = await this.permissions.receipt();
      const message = `${permissionSummary}\n\nFirst requested operation: ${plan.operation.replaceAll('_', ' ')}.\nWorkspace: ${plan.workspaceRoot}\nPython: ${plan.interpreter}\n${plan.detail}\n\n${JSON.stringify(plan.facts, null, 2)}`;
      const native = await this.native(message, combined);
      const approved = native ?? await this.browser(frozen, combined);
      combined.throwIfAborted();
      check();
      const after = await this.permissions.receipt();
      if (after.status === 'denied' && after.revision !== before.revision) throw new HostError('APPROVAL_DECLINED', 'TensorViz permission was revoked while this decision was pending.');
      await this.permissions.save(approved ? 'granted' : 'denied');
      if (!approved) {
        throw new HostError('APPROVAL_DECLINED', 'The user declined TensorViz permission. Nothing was started.');
      }
    } catch (error) {
      if (combined.aborted) throw new HostError('APPROVAL_CANCELLED', 'Approval expired or was cancelled. Nothing was started.');
      throw error;
    } finally {
      clearTimeout(timer);
      this.server?.closeAllConnections();
      this.server?.close();
      this.server = undefined;
      this.pending = undefined;
      this.cancel = undefined;
    }
  }

  private async browser(plan: OperationPlan, signal: AbortSignal): Promise<boolean> {
    signal.throwIfAborted();
    const token = randomBytes(32).toString('base64url');
    const nonce = randomBytes(24).toString('base64url');
    let origin = '';
    let finish!: (approved: boolean) => void;
    const decision = new Promise<boolean>((resolve, reject) => {
      finish = resolve;
      signal.addEventListener('abort', () => reject(new Error('cancelled')), { once: true });
    });
    // Aborts can arrive during listen(), before the caller begins awaiting the decision.
    void decision.catch(() => {});
    // Both token and origin are required. No token appears in a query, cookie, log or file.
    this.server = createServer(async (request, response) => {
      response.setHeader('Cache-Control', 'no-store');
      response.setHeader('Referrer-Policy', 'no-referrer');
      response.setHeader('X-Content-Type-Options', 'nosniff');
      response.setHeader('Content-Security-Policy', `default-src 'none'; script-src 'nonce-${nonce}'; style-src 'nonce-${nonce}'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'`);
      if (request.headers.host !== new URL(origin).host) { response.writeHead(403).end(); return; }
      if (request.method === 'GET' && request.url === '/approval') {
        response.setHeader('Content-Type', 'text/html; charset=utf-8');
        response.end(`<!doctype html><html><meta charset="utf-8"><title>TensorViz approval</title><style nonce="${nonce}">body{font:16px system-ui;max-width:850px;margin:48px auto;padding:24px;background:#111;color:#eee}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:12px 24px;margin:12px 12px 0 0}</style><h1>Enable TensorViz</h1><pre id="plan">Loading…</pre><button id="yes" disabled>Allow TensorViz</button><button id="no" disabled>Decline</button><script nonce="${nonce}">const token=location.hash.slice(1);history.replaceState(null,'',location.pathname);const call=async(path,body)=>{const r=await fetch(path,{method:'POST',headers:{'Authorization':'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok)throw Error('Approval expired or unavailable.');return r.json()};call('/plan',{}).then(p=>{document.getElementById('plan').textContent=p.permissions+'\\n\\nFirst requested operation: '+JSON.stringify(p.firstOperation,null,2);for(const id of ['yes','no']){const b=document.getElementById(id);b.disabled=false;b.onclick=async()=>{for(const x of ['yes','no'])document.getElementById(x).disabled=true;try{await call('/decision',{approve:id==='yes'});document.getElementById('plan').textContent=id==='yes'?'Permission saved. Return to your coding agent.':'Declined. Nothing was started.'}catch(e){document.getElementById('plan').textContent=e.message}}}}).catch(e=>document.getElementById('plan').textContent=e.message)</script></html>`);
        return;
      }
      const supplied = request.headers.authorization ?? '';
      const expected = `Bearer ${token}`;
      if (request.method !== 'POST' || request.headers.origin !== origin || Buffer.byteLength(supplied) !== Buffer.byteLength(expected)
        || !timingSafeEqual(Buffer.from(supplied), Buffer.from(expected))) { response.writeHead(403).end(); return; }
      let body = '';
      try {
        for await (const chunk of request) {
          body += chunk.toString();
          if (Buffer.byteLength(body) > 1024) { response.writeHead(413).end(); return; }
        }
        const value = JSON.parse(body);
        response.setHeader('Content-Type', 'application/json');
        if (request.url === '/plan') { response.end(JSON.stringify({ permissions: permissionSummary, firstOperation: plan })); return; }
        if (request.url !== '/decision' || typeof value.approve !== 'boolean' || Object.keys(value).length !== 1) { response.writeHead(400).end(); return; }
        response.end('{"recorded":true}', () => finish(value.approve));
      } catch { response.writeHead(400).end(); }
    });
    this.server.requestTimeout = 10_000;
    await new Promise<void>((resolve, reject) => {
      this.server!.once('error', reject);
      this.server!.listen(0, '127.0.0.1', resolve);
    });
    const address = this.server.address();
    if (!address || typeof address === 'string') throw new HostError('APPROVAL_UNAVAILABLE', 'The confirmation page could not start.');
    origin = `http://127.0.0.1:${address.port}`;
    this.pending = { ...this.pending!, method: 'browser', url: `${origin}/approval#${token}` };
    return decision;
  }

  dispose(): void { this.cancel?.(); }
}
