import * as assert from 'node:assert/strict';
import { z } from 'zod';
import { registerCanvasAgentPermissionTools } from '../../canvasAgentPermissionTools';
import type { CanvasAgentPermission, CanvasAgentPermissionPresentation } from '@tensorviz/contracts/protocol';

const presentation: CanvasAgentPermissionPresentation = { title: 'Install Net libraries?', summary: 'These libraries let TensorViz load Net and record tensor shapes.',
  allowLabel: 'Install libraries', denyLabel: 'Keep current Python' };
const permission = (): CanvasAgentPermission => ({ id: 'permission', requestId: 'turn', status: 'pending', operation: 'python.packages.install',
  scope: 'isolated_environment', effects: ['install'], facts: { python: '/storage/net/bin/python', packages: [{ name: 'torch', version: '2.9.0' }] } });

suite('Canvas agent permission tools', () => {
  function rig() {
    const tools: Record<string, { config: any; handler: (args: any, extra: { signal: AbortSignal }) => Promise<any> }> = {};
    let pending: CanvasAgentPermission | undefined = permission(), reads = 0;
    const presentations: Array<{ id: string; presentation: CanvasAgentPermissionPresentation }> = [];
    let refusal: Error | undefined;
    registerCanvasAgentPermissionTools({ registerTool: (name: string, config: any, handler: any) => { tools[name] = { config, handler }; } } as never, {
      pending: () => { reads++; return pending; },
      present: (id, copy) => { if (refusal) throw refusal; presentations.push({ id, presentation: copy }); },
    });
    return { tools, presentations, get reads() { return reads; },
      set pending(value: CanvasAgentPermission | undefined) { pending = value; }, set refusal(value: Error) { refusal = value; },
      call: (name: string, args: unknown, signal = new AbortController().signal) => {
        const tool = tools[name]; return tool.handler(z.object(tool.config.inputSchema).strict().parse(args), { signal });
      } };
  }

  test('the read tool exposes actual operation facts and no approval authority', async () => {
    const r = rig(); assert.deepEqual(Object.keys(r.tools).sort(), ['get_pending_permission', 'present_permission']);
    assert.equal(r.tools.get_pending_permission.config.annotations.readOnlyHint, true);
    const result = await r.call('get_pending_permission', {});
    assert.deepEqual(result.structuredContent, { permission: permission() });
    assert.equal(r.presentations.length, 0); assert.equal(r.reads, 1);
    r.pending = undefined; assert.deepEqual((await r.call('get_pending_permission', {})).structuredContent, { permission: null });
  });

  test('present supplies only wording and reports a still-pending user decision', async () => {
    const r = rig(); const result = await r.call('present_permission', { permissionId: 'permission', presentation });
    assert.equal(result.isError, undefined); assert.deepEqual(r.presentations, [{ id: 'permission', presentation }]);
    assert.deepEqual(result.structuredContent, { presented: true, permissionId: 'permission', waitingForUser: true });
    assert.equal(r.tools.present_permission.config.annotations.destructiveHint, false);
    assert.equal(r.tools.present_permission.config.annotations.openWorldHint, false);
    assert.equal((await r.call('get_pending_permission', {})).structuredContent.permission.status, 'pending');
  });

  test('a host policy approval is reported without letting the wording tool choose it', async () => {
    const r = rig(); r.pending = undefined;
    const result = await r.call('present_permission', { permissionId: 'permission', presentation });
    assert.deepEqual(result.structuredContent, { presented: true, permissionId: 'permission', waitingForUser: false });
    assert.deepEqual(r.presentations[0], { id: 'permission', presentation });
  });

  test('operation, approval and permission-mode fields cannot be smuggled into generated copy', () => {
    const r = rig();
    for (const invalid of [{ permissionId: 'permission', presentation, approved: true },
      { permissionId: 'permission', presentation, operation: 'shell.execute' },
      { permissionId: 'permission', presentation: { ...presentation, effects: ['create'] } },
      { permissionId: 'permission', presentation: { ...presentation, mode: 'full_setup' } },
      { permissionId: '', presentation }, { permissionId: 'permission', presentation: { ...presentation, title: 'x'.repeat(161) } }]) {
      assert.throws(() => r.call('present_permission', invalid));
    }
    assert.equal(r.presentations.length, 0);
  });

  test('stale receipt or invalid presentation refusals become tool errors, not success receipts', async () => {
    const r = rig(); r.refusal = new Error('The permission request is no longer current.');
    const result = await r.call('present_permission', { permissionId: 'stale', presentation });
    assert.equal(result.isError, true); assert.equal(r.presentations.length, 0);
    assert.equal(result.structuredContent?.presented, undefined);
  });

  test('cancelled read and presentation calls never reach the broker', async () => {
    const r = rig(), controller = new AbortController(); controller.abort();
    for (const [name, args] of [['get_pending_permission', {}], ['present_permission', { permissionId: 'permission', presentation }]] as const) {
      const result = await r.call(name, args, controller.signal); assert.equal(result.isError, true);
    }
    assert.equal(r.reads, 0); assert.equal(r.presentations.length, 0);
  });
});
