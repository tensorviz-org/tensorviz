import * as assert from 'node:assert/strict';
import * as os from 'node:os';
import { FEEDBACK_URL, feedbackReport, feedbackUrl, redactPaths, sendFeedback, type FeedbackReport } from '@tensorviz/graph-host/feedback';
import { feedbackPreview, registerMcpFeedbackTool } from '@tensorviz/graph-host/mcpFeedback';
import { WebsiteError } from '../../websiteAccount';
import { setup, signIn } from '../websiteFixture';

const client = { product: 'plugin' as const, version: '1.2.3', os: 'macOS 24.6.0 (arm64)' };
const isFeedback = (url: string) => url.endsWith('/api/feedback');
const accepted = () => Response.json({ id: 'fb_1' }, { status: 201 });

/** A server stand-in: the registered tool, and the client's confirmation form when it has one. */
function toolServer(form?: (message: string) => { action: string; content?: Record<string, unknown> }) {
  let handler!: (args: any, extra: any) => Promise<any>;
  const asked: string[] = [];
  const server = {
    registerTool(_name: string, _config: unknown, registered: typeof handler) { handler = registered; },
    server: {
      getClientCapabilities: () => form ? { elicitation: { form: {} } } : {},
      elicitInput: async (request: { message: string }) => { asked.push(request.message); return form!(request.message); },
    },
  };
  const sent: FeedbackReport[] = [];
  registerMcpFeedbackTool(server as any, {
    graph: () => ({ model: 'TinyNet', nodes: 4 }),
    report: input => feedbackReport(input, 'agent', client, ['/work/proj']),
    send: async report => { sent.push(report); return 'fb_1'; },
  });
  const call = async (args: Record<string, unknown>) =>
    (await handler({ kind: 'bug', includeGraph: false, ...args }, { signal: new AbortController().signal })).structuredContent;
  return { call, sent, asked };
}

suite('Send feedback', () => {
  test('hides workspace and home paths and checks the words', () => {
    const home = os.homedir();
    assert.equal(redactPaths(`Failed at /work/proj/models/net.py and ${home}/x.py and /home/sam/y.py and C:\\Users\\sam\\z.py`, ['/work/proj', home]),
      'Failed at <workspace>/models/net.py and ~/x.py and ~/y.py and ~\\z.py');
    const report = feedbackReport({ kind: 'bug', message: ' Broke in /work/proj/a.py ', graph: { model: 'TinyNet', nodes: 3 } }, 'canvas', client, ['/work/proj']);
    assert.deepEqual(report, { kind: 'bug', message: 'Broke in <workspace>/a.py', source: 'canvas', client, graph: { model: 'TinyNet', nodes: 3 } });
    assert.equal(feedbackReport({ kind: 'idea', message: 'x', graph: { model: '../etc', nodes: 1 } }, 'canvas', client).graph, undefined);
    for (const input of [{ kind: 'bug', message: '   ' }, { kind: 'praise', message: 'x' }, { kind: 'bug', message: 'x'.repeat(4001) }]) {
      assert.throws(() => feedbackReport(input as any, 'canvas', client), (error: WebsiteError) => error.code === 'invalid');
    }
  });

  test('posts as the member when signed in, and without a token otherwise; never starts a sign-in', async () => {
    const world = setup(url => isFeedback(url) ? accepted() : undefined);
    const report = feedbackReport({ kind: 'bug', message: 'Edges overlap.' }, 'canvas', client);
    assert.equal(await sendFeedback(world.account, report, world.fetcher, world.host.env), 'fb_1');
    const anonymous = world.seen.find(request => isFeedback(request.url))!;
    assert.equal(anonymous.url, 'https://feedback.tensorviz.dev/api/feedback');
    assert.equal((anonymous.init?.headers as Record<string, string>).Authorization, undefined);
    assert.deepEqual(JSON.parse(String(anonymous.init?.body)), report);
    assert.deepEqual(world.opened, []);
    await signIn(world);
    await sendFeedback(world.account, report, world.fetcher, world.host.env);
    const signed = world.seen.filter(request => isFeedback(request.url))[1];
    assert.match((signed.init?.headers as Record<string, string>).Authorization, /^Bearer e30\./);
  });

  test('goes to the tensorViz team, and while developing only to a feedback endpoint on this computer', async () => {
    const local = 'http://127.0.0.1:5001/api/feedback';
    assert.equal(feedbackUrl({}), FEEDBACK_URL);
    assert.equal(feedbackUrl({ TENSORVIZ_FEEDBACK_URL: local }), local);
    assert.equal(feedbackUrl({ TENSORVIZ_FEEDBACK_URL: 'http://localhost:5001/local/feedback',
      TENSORVIZ_WEBSITE_EMULATOR: 'http://localhost:5174' }), 'http://localhost:5001/local/feedback');
    for (const env of [
      { TENSORVIZ_FEEDBACK_URL: 'https://feedback.tensorviz.dev.example.com/api/feedback' },
      { TENSORVIZ_FEEDBACK_URL: 'http://192.168.1.20:5001/api/feedback' },
      { TENSORVIZ_FEEDBACK_URL: 'https://127.0.0.1:5001/api/feedback' },
      { TENSORVIZ_FEEDBACK_URL: 'http://user:secret@127.0.0.1:5001/api/feedback' },
      { TENSORVIZ_FEEDBACK_URL: 'not an address' },
      { TENSORVIZ_WEBSITE_EMULATOR: 'http://localhost:5174' },
      { TENSORVIZ_WEBSITE_URL: 'http://localhost:5173' },
    ]) {
      assert.throws(() => feedbackUrl(env), (error: WebsiteError) => error.code === 'setup', JSON.stringify(env));
    }
    const world = setup(() => accepted());
    const report = feedbackReport({ kind: 'bug', message: 'x' }, 'canvas', client);
    await sendFeedback(world.account, report, world.fetcher, { TENSORVIZ_FEEDBACK_URL: local });
    assert.deepEqual(world.seen.map(request => request.url), [local]);
    // A development sign-in never reaches the team's endpoint, and nothing is sent.
    const developing = setup(() => accepted());
    await assert.rejects(sendFeedback(developing.account, report, developing.fetcher, { TENSORVIZ_WEBSITE_EMULATOR: 'http://localhost:5174' }),
      (error: WebsiteError) => error.code === 'setup');
    assert.deepEqual(developing.seen, []);
  });

  test('retries a rejected token once without it, and reports the endpoint\'s refusals', async () => {
    let posts = 0;
    const world = setup((url, init) => {
      if (!isFeedback(url)) return undefined;
      posts++;
      return (init?.headers as Record<string, string>).Authorization ? new Response('{}', { status: 401 }) : accepted();
    });
    await signIn(world);
    assert.equal(await sendFeedback(world.account, feedbackReport({ kind: 'bug', message: 'x' }, 'canvas', client), world.fetcher, world.host.env), 'fb_1');
    assert.equal(posts, 2);
    for (const [status, code] of [[429, 'too-soon'], [400, 'invalid'], [403, 'denied'], [500, 'unavailable']] as const) {
      const refusing = setup(url => isFeedback(url) ? Response.json({ error: { message: 'Nope.' } }, { status }) : undefined);
      await assert.rejects(sendFeedback(refusing.account, feedbackReport({ kind: 'bug', message: 'x' }, 'canvas', client), refusing.fetcher, refusing.host.env),
        (error: WebsiteError) => error.code === code);
    }
  });

  test('the agent tool sends only the exact report the user approved', async () => {
    const tool = toolServer();
    const preview = await tool.call({ message: 'Capture hangs in /work/proj/train.py', includeGraph: true });
    assert.equal(preview.status, 'needs_confirmation');
    assert.deepEqual(tool.sent, []);
    assert.equal(preview.report.message, 'Capture hangs in <workspace>/train.py');
    assert.deepEqual(preview.report.graph, { model: 'TinyNet', nodes: 4 });
    assert.equal(preview.preview, feedbackPreview(preview.report));
    await assert.rejects(Promise.resolve(tool.call({ confirmationId: '00000000-0000-4000-8000-000000000000' })).then(result => {
      if (result.status !== 'sent') throw new Error(result.code);
    }));
    assert.equal((await tool.call({ confirmationId: preview.confirmationId })).status, 'sent');
    assert.deepEqual(tool.sent, [preview.report]);
    assert.notEqual((await tool.call({ confirmationId: preview.confirmationId })).status, 'sent');
    assert.equal(tool.sent.length, 1);
  });

  test('the agent tool asks in the client\'s own form when it has one', async () => {
    const declined = toolServer(() => ({ action: 'accept', content: { send: false } }));
    assert.equal((await declined.call({ message: 'Slow.' })).status, 'not_sent');
    assert.deepEqual(declined.sent, []);
    assert.match(declined.asked[0], /Message: Slow\.\nAlso sent: tensorViz 1\.2\.3 · macOS/);
    const approved = toolServer(() => ({ action: 'accept', content: { send: true } }));
    assert.equal((await approved.call({ message: 'Slow.', kind: 'idea' })).status, 'sent');
    assert.deepEqual(approved.sent.map(report => [report.kind, report.message, report.source]), [['idea', 'Slow.', 'agent']]);
  });
});
