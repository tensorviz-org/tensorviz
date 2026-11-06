const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const { gzipSync, gunzipSync } = require('node:zlib');
const { SessionFile, WebsitePublisher } = require('../../out/host/website.cjs');
const { StandaloneHost } = require('../../out/host/host.cjs');
const { ApprovalGate } = require('../../out/host/approval.cjs');
const { isLocalBrowserMessage, LOCAL_BROWSER_PROTOCOL_VERSION } = require('../../out/host/protocol.cjs');

const GRAPH_ID = 'AbCdEfGhIjKlMnOpQrSt';
// Only an ID token's claims matter here; Firebase verifies the signature.
const firebaseToken = claims => `e30.${Buffer.from(JSON.stringify(claims)).toString('base64url')}.signature`;
const google = { user_id: 'uid-1', name: 'Ada Lovelace', email: 'ada@example.com', firebase: { sign_in_provider: 'google.com' } };
const graphFile = (name = 'TinyNet', nodes = 2) => gzipSync(JSON.stringify({ format: 'tensorviz.graph', version: 3, status: 'static',
  document: { semantic: { name, nodes: Array.from({ length: nodes }, (_, index) => ({ node_id: `layer${index}` })) } } })).toString('base64');

async function storage(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-website-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  return root;
}

/** Firebase Authentication and the website's upload API, answering as the real ones do. */
function website(respond = () => undefined) {
  const seen = [];
  const fetch = async (input, init) => {
    const url = String(input);
    seen.push({ url, init });
    const answer = respond(url, init);
    if (answer) return answer;
    if (url.includes('accounts:signInWithIdp')) return Response.json({ idToken: firebaseToken(google), refreshToken: 'refresh-1', expiresIn: '3600' });
    if (url.includes('securetoken')) return Response.json({ id_token: firebaseToken(google), refresh_token: 'refresh-1', expires_in: '3600' });
    if (url.endsWith('/api/graphs')) return Response.json({ id: GRAPH_ID, thread: null }, { status: 201 });
    return new Response('{}', { status: 404 });
  };
  return { seen, fetch };
}

/** The browser following a link to the loopback callback, optionally with another Host header. */
function visit(href, host) {
  const url = new URL(href);
  return new Promise((resolve, reject) => {
    http.get({ host: url.hostname, port: url.port, path: `${url.pathname}${url.search}`, headers: host ? { Host: host } : {} }, response => {
      let body = '';
      response.on('data', chunk => { body += chunk; });
      response.on('end', () => resolve({ status: response.statusCode, location: response.headers.location, body,
        policy: response.headers['content-security-policy'], cache: response.headers['cache-control'] }));
    }).on('error', reject);
  });
}

/** What the website's connect page does once the member picks a Google account. */
const connected = (signInUrl, idToken = 'google-id-token') => {
  const page = new URL(signInUrl);
  return `${page.searchParams.get('callback')}?${new URLSearchParams({ state: page.searchParams.get('state'), id_token: idToken })}`;
};

function signingIn() {
  let opened;
  const url = new Promise(resolve => { opened = resolve; });
  return { url, signingIn: signInUrl => { opened(signInUrl); return true; } };
}

test('Publish signs in through a one-time loopback callback, keeps only the refresh token and uploads privately', async t => {
  const root = await storage(t);
  const { seen, fetch } = website();
  const publisher = new WebsitePublisher({ storage: root, fetch, env: {} });
  t.after(() => publisher.dispose());
  const phases = [];
  const browser = signingIn();
  const pending = publisher.publish(graphFile(), {
    signingIn: (url, detail) => { phases.push(['signing_in', detail]); return browser.signingIn(url); },
    uploading: detail => phases.push(['uploading', detail]),
  }, new AbortController().signal);

  const page = new URL(await browser.url);
  assert.equal(`${page.origin}${page.pathname}`, 'https://tensorviz.dev/explore/connect/');
  assert.match(page.searchParams.get('state'), /^[A-Za-z0-9_-]{16,128}$/, 'the state the website accepts');
  const callback = new URL(page.searchParams.get('callback'));
  // Exactly the loopback form the website's connect page accepts.
  assert.deepEqual([callback.protocol, callback.hostname, callback.pathname, callback.search], ['http:', '127.0.0.1', '/tensorviz/auth', '']);
  assert.ok(Number(callback.port) >= 1024 && Number(callback.port) <= 65535);

  assert.equal((await visit(connected(page.href), 'attacker.example')).status, 403, 'another host name is refused');
  const stolen = await visit(connected(page.href).replace(/state=[^&]+/, `state=${'x'.repeat(32)}`));
  assert.deepEqual([stolen.status, stolen.location], [303, '/tensorviz/sign-in-expired'], 'another state is not this sign-in');
  const answered = await visit(connected(page.href));
  assert.deepEqual([answered.status, answered.location], [303, '/tensorviz/signed-in'], 'the token leaves the kept page');
  assert.equal(answered.cache, 'no-store');
  const closing = await visit(new URL(answered.location, callback).href);
  assert.match(closing.body, /Signed in to tensorViz/);
  assert.match(closing.policy, /default-src 'none'/);
  assert.equal((await visit(connected(page.href))).location, '/tensorviz/sign-in-expired', 'a replayed callback');

  assert.deepEqual(await pending, { id: GRAPH_ID, title: 'TinyNet', url: `https://tensorviz.dev/explore/graph/?id=${GRAPH_ID}`,
    shareUrl: `https://tensorviz.dev/explore/graph/?id=${GRAPH_ID}&share=1` });
  assert.deepEqual(phases, [['signing_in', 'Sign in to tensorViz in your browser to publish TinyNet.'], ['uploading', 'Publishing TinyNet to tensorViz…']]);
  const exchange = seen.find(request => request.url.includes('accounts:signInWithIdp'));
  assert.equal(JSON.parse(exchange.init.body).postBody, 'id_token=google-id-token&providerId=google.com');
  const upload = seen.find(request => request.url === 'https://tensorviz.dev/api/graphs');
  assert.equal(upload.init.headers.Authorization, `Bearer ${firebaseToken(google)}`);
  const body = JSON.parse(upload.init.body);
  assert.deepEqual([body.title, body.visibility], ['TinyNet', 'private']);
  assert.equal(JSON.parse(gunzipSync(Buffer.from(body.file, 'base64'))).document.semantic.name, 'TinyNet');

  const sessions = new SessionFile(root);
  const stored = await fs.readFile(sessions.file, 'utf8');
  assert.match(stored, /refresh-1/);
  assert.doesNotMatch(stored, /google-id-token|signature/, 'no ID token is stored');
  if (process.platform !== 'win32') {
    assert.equal((await fs.stat(sessions.file)).mode & 0o777, 0o600);
    assert.equal((await fs.stat(sessions.directory)).mode & 0o777, 0o700);
  }

  // Another chat's plugin host reuses the sign-in without opening the website.
  const next = new WebsitePublisher({ storage: root, fetch, env: {} });
  const again = await next.publish(graphFile('Other'), { signingIn: () => assert.fail('signed in already'), uploading: () => {} },
    new AbortController().signal);
  assert.equal(again.title, 'Other');
  assert.equal(await next.signOut(), true);
  await assert.rejects(fs.stat(sessions.file), { code: 'ENOENT' });
  assert.equal(await next.signOut(), false);
});

test('cancelling Publish while signing in stops waiting, and its late callback is refused', async t => {
  const root = await storage(t);
  const { seen, fetch } = website();
  const publisher = new WebsitePublisher({ storage: root, fetch, env: {} });
  t.after(() => publisher.dispose());
  const controller = new AbortController();
  const browser = signingIn();
  const pending = publisher.publish(graphFile(), { signingIn: browser.signingIn, uploading: () => assert.fail('nothing uploads') }, controller.signal);
  const signInUrl = await browser.url;
  assert.equal(publisher.publishing, true);
  controller.abort();
  await assert.rejects(pending, error => error.code === 'cancelled');
  assert.equal((await visit(connected(signInUrl))).location, '/tensorviz/sign-in-expired');
  assert.deepEqual(seen, [], 'nothing reached Firebase or tensorViz');
  assert.equal(publisher.publishing, false);
});

test('the graph file is checked before any sign-in, and one publish runs at a time', async t => {
  const root = await storage(t);
  const { seen, fetch } = website();
  const publisher = new WebsitePublisher({ storage: root, fetch, env: {} });
  t.after(() => publisher.dispose());
  const unused = { signingIn: () => assert.fail('no sign-in'), uploading: () => assert.fail('no upload') };
  const signal = new AbortController().signal;
  const bomb = gzipSync(Buffer.alloc(10 * 1024 * 1024 + 2, 32)).toString('base64');
  await assert.rejects(publisher.publish(bomb, unused, signal), error => error.code === 'invalid' && /10 MB/.test(error.message));
  await assert.rejects(publisher.publish('bm90IGd6aXA=', unused, signal), error => error.code === 'invalid');
  await assert.rejects(publisher.publish(gzipSync('{}').toString('base64'), unused, signal), error => error.code === 'invalid');
  await assert.rejects(publisher.publish(graphFile('Net', 0), unused, signal), error => error.code === 'invalid' && /5,000 blocks/.test(error.message));
  assert.deepEqual(seen, []);

  const controller = new AbortController();
  const browser = signingIn();
  const first = publisher.publish(graphFile(), { signingIn: browser.signingIn, uploading: () => {} }, controller.signal);
  await browser.url;
  await assert.rejects(publisher.publish(graphFile(), unused, signal), error => error.code === 'too-soon');
  controller.abort();
  await assert.rejects(first, error => error.code === 'cancelled');
});

test('the sign-in file must be private and owned by you', { skip: process.platform === 'win32' }, async t => {
  const root = await storage(t);
  const sessions = new SessionFile(root);
  await fs.mkdir(sessions.directory);
  await fs.chmod(sessions.directory, 0o755);
  await assert.rejects(sessions.get('key'), error => error.code === 'setup');
  await fs.chmod(sessions.directory, 0o700);
  await sessions.store('key', 'value');
  assert.equal(await sessions.get('key'), 'value');
  await fs.chmod(sessions.file, 0o644);
  await assert.rejects(sessions.get('key'), error => error.code === 'setup');
});

test('the browser connection carries Publish and capture export to this host and answers only with checked messages', async () => {
  const host = new StandaloneHost('/unused', '/unused', new ApprovalGate(async () => false, '/unused'));
  const target = { workspaceFolder: '/workspace', file: 'model.py', cls: 'Tiny', documentVersion: 0,
    sourceDigest: 'a'.repeat(64), backendRevision: 1 };
  const published = [];
  let onMessage, onStatus;
  const transport = {
    status: { phase: 'connected', epoch: 1 },
    publish(message) {
      const full = { ...message, protocolVersion: LOCAL_BROWSER_PROTOCOL_VERSION, workspaceSessionId: 'session', messageId: 'message' };
      if (['activeTarget', 'projection', 'syncState', 'graphOpening'].includes(message.type)) full.hostSequence = 1;
      assert.ok(isLocalBrowserMessage(full, 'session'), `invalid browser message: ${message.type}`);
      published.push(message);
      return full;
    },
    onMessage(listener) { onMessage = listener; }, onStatusChange(listener) { onStatus = listener; return () => { onStatus = undefined; }; }, dispose() {},
  };
  host.backend.connect = async () => { host.backend.transport = transport; return transport; };
  const replies = type => published.filter(message => message.type === type);
  const settled = () => new Promise(resolve => setImmediate(resolve));
  try {
    await host.connectBrowser(new AbortController().signal);

    // Publish: progress and the result go back to the asking request only.
    let signIn;
    host.website.publish = async (file, progress, signal) => {
      assert.equal(file, 'H4sIAAAAAAAA');
      assert.equal(progress.signingIn('https://tensorviz.dev/explore/connect/?state=abc', 'Sign in to publish Tiny.'), true);
      await new Promise((resolve, reject) => { signIn = resolve; signal.addEventListener('abort', () => reject(new Error('aborted'))); });
      progress.uploading('Publishing Tiny to tensorViz…');
      return { id: GRAPH_ID, title: 'Tiny', url: `https://tensorviz.dev/explore/graph/?id=${GRAPH_ID}`,
        shareUrl: `https://tensorviz.dev/explore/graph/?id=${GRAPH_ID}&share=1` };
    };
    onMessage({ type: 'graphPublishRequest', target, requestId: 'publish', file: 'H4sIAAAAAAAA' });
    await settled();
    assert.deepEqual(replies('graphPublishState').map(message => message.phase), ['signing_in']);
    onMessage({ type: 'graphPublishRequest', target, requestId: 'second', file: 'H4sIAAAAAAAA' });
    assert.deepEqual([replies('graphPublishResult')[0].requestId, replies('graphPublishResult')[0].status], ['second', 'failed'], 'one publish at a time');
    signIn();
    await settled(); await settled();
    const result = replies('graphPublishResult').at(-1);
    assert.deepEqual([result.requestId, result.status, result.graph.id], ['publish', 'published', GRAPH_ID]);
    assert.equal(onStatus, undefined, 'the finished request stops watching the connection');

    // A cancel from the view, or a new broker epoch, stops a pending sign-in.
    onMessage({ type: 'graphPublishRequest', target, requestId: 'cancelled', file: 'H4sIAAAAAAAA' });
    await settled();
    onMessage({ type: 'graphPublishCancel', target, requestId: 'cancelled' });
    await settled(); await settled();
    assert.deepEqual([replies('graphPublishResult').at(-1).requestId, replies('graphPublishResult').at(-1).status], ['cancelled', 'cancelled']);
    onMessage({ type: 'graphPublishRequest', target, requestId: 'reconnected', file: 'H4sIAAAAAAAA' });
    await settled();
    onStatus({ phase: 'connected', epoch: 2 });
    await settled(); await settled();
    assert.equal(replies('graphPublishResult').at(-1).status, 'cancelled');

    // Capture export answers only for the displayed capture of the current model.
    onMessage({ type: 'captureExportRequest', target, requestId: 'export', reference: { captureId: 'capture', sourceId: 'source' } });
    await settled();
    assert.deepEqual([replies('captureExportResult').at(-1).capture, replies('captureExportResult').at(-1).detail],
      [null, 'The displayed capture changed. Export the current graph again.']);
    host.model = { target, current: true, projection: { traced: true } };
    host.captures.list = () => ({ displayedCaptureId: 'capture' });
    host.captures.retained = () => ({ id: 'capture' });
    const portable = { version: 1, sourceNames: 'anonymized', capture: { id: 'capture' } };
    host.backend.portable = async capture => { assert.deepEqual(capture, { id: 'capture' }); return portable; };
    onMessage({ type: 'captureExportRequest', target, requestId: 'export-2', reference: { captureId: 'capture', sourceId: 'source' } });
    await settled(); await settled();
    assert.deepEqual(replies('captureExportResult').at(-1).capture, portable);
  } finally { host.dispose(); }
});
