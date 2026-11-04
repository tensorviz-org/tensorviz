import * as assert from 'node:assert/strict';
import { connectUrl, editorCallback, EMULATOR_VARIABLE, SITE_VARIABLE, WEBSITE, WebsiteError, websiteConfig } from '../../websiteAccount';
import { CALLBACK, cancellation, google, SECRET, settled, setup, signIn, WEBSITE_STATE } from '../websiteFixture';

/** What vscode.env.openExternal hands the browser for a Uri, as VS Code's own URI code does
 *  it: the query is percent-decoded, Uri.toString(true) escapes only `?` and `#` in it, and
 *  encodeURI then escapes the result once more. */
function openedByVsCode(url: string): string {
  const at = url.indexOf('?');
  const query = decodeURIComponent(url.slice(at + 1)).replace(/[?#]/g, character => encodeURIComponent(character));
  return encodeURI(`${url.slice(0, at)}?${query}`);
}

suite('tensorViz website sign-in', () => {
  test('signs in through the connect page and keeps only the refresh token', async () => {
    const world = setup();
    const pending = world.account.signIn();
    await settled();
    const page = new URL(world.opened[0]);
    assert.equal(`${page.origin}${page.pathname}`, 'https://tensorviz.dev/explore/connect/');
    const state = page.searchParams.get('state')!;
    assert.match(state, WEBSITE_STATE);
    assert.equal(page.searchParams.get('callback'), CALLBACK);

    assert.equal(world.account.handleUri('/auth', `state=${'x'.repeat(state.length)}&id_token=stolen`), false, 'another state');
    assert.equal(world.account.handleUri('/elsewhere', `state=${state}&id_token=google-id-token`), false);
    assert.ok(world.account.handleUri('/auth', `state=${state}&id_token=google-id-token`));
    const session = await pending;
    assert.deepEqual([session.uid, session.authorName, session.authorPhoto], ['uid-1', 'Ada Lovelace', 'https://example.com/ada.png']);

    const exchange = world.seen.find(request => request.url.includes('accounts:signInWithIdp'))!;
    assert.equal(exchange.url, `${WEBSITE.identity}/v1/accounts:signInWithIdp?key=${WEBSITE.apiKey}`);
    assert.equal(JSON.parse(String(exchange.init?.body)).postBody, 'id_token=google-id-token&providerId=google.com');
    const stored = world.secrets.get(SECRET)!;
    assert.deepEqual(JSON.parse(stored), { uid: 'uid-1', name: 'Ada Lovelace', email: 'ada@example.com', refreshToken: 'refresh-1' });
    assert.doesNotMatch(stored, /google-id-token|signature/, 'no ID token is stored');
    assert.equal(world.account.handleUri('/auth', `state=${state}&id_token=google-id-token`), false, 'a replayed callback');
    assert.deepEqual(await world.account.member(), { uid: 'uid-1', name: 'Ada Lovelace', email: 'ada@example.com' });
  });

  test('the connect link reaches the browser as the website expects it', () => {
    const state = 'Zq3_state-abcdefghij';
    assert.equal(editorCallback('vscode', 'tensorviz.tensorviz-vscode'), CALLBACK);
    const opened = new URL(openedByVsCode(connectUrl(WEBSITE, CALLBACK, state)));
    assert.equal(opened.searchParams.get('state'), state);
    assert.equal(opened.searchParams.get('callback'), CALLBACK);
    // Why the callback carries no ?windowId= from asExternalUri: the page got auth%3FwindowId=1.
    const windowed = `${CALLBACK}?windowId=1`;
    assert.equal(new URL(openedByVsCode(connectUrl(WEBSITE, windowed, state))).searchParams.get('callback'), `${CALLBACK}%3FwindowId=1`);
  });

  test('a second sign-in joins the one already opening the browser', async () => {
    const world = setup();
    const first = world.account.signIn();
    const second = world.account.signIn();
    await settled();
    assert.equal(world.opened.length, 1);
    const state = new URL(world.opened[0]).searchParams.get('state')!;
    assert.ok(world.account.handleUri('/auth', `state=${state}&id_token=google-id-token`));
    const [left, right] = await Promise.all([first, second]);
    assert.equal(left, right);
  });

  test('a cancelled sign-in stops waiting and ignores a late callback', async () => {
    const world = setup();
    const token = cancellation();
    const pending = world.account.signIn(token);
    await settled();
    const state = new URL(world.opened[0]).searchParams.get('state')!;
    token.cancel();
    await assert.rejects(pending, (error: WebsiteError) => error.code === 'cancelled');
    assert.equal(world.account.handleUri('/auth', `state=${state}&id_token=google-id-token`), false);
    assert.equal(world.secrets.size, 0);
  });

  test('accepts Google accounts only, as the website rules do', async () => {
    const world = setup();
    world.setClaims({ ...google, firebase: { sign_in_provider: 'password' } });
    await assert.rejects(signIn(world), (error: WebsiteError) => error.code === 'denied');
    assert.equal(world.secrets.size, 0);
  });

  test('renews the ID token once for concurrent callers, and signs out when Firebase drops the sign-in', async () => {
    let expired = false;
    const world = setup(url => expired && url.includes('securetoken')
      ? Response.json({ error: { code: 400, message: 'TOKEN_EXPIRED' } }, { status: 400 }) : undefined);
    const first = await signIn(world);
    assert.equal(await world.account.session(), first, 'a fresh token is reused');

    world.tick(56 * 60_000);
    const [a, b] = await Promise.all([world.account.session(), world.account.session()]);
    assert.equal(a, b);
    const renewals = world.seen.filter(request => request.url.includes('securetoken'));
    assert.equal(renewals.length, 1);
    assert.equal(renewals[0].url, `${WEBSITE.secureToken}/v1/token?key=${WEBSITE.apiKey}`);
    assert.equal(String(renewals[0].init?.body), 'grant_type=refresh_token&refresh_token=refresh-1');

    expired = true;
    world.tick(56 * 60_000);
    assert.equal(await world.account.session(), undefined);
    assert.equal(world.secrets.has(SECRET), false);
  });

  test('sign-out forgets this computer\'s sign-in', async () => {
    const world = setup();
    await signIn(world);
    assert.equal(await world.account.signOut(), true);
    assert.equal(world.secrets.size, 0);
    assert.equal(await world.account.session(), undefined);
    assert.equal(await world.account.signOut(), false);
  });

  test('the development variables point the website at this machine only', () => {
    assert.equal(websiteConfig({}), WEBSITE);
    // A website branch in front of the real project: real sign-in, local pages and /api.
    assert.deepEqual(websiteConfig({ [SITE_VARIABLE]: 'http://localhost:5173' }), { ...WEBSITE, site: 'http://localhost:5173' });
    assert.throws(() => websiteConfig({ [SITE_VARIABLE]: 'https://staging.example.com' }), (error: WebsiteError) => error.code === 'setup');
    assert.throws(() => websiteConfig({ [SITE_VARIABLE]: 'http://localhost:5173', [EMULATOR_VARIABLE]: 'http://localhost:5174' }),
      (error: WebsiteError) => error.code === 'setup', 'one or the other');
    const local = websiteConfig({ [EMULATOR_VARIABLE]: 'http://localhost:5174' });
    assert.deepEqual([local.site, local.projectId, local.identity], ['http://localhost:5174', 'demo-tensorviz', 'http://127.0.0.1:9099/identitytoolkit.googleapis.com']);
    for (const value of ['https://tensorviz.dev', 'http://192.168.1.4:5174', 'http://localhost:5174/explore/', 'not a url']) {
      assert.throws(() => websiteConfig({ [EMULATOR_VARIABLE]: value }), (error: WebsiteError) => error.code === 'setup', value);
    }
  });
});
