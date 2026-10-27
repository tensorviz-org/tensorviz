import * as assert from 'node:assert/strict';

import {
  BackendLaunch,
  backendEnv,
  backendOrigin,
  backendReload,
  devServerUrl,
  fixedBackendPort,
  withLaunchToken,
} from '../../config';

const LAUNCH: BackendLaunch = {
  port: 51234,
  staticDir: '/ext/media/app',
  token: 'deadbeef',
  stateDbPath: '/ws/.tensorviz/state.sqlite3',
};

/** Why these assertions exist: every value here is injected precisely because it must BEAT the
 *  user's own .env. server_v2.py calls load_dotenv with override=False, so a value already in
 *  the process environment wins. If injection ever stopped overriding, the extension would
 *  launch a backend configured by whatever the checkout's .env happens to say — auth on, the
 *  static mount off, no token — and the canvas would fail at request time, not at startup. */
suite('config.backendEnv', () => {
  test('keeps source identity separate from private state and overrides inherited identity', () => {
    const env = backendEnv({ TENSORVIZ_WORKSPACE_DIR: '/other/workspace' }, {
      ...LAUNCH, workspaceDir: '/source/model', stateDbPath: '/vscode/storage/.tensorviz/state.sqlite3',
    });
    assert.equal(env.TENSORVIZ_WORKSPACE_DIR, '/source/model');
    assert.equal(env.TENSORVIZ_ALLOWED_ROOTS, '/source/model');
    assert.equal(env.USER_HISTORY_DB_PATH, '/vscode/storage/.tensorviz/state.sqlite3');
    assert.equal(backendEnv({ TENSORVIZ_WORKSPACE_DIR: '/other/workspace' }, LAUNCH).TENSORVIZ_WORKSPACE_DIR, '');
  });
  test('forces the values a user .env must not be able to change', () => {
    const env = backendEnv(
      {
        AUTH_DISABLED: 'false',
        CORS_ORIGINS: 'https://app.example.com',
        TENSORVIZ_STATIC_DIR: '/somewhere/else',
        TENSORVIZ_LOCAL_TOKEN: 'stale',
        USER_HISTORY_DB_PATH: '/shared/dev.sqlite3',
        KMP_DUPLICATE_LIB_OK: 'FALSE',
        PYTHONUNBUFFERED: '0',
      },
      LAUNCH,
    );

    // auth.py accepts 1|true|yes|on; the panel has no way to sign in, so this must be on —
    // which is exactly why the launch token below is not optional.
    assert.equal(env.AUTH_DISABLED, 'true');
    // Same-origin in Stage A, so this widens nothing; it exists so a production origin in a
    // user .env cannot widen what a loopback, auth-disabled process accepts.
    assert.equal(env.CORS_ORIGINS, backendOrigin(LAUNCH.port));
    // The app the panel iframes. Without it /health reports capabilities.static=false and the
    // extension refuses to open a panel at all.
    assert.equal(env.TENSORVIZ_STATIC_DIR, LAUNCH.staticDir);
    // The per-launch secret. A stale one from the environment would 401 every request.
    assert.equal(env.TENSORVIZ_LOCAL_TOKEN, LAUNCH.token);
    // The AI harness's store, inside the workspace rather than the shared dev database.
    assert.equal(env.USER_HISTORY_DB_PATH, LAUNCH.stateDbPath);
    // Without it torch + MKL aborts on some machines.
    assert.equal(env.KMP_DUPLICATE_LIB_OK, 'TRUE');
    // Without it a crash's traceback sits in a 4 KB pipe buffer and the output channel is empty.
    assert.equal(env.PYTHONUNBUFFERED, '1');
  });

  test('carries the launch port to both places that read one', () => {
    const env = backendEnv({}, LAUNCH);
    assert.equal(env.BACKEND_PORT, String(LAUNCH.port));
    assert.equal(env.CORS_ORIGINS, `http://127.0.0.1:${LAUNCH.port}`);
  });

  test('passes everything else through untouched', () => {
    const env = backendEnv({ PATH: '/usr/bin', OPENAI_API_KEY: 'sk-test' }, LAUNCH);
    assert.equal(env.PATH, '/usr/bin');
    assert.equal(env.OPENAI_API_KEY, 'sk-test');
  });

  test('does not arm the worker parent-death pipe on the backend process', () => {
    const env = backendEnv({ TENSORVIZ_PARENT_PIPE: '1', PATH: '/usr/bin' }, LAUNCH);
    assert.equal(env.TENSORVIZ_PARENT_PIPE, undefined);
    assert.equal(env.PATH, '/usr/bin');
  });
});

/** Why these exist: ide/dev.sh depends on three environment variables lining up, and a
 *  misread of any one of them fails at request time with a symptom that points elsewhere —
 *  a frame at the wrong URL looks like a blank panel, a proxy at the wrong port looks like a
 *  dead backend, a missing token looks like the same dead backend. */
suite('config: the dev loop overrides', () => {
  test('devServerUrl prefers the setting, falls back to the environment, else empty', () => {
    assert.equal(devServerUrl('  http://a/ ', { TENSORVIZ_DEV_SERVER_URL: 'http://b/' }), 'http://a/');
    assert.equal(devServerUrl('', { TENSORVIZ_DEV_SERVER_URL: ' http://b/ ' }), 'http://b/');
    assert.equal(devServerUrl('', {}), '');
  });

  test('withLaunchToken adds the token once, with the right separator', () => {
    assert.equal(withLaunchToken('http://h/', 'a b'), 'http://h/?token=a%20b');
    assert.equal(withLaunchToken('http://h/?x=1', 't'), 'http://h/?x=1&token=t');
    // Already there: the caller's token is the one the backend was started with.
    assert.equal(withLaunchToken('http://h/?token=x', 't'), 'http://h/?token=x');
    assert.equal(withLaunchToken('http://h/?x=1&token=x', 't'), 'http://h/?x=1&token=x');
  });

  test('fixedBackendPort is undefined when unset and refuses a non-port', () => {
    assert.equal(fixedBackendPort({}), undefined);
    assert.equal(fixedBackendPort({ TENSORVIZ_BACKEND_PORT: '' }), undefined);
    assert.equal(fixedBackendPort({ TENSORVIZ_BACKEND_PORT: ' 8765 ' }), 8765);
    // Silently allocating a random port here would leave the CRA proxy forwarding into
    // nothing, so a bad value must stop the launch and say why.
    assert.throws(() => fixedBackendPort({ TENSORVIZ_BACKEND_PORT: 'eight' }), /TENSORVIZ_BACKEND_PORT/);
    assert.throws(() => fixedBackendPort({ TENSORVIZ_BACKEND_PORT: '70000' }), /TENSORVIZ_BACKEND_PORT/);
  });

  test('backendReload accepts the same spellings auth.py does', () => {
    for (const on of ['1', 'true', 'YES', ' on ']) {
      assert.equal(backendReload({ TENSORVIZ_BACKEND_RELOAD: on }), true, on);
    }
    for (const off of [undefined, '', '0', 'false', 'reload']) {
      assert.equal(backendReload({ TENSORVIZ_BACKEND_RELOAD: off }), false, String(off));
    }
  });
});
