const { test } = require('node:test');
const assert = require('node:assert/strict');
const { browserAccess, requestedBrowserPort } = require('../../out/host/backend.cjs');
const { launcherCommand, permissionSummary } = require('../../out/host/permissions.cjs');

const url = 'http://127.0.0.1:45791/#session=s&credential=c';

test('the revoke instruction names a launcher this installation has', () => {
  assert.equal(launcherCommand('--revoke-permission', 'linux', true), 'sh scripts/launch.sh --revoke-permission');
  assert.equal(launcherCommand('--revoke-permission', 'darwin', false), 'sh scripts/launch.sh --revoke-permission');
  assert.equal(launcherCommand('--revoke-permission', 'win32', true), '.\\scripts\\tensorviz-bootstrap.exe --revoke-permission');
  assert.equal(launcherCommand('--revoke-permission', 'win32', false), 'node scripts\\launch.cjs --revoke-permission');
  assert.match(permissionSummary, new RegExp(`Revoke it by running ${launcherCommand('--revoke-permission').replace(/[\\.]/g, '\\$&')} from`));
  if (process.platform !== 'win32') assert.doesNotMatch(permissionSummary, /launch\.cjs/);
});

test('TENSORVIZ_BROWSER_PORT accepts only an unprivileged port number', () => {
  assert.equal(requestedBrowserPort({}), undefined);
  assert.equal(requestedBrowserPort({ TENSORVIZ_BROWSER_PORT: ' ' }), undefined);
  assert.equal(requestedBrowserPort({ TENSORVIZ_BROWSER_PORT: '45791' }), 45791);
  assert.equal(requestedBrowserPort({ TENSORVIZ_BROWSER_PORT: ' 1024 ' }), 1024);
  for (const value of ['80', '65536', '4579x', '-1', '1e4', '0x50']) {
    assert.equal(requestedBrowserPort({ TENSORVIZ_BROWSER_PORT: value }), 'invalid', value);
  }
});

test('an SSH session gets forwarding guidance with the link port; a local one gets nothing', () => {
  assert.equal(browserAccess(url, undefined, {}), undefined);
  const ssh = browserAccess(url, undefined, { SSH_CONNECTION: '10.0.0.2 52144 10.0.0.9 22' });
  assert.match(ssh, /ssh -L 45791:127\.0\.0\.1:45791 <server>/);
  assert.match(ssh, /change the port in url/);
  assert.match(ssh, /within 2 minutes/);
  assert.match(ssh, /TENSORVIZ_BROWSER_PORT/);
  assert.match(browserAccess(url, undefined, { SSH_CLIENT: '10.0.0.2 52144 22' }), /ssh -L 45791/);
});

test('a port notice reaches the agent even outside SSH', () => {
  const notice = 'TENSORVIZ_BROWSER_PORT 45791 is in use or not permitted (another TensorViz session may hold it), so this view uses port 51234.';
  assert.equal(browserAccess(url, notice, {}), notice);
  assert.ok(browserAccess(url, notice, { SSH_CONNECTION: 'a b c d' }).startsWith(notice + ' TensorViz is running over SSH'));
});
