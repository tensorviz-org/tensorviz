import * as assert from 'node:assert/strict';
import { BACKEND_RESTART_LIMIT, planBackendExit } from '../../backendExit';

suite('backend exit', () => {
  test('names the signal when the process was killed, and the code otherwise', () => {
    assert.equal(planBackendExit(null, 'SIGKILL', 0).notice, 'The tensorViz backend exited unexpectedly (signal SIGKILL).');
    assert.equal(planBackendExit(4, null, 0).notice, 'The tensorViz backend exited unexpectedly (code 4).');
    assert.equal(planBackendExit(null, null, 0).notice, 'The tensorViz backend exited unexpectedly (code null).');
  });

  test('restarts only until the limit', () => {
    assert.equal(planBackendExit(null, 'SIGTERM', 0).restart, true);
    assert.equal(planBackendExit(null, 'SIGTERM', BACKEND_RESTART_LIMIT - 1).restart, true);
    assert.equal(planBackendExit(null, 'SIGTERM', BACKEND_RESTART_LIMIT).restart, false);
  });
});
