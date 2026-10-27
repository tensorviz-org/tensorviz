import * as assert from 'node:assert/strict';
import * as net from 'node:net';

import { HOST } from '../../config';
import { reserveLoopbackPort } from '../../preflight';

function close(server: net.Server): Promise<void> {
  return new Promise((resolve) => server.close(() => resolve()));
}

/** A real bind, not a mock.
 *
 * The port is what the webview's CSP, the iframe URL and the port mapping are all
 * built from, and they are decided before uvicorn has bound anything — so the
 * number this returns has to be one the OS will actually hand over, and the
 * probe socket has to be gone by the time it does. A mocked allocator would prove
 * neither. */
suite('preflight.reserveLoopbackPort', () => {
  test('returns a loopback port that is free by the time it answers', async () => {
    const port = await reserveLoopbackPort();
    assert.ok(port > 0 && port < 65536, `expected a real port, got ${port}`);

    // The reservation must release its own listener, or the child that was given
    // the number could never bind it — which is exactly the failure this test
    // exists to catch, and it is silent until uvicorn exits with EADDRINUSE.
    const server = net.createServer();
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(port, HOST, () => resolve());
    });
    await close(server);
  });

  test('does not hand out the same port twice in a row', async () => {
    // Not a guarantee the OS makes in general, but two allocations in flight
    // together must differ, and that is the case the extension can actually hit
    // (two windows opening the canvas at once).
    const [a, b] = await Promise.all([reserveLoopbackPort(), reserveLoopbackPort()]);
    assert.notEqual(a, b);
  });
});
