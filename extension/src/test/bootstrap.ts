/** Mocha's `--require` for the unit run: makes `require('vscode')` resolve to vscodeStub.js.
 *
 * The unit tests deliberately do NOT run in an extension host — that is what test:electron is
 * for. But config.ts / preflight.ts / stack.ts sit in the same module graph as `import * as
 * vscode from 'vscode'`, and Node cannot resolve that bare specifier outside VS Code. Patching
 * the resolver is the one hook that works without a bundler, a path alias, or a proxyquire
 * dependency (the extension ships zero runtime deps and keeps its dev tree small).
 *
 * The hook is deliberately narrow: only the exact specifier `vscode` is redirected; everything
 * else goes to the real resolver, so a typo in a relative import still fails loudly.
 */
export {};

// eslint-disable-next-line @typescript-eslint/no-var-requires
const Module = require('node:module') as {
  _resolveFilename(request: string, ...rest: unknown[]): string;
};

const stubPath = require.resolve('./vscodeStub.js');
const original = Module._resolveFilename;

Module._resolveFilename = function patched(
  this: unknown,
  request: string,
  ...rest: unknown[]
): string {
  if (request === 'vscode') {
    return stubPath;
  }
  return original.call(this, request, ...rest);
};
