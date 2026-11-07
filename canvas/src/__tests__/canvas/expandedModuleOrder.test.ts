import { execFileSync } from 'child_process';
import path from 'path';

jest.setTimeout(30000);

it('matches installed tldraw singleton ordering and bounded sibling reads', () => {
  // Like tldrawGestures.cjs, bypass CRA's incompatible tldraw transform.
  // Node's test runner exits nonzero on any failed differential assertion.
  execFileSync(process.execPath, [path.join(__dirname, '../../test/expandedModuleOrderChecks.cjs')], {
    cwd: process.cwd(), encoding: 'utf8', timeout: 25000,
    env: { ...process.env, NODE_ENV: 'test' },
  });
});

it('preserves real-store and projection-wrapper behavior with production ShapeUtils', () => {
  execFileSync(process.execPath, [path.join(__dirname, '../../test/expandedModuleOrderIntegration.cjs')], {
    cwd: process.cwd(), encoding: 'utf8', timeout: 25000,
    env: { ...process.env, NODE_ENV: 'test' },
  });
});
