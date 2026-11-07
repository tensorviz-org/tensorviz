import { execFileSync } from 'child_process';
import path from 'path';

jest.setTimeout(30000);

it('preserves real tldraw parent discovery and bridge behavior with bounded parent checks', () => {
  // CRA cannot load tldraw's modules directly. Follow tldrawGestures.cjs and use
  // the installed Editor/store and production ShapeUtils in a Node child.
  execFileSync(process.execPath, [path.join(__dirname, '../../test/graphShapeCreationIntegration.cjs')], {
    cwd: process.cwd(), encoding: 'utf8', timeout: 25000,
    env: { ...process.env, NODE_ENV: 'test' },
  });
});
