import { lstatSync } from 'node:fs';
import * as path from 'node:path';

const PROJECT_MARKERS = ['tensorviz.toml', 'pyproject.toml', 'setup.py', 'setup.cfg', '.git'];

/** A VS Code folder can contain several independent Python projects. Keep model
 * paths, local imports, setup files and source budgets relative to the selected
 * project's boundary, without ever searching above the owning workspace. */
export function modelProjectRoot(file: string, workspace?: string): string {
  const directory = path.dirname(file);
  if (!workspace) return directory;
  const relative = path.relative(workspace, file);
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error('The model is outside its owning workspace.');
  }
  for (let current = directory; current !== workspace; current = path.dirname(current)) {
    for (const marker of PROJECT_MARKERS) {
      try {
        // Even a malformed or linked config establishes the boundary: setup
        // validation must report it, rather than bypass it at an outer root.
        lstatSync(path.join(current, marker));
        return current;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      }
    }
    // A checkout named "my-model" or "my model" cannot be a dotted Python
    // module prefix. It is also a boundary for projects without metadata.
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(path.basename(current))) return current;
  }
  return workspace;
}
