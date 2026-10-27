#!/usr/bin/env node
/**
 * Is what the dev host would run actually what the repository says?
 *
 * Build outputs feed one launch, and they go stale independently:
 *
 *   extension/backend/   <- backend/        (stage:backend)
 *   extension/media/app/ <- canvas/src/    (build:webview)
 *   extension/out/       <- extension/src/      (compile, or the watcher)
 *   packages/<name>/dist/ <- packages/<name>/src/ (build:shared, before compile)
 *
 * A stale one does not fail, warn, or look any different. It just runs yesterday's
 * code — which cost an afternoon: backend fixes were written, committed and
 * reported as ready while the dev host went on serving a copy staged four hours
 * earlier, and every screenshot showed the defect still there.
 *
 * `npm run check:staged` says which of the three is behind, and `npm run dev:build`
 * makes all three current.
 */
import { execSync } from 'node:child_process';
import { existsSync, statSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

/** The newest mtime under a directory, for the files that matter. */
function newest(dir, match) {
  let latest = 0;
  const walk = (current) => {
    let entries;
    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.name === 'node_modules' || entry.name === '__pycache__') continue;
      const path = join(current, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (match.test(entry.name)) latest = Math.max(latest, statSync(path).mtimeMs);
    }
  };
  walk(dir);
  return latest;
}

const checks = [
  ...['contracts', 'graph-host'].map(name => ({
    what: `shared ${name} (build:shared)`,
    source: join(root, 'packages', name, 'src'),
    built: join(root, 'packages', name, 'dist'),
    match: /\.ts$/,
  })),
  {
    what: 'backend   (stage:backend)',
    source: join(root, 'backend'),
    built: join(root, 'extension', 'backend'),
    match: /\.py$/,
  },
  {
    what: 'webview   (build:webview)',
    source: join(root, 'canvas', 'src'),
    built: join(root, 'extension', 'media', 'app'),
    match: /\.(tsx?|css)$/,
  },
  {
    what: 'extension (compile)',
    source: join(root, 'extension', 'src'),
    built: join(root, 'extension', 'out'),
    match: /\.ts$/,
  },
];

let stale = 0;
for (const check of checks) {
  if (!existsSync(check.built)) {
    console.log(`  MISSING  ${check.what} — never built`);
    stale += 1;
    continue;
  }
  const sourceAt = newest(check.source, check.match);
  const builtAt = newest(check.built, /\.(py|js|html|css)$/);
  if (sourceAt > builtAt) {
    const behind = Math.round((sourceAt - builtAt) / 60000);
    console.log(`  STALE    ${check.what} — ${behind} min behind its source`);
    stale += 1;
  } else {
    console.log(`  ok       ${check.what}`);
  }
}

if (stale > 0) {
  console.log(`\n${stale} of ${checks.length} build outputs are behind. Run: npm run dev:build --prefix extension`);
  process.exit(1);
}
console.log('\nWhat the dev host runs is what the repository says.');
