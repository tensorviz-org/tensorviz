import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';

if (!process.env.TENSORVIZ_QA_PAGE_FILE || !existsSync(process.env.TENSORVIZ_QA_PAGE_FILE)) {
  console.error('Set TENSORVIZ_QA_PAGE_FILE to the running disposable installed host’s control/page.json.');
  process.exit(1);
}
const specs = process.argv.slice(2);
const result = spawnSync('npx', ['playwright', 'test', ...(specs.length ? specs : ['tests/e2e/local-browser-installed.spec.ts'])], {
  stdio: 'inherit',
  env: { ...process.env, E2E_DISABLE_WEBSERVER: '1' },
});
process.exit(result.status ?? 1);
