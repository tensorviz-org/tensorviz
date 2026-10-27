const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

for (const surface of ['browser', 'panel']) test(`the built standalone ${surface} excludes hosted authentication and analytics dependencies`, () => {
  const stats = JSON.parse(fs.readFileSync(path.resolve(__dirname, `../../out/${surface}-modules.json`), 'utf8'));
  const resources = [];
  const visit = modules => { for (const module of modules ?? []) {
    if (module.nameForCondition) resources.push(module.nameForCondition);
    visit(module.modules); visit(module.children);
  } };
  visit(stats.modules);
  assert.ok(resources.length > 100, 'Dependency inventory must include cached and concatenated modules');
  assert.ok(resources.some(file => file.endsWith('/ide/LocalBrowserApp.tsx')));
  assert.deepEqual(resources.filter(file => /\/(?:@firebase|firebase|posthog-js)\//.test(file)
    || /\/(?:lib\/firebase|contexts\/AuthContext|services\/(?:authService|graphStorage|projectsApi|verificationStore))\./.test(file)), []);
  assert.deepEqual(resources.filter(file => /\/hosted\//.test(file)), []);
});
