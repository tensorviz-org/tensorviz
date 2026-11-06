import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { annotatePublicFirebaseConfig } from '../../extension/scripts/public-firebase-config.mjs';

const key = `AIza${'0'.repeat(35)}`;
const source = `export const WEBSITE = {\n  apiKey: '${key}',\n};`;
const bundle = `const WEBSITE = {\n  apiKey: "${key}",\n  other: "unrelated-credential",\n};\nWEBSITE.apiKey;`;

test('annotates only the declared public identifier without changing its runtime value', () => {
  const result = annotatePublicFirebaseConfig(bundle, source);
  assert.equal(runInNewContext(result), runInNewContext(bundle));
  assert.equal(result.split('\n').filter(line => line.includes('secret-detector:ignore')).length, 1);
  assert.match(result.split('\n').find(line => line.includes(key)), /secret-detector:ignore/);
  assert.ok(result.includes('  other: "unrelated-credential",\n'), 'Other credentials receive no suppression.');
});

test('refuses missing, duplicated or combined lines instead of broadening suppression', () => {
  for (const input of ['', bundle + `\n"${key}";`, bundle.replace(`"${key}",`, `"${key}", other: "private"` )]) {
    assert.throws(() => annotatePublicFirebaseConfig(input, source), /exactly one standalone/);
  }
  assert.throws(() => annotatePublicFirebaseConfig(bundle, 'const unrelated = {};'), /Expected the website/);
});

test('the plugin publishes to the same website as the extension', async () => {
  const { readFileSync } = await import('node:fs');
  const config = file => {
    const source = readFileSync(new URL(`../../${file}`, import.meta.url), 'utf8');
    const block = source.match(/^export const WEBSITE: WebsiteConfig = \{\n([\s\S]*?)\n\};$/m)?.[1];
    assert.ok(block, `${file} declares the website's public configuration`);
    return Object.fromEntries([...block.matchAll(/^  (\w+): '([^']+)',$/gm)].map(([, key, value]) => [key, value]));
  };
  const extension = config('extension/src/websiteAccount.ts');
  assert.deepEqual(Object.keys(extension).sort(), ['apiKey', 'identity', 'projectId', 'secureToken', 'site']);
  assert.deepEqual(config('plugin/src/website.ts'), extension);
});
