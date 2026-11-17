import test from 'node:test';
import assert from 'node:assert/strict';
import {ensureUnifiedReleaseTag} from '../../scripts/ensure-unified-release-tag.mjs';
const source = 'a'.repeat(40), identity = {repository: 'tensorviz-org/tensorviz', version: '1.0.0', source};
const tag = {ref: 'refs/tags/v1.0.0', object: {type: 'commit', sha: source}};
test('only a missing tag permits one exact lightweight ref creation', () => {
  const calls = [];
  const result = ensureUnifiedReleaseTag(identity, (...args) => { calls.push(args); return calls.length === 1 ? null : tag; });
  assert.equal(result.created, true);
  assert.deepEqual(calls, [['GET', 'repos/tensorviz-org/tensorviz/git/ref/tags/v1.0.0'], ['POST', 'repos/tensorviz-org/tensorviz/git/refs', {ref: tag.ref, sha: source}]]);
});
test('matching lightweight tag is verified without a write', () => {
  let count = 0;
  assert.equal(ensureUnifiedReleaseTag(identity, method => { count++; assert.equal(method, 'GET'); return tag; }).created, false);
  assert.equal(count, 1);
});
test('moved, annotated, unexpected and malformed tags fail without writes', () => {
  for (const value of [{...tag, object: {type: 'commit', sha: 'b'.repeat(40)}}, {...tag, object: {type: 'tag', sha: source}}, {...tag, ref: 'refs/tags/other'}, undefined, {}]) {
    let count = 0;
    assert.throws(() => ensureUnifiedReleaseTag(identity, method => { count++; assert.equal(method, 'GET'); return value; }), /Existing release tag/);
    assert.equal(count, 1);
  }
});
test('invalid identities and API failures never trigger another API mutation', () => {
  for (const bad of [{repository: 'other/repo'}, {version: '1.0.0;echo bad'}, {source: 'main'}]) assert.throws(() => ensureUnifiedReleaseTag({...identity, ...bad}, () => assert.fail('must not call API')), /identity required/);
  let count = 0;
  assert.throws(() => ensureUnifiedReleaseTag(identity, () => { count++; throw Error('network failure'); }), /network failure/);
  assert.equal(count, 1);
});
