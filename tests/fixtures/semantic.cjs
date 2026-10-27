const { readFileSync } = require('node:fs');
const path = require('node:path');

function appliedFixture() {
  const corpus = JSON.parse(readFileSync(path.join(__dirname, 'contracts/local-browser-protocol.json'), 'utf8'));
  return corpus.valid.find(item => item.messageId === 'applied-source-checked').result.applied;
}

function canonicalFixture(attemptId = 'attempt:0') {
  const capture = JSON.parse(readFileSync(path.join(__dirname, 'contracts/semantic-capture.json'), 'utf8'));
  capture.attemptId = attemptId;
  const plan = { id: attemptId, source: structuredClone(capture.source), captureLimits: structuredClone(capture.limits) };
  return { capture, plan };
}

function automaticFixture(attemptId = 'attempt:auto') {
  const { capture: planned, plan } = canonicalFixture(attemptId);
  const capture = { ...planned, planDigest: null, preparationDigest: planned.planDigest,
    automatic: { provenance: 'seed-from-first-layer', candidateIndex: 1, candidatesTried: 1 } };
  const preparation = { id: attemptId, source: plan.source, captureLimits: plan.captureLimits,
    policy: 'legacy-input-search-v1' };
  return { capture, preparation };
}

module.exports = { appliedFixture, canonicalFixture, automaticFixture };
