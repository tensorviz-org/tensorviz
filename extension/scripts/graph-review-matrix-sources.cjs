// The Python matrix uses the production revision reader, with only VS Code's
// unavailable process API stubbed by the existing unit bootstrap.
const fs = require('node:fs/promises');
const { gitReviewSourceSet, changedReviewFiles } = require('@tensorviz/graph-host/gitReviewSources');
(async () => {
  const cases = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
  for (const [index, item] of cases.entries()) {
    const started = performance.now();
    item.before = await gitReviewSourceSet(item.repo, 'model.py', item.base);
    item.after = await gitReviewSourceSet(item.repo, 'model.py', item.head);
    item.gitReadMs = Math.round(performance.now() - started);
    item.changedFiles = changedReviewFiles(item.before, item.after);
    if (item.before.truncated || item.after.truncated) throw new Error(`Unexpected truncation: ${item.id}`);
    if ((index + 1) % 100 === 0) process.stderr.write(`Read Git sources ${index + 1}/${cases.length}\n`);
  }
  await fs.writeFile(process.argv[3], JSON.stringify(cases));
})().catch(error => { console.error(error); process.exitCode = 1; });
