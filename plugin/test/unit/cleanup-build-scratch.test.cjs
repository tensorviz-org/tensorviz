const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
test('cleanup removes only completed build intermediates, preserving all acceptance and release inputs',async t=>{
 const {cleanupBuildScratch,disposableBuildPaths}=await import('../../scripts/cleanup-build-scratch.mjs');
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'build-scratch-test-'));
 t.after(()=>fs.rm(root,{recursive:true,force:true}));
 const retained=['plugin/out/test-venv','plugin/out/native-runtime','plugin/out/audit-clients','plugin/out/integration','plugin/dist','plugin/node_modules','backend','canvas/src'];
 for(const p of [...disposableBuildPaths,...retained]) {await fs.mkdir(path.join(root,p),{recursive:true});await fs.writeFile(path.join(root,p,'sentinel'),'retain');}
 const result=await cleanupBuildScratch(root);
 assert.deepEqual(result.removed,disposableBuildPaths);assert.ok(Number.isFinite(result.afterBytes));
 for(const p of disposableBuildPaths) await assert.rejects(fs.stat(path.join(root,p)),{code:'ENOENT'});
 for(const p of retained) assert.equal(await fs.readFile(path.join(root,p,'sentinel'),'utf8'),'retain');
 await cleanupBuildScratch(root); // Missing intermediates are safe on retries.
});
test('cleanup never follows a build-output symlink into unrelated data',async t=>{
 const {cleanupBuildScratch}=await import('../../scripts/cleanup-build-scratch.mjs');
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'build-scratch-link-'));
 t.after(()=>fs.rm(root,{recursive:true,force:true}));
 await fs.mkdir(path.join(root,'retained'));await fs.writeFile(path.join(root,'retained/sentinel'),'keep');
 await fs.mkdir(path.join(root,'plugin/out'),{recursive:true});await fs.symlink(path.join(root,'retained'),path.join(root,'plugin/out/native-backend'));
 await cleanupBuildScratch(root);assert.equal(await fs.readFile(path.join(root,'retained/sentinel'),'utf8'),'keep');
});
test('both native workflows reclaim scratch only after packaged-runtime acceptance',async()=>{
 for(const name of ['release-plugin','release-checks']) {
  const text=await fs.readFile(path.resolve(__dirname,`../../../.github/workflows/${name}.yml`),'utf8');
  const cleanup=text.indexOf('run: node plugin/scripts/cleanup-build-scratch.mjs');
  const acceptance=text.indexOf('npm run test:release --prefix plugin');
  const migration=text.indexOf('run: node --test plugin/test/migration/native-version-transition.cjs');
  assert.ok(acceptance>=0&&cleanup>acceptance&&migration>cleanup,name);
  assert.match(text,/Retain verified migration receipt/);
 }
});
