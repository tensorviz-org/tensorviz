import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {lockedCLI} from '../../scripts/production-release-adapters.mjs';

async function fixture(bin, fn) {
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'locked-real-bin-test-'));
  try {
    const directory=path.join(root,'node_modules/firebase-tools');
    await fs.mkdir(path.join(directory,'lib/bin'),{recursive:true});
    await fs.writeFile(path.join(root,'package-lock.json'),JSON.stringify({packages:{'node_modules/firebase-tools':{version:'15.36.0',integrity:'sha512-test'}}}));
    await fs.writeFile(path.join(directory,'package.json'),JSON.stringify({name:'firebase-tools',version:'15.36.0',bin:{firebase:bin}}));
    await fs.writeFile(path.join(directory,'lib/bin/firebase.js'),'// fixture');
    return await fn({root,directory});
  } finally {await fs.rm(root,{recursive:true,force:true});}
}

test('the locked Firebase package retains its valid ./ bin prefix',async()=>{
  await fixture('./lib/bin/firebase.js',async({root,directory})=>{
    assert.deepEqual(await lockedCLI(root,'firebase-tools','15.36.0'),[process.execPath,await fs.realpath(path.join(directory,'lib/bin/firebase.js'))]);
  });
});

for(const bin of ['../firebase.js','./../firebase.js','./lib/../bin/firebase.js','././lib/bin/firebase.js','/tmp/firebase.js','lib//bin/firebase.js','lib\\bin\\firebase.js','https://example.com/cli.js']) {
  test(`locked CLI rejects unsafe entrypoint ${JSON.stringify(bin)}`,async()=>{
    await fixture(bin,async({root})=>assert.rejects(lockedCLI(root,'firebase-tools','15.36.0'),/Invalid release CLI entrypoint/));
  });
}

test('a valid relative bin cannot escape through a symbolic link',async()=>{
  await fixture('./lib/bin/firebase.js',async({root,directory})=>{
    const outside=path.join(root,'outside.js');await fs.writeFile(outside,'// outside');
    await fs.unlink(path.join(directory,'lib/bin/firebase.js'));
    await fs.symlink(outside,path.join(directory,'lib/bin/firebase.js'));
    await assert.rejects(lockedCLI(root,'firebase-tools','15.36.0'),/CLI entrypoint escaped package/);
  });
});
