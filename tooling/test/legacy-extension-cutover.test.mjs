import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {prepareExtensionRelease,readReleaseResult,releaseRepository} from '../release/extension-release.mjs';
import {syncGithubRelease,shouldUpdateRollingRelease} from '../../extension/scripts/sync-github-release.mjs';
import {syncOpenVsxRelease} from '../../extension/scripts/sync-openvsx-release.mjs';
const source='a'.repeat(40);
test('legacy workflow preparation rejects1.x before publication handoff',async()=>{
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'legacy-extension-guard-'));
 try{
  await fs.mkdir(path.join(root,'extension'));
  const manifest={name:'tensorviz-vscode',publisher:'tensorviz',version:'1.0.0'};
  await fs.writeFile(path.join(root,'extension/package.json'),JSON.stringify(manifest));
  await fs.writeFile(path.join(root,'extension/package-lock.json'),JSON.stringify({name:manifest.name,version:manifest.version,packages:{'':{version:manifest.version}}}));
  assert.throws(()=>prepareExtensionRelease({root,source,repository:releaseRepository,publish:true,command:(_tool,args)=>args[0]==='rev-parse'?source:''}),/unified source publisher/);
  const result=path.join(root,'result.json');await fs.writeFile(result,JSON.stringify({schema:1,repository:releaseRepository,publish:true,source,version:'1.0.0'}));
  assert.throws(()=>readReleaseResult(result,source),/unified source publisher/);
 }finally{await fs.rm(root,{recursive:true,force:true});}
});
test('legacy GitHub sync rejects1.x after only local archive metadata read',()=>{
 const calls=[];
 assert.throws(()=>syncGithubRelease({repository:releaseRepository,source,vsix:'fixture.vsix',command:(tool)=>{calls.push(tool);assert.equal(tool,'unzip');return JSON.stringify({name:'tensorviz-vscode',publisher:'tensorviz',version:'1.0.0'});}}),/unified source publisher/);
 assert.deepEqual(calls,['unzip']);
});
test('legacy rolling pointer cannot target1.x',()=>{
 let called=false;assert.throws(()=>shouldUpdateRollingRelease({repository:releaseRepository,version:'1.0.0',command:()=>{called=true;}}),/unified source publisher/);assert.equal(called,false);
});
test('legacy OpenVSX direct1.x rejects without request or command',async()=>{
 let called=false;await assert.rejects(syncOpenVsxRelease({repository:releaseRepository,version:'1.0.0',request:()=>{called=true;},command:()=>{called=true;}}),/unified source publisher/);assert.equal(called,false);
});
test('legacy OpenVSX source-only1.x allows only source metadata read, no download/publish',async()=>{
 let requests=0,commands=0;await assert.rejects(syncOpenVsxRelease({repository:releaseRepository,source,request:()=>{requests++;},command:(tool,args)=>{commands++;assert.equal(tool,'gh');assert.ok(args.some(x=>x.includes('contents/extension/package.json')));return JSON.stringify({content:Buffer.from('{"version":"1.0.0"}').toString('base64')});}}),/unified source publisher/);assert.equal(requests,0);assert.equal(commands,1);
});
