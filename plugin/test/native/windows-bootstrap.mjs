// Real Windows binaries and PowerShell stdio. No runtime verification bypasses.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
import assert from 'node:assert/strict';
import {spawn,execFileSync} from 'node:child_process';
import {buildBootstrap,windowsCatalog,windowsMcp} from '../../scripts/windows-bootstrap.mjs';
if(process.platform!=='win32') throw new Error('Native Windows execution required; no emulated pass');
const repo=process.cwd(),out=path.join(repo,'plugin/out/windows-bootstrap');
await fs.mkdir(out,{recursive:true});
const pkg=path.join(repo,'.bootstrap-build/node_modules/@yao-pkg/pkg/lib-es5/bin.js');
const binary=await buildBootstrap({launcher:path.join(repo,'plugin/integration/launch.cjs'),pkg,output:out,arch:process.arch});
function checkPE(bytes){
 assert.equal(bytes.toString('ascii',0,2),'MZ');const pe=bytes.readUInt32LE(0x3c);
 assert.equal(bytes.toString('binary',pe,pe+4),'PE\0\0');
 assert.equal(bytes.readUInt16LE(pe+4),process.arch==='arm64'?0xaa64:0x8664);
}
checkPE(await fs.readFile(binary));
const fixture=path.join(out,'fixture.cjs'), host=path.join(out,'fixture.exe');
await fs.writeFile(fixture,`'use strict';
console.error(JSON.stringify({pid:process.pid,exe:process.execPath}));
let text='';process.stdin.setEncoding('utf8');process.stdin.on('data',x=>text+=x);
process.stdin.on('end',()=>{const p=JSON.parse(text);process.stdout.write(JSON.stringify({echo:p.text})+'\\n',()=>process.exit(p.exit));});
`);
execFileSync(process.execPath,[pkg,fixture,'--target',`node22.23.4-win-${process.arch}`,'--output',host],{stdio:'inherit'});
const hostBytes=await fs.readFile(host);checkPE(hostBytes);
function tar(name,data){
 const h=Buffer.alloc(512);h.write(name);h.write('0000700\0',100);h.write(data.length.toString(8).padStart(11,'0')+'\0',124);h.fill(32,148,156);h.write('0',156);h.write('ustar\0',257);h.write('00',263);
 h.write(h.reduce((n,v)=>n+v,0).toString(8).padStart(6,'0')+'\0 ',148);
 return zlib.gzipSync(Buffer.concat([h,data,Buffer.alloc((512-data.length%512)%512),Buffer.alloc(1024)]));
}
const archive=tar('tensorviz-host.exe',hostBytes),sha256=crypto.createHash('sha256').update(archive).digest('hex');
const lock={formatVersion:1,platform:`win32-${process.arch}`,version:'0.1.44',pythonABI:`cp312-win_${process.arch==='x64'?'amd64':'arm64'}`,sha256,bytes:archive.length,url:'https://example.invalid/never-downloaded.tar.gz'};
const work=await fs.mkdtemp(path.join(os.tmpdir(),'TensorViz 测试 space ')),cache=path.join(work,'cache');await fs.mkdir(cache);
await fs.writeFile(path.join(cache,sha256+'.tar.gz'),archive);
const notices=path.join(out,'NOTICES-test-only.md');await fs.writeFile(notices,'CI fixture, not a distributable release.');
const catalog=path.join(work,'catalog');await windowsCatalog({destination:catalog,version:lock.version,lock,binaries:{[process.arch]:binary},notices});
const root=path.join(catalog,'plugins',`tensorviz-windows-${process.arch}`),mcp=windowsMcp();
const env={...process.env,TENSORVIZ_RUNTIME_CACHE_DIR:cache,PATH:process.env.SystemRoot+'\\System32;'+process.env.SystemRoot+'\\System32\\WindowsPowerShell\\v1.0'};
function run(payload){return new Promise((resolve,reject)=>{
 const child=spawn(mcp.command,mcp.args,{cwd:root,env,windowsHide:true,stdio:'pipe'});let stdout='',stderr='';
 const timer=setTimeout(()=>{child.kill();reject(new Error('PowerShell bootstrap timeout (EOF not forwarded?)'));},60000);
 child.stdout.setEncoding('utf8').on('data',s=>stdout+=s);child.stderr.setEncoding('utf8').on('data',s=>stderr+=s);
 child.on('error',e=>{clearTimeout(timer);reject(e)});child.on('close',code=>{clearTimeout(timer);resolve({code,stdout,stderr})});child.stdin.on('error',()=>{});child.stdin.end(JSON.stringify(payload));
});}
try{
 for(const payload of [{text:'héllo हिन्दी 中文 🦙',exit:0},{text:'漢🦙'.repeat(300000),exit:0},{text:'nonzero',exit:17}]){
 const r=await run(payload);assert.equal(r.code,payload.exit,r.stderr);assert.equal(JSON.parse(r.stdout).echo,payload.text);
 const state=r.stderr.split(/\r?\n/).map(x=>{try{return JSON.parse(x)}catch{return null}}).find(x=>x?.exe);
 assert.ok(state,'fixture did not report native exe');await assert.rejects(fs.access(path.dirname(state.exe)),{code:'ENOENT'});
 }
 // Same-size cache mutation must fail BEFORE fixture execution; URL intentionally cannot resolve.
 const corrupt=Buffer.from(archive);corrupt[corrupt.length-1]^=1;await fs.writeFile(path.join(cache,sha256+'.tar.gz'),corrupt);
 let r=await run({text:'must not run',exit:0});assert.notEqual(r.code,0);assert.match(r.stderr,/checksum mismatch/);assert.equal(r.stdout,'');
 await fs.writeFile(path.join(cache,sha256+'.tar.gz'),archive.subarray(0,-1));
 r=await run({text:'must not run',exit:0});assert.notEqual(r.code,0);assert.match(r.stderr,/size mismatch/);assert.equal(r.stdout,'');
 await fs.writeFile(path.join(cache,sha256+'.tar.gz'),archive);
 // Optional official installed Codex CLI discovery, with isolated home and no login/model request.
 if(process.env.TENSORVIZ_TEST_CODEX){
  const codex=process.env.TENSORVIZ_TEST_CODEX,cenv={...process.env,CODEX_HOME:path.join(work,'codex-home')};await fs.mkdir(cenv.CODEX_HOME);
  for(const args of [['--version'],['plugin','marketplace','add',catalog,'--json'],['plugin','add',`tensorviz-windows-${process.arch}@tensorviz-windows-candidate`,'--json'],['mcp','list','--json']]){
   const result=execFileSync(codex,args,{env:cenv,encoding:'utf8'});console.log(result);
   if(args[0]==='mcp'){const row=JSON.parse(result).find(x=>x.name==='tensorviz');assert.ok(row?.enabled);assert.equal(row.transport.command,'powershell.exe');assert.ok(path.isAbsolute(row.transport.cwd));}
  }
 }
 await fs.writeFile(path.join(out,'result.json'),JSON.stringify({platform:process.platform,arch:process.arch,bootstrapBytes:(await fs.stat(binary)).size,passed:['PE architecture','no Node/Python PATH','PowerShell Unicode stdio','large JSON','EOF','nonzero exit','session cleanup','corrupt digest','truncated cache'],codexDiscovery:!!process.env.TENSORVIZ_TEST_CODEX},null,2));
 console.log('Native Windows bootstrap fixtures PASS. This is not the production TensorViz MCP host.');
}finally{await fs.rm(work,{recursive:true,force:true,maxRetries:5,retryDelay:200})}
