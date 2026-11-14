/** Acceptance test: actual packaged TensorViz, not the bootstrap echo fixture.
 * Requires Windows, built bootstrap, package-native archive+lock, official Codex,
 * and a clean service-only Python. Installs into an isolated Codex home only.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {execFileSync,execFile} from 'node:child_process';
import https from 'node:https';
import {promisify} from 'node:util';
import {Client} from '@modelcontextprotocol/sdk/client/index.js';
import {StdioClientTransport} from '@modelcontextprotocol/sdk/client/stdio.js';
import {exportPublic} from '../../scripts/export-public.mjs';
import {auditIntegration} from '../../scripts/release-candidate.mjs';
if(process.platform!=='win32') throw new Error('Real Windows runner required');
const required=name=>{assert.ok(process.env[name],`Set ${name}`);return path.resolve(process.env[name])};
const lockFile=required('TENSORVIZ_PRODUCTION_RUNTIME_LOCK');
const bootstrap=required('TENSORVIZ_WINDOWS_BOOTSTRAP');
const codex=required('TENSORVIZ_TEST_CODEX');
const python=required('TENSORVIZ_SERVICE_TEST_PYTHON');
const certificate=required('TENSORVIZ_TEST_TLS_CERT'),key=required('TENSORVIZ_TEST_TLS_KEY');
const output=path.resolve('plugin/out/windows-production-mcp');await fs.mkdir(output,{recursive:true});
execFileSync(python,['-I','-c','import importlib.util; assert importlib.util.find_spec("torch") is None'],{stdio:'inherit',timeout:30000});
const lock=JSON.parse(await fs.readFile(lockFile,'utf8'));
assert.equal(lock.platform,`win32-${process.arch}`);assert.equal(lock.formatVersion,1);
const archive=await fs.readFile(lockFile.replace(/\.json$/,''));
assert.equal(archive.length,lock.bytes);assert.equal(crypto.createHash('sha256').update(archive).digest('hex'),lock.sha256);
const native=path.resolve('plugin/out/native-runtime');
const info=JSON.parse(execFileSync(path.join(native,'tensorviz-host.exe'),['--verify-native-package'],{encoding:'utf8',timeout:30000}));
assert.equal(info.sourceIncluded,false);assert.equal(info.platform,'win32');assert.equal(info.architecture,process.arch);
const release=JSON.parse(await fs.readFile(path.join(native,'release-manifest.json'),'utf8'));
assert.equal(release.coreSourceIncluded,false);assert.equal(release.version,lock.version);
assert.equal(release.platform,lock.platform);
const root=await fs.mkdtemp(path.join(os.tmpdir(),'TensorViz production 测试 space '));
let client,transport,diagnostics='',primaryError,server;let downloads=0;
const killTree=pid=>{if(pid)try{execFileSync(path.join(process.env.SystemRoot,'System32/taskkill.exe'),['/PID',String(pid),'/T','/F'],{timeout:10000,stdio:'ignore'})}catch{}};
const bounded=async(promise,label,ms=30000)=>{let timer;try{return await Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error(`${label} timed out`)),ms)})])}finally{clearTimeout(timer)}};
try{
 const cache=path.join(root,'cache'),data=path.join(root,'data'),home=path.join(root,'codex-home');
 for(const dir of [cache,data,home])await fs.mkdir(dir);
 // A fresh cache must use the real HTTPS/digest downloader, not a test shortcut.
 server=https.createServer({key:await fs.readFile(key),cert:await fs.readFile(certificate),minVersion:'TLSv1.2'},(req,res)=>{
  if(req.url!=='/runtime.tar.gz'){res.writeHead(404).end();return}downloads++;res.writeHead(200,{'content-type':'application/gzip','content-length':archive.length});res.end(archive);
 });
 await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve)});
 const testLock={...lock,url:`https://127.0.0.1:${server.address().port}/runtime.tar.gz`};
 const catalog=path.join(root,'catalog');
 const bootstrapBytes=await fs.readFile(bootstrap), notices=path.join(native,'DEPENDENCY_NOTICES.md');
 // Exercise the production exporter and full marketplace, not a candidate-only catalog.
 const receipt={formatVersion:1,arch:process.arch,version:lock.version,sourceRevision:release.sourceRevision,
  sha256:crypto.createHash('sha256').update(bootstrapBytes).digest('hex'),bytes:bootstrapBytes.length,
  noticesSha256:crypto.createHash('sha256').update(await fs.readFile(notices)).digest('hex')};
 await exportPublic(catalog,{runtimeLock:testLock,windowsBootstraps:{[process.arch]:{binary:bootstrap,notices,receipt}}});
 await auditIntegration(catalog);
 const cenv={...process.env,CODEX_HOME:home};
 const codexVersion=execFileSync(codex,['--version'],{env:cenv,encoding:'utf8',timeout:30000}).trim();
 execFileSync(codex,['plugin','marketplace','add',catalog,'--json'],{env:cenv,encoding:'utf8',timeout:30000});
 const installed=JSON.parse(execFileSync(codex,['plugin','add',`tensorviz-windows-${process.arch}@tensorviz-standalone`,'--json'],{env:cenv,encoding:'utf8',timeout:30000}));
 const bootstrapSha256=crypto.createHash('sha256').update(await fs.readFile(path.join(installed.installedPath,'scripts/tensorviz-bootstrap.exe'))).digest('hex');
 assert.equal(bootstrapSha256,crypto.createHash('sha256').update(await fs.readFile(bootstrap)).digest('hex'),'Installed bootstrap differs from release input');
 const discovered=JSON.parse(execFileSync(codex,['mcp','list','--json'],{env:cenv,encoding:'utf8',timeout:30000})).find(x=>x.name==='tensorviz');
 assert.ok(discovered?.enabled);assert.equal(discovered.transport.command,'cmd.exe');
 assert.deepEqual(discovered.transport.args,['/d','/c','.\\scripts\\tensorviz-bootstrap.exe']);
 assert.equal(path.resolve(discovered.transport.cwd),path.resolve(installed.installedPath));
 // Deliberately exclude PATH Node/Python, compiler env, project, consent and saved selections.
 const env={SystemRoot:process.env.SystemRoot,PATH:path.join(process.env.SystemRoot,'System32'),
  TEMP:root,TMP:root,USERPROFILE:root,LOCALAPPDATA:root,APPDATA:root,
  TENSORVIZ_RUNTIME_CACHE_DIR:cache,TENSORVIZ_DATA_DIR:data,NODE_EXTRA_CA_CERTS:certificate};
 const untrusted={...env};delete untrusted.NODE_EXTRA_CA_CERTS;
 await assert.rejects(promisify(execFile)(path.join(installed.installedPath,'scripts/tensorviz-bootstrap.exe'),['--prepare-runtime'],{env:untrusted,cwd:installed.installedPath,timeout:30000}),e=>e.code===1,'Untrusted HTTPS certificate must be rejected');
 assert.equal(downloads,0,'Untrusted client must not request archive');
 assert.deepEqual(await fs.readdir(cache),[],'TLS failure must not populate cache');
 transport=new StdioClientTransport({command:discovered.transport.command,args:discovered.transport.args,cwd:discovered.transport.cwd,env,stderr:'pipe'});
 transport.stderr.on('data',chunk=>{diagnostics+=chunk.toString();if(diagnostics.length>2*1024*1024)diagnostics=diagnostics.slice(-2*1024*1024)});
 client=new Client({name:'tensorviz-production-windows-acceptance',version:'1.0.0'});
 await bounded(client.connect(transport),'MCP initialize',60000);
 const serverInfo=client.getServerVersion();assert.equal(serverInfo.name,'tensorviz');assert.equal(serverInfo.version,lock.version);
 assert.equal(downloads,1,'Cold startup must download exactly once');
 assert.equal(crypto.createHash('sha256').update(await fs.readFile(path.join(cache,lock.sha256+'.tar.gz'))).digest('hex'),lock.sha256);
 const {tools}=await bounded(client.listTools(),'tools/list');
 for(const name of ['list_models','get_model_context','open_graph','open_graph_panel','get_execution_environment'])assert.ok(tools.some(t=>t.name===name),`Missing production tool ${name}`);
 for(const name of ['get_pending_permission','present_permission','get_machine_info','read_project_file','request_project_command','get_project_command'])assert.ok(!tools.some(t=>t.name===name),`Private tool leaked: ${name}`);
 assert.equal(tools.find(t=>t.name==='list_models').annotations.readOnlyHint,true);
 const listing=await bounded(client.callTool({name:'list_models',arguments:{}}),'list_models');
 assert.ok(!listing.isError,JSON.stringify(listing));assert.equal(listing.structuredContent.host.kind,'standalone');
 assert.equal(listing.structuredContent.host.version,lock.version);assert.equal(listing.structuredContent.workspaceSelectionRequired,true);
 const unopened=await bounded(client.callTool({name:'get_model_context',arguments:{modelId:'unopened',captureId:'missing',sourceId:'missing',revision:'a'.repeat(64)}}),'safe unopened model lookup');
 assert.equal(unopened.structuredContent.error.code,'NO_MODEL');
 // Real UI asset is read but not opened in a browser or published.
 const panel=tools.find(t=>t.name==='open_graph_panel');assert.ok(panel._meta?.ui?.resourceUri);
 const resource=await bounded(client.readResource({uri:panel._meta.ui.resourceUri}),'static UI resource');
 assert.ok(resource.contents.some(x=>x.mimeType==='text/html;profile=mcp-app'&&x.text?.includes('<title>Model graph</title>')));
 // Discovery must not silently consent, provision a runtime or select a workspace.
 const state=await fs.readdir(data,{recursive:true});
 assert.ok(!state.some(p=>/(^|[\\/])(permissions|workspaces|runtimes|venvs)([\\/]|$)/.test(p)),`Discovery provisioned state: ${state}`);
 await bounded(client.close(),'MCP EOF shutdown',15000);
 const sessions=(await fs.readdir(root)).filter(x=>x.startsWith('tensorviz-native-'));
 assert.deepEqual(sessions,[],'Bootstrap extracted session leaked after EOF');
 const result={platform:process.platform,arch:process.arch,version:lock.version,runtimeSha256:lock.sha256,bootstrapSha256,sourceRevision:release.sourceRevision,codexVersion,toolCount:tools.length,
  passed:['cold HTTPS archive download','untrusted TLS rejection','verified production archive checksum','real compiled host','Codex installed catalog command','MCP initialize','tools/list','list_models','NO_MODEL static lookup','static graph UI resource','no Torch service environment','no PATH Node/Python','no consent or provisioning','EOF session cleanup'],limitations:['No browser rendering or capture test','No forced termination or executable signing test','HTTPS uses temporary local trusted certificate, not production CDN']};
 await fs.writeFile(path.join(output,'result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}catch(error){primaryError=error;console.error(error);await fs.writeFile(path.join(output,'failure.txt'),String(error.stack??error));killTree(transport?.pid);throw error}
finally{
 await fs.writeFile(path.join(output,'stderr.txt'),diagnostics);
 if(primaryError)killTree(transport?.pid);
 if(server){server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
 await fs.rm(root,{recursive:true,force:true,maxRetries:1,retryDelay:100}).catch(async error=>{
  console.error('Cleanup error:',error);
  if(!primaryError){await fs.writeFile(path.join(output,'failure.txt'),String(error.stack??error));await fs.rm(path.join(output,'result.json'),{force:true});throw error}
 });
}

