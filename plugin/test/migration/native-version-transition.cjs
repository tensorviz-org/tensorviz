// Tests an actual released binary and the new candidate, not relabeled/source fixtures.
// Invoked explicitly after native build. Never runs a model/provider turn in either CLI.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { ElicitRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
// Only reviewed immutable transitions are accepted; callers cannot substitute arbitrary old binaries.
// `canonical` marks an old catalog published under distribution/plugin-catalog/ in this repository.
const transitions = {
  '1.0.2': { fromVersion: '1.0.1', previousExport: 'fc654a7c77b1d30580438b69ab77ae88bdf9e1f7', canonical: true },
  '1.0.1': { fromVersion: '1.0.0', previousExport: '664d5615d71c10b5d9f84695c4c40011dddbdc7f', canonical: true },
  '1.0.0': { fromVersion: '0.1.48', previousExport: '8b0b539953c5abe693803c15854d953a00bcd5fe' },
  '0.1.48': { fromVersion: '0.1.47', previousExport: '9c20428f515dda38b25d1b9436b7681969e41472' },
  '0.1.47': { fromVersion: '0.1.46', previousExport: '3d311272047e649a9ba9829c6e1b21ca95fc0655' },
  '0.1.46': { fromVersion: '0.1.45', previousExport: '1b5da75b72a38c903deb7ea749eda71996284986' },
  '0.1.45': { fromVersion: '0.1.44', previousExport: '6fff2d501cbe037e2d90b252ee484a446c37bd61' },
  '0.1.43': { fromVersion: '0.3.110', previousExport: '1a3c3618082603030b5ef68b793d9834b8bb7f3d' },
  '0.1.44': { fromVersion: '0.1.43', previousExport: 'de268bb05a44d21fba51dc53b2d711ccee8a2738' },
};
const toVersion = process.env.TENSORVIZ_MIGRATION_TARGET_VERSION || '0.1.43';
assert.ok(Object.hasOwn(transitions, toVersion), 'Unreviewed migration target');
const { fromVersion, previousExport, canonical } = transitions[toVersion];
const digest = b => createHash('sha256').update(b).digest('hex');
const json = async p => JSON.parse(await fs.readFile(p, 'utf8'));
const required = name => { const value = process.env[name]; assert.ok(value && path.isAbsolute(value), `${name} must be an absolute path`); return value; };

// Test profiles never inherit credentials. Also scrub launcher URLs/tokens before
// emitting bounded diagnostics, including native-generated local launch tokens.
const redactNativeStderr = value => String(value)
  .replace(/https?:\/\/[^\s"'<>]+/g, '[URL omitted]')
  .replace(/((?:authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|token|secret|password)["']?\s*[:=]\s*["']?)(?:Bearer\s+)?[^\s"',;}]+/gi, '$1[redacted]')
  .replace(/\bBearer\s+[^\s"',;}]+/gi, 'Bearer [redacted]')
  .replace(/[A-Za-z0-9_+\/.=-]{32,}/g, '[opaque value omitted]');
const logDisk = async (directory, stage) => {
  try {
    const stat = await fs.statfs(directory);
    const availableMiB = Math.floor(Number(stat.bavail) * Number(stat.bsize) / 1048576);
    const totalMiB = Math.floor(Number(stat.blocks) * Number(stat.bsize) / 1048576);
    console.log(`Migration disk: ${stage}; available=${availableMiB} MiB total=${totalMiB} MiB`);
  } catch (error) {
    // Diagnostics must neither hide the real test failure nor turn a pass into failure.
    console.log(`Migration disk: ${stage}; unavailable (${error.code ?? 'unknown'})`);
  }
};


test('released native plugin upgrades to aligned version and rolls back in both real client caches', { timeout: 600000 }, async () => {
  const oldCatalog = required('TENSORVIZ_MIGRATION_OLD_CATALOG');
  const newCatalog = required('TENSORVIZ_MIGRATION_NEW_CATALOG');
  const oldArchive = required('TENSORVIZ_MIGRATION_OLD_ARCHIVE');
  const newArchive = required('TENSORVIZ_MIGRATION_NEW_ARCHIVE');
  const receiptPath = required('TENSORVIZ_MIGRATION_RECEIPT');
  await fs.rm(receiptPath,{force:true}); // A failed rerun must not leave a prior success receipt.
  const python = required('TENSORVIZ_TEST_PYTHON');
  const sourceRevision = process.env.TENSORVIZ_REVIEWED_SOURCE_SHA;
  assert.match(sourceRevision ?? '', /^[a-f0-9]{40}$/);
  const platform = `${process.platform}-${process.arch}`;
  assert.ok(['linux-x64', 'darwin-arm64'].includes(platform));
  // Previous release provenance is checked before executing any old artifact.
  assert.equal(execFileSync('git', ['rev-parse', 'HEAD'], {cwd:oldCatalog,encoding:'utf8'}).trim(), previousExport);
  assert.equal(execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {cwd:oldCatalog,encoding:'utf8'}).trim(), '');
  const { auditIntegration } = await import('../../scripts/release-candidate.mjs');
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorviz-native-transition-')));
  const sessions = [];
  const diagnostics = [];
  try {
    const cleanOld = path.join(root, 'audited-old');
    if (canonical) {
      // Canonical catalogs preserve the original export under a namespace.
      // Verify the complete overlay before removing its added metadata from the test copy.
      const {canonicalCatalog,catalogPrefix} = await import('../../scripts/canonical-catalog.mjs');
      const checkout = execFileSync('git',['rev-parse','--show-toplevel'],{cwd:oldCatalog,encoding:'utf8'}).trim();
      assert.equal(oldCatalog,path.join(checkout,catalogPrefix));
      const publication = await json(path.join(oldCatalog,'PUBLICATION.json'));
      const entries = Object.fromEntries(await Promise.all(Object.keys(publication.files).map(async name=>[name,await fs.readFile(path.join(oldCatalog,name))])));
      for (const [name,value] of Object.entries(canonicalCatalog(publication,entries)))
        assert.deepEqual(await fs.readFile(path.join(checkout,name)),value.bytes,`Canonical catalog differs: ${name}`);
    }
    await fs.cp(oldCatalog, cleanOld, {recursive:true,filter:source=>path.basename(source)!=='.git'
      && !(canonical&&source===path.join(oldCatalog,'CANONICAL.json'))});
    const oldPublication = await auditIntegration(cleanOld,{allowPrevious:true});
    const newPublication = await auditIntegration(newCatalog);
    assert.equal(oldPublication.version,fromVersion); assert.equal(newPublication.version,toVersion);
    assert.equal(newPublication.sourceRevision,sourceRevision);
    const artifacts=[];
    for (const [catalog,publication,archive] of [[cleanOld,oldPublication,oldArchive],[newCatalog,newPublication,newArchive]]) {
      const lock=publication.runtime.formatVersion===2?publication.runtime.runtimes[platform]:publication.runtime;
      assert.equal(lock.platform,platform); assert.equal(lock.version,publication.version);
      const bytes=await fs.readFile(archive); assert.equal(bytes.length,lock.bytes); assert.equal(digest(bytes),lock.sha256);
      const manifest=JSON.parse(execFileSync('tar',['-xOf',archive,'release-manifest.json'],{encoding:'utf8'}));
      assert.equal(manifest.version,publication.version); assert.equal(manifest.sourceRevision,publication.sourceRevision);
      assert.equal(manifest.platform,platform); assert.equal(manifest.coreSourceIncluded,false);
      artifacts.push({catalog,lock,bytes,publication});
    }
    const receipt={schemaVersion:1,fromVersion,toVersion,sourceRevision,previousExport,platform,runtimeSha256:artifacts[1].lock.sha256,clients:{}};
    for (const [name,version] of [['codex','0.165.0'],['claude','2.1.308']]) {
      await logDisk(root, `before ${name}`);
      const fixture=path.join(root,name),home=path.join(fixture,'home'),project=path.join(fixture,'project'),catalog=path.join(fixture,'catalog');
      const state=path.join(fixture,'state'),codexHome=path.join(home,'.codex'),claudeHome=path.join(home,'.claude');
      for(const dir of [home,project,codexHome,claudeHome]) await fs.mkdir(dir,{recursive:true});
      const cache=path.join(home,process.platform==='darwin'?'Library/Caches/dev.tensorviz/runtime':'.cache/dev.tensorviz/runtime');
      await fs.mkdir(cache,{recursive:true});
      for(const a of artifacts) await fs.writeFile(path.join(cache,`${a.lock.sha256}.tar.gz`),a.bytes);
      // Minimal environment: no inherited credentials, proxy settings, cloud profiles or live client state.
      const env={PATH:process.env.PATH,HOME:home,TMPDIR:root,CODEX_HOME:codexHome,CLAUDE_CONFIG_DIR:claudeHome,
        TENSORVIZ_DATA_DIR:state,TENSORVIZ_RUNTIME_CACHE_DIR:cache,CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:'1'};
      const cli=process.env[name==='codex'?'TENSORVIZ_TEST_CODEX':'TENSORVIZ_TEST_CLAUDE']||name;
      const run=args=>execFileSync(cli,name==='claude'?['--bare',...args]:args,{cwd:project,env,encoding:'utf8',timeout:45000,stdio:['ignore','pipe','pipe']});
      assert.match(run(['--version']),new RegExp(`(?:^|\\s)${version.replaceAll('.','\\.')}($|\\s|\\()`),`${name} version must be exactly pinned`);
      const settings=name==='codex'?path.join(codexHome,'config.toml'):path.join(claudeHome,'settings.json');
      await fs.writeFile(settings,name==='codex'?'# preserved migration preference\nmodel = "migration-preserved"\n':JSON.stringify({env:{TENSORVIZ_MIGRATION_SENTINEL:'preserved'}}));
      const source='from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc = nn.Linear(4, 3)\n def forward(self, x):\n  return self.fc(x)\n';
      const recipe='version = 1\n[profiles.tiny]\nkind = "model"\nmodel = "model:Tiny"\nmode = "eval"\ngrad_enabled = false\ndevice = "cpu"\ninit = { args = [], kwargs = {} }\ninputs = { args = [{kind = "tensor", shape = [2, 4], dtype = "float32", requires_grad = false, generator = {kind = "full", value = 1.0}}], kwargs = {} }\n';
      await fs.writeFile(path.join(project,'model.py'),source); await fs.writeFile(path.join(project,'tensorviz.toml'),recipe);
      const pythonBefore=await fs.stat(python);
      let previousSession;
      for(const [index,artifact] of [artifacts[0],artifacts[1],artifacts[0]].entries()) {
        // Replace only this disposable catalog. Neither published bytes nor user caches are touched.
        await fs.rm(catalog,{recursive:true,force:true}); await fs.cp(artifact.catalog,catalog,{recursive:true});
        if(index===0) run(['plugin','marketplace','add',catalog]);
        let installation;
        if(name==='codex') installation=JSON.parse(run(['plugin','add','tensorviz@tensorviz-standalone','--json']));
        else {
          if(index>0) run(['plugin','marketplace','update','tensorviz-standalone']);
          run(index===0?['plugin','install','tensorviz@tensorviz-standalone','--scope','user']:['plugin','update','tensorviz@tensorviz-standalone','--scope','user']);
        }
        const listing=JSON.parse(run(['plugin','list','--json']));
        const installed=(name==='codex'?listing.installed:listing).find(p=>(p.pluginId||p.id)==='tensorviz@tensorviz-standalone');
        assert.ok(installed?.enabled,`${name} must remain enabled`); assert.equal(installed.version,artifact.lock.version);
        let installedPath=installed.installPath||installed.installedPath||installation?.installedPath;
        // Codex list may not expose a path; its MCP registration carries the actual cache path.
        if(!installedPath&&name==='codex') {
          const servers=JSON.parse(run(['mcp','list','--json']));
          installedPath=servers.find(s=>s.transport?.env?.PLUGIN_ROOT)?.transport.env.PLUGIN_ROOT;
        }
        assert.ok(installedPath&&path.isAbsolute(installedPath),'actual client install path required; never substitute source catalog');
        installedPath=await fs.realpath(installedPath);
        const relative=path.relative(home,installedPath);
        assert.ok(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative),'installed files must be inside this disposable client home');
        assert.deepEqual(await json(path.join(installedPath,'runtime-lock.json')),artifact.publication.runtime,'installed runtime provenance must equal the audited catalog, not just its version');
        const client=new Client({name:'native-version-transition',version:'1'},{capabilities:{elicitation:{form:{}}}});
        sessions.push(client);
        client.setRequestHandler(ElicitRequestSchema,async request=>{
          assert.ok(request.params.message.includes(project)&&request.params.message.includes(python));
          assert.match(request.params.message,/inspect environment|capture/);
          return {action:'accept',content:{confirm:true}};
        });
        const transport=new StdioClientTransport({command:'/bin/sh',args:[path.join(installedPath,'scripts/launch.sh')],cwd:project,env,stderr:'pipe'});
        const diagnostic = { label: `${name}/${index}/${artifact.lock.version}`, tail: '', truncated: false };
        diagnostics.push(diagnostic);
        const retainStderr = chunk => {
          const combined = diagnostic.tail + String(chunk);
          diagnostic.truncated ||= combined.length > 4096;
          diagnostic.tail = combined.slice(-4096);
        };
        // The SDK exposes a pre-start passthrough for stderr:'pipe'. Attach before
        // connecting so startup failures are recorded as well as capture failures.
        const earlyStderr = transport.stderr;
        earlyStderr?.on('data', retainStderr);
        try { await client.connect(transport); }
        finally {
          if (!earlyStderr) transport.stderr?.on('data', retainStderr);
        }
        const call=async(name,args={})=>{const r=await client.callTool({name,arguments:args});assert.ok(!r.isError,JSON.stringify(r));return r.structuredContent;};
        assert.equal((await call('list_models')).host.version,artifact.lock.version);
        const capture=async()=>{
          await logDisk(root, `before ${name}/${index}/${artifact.lock.version} capture`);
          let result=await call('open_graph',{workspaceRoot:project,file:'model.py',cls:'Tiny',interpreter:python});
          for(let n=0;n<450&&!['ready','failed','blocked','needs_setup','cancelled'].includes(result.status);n++) {
            await new Promise(r=>setTimeout(r,100)); result=await call('open_graph',{requestId:result.requestId});
          }
          assert.equal(result.status,'ready',JSON.stringify(result));
          const ref={modelId:result.models[0].modelId,revision:result.models[0].revision};
          const graph=await call('get_graph',{...ref,mode:'graph',limit:100});
          assert.ok(graph.nodes.some(node=>JSON.stringify(node.shapes).includes('3')));
          return ref;
        };
        const ref=await capture();
        if(previousSession) {
          assert.equal((await previousSession.call('list_models')).host.version,previousSession.version);
          assert.ok((await previousSession.call('get_graph',{...previousSession.ref,mode:'graph',limit:100})).nodes.length);
          await previousSession.capture(); // Old native backend still launches workers after cache replacement.
        }
        previousSession={call,capture,ref,version:artifact.lock.version};
        const current=await fs.readFile(settings,'utf8');
        if(name==='codex') assert.match(current,/# preserved migration preference\nmodel = "migration-preserved"/);
        else assert.equal(JSON.parse(current).env.TENSORVIZ_MIGRATION_SENTINEL,'preserved');
        assert.equal(await fs.readFile(path.join(project,'model.py'),'utf8'),source);
        assert.equal(await fs.readFile(path.join(project,'tensorviz.toml'),'utf8'),recipe);
        assert.equal((await fs.stat(python)).mtimeMs,pythonBefore.mtimeMs);
      }
      // Keep every old/new host alive through all migration assertions above.
      // A failed close leaves the session tracked for the outer cleanup retry.
      for(const session of sessions) await session.close();
      sessions.length = 0;
      // This client's archives, installed caches and model capture state are no
      // longer needed. Release them before the next client's first extraction.
      await fs.rm(fixture,{recursive:true,force:true,maxRetries:5,retryDelay:100});
      await logDisk(root, `after ${name} cleanup`);
      receipt.clients[name]={version,upgrade:true,rollback:true,nativeCapture:true,preservedSettings:true};
    }
    // Atomic success receipt only; no partial receipt can satisfy the publishing gate.
    await fs.mkdir(path.dirname(receiptPath),{recursive:true});
    await fs.writeFile(`${receiptPath}.tmp`,JSON.stringify(receipt,null,2)+'\n'); await fs.rename(`${receiptPath}.tmp`,receiptPath);
    console.log(`Verified compiled migration and rollback for ${platform}: ${fromVersion} -> ${toVersion} -> ${fromVersion}`);
  } catch (error) {
    await logDisk(root, 'failure');
    for (const diagnostic of diagnostics) {
      // Drop the first partial line after truncation, avoiding a token suffix
      // whose key/prefix may have fallen outside the retained window.
      const tail = diagnostic.truncated ? (diagnostic.tail.includes('\n') ? diagnostic.tail.slice(diagnostic.tail.indexOf('\n') + 1) : '') : diagnostic.tail;
      if (tail) console.error(`Migration native stderr ${diagnostic.label}:\n${redactNativeStderr(tail).slice(-4096)}`);
    }
    throw error;
  } finally {
    for(const session of sessions) await session.close().catch(()=>{});
    await fs.rm(root,{recursive:true,force:true,maxRetries:5,retryDelay:100});
  }
});


