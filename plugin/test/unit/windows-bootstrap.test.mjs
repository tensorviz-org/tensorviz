import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {windowsCatalog,buildBootstrap} from '../../scripts/windows-bootstrap.mjs';
test('candidate catalog has literal native command and per-architecture pinned lock',async()=>{
 const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-win-catalog-'));
 try {
 const exe=path.join(tmp,'fake.exe'),notices=path.join(tmp,'notices');
 await fs.writeFile(exe,'MZfixture-not-executable');await fs.writeFile(notices,'fixture');
 const entry={formatVersion:1,version:'0.1.43',platform:'win32-x64',pythonABI:'cp312-win_amd64',sha256:'a'.repeat(64),bytes:10,url:'https://example.com/runtime.tar.gz'};
 const out=path.join(tmp,'catalog');await windowsCatalog({destination:out,version:entry.version,lock:entry,binaries:{x64:exe},notices});
 const root=path.join(out,'plugins/tensorviz-windows-x64');
 const mcp=JSON.parse(await fs.readFile(path.join(root,'mcp.json')));
 assert.equal(mcp.mcpServers.tensorviz.command,'powershell.exe');
 assert.equal(mcp.mcpServers.tensorviz.cwd,'.');
 assert.ok(!JSON.stringify(mcp).includes('PLUGIN_ROOT'));
 assert.ok(!JSON.stringify(mcp).includes('ExecutionPolicy'));
 assert.deepEqual(JSON.parse(await fs.readFile(path.join(root,'runtime-lock.json'))),entry);
 assert.equal(JSON.parse(await fs.readFile(path.join(root,'.codex-plugin/plugin.json'))).mcpServers,'./mcp.json');
 }finally{await fs.rm(tmp,{recursive:true,force:true})}
});
test('refuses cross-compilation masquerading as native verification',{skip:process.platform==='win32'},async()=>{
 await assert.rejects(buildBootstrap({arch:'x64'}),/matching native Windows/);
});
