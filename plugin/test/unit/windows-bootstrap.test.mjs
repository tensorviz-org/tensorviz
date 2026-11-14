import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {windowsCatalog,buildBootstrap,validateWindowsPE} from '../../scripts/windows-bootstrap.mjs';
test('candidate catalog has literal native command and per-architecture pinned lock',async()=>{
 const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-win-catalog-'));
 try {
 const exe=path.join(tmp,'fake.exe'),notices=path.join(tmp,'notices');
 // Synthetic PE header only: never executed and never reported as native evidence.
 const mock=Buffer.alloc(128);mock.write('MZ');mock.writeUInt32LE(64,0x3c);mock.write('PE\0\0',64);mock.writeUInt16LE(0x8664,68);
 await fs.writeFile(exe,mock);await fs.writeFile(notices,'fixture');
 const entry={formatVersion:1,version:'0.1.43',platform:'win32-x64',pythonABI:'cp312-win_amd64',sha256:'a'.repeat(64),bytes:10,url:'https://example.com/runtime.tar.gz'};
 const out=path.join(tmp,'catalog');await windowsCatalog({destination:out,version:entry.version,lock:entry,binaries:{x64:exe},notices});
 const root=path.join(out,'plugins/tensorviz-windows-x64');
 const mcp=JSON.parse(await fs.readFile(path.join(root,'mcp.json')));
 assert.equal(mcp.mcpServers.tensorviz.command,'cmd.exe');
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

test('PE validation rejects wrong architecture, truncated headers and invalid offsets',()=>{
 const mock=Buffer.alloc(128);mock.write('MZ');mock.writeUInt32LE(64,0x3c);mock.write('PE\0\0',64);mock.writeUInt16LE(0x8664,68);
 validateWindowsPE(mock,'x64');assert.throws(()=>validateWindowsPE(mock,'arm64'),/architecture/);
 for(const n of [0,2,63,64,87]) assert.throws(()=>validateWindowsPE(mock.subarray(0,n),'x64'));
 for(const offset of [0,63,105,0xffffffff]) {const bad=Buffer.from(mock);bad.writeUInt32LE(offset,0x3c);assert.throws(()=>validateWindowsPE(bad,'x64'));}
 const bad=Buffer.from(mock);bad[64]=0;assert.throws(()=>validateWindowsPE(bad,'x64'));
 mock.writeUInt16LE(0xaa64,68);validateWindowsPE(mock,'arm64');
});
