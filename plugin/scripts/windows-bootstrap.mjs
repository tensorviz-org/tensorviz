/** Candidate only: creates a separate Windows catalog, never changes Unix export. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
const targets = {x64:'node22.23.4-win-x64',arm64:'node22.23.4-win-arm64'};
export async function buildBootstrap({launcher, pkg, output, arch}) {
  if (!targets[arch]) throw new Error('Unsupported Windows architecture');
  // Build natively so bytecode, signing and smoke tests match the target.
  if (process.platform !== 'win32' || process.arch !== arch) throw new Error('Build and test on the matching native Windows runner');
  await fs.mkdir(output,{recursive:true});
  let source = await fs.readFile(launcher,'utf8');
  const origin = "const plugin = path.resolve(__dirname, '..');";
  if (source.split(origin).length !== 2) throw new Error('Launcher entrypoint changed; review bootstrap adaptation');
  source = source.replace(origin,"const plugin = path.resolve(path.dirname(process.execPath), '..');");
  const entry = path.join(output,'bootstrap.cjs');
  await fs.writeFile(entry, source);
  const binary = path.join(output,`tensorviz-bootstrap-${arch}.exe`);
  execFileSync(process.execPath,[pkg,entry,'--target',targets[arch],'--output',binary],{stdio:'inherit'});
  return binary;
}
// Codex roots cwd, not PLUGIN_ROOT in command. cmd's fixed relative invocation
// forwards native stdio handles without PowerShell's string pipelines. /d blocks
// AutoRun commands; no user data or absolute path is interpolated into the shell.
export const windowsMcp = () => ({type:'stdio', command:'cmd.exe',
  args:['/d','/c','.\\scripts\\tensorviz-bootstrap.exe'], cwd:'.'});
export function validateWindowsPE(bytes, arch) {
  const machine = {x64:0x8664, arm64:0xaa64}[arch];
  if (!machine || bytes.length < 64 || bytes.toString('ascii',0,2) !== 'MZ') throw new Error('Invalid Windows bootstrap PE header');
  const offset=bytes.readUInt32LE(0x3c);
  if (offset < 64 || offset > bytes.length-24 || bytes.toString('binary',offset,offset+4) !== 'PE\0\0'
    || bytes.readUInt16LE(offset+4) !== machine) throw new Error(`Windows bootstrap PE architecture mismatch: ${arch}`);
}
export async function windowsCatalog({destination, version, lock, binaries, notices}) {
  if (!/^\d+\.\d+\.\d+$/.test(version) || lock.version !== version) throw new Error('Invalid/mismatched version');
  // Caller provides native-CI-approved binaries, not raw or opportunistically built files.
  const plugins=[];
  await fs.mkdir(destination,{recursive:false});
  for (const arch of ['x64','arm64']) {
    if (!binaries[arch]) continue;
    const key=`win32-${arch}`;
    const entry=lock.formatVersion===2?lock.runtimes?.[key]:lock;
    if(entry?.formatVersion!==1 || !/^[a-f0-9]{64}$/.test(entry.sha256) || !Number.isSafeInteger(entry.bytes) || entry.bytes<1 || entry.bytes>256*1024*1024 || entry?.platform!==key || entry.version!==version || entry.pythonABI!==`cp312-win_${arch==='x64'?'amd64':'arm64'}`) throw new Error('Missing matching Windows runtime');
    const url=new URL(entry.url); if(url.protocol!=='https:' || url.username || url.password || url.hash) throw new Error('Unsafe runtime URL');
    const name=`tensorviz-windows-${arch}`, root=path.join(destination,'plugins',name);
    await fs.mkdir(path.join(root,'scripts'),{recursive:true});
    await fs.mkdir(path.join(root,'.codex-plugin'));
    const bytes=await fs.readFile(binaries[arch]);
    validateWindowsPE(bytes,arch);
    await fs.writeFile(path.join(root,'scripts/tensorviz-bootstrap.exe'),bytes);
    await fs.copyFile(notices,path.join(root,'DEPENDENCY_NOTICES.md'));
    const manifest={name,version,description:`TensorViz local MCP for native Windows ${arch}. No separate Node.js installation.`,mcpServers:'./mcp.json',requires_local_executor:true};
    for(const file of ['plugin.json','.codex-plugin/plugin.json']) await fs.writeFile(path.join(root,file),JSON.stringify(manifest,null,2));
    await fs.writeFile(path.join(root,'mcp.json'),JSON.stringify({mcpServers:{tensorviz:windowsMcp()}},null,2));
    await fs.writeFile(path.join(root,'runtime-lock.json'),JSON.stringify(entry,null,2));
    await fs.writeFile(path.join(root,'bootstrap-sha256.txt'),createHash('sha256').update(bytes).digest('hex')+'\n');
    plugins.push({name,source:{source:'local',path:`./plugins/${name}`},policy:{installation:'AVAILABLE',authentication:'ON_INSTALL'},category:'Developer Tools'});
  }
  if(!plugins.length) throw new Error('No verified Windows bootstrap supplied');
  await fs.mkdir(path.join(destination,'.agents/plugins'),{recursive:true});
  await fs.writeFile(path.join(destination,'.agents/plugins/marketplace.json'),JSON.stringify({name:'tensorviz-windows-candidate',plugins},null,2));
}
