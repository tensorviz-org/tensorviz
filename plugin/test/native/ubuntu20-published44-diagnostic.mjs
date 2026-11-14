/** Diagnostic only: published 0.1.44 transport on Ubuntu 20.04. Never release evidence. */
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
const sourceFile = new URL('./unix-production-mcp.mjs', import.meta.url);
const generated = new URL('./.ubuntu20-diagnostic.generated.mjs', import.meta.url);
assert.equal(process.platform, 'linux'); assert.equal(process.arch, 'x64');
assert.equal(execFileSync('getconf',['GNU_LIBC_VERSION'],{encoding:'utf8'}).trim(), 'glibc 2.31');
const source = await fs.readFile(sourceFile, 'utf8');
const imports = source.slice(source.indexOf("import fs from"), source.indexOf("assert.ok(['darwin', 'linux']"));
assert.ok(imports.includes('waitForInstalledMcp'));
let body = source.slice(source.indexOf('const root ='));
assert.ok(body.startsWith('const root ='));
// Keep the real installed-client path and assertions identical, but do not claim
// new-backend no-Torch probes for an older, already published binary.
body = body.replace("'no Torch or model execution', 'compiled missing/broken Torch diagnostics',", "'static tools without model execution',");
body = body.replace('const result = {platform:', 'const result = {diagnosticOnly: true, releaseProof: false, platform:');
body = body.replace("limitations: [", "limitations: ['Published 0.1.44 diagnostic only; NOT acceptance of 0.1.45', 'Compiled no-Torch probes intentionally not run against historical backend', ");
const setup = `
const required = name => { assert.ok(process.env[name], 'Missing '+name); return path.resolve(process.env[name]); };
const codex = required('TENSORVIZ_TEST_CODEX');
const certificate = required('TENSORVIZ_TEST_TLS_CERT'), key = required('TENSORVIZ_TEST_TLS_KEY');
const target = nativePlatform();
const native = path.resolve('plugin/out/native-runtime');
const lock = {formatVersion:1,version:'0.1.44',platform:'linux-x64',pythonABI:'cpython-312-x86_64-linux-gnu',bytes:43516066,sha256:'78badf11ae2be88648c8459836cc95dd5fa20a1ee6481d81046de903640c06d2',url:'https://github.com/tensorviz-org/tensorviz-plugins/releases/download/v0.1.44/tensorviz-runtime-0.1.44-linux-x64.tar.gz'};
const archive = await fs.readFile('plugin/out/published44.tar.gz');
assert.equal(archive.length,lock.bytes);assert.equal(crypto.createHash('sha256').update(archive).digest('hex'),lock.sha256);
const info=JSON.parse(execFileSync(path.join(native,target.host),['--verify-native-package'],{encoding:'utf8',timeout:30000}));
assert.equal(info.platform,'linux');assert.equal(info.architecture,'x64');assert.equal(info.sourceIncluded,false);
const release=JSON.parse(await fs.readFile(path.join(native,'release-manifest.json'),'utf8'));
assert.equal(release.version,lock.version);
const output=path.resolve('plugin/out/ubuntu20-codex-diagnostic');await fs.mkdir(output,{recursive:true});
console.log('DIAGNOSTIC ONLY: published 0.1.44; not 0.1.45 release acceptance');
`;
try {
 await fs.writeFile(generated, imports + setup + body, {flag:'wx'});
 execFileSync(process.execPath,[generated.pathname],{stdio:'inherit',env:{...process.env,TENSORVIZ_RELEASE_VERSION:'0.1.44'},timeout:180000});
} finally { await fs.rm(generated,{force:true}); }
