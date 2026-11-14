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
body = body.replace('const result = {platform:', 'const result = {diagnosticOnly: true, releaseProof: false, diagnosticMode: mode, platform:');
body = body.replace("limitations: [", "limitations: ['Published 0.1.44 diagnostic only; NOT acceptance of 0.1.45', 'Compiled no-Torch probes intentionally not run against historical backend', ");
// Instrument only the disposable diagnostic catalog, never the production launcher.
const diagnosticPreamble = "case \",${NO_PROXY:-${no_proxy:-}},\" in *127.0.0.1*|*localhost*|*,\\**,*) diagnostic_exemption=yes;; *) diagnostic_exemption=no;; esac\nprintf 'DIAGNOSTIC CHILD: CA=%s SSL_CERT_FILE=%s HTTPS_PROXY=%s https_proxy=%s ALL_PROXY=%s all_proxy=%s localhost_exempt=%s\\n' \"${CURL_CA_BUNDLE:-unset}\" \"${SSL_CERT_FILE:-unset}\" \"${HTTPS_PROXY:+set}\" \"${https_proxy:+set}\" \"${ALL_PROXY:+set}\" \"${all_proxy:+set}\" \"$diagnostic_exemption\" >&2\nif [ -r \"${CURL_CA_BUNDLE:-/nonexistent}\" ]; then echo 'DIAGNOSTIC CHILD: CA readable' >&2; fi";
body = body.replace("  child = spawn(codex,", `  assert.equal(discovered.transport.env.CURL_CA_BUNDLE, certificate, 'Codex installed env must retain the exact fixture CA');
  const directCache = path.join(root, 'diagnostic-direct-cache');
  const directStart = downloads;
  const direct = await promisify(execFile)('/bin/sh', [path.join(installed.installedPath, 'scripts/launch.sh'), '--prepare-runtime'],
    {env: {...env, CURL_CA_BUNDLE: certificate, TENSORVIZ_RUNTIME_CACHE_DIR: directCache}, cwd: project, timeout: 30000});
  assert.equal(downloads, directStart + 1, 'Direct trusted launcher must fetch from the fixture');
  assert.equal(crypto.createHash('sha256').update(await fs.readFile(path.join(directCache, lock.sha256+'.tar.gz'))).digest('hex'), lock.sha256);
  assert.deepEqual(await fs.readdir(cache), [], 'Direct diagnostic must not warm the Codex cache');
  await fs.writeFile(path.join(output, 'direct-trust.json'), JSON.stringify({passed:true,installedCaMatches:true,downloadCount:downloads-directStart,stderr:direct.stderr}));
  console.log('DIRECT TRUSTED LAUNCHER PASSED; separate cache; configured Codex CA matches fixture');
  downloads = 0; // Begin the separately measured cold Codex phase.
  const launchFile = path.join(installed.installedPath, 'scripts/launch.sh');
  const launchSource = await fs.readFile(launchFile, 'utf8');
  await fs.writeFile(launchFile, launchSource.replace('set -eu', 'set -eu\\n'+${JSON.stringify(diagnosticPreamble)}));
  child = spawn(codex,`);
// Apply one-variable-at-a-time test trust/proxy overrides only to trusted paths.
// The untrusted TLS negative check deliberately keeps the original CA-free env.
body = body.replace('  server = https.createServer', `  const fixtureOverrides = {
    ...(['ssl-cert-file','combined'].includes(mode) ? {SSL_CERT_FILE:certificate} : {}),
    ...(['localhost-no-proxy','combined'].includes(mode) ? {NO_PROXY:'127.0.0.1,localhost',no_proxy:'127.0.0.1,localhost'} : {}),
  };
  server = https.createServer`);
body = body.replaceAll('{...env, CURL_CA_BUNDLE: certificate', '{...env, ...fixtureOverrides, CURL_CA_BUNDLE: certificate');
body = body.replace('const cenv = {...process.env, ...env, PATH:', 'const cenv = {...process.env, ...env, ...fixtureOverrides, PATH:');
const setup = `
const mode = process.env.TENSORVIZ_DIAGNOSTIC_CA_MODE;
assert.ok(['baseline','ssl-cert-file','localhost-no-proxy','combined'].includes(mode));
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
const output=path.resolve('plugin/out/ubuntu20-codex-diagnostic',mode);await fs.mkdir(output,{recursive:true});
console.log('DIAGNOSTIC ONLY: published 0.1.44; not 0.1.45 release acceptance');
`;
const cases = [];
try {
 await fs.writeFile(generated, imports + setup + body, {flag:'wx'});
 for (const mode of ['baseline','ssl-cert-file','localhost-no-proxy','combined']) {
   console.log('DIAGNOSTIC CASE:', mode);
   try {
     execFileSync(process.execPath,[generated.pathname],{stdio:'inherit',env:{...process.env,TENSORVIZ_RELEASE_VERSION:'0.1.44',TENSORVIZ_DIAGNOSTIC_CA_MODE:mode},timeout:180000});
     cases.push({mode,passed:true});
     if (mode === 'baseline') break;
   } catch (error) { cases.push({mode,passed:false,exitCode:error.status ?? null}); }
 }
 const report={diagnosticOnly:true,releaseProof:false,publishedVersion:'0.1.44',cases};
 await fs.mkdir('plugin/out/ubuntu20-codex-diagnostic',{recursive:true});
 await fs.writeFile('plugin/out/ubuntu20-codex-diagnostic/cases.json',JSON.stringify(report,null,2));
 console.log(JSON.stringify(report));
 if (!cases.some(item=>item.passed)) process.exitCode=1;
} finally { await fs.rm(generated,{force:true}); }
