const {test}=require('node:test');
const assert=require('node:assert/strict');
const musl={header:{},sharedObjects:['/lib/ld-musl-x86_64.so.1']};
test('musl is explicit, ABI exact, unknown libc rejected',async()=>{
 const {nativePlatform,linuxLibc}=await import('../../scripts/native-platforms.mjs');
 assert.equal(linuxLibc(musl),'musl');
 assert.equal(linuxLibc({header:{glibcVersionRuntime:'2.31'}}),'glibc');
 assert.throws(()=>linuxLibc({header:{}}),/Unknown Linux libc/);
 const target=nativePlatform('linux','x64','musl');
 assert.equal(target.key,'linux-x64-musl');assert.equal(target.abi,'cpython-312-x86_64-linux-musl');
 assert.equal(target.pkg,'node22.23.4-alpine-x64');
 assert.throws(()=>nativePlatform('linux','arm64','musl'),/Unsupported/);
});
test('musl audit cannot admit GNU ELF or unexpected dependencies',async()=>{
 const {verifyLinuxMusl}=await import('../../scripts/package-native.mjs');
 const output={ '-h':'Machine: Advanced Micro Devices X86-64','--version-info':'No version information found','-d':'(NEEDED) Shared library: [libc.musl-x86_64.so.1]','-l':'Requesting program interpreter: /lib/ld-musl-x86_64.so.1'};
 const run=(_,args)=>output[args[0]];
 verifyLinuxMusl('host',true,run);
 output['--version-info']='GLIBC_2.31';assert.throws(()=>verifyLinuxMusl('host',true,run),/GNU/);
 output['--version-info']='';output['-d']+='\n(NEEDED) Shared library: [libstdc++.so.6]';assert.throws(()=>verifyLinuxMusl('host',true,run),/Unexpected/);
});

test('bootstrap selects only the explicit musl catalog, never GNU fallback',()=>{
 const {selectRuntime,checkHost}=require('../../integration/launch.cjs');
 const entry={formatVersion:1,version:'0.1.44',platform:'linux-x64-musl',pythonABI:'cpython-312-x86_64-linux-musl',sha256:'a'.repeat(64),bytes:123,url:'https://example.test/native.tar.gz'};
 const lock={formatVersion:2,version:'0.1.44',runtimes:{'linux-x64-musl':entry}};
 assert.equal(selectRuntime(lock,'linux','x64',musl),entry);
 assert.throws(()=>selectRuntime({...lock,runtimes:{}},'linux','x64',musl),/No valid linux-x64-musl/);
 assert.throws(()=>selectRuntime(lock,'linux','x64',{header:{}}),/Unknown Linux libc/);
 checkHost('linux',musl,'x64');
 assert.throws(()=>checkHost('linux',musl,'arm64'),/requires glibc/);
});
