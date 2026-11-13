const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test}=require('node:test');
test('migration diagnostics redact secrets and fixture cleanup follows all active-session assertions',async()=>{
const s=readFileSync(require('node:path').resolve(__dirname,'../migration/native-version-transition.cjs'),'utf8');
const helper=s.slice(s.indexOf('const redactNativeStderr'),s.indexOf("\ntest('released native"));
const logs=[];
const fs={statfs:async()=>({bavail:81*256,bsize:4096,blocks:1000*256})};
const c={fs,console:{log:s=>logs.push(s)}};
vm.createContext(c); vm.runInContext(helper+'\nglobalThis.scrub=redactNativeStderr; globalThis.disk=logDisk;',c);
for(const secret of ['token=secret-token','"api_key":"private-value"','Authorization: Bearer secret-key','https://localhost/test?token=sensitive','a'.repeat(64)]){
 const out=c.scrub(secret); assert.ok(!out.includes('secret-token')&&!out.includes('private-value')&&!out.includes('secret-key')&&!out.includes('sensitive')&&!out.includes('a'.repeat(64)),out);
}
assert.match(c.scrub('OSError: [Errno 28] No space left on device'),/No space left on device/);

 await c.disk('/tmp','before claude'); assert.match(logs[0],/available=81 MiB total=1000 MiB/);
 fs.statfs=async()=>{throw Object.assign(new Error('sensitive error'),{code:'ENOENT'})};
 await c.disk('/tmp','failure'); assert.match(logs[1],/unavailable \(ENOENT\)/);assert.ok(!logs[1].includes('sensitive'));
 // Cleanup is after all assertions and all closes, before recording client success.
 const close=s.indexOf('for(const session of sessions) await session.close();');
 const remove=s.indexOf('await fs.rm(fixture,'); const success=s.indexOf('receipt.clients[name]');
 assert.ok(close<s.indexOf('sessions.length = 0;')&&close<remove&&remove<success);
 assert.ok(s.indexOf('await previousSession.capture();')<close);
 console.log('Diagnostics redaction, filesystem reporting and cleanup order smoke checks passed.');
});
