'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const os=require('node:os');
const path=require('node:path');
const {execFileSync}=require('node:child_process');
const {ManagedRuntime}=require('../../out/host/runtime.cjs');

test('cold managed service downloads verified Python, installs wheels and starts without Torch', {timeout:600000}, async()=>{
 const state=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-cold-managed-service-'));
 const backend=path.resolve(__dirname,'../../out/native-runtime/backend');
 const approvals=[];
 const runtime=new ManagedRuntime(backend,state,{approve:async request=>{
   // Explicit test-only approval inside this disposable checkout; no user models.
   assert.notEqual(request.operation,'build_dependencies','Service must install from wheels, never source builds');
   assert.ok(['managed_runtime','install_dependencies'].includes(request.operation));
   approvals.push(request.operation);
 }});
 try {
   const python=await runtime.provisionService(state,AbortSignal.timeout(550000));
   assert.ok(approvals.includes('managed_runtime'));
   assert.ok(approvals.includes('install_dependencies'));
   execFileSync(python,['-I','-c','import importlib.util; assert importlib.util.find_spec("torch") is None'],{stdio:'inherit'});
   execFileSync(python,[path.resolve(__dirname,'../../../backend/tests/runtime/test_service_without_torch.py')],{
     stdio:'inherit',env:{...process.env,TENSORVIZ_SERVICE_TEST_BACKEND:backend},timeout:120000,
   });
   const previous=approvals.length;
   assert.equal(await runtime.provisionService(state,AbortSignal.timeout(10000)),python);
   assert.equal(approvals.length,previous,'Warm reuse must not reinstall or request fresh approval');
 } finally {await fs.rm(state,{recursive:true,force:true});}
});
