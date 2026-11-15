const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const {StandaloneHost,readHostSnapshot,disposeSnapshotHelpers}=require(process.env.TENSORVIZ_TEST_SNAPSHOT_ADAPTER || '../../out/host/snapshotCleanup.cjs');
const python=process.env.TENSORVIZ_TEST_PYTHON;
assert.ok(python&&path.isAbsolute(python),'Set TENSORVIZ_TEST_PYTHON to an existing Python interpreter');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const alive=pid=>{try{process.kill(pid,0);return true}catch(e){if(e.code==='ESRCH')return false;throw e}};
test('standalone disposal ends an idle snapshot interpreter before assets are removed',async()=>{
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-snapshot-cleanup-'));
 let pid,host;
 try{
  const backend=path.join(root,'backend');await fs.mkdir(path.join(backend,'services/validation'),{recursive:true});
  await fs.writeFile(path.join(backend,'services/validation/host_snapshot.py'),'import sys,json,os,threading\nthreading.Timer(8,lambda:os._exit(2)).start()\nfor line in sys.stdin:\n print(json.dumps({"pid":os.getpid()}),flush=True)\nos._exit(0)\n');
  host=new StandaloneHost(backend,path.join(root,'browser'),{dispose(){}},path.join(root,'data'));
  const response=await readHostSnapshot(python,backend,{},new AbortController().signal);pid=response.pid;assert.ok(alive(pid));
  host.dispose();
  const deadline=Date.now()+3000;while(alive(pid)&&Date.now()<deadline)await sleep(20);
  assert.equal(alive(pid),false,'Idle helper survived host disposal');
  await fs.rm(backend,{recursive:true});
  assert.equal(await fs.stat(backend).catch(()=>null),null);
 }finally{
  host?.dispose();disposeSnapshotHelpers();
  const deadline=Date.now()+9000;while(pid&&alive(pid)&&Date.now()<deadline)await sleep(50);
  if(pid)assert.equal(alive(pid),false,'Fixture cleanup failed');
  await fs.rm(root,{recursive:true,force:true});
 }
});
