const {test}=require('node:test');const assert=require('node:assert/strict');const path=require('node:path');const {spawn}=require('node:child_process');
const python=process.env.TENSORVIZ_TEST_CAPTURE_PYTHON,backend=process.env.TENSORVIZ_TEST_BACKEND,source=process.env.TENSORVIZ_TEST_GUARD_SOURCE;
assert.ok(python&&path.isAbsolute(python)&&backend&&path.isAbsolute(backend),'Set explicit capture Python and backend paths.');
test('Windows lifetime watcher permits native imports with its owner pipe held open',{skip:process.platform!=='win32'},async()=>{
 const code=`import sys,importlib,json,runpy\nsys.path.insert(0,sys.argv[1])\nif sys.argv[2]:\n guard=runpy.run_path(sys.argv[2])['guard_parent_pipe']\nelse:\n guard=importlib.import_module('_tensorviz_backend.services.ingest.worker_lifetime').guard_parent_pipe\nguard()\nimport torch,numpy\nprint(json.dumps({'torch':torch.__version__,'numpy':numpy.__version__}),flush=True)\n`;
 const child=spawn(python,['-I','-B','-c',code,backend,source||''],{windowsHide:true,stdio:['pipe','pipe','pipe'],env:{...process.env,TENSORVIZ_PARENT_PIPE:'1'}});let stdout='',stderr='';child.stdout.on('data',x=>stdout+=x);child.stderr.on('data',x=>stderr+=x);
 let timer;const closed=new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>resolve({code,signal}));});
 try{const result=await Promise.race([closed,new Promise(resolve=>{timer=setTimeout(()=>resolve({timeout:true}),20000)})]);assert.equal(result.timeout,undefined,'Native import stalled with lifetime stdin held open');assert.equal(result.code,0,stderr);const versions=JSON.parse(stdout);assert.ok(versions.torch&&versions.numpy);}
 finally{clearTimeout(timer);if(child.exitCode===null){child.stdin.end();const cleanup=setTimeout(()=>child.kill(),2000);await closed;clearTimeout(cleanup);}}
});
