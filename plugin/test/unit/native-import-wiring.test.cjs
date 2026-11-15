const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {EventEmitter}=require('node:events');
const filename=process.env.TENSORVIZ_TEST_IMPORT_FIXTURE || path.resolve(__dirname,'../native/parent-pipe-native-import.cjs');
function load(platform,env={}) {
 let registered,command;
 vm.runInNewContext(fs.readFileSync(filename,'utf8'),{
  process:{platform,env},setTimeout,clearTimeout,
  require(name){
   if(name==='node:test')return {test(title,options,run){registered={options,run}}};
   if(name==='node:child_process')return {spawn(executable){
    command=executable;const child=new EventEmitter();child.stdout=new EventEmitter();child.stderr=new EventEmitter();child.stdin={end(){}};child.exitCode=null;
    queueMicrotask(()=>{child.stdout.emit('data',JSON.stringify({torch:'fixture',numpy:'fixture'}));child.exitCode=0;child.emit('close',0,null)});return child;
   }};
   return require(name);
  }
 },{filename});
 return {registered,command:()=>command};
}
for(const platform of ['linux','darwin'])test(`${platform} skips Windows import acceptance without Windows fixture variables`,()=>{
 const loaded=load(platform);assert.equal(loaded.registered.options.skip,true);assert.equal(loaded.command(),undefined);
});
test('Windows native imports accept the release runner Python variable',async()=>{
 const python=path.resolve('release-python'),backend=path.resolve('native-backend');
 const loaded=load('win32',{TENSORVIZ_TEST_PYTHON:python,TENSORVIZ_TEST_BACKEND:backend});
 assert.equal(loaded.registered.options.skip,false);await loaded.registered.run();assert.equal(loaded.command(),python);
});
test('Windows validates fixture paths when its test executes',async()=>{
 const loaded=load('win32');await assert.rejects(loaded.registered.run(),/TENSORVIZ_TEST_PYTHON/);assert.equal(loaded.command(),undefined);
});
