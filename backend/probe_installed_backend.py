#!/usr/bin/env python3
"""Probe installed VSIX HTTP services in an existing isolated no-Torch Python.

No settings are changed. Each service binds loopback and is terminated afterward.
Only allowlisted results are exported. Full child logs remain under private/.
"""
import argparse, concurrent.futures, datetime, hashlib, json, os, pathlib, secrets, signal, socket, subprocess, sys, time, urllib.error, urllib.request
p=argparse.ArgumentParser();p.add_argument('--backend',required=True,type=pathlib.Path);p.add_argument('--python',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args();backend=a.backend.resolve();out=a.output.resolve();private=out/'private';private.mkdir(parents=True,exist_ok=True,mode=0o700)
home=private/'installed-backend-home';home.mkdir(exist_ok=True);tmp=private/'installed-backend-tmp';tmp.mkdir(exist_ok=True)
baseenv={'PATH':'/usr/bin:/bin','HOME':str(home),'TMPDIR':str(tmp),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','TENSORVIZ_LOCAL_TOKEN':secrets.token_urlsafe(32),'TENSORVIZ_WORKSPACE_DIR':str(home),'TENSORVIZ_ALLOWED_ROOTS':str(home),'USER_HISTORY_DB_PATH':str(home/'state.db')}
version=json.loads(subprocess.run([str(a.python.absolute()),'-I','-c',"import platform,sys,importlib.util,json;print(json.dumps({'python':sys.version.split()[0],'architecture':platform.machine(),'torch_importable':importlib.util.find_spec('torch') is not None}))"],capture_output=True,text=True,check=True).stdout)
assert not version['torch_importable'],'Use an isolated interpreter without Torch.'
report={'artifact':'installed tensorviz.tensorviz-vscode 0.1.44 VSIX backend','environment':version,'tested_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'backend_files':{n:hashlib.sha256((backend/n).read_bytes()).hexdigest() for n in ['server_v2.py','standalone_server.py']},'checks':[]}
source='from torch import nn\nclass Tiny(nn.Module):\n def __init__(self):\n  super().__init__()\n  self.fc=nn.Linear(4,3)\n def forward(self,x): return self.fc(x)\n'
digest=hashlib.sha256(source.encode()).hexdigest();snapshot={'protocol_version':1,'root':{'file':'model.py','cls':'Tiny','source_digest':digest},'sources':[{'file':'model.py','source':source,'digest':digest}],'truncated':False}
client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name in ['extension_server_v2','standalone_service']:
 env=dict(baseenv)
 if name=='extension_server_v2':
  env['AUTH_DISABLED']='true' # The normal extension IDE mode, still protected by launch token.
  runner="import sys,socket,json;sys.path.insert(0,sys.argv[1]);import uvicorn;s=socket.socket();s.bind(('127.0.0.1',0));print(json.dumps({'port':s.getsockname()[1]}),flush=True);uvicorn.Server(uvicorn.Config('server_v2:app',log_level='warning',access_log=False)).run(sockets=[s])"
  cmd=[str(a.python.absolute()),'-I','-B','-c',runner,str(backend)]
  health='/health'
 else:
  cmd=[str(a.python.absolute()),'-I','-B',str(backend/'standalone_server.py'),'--parent-pipe'];health='/api/health'
 logpath=private/(name+'.stderr.log')
 with logpath.open('w') as logfile:
  child=subprocess.Popen(cmd,env=env,cwd=private,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logfile,text=True,start_new_session=True)
  record={'service':name,'startup':'not_run','static_projection':'not_run'}
  try:
   with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    future=pool.submit(child.stdout.readline)
    try:line=future.result(timeout=30)
    except concurrent.futures.TimeoutError:
     os.killpg(child.pid,signal.SIGTERM);raise RuntimeError('No startup address within 30 seconds')
   port=json.loads(line)['port'];origin=f'http://127.0.0.1:{port}'
   def request(path,body=None):
    headers={'X-tensorViz-Token':env['TENSORVIZ_LOCAL_TOKEN']};data=None
    if body is not None:headers['Content-Type']='application/json';data=json.dumps(body).encode()
    with client.open(urllib.request.Request(origin+path,data=data,headers=headers),timeout=3) as r:return r.status,json.load(r)
   for retry in range(100):
    try:status,body=request(health);break
    except (OSError,urllib.error.URLError):
     if child.poll() is not None:raise RuntimeError('Service process exited before HTTP health became available')
     if retry==99:raise
     time.sleep(.05)
   assert status==200;record['startup']='passed';record['health_status']=status
   status,body=request('/api/ide/multifile/project',snapshot);assert status==200 and body['document']['semantic']['nodes'] and body['traced'] is False
   record['static_projection']='passed';record['projection_nodes']=len(body['document']['semantic']['nodes'])
   status,body=request('/api/ide/review/index',{'protocol_version':1,'sources':snapshot['sources'],'truncated':False});assert status==200 and body['models'][0]['cls']=='Tiny';record['static_discovery']='passed'
   status,body=request('/api/ide/graph-setup/discover',snapshot);assert status==200;record['static_setup']='passed'
  except Exception as error:
   record['startup']='failed' if record['startup']=='not_run' else record['startup'];record['failure_kind']=type(error).__name__;record['failure_summary']=str(error).replace(str(backend),'<INSTALLED_BACKEND>')[:300]
   if record['static_projection']=='not_run':record['static_projection']='blocked'
  finally:
   if child.poll() is None:
    child.stdin.close()
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:
     os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=5)
   child.stdout.close()
 with logpath.open() as f:log=f.read()
 for line in log.splitlines():
  if 'ModuleNotFoundError:' in line or 'ImportError:' in line:record['import_error']=line.strip()[:200]
 report['checks'].append(record)
(out/'installed-vsix-service-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
