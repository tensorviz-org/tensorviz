#!/usr/bin/env python3
"""Installed Codex/Claude static MCP checks using a verified warm runtime cache.

No model turn, account login, permission reply, capture or GUI is performed.
"""
import argparse, hashlib, json, os, queue, shutil, signal, subprocess, threading, time
from pathlib import Path
p=argparse.ArgumentParser()
for n in ['evidence','catalog','label']:p.add_argument('--'+n,required=True)
p.add_argument('--archive')
p.add_argument('--source',action='store_true')
a=p.parse_args();base=Path(a.evidence).resolve();root=base/'private'/(a.label+'-client-mcp');root.mkdir(mode=0o700)
catalog=root/'catalog';shutil.copytree(Path(a.catalog).resolve(),catalog)
cache=root/'cache';cache.mkdir()
if a.source:
    manifest=json.loads((catalog/'release-manifest.json').read_text())
    entry={'version':manifest['version'],'sha256':hashlib.sha256((catalog/'plugins/tensorviz/runtime/host.cjs').read_bytes()).hexdigest()}
    for name,expected in manifest['files'].items():assert hashlib.sha256((catalog/name).read_bytes()).hexdigest()==expected
else:
    lock=json.loads((catalog/'plugins/tensorviz/runtime-lock.json').read_text())
    entry=lock['runtimes']['darwin-arm64'] if lock['formatVersion']==2 else lock
    archive=Path(a.archive).read_bytes();assert len(archive)==entry['bytes'];assert hashlib.sha256(archive).hexdigest()==entry['sha256']
    (cache/(entry['sha256']+'.tar.gz')).write_bytes(archive)
for f in ['mcp.json','.mcp.json']:
    path=catalog/'plugins/tensorviz'/f;j=json.loads(path.read_text())
    j['mcpServers']['tensorviz']['env']={'TENSORVIZ_DATA_DIR':str(root/'state'),'TENSORVIZ_RUNTIME_CACHE_DIR':str(cache)}
    path.write_text(json.dumps(j,indent=2)+'\n')
project=root/'empty-project';project.mkdir();temp=root/'tmp';temp.mkdir()
env={k:os.environ[k] for k in ['PATH','HOME','USER','LOGNAME','LANG'] if k in os.environ}
env.update({'CODEX_HOME':str(root/'codex-profile'),'CLAUDE_CONFIG_DIR':str(root/'claude-profile'),'TMPDIR':str(temp),
 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1','ANTHROPIC_API_KEY':'test-no-provider-calls','CLAUDE_CODE_OAUTH_TOKEN':'',
 'TENSORVIZ_DATA_DIR':str(root/'state'),'TENSORVIZ_RUNTIME_CACHE_DIR':str(cache),'NO_COLOR':'1'})
for n in ['CODEX_HOME','CLAUDE_CONFIG_DIR']:Path(env[n]).mkdir(mode=0o700)
report={'label':a.label,'runtime_cache':'not applicable; local Node source build' if a.source else 'preverified published archive; warm startup','runtime_sha256':entry['sha256'],
        'provider_calls':False,'permission_replies':0,'checks':[]}
if a.source:report.update({'classification':'source-development only; not a published/native release','source_revision':manifest['sourceRevision'],'source_working_tree_dirty':manifest['workingTreeDirty']})

def cli(label,args):
    r=subprocess.run(args,env=env,cwd=project,capture_output=True,text=True,timeout=40)
    (root/(label+'.stdout')).write_text(r.stdout);(root/(label+'.stderr')).write_text(r.stderr)
    if r.returncode:raise RuntimeError(label+' failed; inspect private output')
    try:return json.loads(r.stdout)
    except ValueError:return r.stdout

class Bridge:
    def __init__(self,kind,command):
        self.kind=kind;self.i=0;self.q=queue.Queue();self.closed_threads=set();self.shutdown_mode='eof';self.raw=open(root/(kind+'-protocol.jsonl'),'w');self.err=open(root/(kind+'-stderr'),'w')
        self.proc=subprocess.Popen(command,env=env,cwd=project,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,start_new_session=True,bufsize=1)
        def read():
            for line in self.proc.stdout:
                self.raw.write(line);self.raw.flush()
                try:
                    message=json.loads(line)
                    if message.get('method')=='thread/closed' and message.get('params',{}).get('threadId'):self.closed_threads.add(message['params']['threadId'])
                    self.q.put(message)
                except ValueError:pass
        threading.Thread(target=read,daemon=True).start()
    def send(self,j):self.proc.stdin.write(json.dumps(j)+'\n');self.proc.stdin.flush()
    def call(self,method,params=None):
        self.i+=1;ident=self.i if self.kind=='codex' else str(self.i)
        self.send({'id':ident,'method':method,'params':params or {}} if self.kind=='codex' else {'type':'control_request','request_id':ident,'request':{'subtype':method,**(params or {})}})
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            try:m=self.q.get(timeout=max(.01,deadline-time.monotonic()))
            except queue.Empty:raise TimeoutError(method)
            if self.kind=='codex':
                if 'method' in m and 'id' in m:raise RuntimeError('blocked on client permission/request: '+m['method'])
                if m.get('id')==ident:
                    if 'error' in m:raise RuntimeError('client response error: '+str(m['error'].get('code')))
                    return m.get('result',{})
            else:
                if m.get('type')=='control_request':raise RuntimeError('blocked on client permission/request')
                response=m.get('response',{})
                if m.get('type')=='control_response' and response.get('request_id')==ident:
                    if response.get('subtype')=='error':raise RuntimeError('Claude control response error')
                    return response.get('response',{})
        raise TimeoutError(method)
    def close(self):
        try:self.proc.stdin.close();self.proc.wait(timeout=5)
        except (BrokenPipeError,subprocess.TimeoutExpired):
            self.shutdown_mode='sigterm'
            try:os.killpg(self.proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:self.shutdown_mode='sigkill';os.killpg(self.proc.pid,signal.SIGKILL);self.proc.wait()
        self.raw.close();self.err.close()

c=shutil.which('codex');cl=shutil.which('claude')
for kind in ['codex','claude']:
    bridge=None;result={'client':kind};started=time.monotonic()
    try:
        if kind=='codex':
            cli('codex-marketplace',[c,'plugin','marketplace','add',str(catalog),'--json'])
            installed=cli('codex-install',[c,'plugin','add','tensorviz@tensorviz-standalone','--json'])
            bridge=Bridge('codex',[c,'app-server','--stdio','-c','thread_unload_delay_secs=0'])
            bridge.call('initialize',{'clientInfo':{'name':'tensorviz-static-acceptance','version':'1'},'capabilities':{'experimentalApi':True}})
            bridge.send({'method':'initialized'})
            tid=bridge.call('thread/start',{'cwd':str(project),'ephemeral':True,'approvalPolicy':'on-request','sandbox':'workspace-write'})['thread']['id']
            deadline=time.monotonic()+30
            while True:
                status=bridge.call('mcpServerStatus/list',{'threadId':tid,'serverName':'tensorviz'})
                servers=status.get('data',[])
                if len(servers)!=1:raise RuntimeError('unexpected MCP server count')
                server=servers[0];tools=list(server.get('tools',{}).values())
                if server.get('toolsError') or server.get('runtimeStatus') in ['failed','cancelled','disabled','authenticationRequired']:raise RuntimeError('MCP startup failed')
                if any(t.get('name')=='open_graph' for t in tools):break
                if time.monotonic()>deadline:raise TimeoutError('installed MCP readiness')
                time.sleep(.2)
            listing=bridge.call('mcpServer/tool/call',{'threadId':tid,'server':server['name'],'tool':'list_models','arguments':{}})['structuredContent']
            skills=bridge.call('skills/list',{'cwds':[str(project)],'forceReload':True})
            result['skill_advertised']=any(s.get('enabled') and s.get('name','').endswith('model-workflow') for d in skills.get('data',[]) for s in d.get('skills',[]))
            result.update({'tool_count':len(tools),'plugin_version':listing['host']['version'],'model_count':len(listing['models'])})
            assert listing['host']['version']==entry['version'] and len(tools)==29 and not listing['models'] and result['skill_advertised']
            bridge.call('thread/unsubscribe',{'threadId':tid})
            # The client owns separate MCP process groups. Confirm managed teardown
            # before stopping app-server; a fixed delay can abandon the launcher.
            deadline=time.monotonic()+30
            while tid not in bridge.closed_threads and time.monotonic()<deadline:time.sleep(.1)
            if tid not in bridge.closed_threads:raise TimeoutError('managed thread shutdown')
            result['managed_thread_closed']=True
        else:
            cli('claude-marketplace',[cl,'--bare','plugin','marketplace','add',str(catalog)])
            cli('claude-install',[cl,'--bare','plugin','install','tensorviz@tensorviz-standalone','--scope','user'])
            bridge=Bridge('claude',[cl,'-p','--input-format','stream-json','--output-format','stream-json','--verbose','--no-session-persistence','--setting-sources','user'])
            init=bridge.call('initialize')
            result['skill_advertised']='tensorviz:model-workflow' in json.dumps(init.get('commands',[]))
            deadline=time.monotonic()+30
            while True:
                status=bridge.call('mcp_status');servers=status.get('mcpServers',[])
                if len(servers)==1 and servers[0].get('status')=='connected':break
                if any(s.get('status') in ['failed','needs-auth'] for s in servers):raise RuntimeError('MCP startup failed')
                if time.monotonic()>deadline:raise TimeoutError('installed MCP readiness')
                time.sleep(.2)
            listing=bridge.call('mcp_call',{'tool':'mcp__plugin_tensorviz_tensorviz__list_models','arguments':{}})['structuredContent']
            result.update({'tool_count':len(servers[0]['tools']),'plugin_version':listing['host']['version'],'model_count':len(listing['models'])})
            assert listing['host']['version']==entry['version'] and len(servers[0]['tools'])==29 and not listing['models'] and result['skill_advertised']
        result['status']='passed'
    except Exception as e:
        result.update({'status':'failed','error_type':type(e).__name__,'error_summary':str(e) if '/' not in str(e) else 'private diagnostic'})
    finally:
        if bridge:
            bridge.close();result['shutdown_mode']=bridge.shutdown_mode
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                runtime_dirs=[x for x in temp.iterdir() if x.name.startswith(('tensorviz-native.','tensorviz-release-session-'))]
                if not runtime_dirs:break
                time.sleep(.1)
            result['runtime_temp_directories_remaining']=len(runtime_dirs)
            result['runtime_cleanup_status']='passed' if not runtime_dirs else 'failed'
        try:
            cmd=[c,'plugin','remove','tensorviz@tensorviz-standalone','--json'] if kind=='codex' else [cl,'--bare','plugin','uninstall','tensorviz@tensorviz-standalone','--scope','user']
            cli(kind+'-remove',cmd);result['uninstall_command_succeeded']=True
        except Exception:result['uninstall_command_succeeded']=False
        result['seconds']=round(time.monotonic()-started,3);report['checks'].append(result);print(json.dumps(result),flush=True)
report['permission_record_created']=(root/'state/permissions/standalone.json').exists()
report['capture']='not_run; static discovery only'
report['desktop_rendering']='not_run'
# Codex can launch a curated-plugin fetch that outlives app-server. Stop only
# Git processes whose arguments identify this exact disposable profile's cache.
background=[]
for line in subprocess.check_output(['ps','-ww','-axo','pid=,ppid=,args='],text=True).splitlines():
    fields=line.strip().split(None,2)
    if len(fields)==3 and str(root/'codex-profile/.tmp') in fields[2] and Path(fields[2].split()[0]).name in ['git','git-remote-https']:
        try:os.kill(int(fields[0]),signal.SIGTERM);background.append(int(fields[0]))
        except ProcessLookupError:pass
report['isolated_background_fetch_processes_stopped']=len(background)
(base/'public'/(a.label+'-client-mcp.json')).write_text(json.dumps(report,indent=2)+'\n')
