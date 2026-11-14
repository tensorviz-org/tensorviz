#!/usr/bin/env python3
"""Capture a tiny random GPT-2 offline without changing its cache default."""
import argparse, json, os, pathlib, subprocess, sys, tempfile
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();backend=a.source.resolve()/'backend';out=a.output.resolve();sys.path.insert(0,str(backend))
from services.ide.graph_setup import discover
from services.ide.graph_setup_profiles import prepare_profile
from services.semantic.execution_plan import profile_execution
from services.semantic.projection import project_capture
from services.validation.contracts import ModelProfile
from services.validation.host_snapshot import snapshot
from tests.discovery.test_graph_setup_profiles import snapshot as sources
source='''from transformers import GPT2Model, GPT2Config
class Net(GPT2Model): pass

def make_model() -> Net:
    return Net(GPT2Config(vocab_size=256, n_embd=16, n_layer=1, n_head=4, n_positions=16))
'''
private=out/'private';private.mkdir(parents=True,exist_ok=True,mode=0o700)
with tempfile.TemporaryDirectory(prefix='gpt2-',dir=private) as name:
 root=pathlib.Path(name);workspace=root/'workspace';workspace.mkdir();(workspace/'model.py').write_text(source)
 req=sources(source);resolution=discover(req)['resolution'];profile=resolution['draft']['profile']
 def run(probe):
  raw=snapshot(dict(root=str(workspace),executable=sys.executable,environmentRevision=0,session='gpt2',workspaceId='gpt2',runId='gpt2',generation=1,modelFile='model.py',className='Net',timeoutMs=45000,execution=profile_execution(ModelProfile.model_validate(profile))))
  plan=root/'plan.json';plan.write_text(json.dumps({'version':1,'plan':raw['plan'],'applicabilityDigest':raw['applicabilityDigest']}));plan.chmod(0o600)
  command=[sys.executable,'-B',str(backend/'services/ingest/bootstrap.py'),'--file',str(workspace/'model.py'),'--cls','Net','--workspace',str(workspace),'--resolved-execution',*(['--prepare-inputs'] if probe else []),'--json']
  env={'PATH':'/usr/bin:/bin','HOME':str(root),'TMPDIR':str(root),'XDG_CACHE_HOME':str(root/'cache'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','TENSORVIZ_EXECUTION_PLAN':str(plan),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1'}
  result=subprocess.run(command,cwd=backend,env=env,capture_output=True,text=True,timeout=60)
  (private/('gpt2-probe.log' if probe else 'gpt2-capture.log')).write_text(result.stdout+'\nSTDERR\n'+result.stderr)
  envelope=json.loads(result.stdout)
  if result.returncode!=0:raise RuntimeError(envelope.get('error'))
  return envelope['result']
 proposal=run(True);assert 'graph' not in proposal
 profile['inputs']=proposal['graphExample']['binding']
 prepared=prepare_profile(req,'',profile);assert prepared['ok'];(workspace/'tensorviz.toml').write_text(prepared['config'])
 reopened=snapshot(dict(root=str(workspace),executable=sys.executable,environmentRevision=0,session='gpt2',workspaceId='gpt2',runId='reopen',generation=2,modelFile='model.py',className='Net',timeoutMs=45000,profile=prepared['profile']))
 profile=reopened['resolution']['spec']['profile'];result=run(False);capture=result['diagnostics']['semantic_capture'];projection=project_capture(capture)
 assert capture['operations'] and projection.nodes
 call=next(x for x in capture['invocations'] if x['id']==capture['rootInvocationId']);assert call['status']=='completed'
 record={'family':'GPT2','source_sha':subprocess.run(['git','-C',str(a.source),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),'status':'passed','capture_status':capture['status'],'omissions':capture['coverage']['omissions'],'inputs':{k:{key:v[key] for key in ['shape','dtype']} for k,v in profile['inputs']['kwargs'].items()},'nodes':len(projection.nodes),'edges':len(projection.edges),'output':capture['output'],'scope':'Tiny randomly initialized GPT2Model; cache default unchanged; offline; no pretrained weights or UI.'}
 values={v['id']:v for v in capture['values']};record['output_shapes']={e['key']['value']:values[e['value']['id']]['shape'] for e in capture['output']['entries'] if e['value']['kind']=='value'}
 (out/'gpt2-result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
