#!/usr/bin/env python3
"""Read-only compiled capture-dependency probes; no model execution or permission grants."""
import argparse, json, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser()
for key in ['evidence','backend','no-torch-python','capture-python','label']:p.add_argument('--'+key,required=True)
a=p.parse_args();base=Path(a.evidence).resolve();root=base/'private'/(a.label+'-capture-probe');root.mkdir(mode=0o700)
env={k:os.environ[k] for k in ['PATH','LANG'] if k in os.environ};env.update({'TMPDIR':str(root),'TORCHINDUCTOR_CACHE_DIR':str(root/'inductor')})
report={'label':a.label,'classification':'compiled dependency probe only; no capture, client or desktop acceptance','checks':[]}
for mode,python in [('missing-torch',a.no_torch_python),('broken-torch',a.no_torch_python),('valid-torch',a.capture_python)]:
    setup="sys.modules['torch']=types.ModuleType('torch')\n" if mode=='broken-torch' else ''
    code='import sys,types\nsys.path.insert(0,sys.argv[1])\nimport _tensorviz_backend\n'+setup+"sys.exit(_tensorviz_backend.run('services/ide/environment_setup.py',['probe-capture']))\n"
    try:
        r=subprocess.run([python,'-I','-B','-c',code,a.backend],env=env,cwd=root,capture_output=True,text=True,timeout=60)
        (root/(mode+'.stdout')).write_text(r.stdout);(root/(mode+'.stderr')).write_text(r.stderr)
        try: structured=json.loads(r.stdout)
        except ValueError: structured={}
        expected='MISSING_DEPENDENCIES' if mode=='missing-torch' else 'CAPTURE_RUNTIME_UNSUPPORTED'
        success=(r.returncode==0 and structured.get('ok') is True) if mode=='valid-torch' else (r.returncode==1 and structured.get('code')==expected and not r.stderr)
        item={'mode':mode,'status':'passed' if success else 'failed','exit_code':r.returncode,'response_ok':structured.get('ok'),'response_code':structured.get('code'),'stderr_present':bool(r.stderr)}
    except subprocess.TimeoutExpired:
        item={'mode':mode,'status':'timeout'}
    report['checks'].append(item)
(base/'public'/(a.label+'-capture-probe.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
