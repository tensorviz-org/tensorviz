#!/usr/bin/env python3
"""Offline optional-library captures. Installs nothing and publishes nothing."""
import argparse, datetime, importlib.util, json, os, pathlib, platform, re, signal, subprocess, time
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--python',type=pathlib.Path,required=True);a=p.parse_args()
src=a.source.resolve();out=a.output.resolve();private=out/'private';private.mkdir(exist_ok=True,parents=True,mode=0o700)
home=private/'library-home';home.mkdir(exist_ok=True);tmp=private/'library-tmp';tmp.mkdir(exist_ok=True)
env={'PATH':'/usr/bin:/bin','HOME':str(home),'TMPDIR':str(tmp),'XDG_CACHE_HOME':str(home/'cache'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1','LANG':'en_US.UTF-8'}
python=str(a.python.absolute())
sha=subprocess.run(['git','-C',str(src),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
versions=json.loads(subprocess.run([python,'-I','-c',"import sys,json,platform,importlib.metadata as m; print(json.dumps({'python':sys.version.split()[0], 'architecture':platform.machine(),'packages':{x.metadata['Name']:x.version for x in m.distributions()}}))"],env=env,capture_output=True,text=True,check=True).stdout)
report={'source_sha':sha,'versions':versions,'scope':'Tiny random-weight constructors; offline; no pretrained models, text, image data, or interactive GUI.','checks':[]}
plans=[('transformer_families',['tests/capture/test_graph_library_inputs.py'],240),('conditional_unet',['tests/capture/test_graph_diffusion_inputs.py','-k','not full_defaults'],360)]
for name,args,limit in plans:
 command=[python,'-m','pytest','-q','-s','-p','no:cacheprovider',*args];start=time.time();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  child=subprocess.Popen(command,cwd=src/'backend',env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
  stdout,stderr=child.communicate(timeout=limit);exitcode=child.returncode;status='passed' if exitcode==0 else 'failed'
 except subprocess.TimeoutExpired as e:
  os.killpg(child.pid,signal.SIGTERM)
  try:stdout,stderr=child.communicate(timeout=10)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGKILL);stdout,stderr=child.communicate()
  if isinstance(stdout,bytes):stdout=stdout.decode(errors='replace')
  if isinstance(stderr,bytes):stderr=stderr.decode(errors='replace')
  exitcode=None;status='blocked'
 (private/(name+'.log')).write_text(stdout+'\nSTDERR\n'+stderr)
 measurements=[]
 for line in stdout.splitlines():
  line=line.lstrip('.')
  if line.startswith('{'):
   try:measurements.append(json.loads(line))
   except ValueError:pass
 counts=[x for x in stdout.splitlines() if re.search(r'^\d+ (passed|failed|skipped)|^=+.*\d+ (passed|failed|skipped)|^FAILED tests/',x)][-8:]
 item={'name':name,'status':status,'exit_code':exitcode,'started_utc':stamp,'duration_seconds':round(time.time()-start,2),'command':'<CAPTURE_PYTHON> -m pytest -q -s -p no:cacheprovider '+' '.join(args),'working_directory':'<SOURCE>/backend','result_summary':counts,'measurements':measurements}
 report['checks'].append(item);(out/'library-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(item),flush=True)
