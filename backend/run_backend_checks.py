#!/usr/bin/env python3
"""Run bounded TensorViz backend checks using pre-created isolated interpreters.

This runner installs nothing, never reads normal user profiles, and publishes
nothing. Raw outputs remain under output/private; summary fields are allowlisted.
"""
import argparse, datetime, hashlib, json, os, pathlib, platform, re, signal, subprocess, time
p=argparse.ArgumentParser()
p.add_argument('--source',required=True,type=pathlib.Path)
p.add_argument('--output',required=True,type=pathlib.Path)
p.add_argument('--no-torch-python',required=True,type=pathlib.Path)
p.add_argument('--capture-python',required=True,type=pathlib.Path)
a=p.parse_args();src=a.source.resolve();out=a.output.resolve();private=out/'private';private.mkdir(parents=True,exist_ok=True,mode=0o700)
basehome=private/'home';basehome.mkdir(exist_ok=True,mode=0o700)
tmp=private/'tmp';tmp.mkdir(exist_ok=True,mode=0o700)
env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':str(basehome),'TMPDIR':str(tmp),'XDG_CACHE_HOME':str(basehome/'cache'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','LANG':'en_US.UTF-8'}
# Keep source acquisition and GitHub authentication outside this environment.
def run(c,**kw):return subprocess.run(c,text=True,capture_output=True,timeout=30,**kw)
sha=run(['git','-C',str(src),'rev-parse','HEAD']).stdout.strip()
versions={}
for name,interpreter in [('no_torch',a.no_torch_python.absolute()),('capture',a.capture_python.absolute())]:
 code="import sys,json,platform,importlib.util,importlib.metadata as m; print(json.dumps({'python':sys.version.split()[0], 'architecture':platform.machine(), 'torch_importable':importlib.util.find_spec('torch') is not None,'packages':{x.metadata['Name']:x.version for x in m.distributions()}}))"
 versions[name]=json.loads(run([str(interpreter),'-I','-c',code],env=env).stdout)
report={'source_sha':sha,'platform':platform.system(),'architecture':platform.machine(),'macos':platform.mac_ver()[0],'versions':versions,'checks':[],'scope':'Synthetic models and selected regression suites; no GUI, weights, provider calls, or sign-ins.'}
plans=[('static_without_torch',a.no_torch_python,['tests/runtime/test_service_without_torch.py','tests/runtime/test_server_v2_without_torch.py'],180),('capture_and_inputs',a.capture_python,['tests/runtime/test_capture_environment.py','tests/capture/test_graph_runtime_inputs.py','tests/graph/test_graph_diff_service.py','tests/runtime/test_torch_cache.py'],240),('environment_diagnostics',a.no_torch_python,['tests/runtime/test_environment_setup.py','-k','not private_capture_worker_executes_without_importing_server_dependencies'],240),('capture_without_server_dependencies',a.capture_python,['tests/runtime/test_environment_setup.py::test_private_capture_worker_executes_without_importing_server_dependencies'],120)]
for name,interpreter,tests,limit in plans:
 command=[str(interpreter.absolute()),'-m','pytest','-q','-p','no:cacheprovider',*tests]
 start=time.time();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
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
  status='blocked';exitcode=None
 log=stdout+'\nSTDERR\n'+stderr;(private/(name+'.log')).write_text(log)
 item={'name':name,'status':status,'exit_code':exitcode,'started_utc':stamp,'duration_seconds':round(time.time()-start,2),'command':'<'+('NO_TORCH' if interpreter==a.no_torch_python else 'CAPTURE')+'_PYTHON> -m pytest -q -p no:cacheprovider '+' '.join(tests),'working_directory':'<SOURCE>/backend','result_summary':[x for x in stdout.splitlines() if re.search(r'\b(passed|failed|error|errors|skipped|deselected)\b',x)][-4:]}
 # Only pytest counts/short failure test names; full tracebacks are local.
 item['result_summary']=[x.replace(str(src),'<SOURCE>').replace(str(out),'<OUTPUT>').replace(str(pathlib.Path.home()),'<HOME>') for x in item['result_summary']]
 report['checks'].append(item);(out/'backend-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(item),flush=True)
