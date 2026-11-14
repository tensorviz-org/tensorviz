#!/usr/bin/env python3
"""Add synthetic broken-Torch metadata/imports to a new disposable virtualenv.

Install requirements-no-torch.lock.txt in that env first. Refuses an environment
that already has Torch, or an interpreter whose prefix is outside the requested
virtualenv. This is a test fixture, not a real PyTorch installation.
"""
import argparse,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--env',type=pathlib.Path,required=True);a=p.parse_args();root=a.env.resolve();python=root/'bin/python';assert (root/'pyvenv.cfg').is_file(),'Expected a disposable virtualenv'
probe="import sys,sysconfig,importlib.util,importlib.metadata as m,platform,json;print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'site':sysconfig.get_path('purelib'),'torch_importable':importlib.util.find_spec('torch') is not None,'packages':{d.metadata['Name'].lower():d.version for d in m.distributions()},'architecture':platform.machine()}))"
facts=json.loads(subprocess.run([str(python),'-I','-c',probe],text=True,capture_output=True,check=True).stdout)
assert pathlib.Path(facts['prefix']).resolve()==root and facts['prefix']!=facts['base_prefix'],'Interpreter is not in the requested virtualenv'
assert not facts['torch_importable'] and 'torch' not in facts['packages'],'Refusing to overwrite an existing Torch installation'
assert {'pydantic','numpy','packaging'} <= facts['packages'].keys(),'Install capture dependencies first'
site=pathlib.Path(facts['site']);assert site.is_relative_to(root),'Site-packages escapes the virtualenv'
(site/'torch').mkdir();(site/'torch/__init__.py').write_text('# Synthetic broken-import test fixture; not real PyTorch.\nraise OSError("Synthetic PyTorch fixture: dlopen incompatible architecture (have x86_64, need arm64)")\n')
metadata=site/'torch-2.14.1.dist-info';metadata.mkdir();(metadata/'METADATA').write_text('Metadata-Version: 2.1\nName: torch\nVersion: 2.14.1\nSummary: Synthetic broken-import acceptance fixture, not real PyTorch\n');(metadata/'RECORD').write_text('');(metadata/'INSTALLER').write_text('tensorviz-acceptance-fixture\n')
print(json.dumps({'created':'synthetic_broken_torch','metadata_version':'2.14.1','architecture':facts['architecture']}))
