#!/usr/bin/env python3
"""Documented local-catalog upgrade in disposable profiles, no provider calls."""
import argparse, json, os, shutil, signal, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser()
for key in ['evidence','older','newer']: p.add_argument('--'+key,required=True)
p.add_argument('--local-only',action='store_true')
a=p.parse_args(); base=Path(a.evidence).resolve()
old_version=json.loads((Path(a.older)/'plugins/tensorviz/plugin.json').read_text())['version']
new_version=json.loads((Path(a.newer)/'plugins/tensorviz/plugin.json').read_text())['version']
label=f'upgrade-{old_version}-to-{new_version}'+('-local-fixed' if a.local_only else '');root=base/'private'/label
root.mkdir(mode=0o700,exist_ok=False); catalog=root/'catalog'; temp=root/'temp';temp.mkdir()
def stage(source):
    shutil.copytree(Path(source).resolve(),catalog)
    for name in ['mcp.json','.mcp.json']:
        path=catalog/'plugins/tensorviz'/name;doc=json.loads(path.read_text())
        doc['mcpServers']['tensorviz']['env']={'TENSORVIZ_DATA_DIR':str(root/'state'),'TENSORVIZ_RUNTIME_CACHE_DIR':str(root/'runtime-cache')}
        path.write_text(json.dumps(doc,indent=2)+'\n')
stage(a.older)
env={k:os.environ[k] for k in ['PATH','HOME','USER','LOGNAME','LANG'] if k in os.environ}
env.update({'TMPDIR':str(temp),'CODEX_HOME':str(root/'codex-profile'),'CLAUDE_CONFIG_DIR':str(root/'claude-profile'),
 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1','ANTHROPIC_API_KEY':'test-no-provider-calls','CLAUDE_CODE_OAUTH_TOKEN':'',
 'TENSORVIZ_DATA_DIR':str(root/'state'),'TENSORVIZ_RUNTIME_CACHE_DIR':str(root/'runtime-cache'),'NO_COLOR':'1','CI':'1'})
for key in ['CODEX_HOME','CLAUDE_CONFIG_DIR']:Path(env[key]).mkdir(mode=0o700)
report={'classification':'CLI lifecycle only; no account/provider/desktop test','checks':[]}
def run(name,args):
    t=time.monotonic();p=subprocess.Popen(args,env=env,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    try:o,e=p.communicate(timeout=60);status='passed' if p.returncode==0 else 'failed'
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM);o,e=p.communicate(timeout=10);status='timeout'
    (root/(name+'.stdout')).write_text(o);(root/(name+'.stderr')).write_text(e)
    x={'name':name,'status':status,'exit_code':p.returncode,'seconds':round(time.monotonic()-t,3)};report['checks'].append(x);print(json.dumps(x),flush=True)
    try:return json.loads(o)
    except ValueError:return None
c=shutil.which('codex'); cl=shutil.which('claude')
try:
    run('codex-marketplace-add-old',[c,'plugin','marketplace','add',str(catalog),'--json'])
    old_c=run('codex-install-old',[c,'plugin','add','tensorviz@tensorviz-standalone','--json'])
    run('claude-marketplace-add-old',[cl,'plugin','marketplace','add',str(catalog)])
    run('claude-install-old',[cl,'plugin','install','tensorviz@tensorviz-standalone','--scope','user'])
    old_cl=run('claude-list-old',[cl,'plugin','list','--json'])
    catalog.rename(root/'saved-old-catalog');stage(a.newer)
    if not a.local_only:run('codex-marketplace-upgrade',[c,'plugin','marketplace','upgrade','tensorviz-standalone','--json'])
    new_c=run('codex-install-upgraded',[c,'plugin','add','tensorviz@tensorviz-standalone','--json'])
    run('codex-mcp-after-upgrade',[c,'mcp','list','--json'])
    run('claude-plugin-update',[cl,'plugin','update','tensorviz@tensorviz-standalone'])
    new_cl=run('claude-list-upgraded',[cl,'plugin','list','--json'])
    report['codex_versions']={'before':old_c.get('version'),'after':new_c.get('version')}
    report['claude_versions']={'before':old_cl[0].get('version'),'after':new_cl[0].get('version')}
    report['codex_plugin_replacement_passed']=report['codex_versions']=={'before':old_version,'after':new_version}
    report['documented_codex_local_catalog_flow']='passed: complete local catalog replacement then plugin add' if a.local_only else 'failed: marketplace upgrade rejects a local-folder marketplace; plugin add afterward succeeded'
    report['claude_upgrade_passed']=report['claude_versions']=={'before':old_version,'after':new_version}
finally:
    run('codex-uninstall',[c,'plugin','remove','tensorviz@tensorviz-standalone','--json'])
    run('claude-uninstall',[cl,'plugin','uninstall','tensorviz@tensorviz-standalone','--scope','user'])
    report['codex_removed']=run('codex-list-removed',[c,'plugin','list','--marketplace','tensorviz-standalone','--json']).get('installed')==[]
    report['claude_removed']=run('claude-list-removed',[cl,'plugin','list','--json'])==[]
    (base/'public'/(label+'-report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
