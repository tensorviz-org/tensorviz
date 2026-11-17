// Called only by activated release publication. Never install/test during draft prep.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {lockedCLI} from './production-release-adapters.mjs';
export async function installReviewedReleaseTools(sourceRoot,sourceRevision,toolConfig,{command=execFileSync,environment=process.env}={}) {
  if(!/^[a-f0-9]{40}$/.test(sourceRevision??''))throw Error('Reviewed tools source required');
  const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-release-tools-'));
  try {
    const tools={},installed=new Map();
    for(const [channel,pkg] of Object.entries({'vscode-marketplace':'@vscode/vsce','open-vsx':'ovsx',docs:'firebase-tools'})) {
      const tool=toolConfig?.[channel];
      if(!['extension','docs','tooling/release-tools'].includes(tool?.root))throw Error('Reviewed tool root required');
      if(!installed.has(tool.root)) {
        const root=path.join(temporary,`install-${installed.size}`);await fs.mkdir(root);
        for(const name of ['package.json','package-lock.json']) {
          const bytes=command('git',['show',`${sourceRevision}:${tool.root}/${name}`],{cwd:sourceRoot,maxBuffer:16*1024*1024});
          const document=JSON.parse(bytes);
          if(name==='package-lock.json') {
            if(document.lockfileVersion<2||!document.packages)throw Error('Modern npm lock required');
            for(const [name,entry]of Object.entries(document.packages))if(name) {
              if(!name.startsWith('node_modules/')||entry.link||!/^sha512-[A-Za-z0-9+/=]+$/.test(entry.integrity??''))throw Error('Every publisher dependency requires registry integrity');
              const url=new URL(entry.resolved);
              if(url.protocol!=='https:'||url.hostname!=='registry.npmjs.org'||url.username||url.password)throw Error('Only locked official npm registry packages permitted');
            }
          }
          await fs.writeFile(path.join(root,name),bytes,{flag:'wx'});
        }
        const home=path.join(root,'home');await fs.mkdir(home);
        const env={HOME:home,USERPROFILE:home,XDG_CONFIG_HOME:path.join(home,'.config'),npm_config_cache:path.join(root,'cache'),npm_config_userconfig:path.join(root,'empty.npmrc'),npm_config_globalconfig:path.join(root,'empty-global.npmrc')};
        for(const key of ['PATH','SystemRoot','TMP','TEMP','TMPDIR'])if(environment[key])env[key]=environment[key];
        await fs.writeFile(env.npm_config_userconfig,'');await fs.writeFile(env.npm_config_globalconfig,'');
        try{command(process.platform==='win32'?'npm.cmd':'npm',['ci','--ignore-scripts','--no-audit','--no-fund','--registry=https://registry.npmjs.org'],{cwd:root,env,stdio:['ignore','pipe','pipe'],timeout:300000,maxBuffer:8*1024*1024});}
        catch{throw Error('Isolated integrity-checked release tool installation failed');}
        installed.set(tool.root,root);
      }
      tools[channel]=await lockedCLI(installed.get(tool.root),pkg,tool.version);
    }
    return {tools,cleanup:()=>fs.rm(temporary,{recursive:true,force:true})};
  } catch(e){await fs.rm(temporary,{recursive:true,force:true});throw e;}
}
