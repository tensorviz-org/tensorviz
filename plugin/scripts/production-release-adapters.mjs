// Inactive production adapters. Only admitted immutable bytes enter this module.
// No builds, test jobs, package downloads, source/ref edits or rolling assets.
import fs from 'node:fs/promises';
import {mkdirSync} from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {contentHash,inventoryHash} from './source-release-admission.mjs';
const segment=value=>typeof value==='string'&&/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(value);
const versionOK=value=>/^[1-9]\d*\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(value??'');
const source='https://github.com/tensorviz-org/tensorviz';
const safeFile=value=>typeof value==='string'&&/^[A-Za-z0-9_.\/-]+$/.test(value)&&!value.split('/').some(p=>!p||p==='.'||p==='..'||p.toLowerCase()==='.git');

/** Only installed CLI versions exactly pinned by the reviewed npm lock are used. */
export async function lockedCLI(root,packageName,expectedVersion) {
  if(!['@vscode/vsce','ovsx','firebase-tools'].includes(packageName)||typeof expectedVersion!=='string') throw Error('Unapproved release tool');
  root=await fs.realpath(root);
  const lock=JSON.parse(await fs.readFile(path.join(root,'package-lock.json'),'utf8'));
  const record=lock.packages?.[`node_modules/${packageName}`];
  if(record?.version!==expectedVersion||typeof record.integrity!=='string'||!record.integrity.startsWith('sha512-')) throw Error('Release CLI must be pinned in reviewed lockfile');
  const directory=await fs.realpath(path.join(root,'node_modules',packageName));
  if(!directory.startsWith(`${root}${path.sep}`)) throw Error('CLI package escaped locked installation');
  const manifest=JSON.parse(await fs.readFile(path.join(directory,'package.json'),'utf8'));
  if(manifest.name!==packageName||manifest.version!==expectedVersion) throw Error('Installed release CLI version mismatch');
  const name=packageName==='@vscode/vsce'?'vsce':packageName==='firebase-tools'?'firebase':'ovsx';
  const relative=typeof manifest.bin==='string'?manifest.bin:manifest.bin?.[name];
  if(!safeFile(relative)) throw Error('Invalid release CLI entrypoint');
  const file=await fs.realpath(path.join(directory,relative));
  if(!file.startsWith(`${directory}${path.sep}`)||!(await fs.stat(file)).isFile()) throw Error('CLI entrypoint escaped package');
  return [process.execPath,file];
}

async function get(url,{request,hosts,maxBytes=128*1024*1024}) {
  for(let redirects=0;redirects<6;redirects++) {
    const u=new URL(url);
    if(u.protocol!=='https:'||u.username||u.password||u.hash||!hosts.includes(u.hostname)) throw Error('Unapproved public artifact URL');
    const response=await request(u.href,{redirect:'manual',signal:AbortSignal.timeout(30000)});
    if([301,302,303,307,308].includes(response.status)) {const location=response.headers.get('location');if(!location)throw Error('Artifact redirect missing location');url=new URL(location,u).href;continue;}
    if(response.status===404) return null;
    if(!response.ok) throw Error(`Public lookup failed: HTTP ${response.status}`);
    if(Number(response.headers.get('content-length'))>maxBytes) throw Error('Public artifact too large');
    const reader=response.body?.getReader();
    if(!reader) {const bytes=Buffer.from(await response.arrayBuffer());if(bytes.length>maxBytes)throw Error('Public artifact too large');return bytes;}
    const chunks=[];let size=0;
    while(true){const {done,value}=await reader.read();if(done)break;size+=value.length;if(size>maxBytes){await reader.cancel();throw Error('Public artifact too large');}chunks.push(Buffer.from(value));}
    return Buffer.concat(chunks);
  }
  throw Error('Too many artifact redirects');
}

/** Durable upload-intent journal prevents blind re-upload after timeout/unknown result. */
export function publicationJournal(directory) {
  return {
    async intent(channel,identity) {
      await fs.mkdir(directory,{recursive:true,mode:0o700});
      const file=path.join(directory,`${channel}.json`);
      const value={schemaVersion:1,channel,version:identity.version,sourceRevision:identity.sourceRevision,vsixSha256:identity.vsixSha256,state:'submission-started'};
      try {await fs.writeFile(file,JSON.stringify(value),{flag:'wx',mode:0o600});}
      catch(e){if(e.code!=='EEXIST')throw e;throw Error('Prior submission is pending or uncertain; inspect registry instead of re-uploading');}
    },
  };
}

/** Config comes from reviewed policy; credentials are only supplied to their CLI. */
export function productionReleaseAdapters({identity,vsixBytes,docs,config,tools,credentials={},journal,github,
  request=fetch,command=execFileSync,baseEnvironment=process.env}) {
  if(!versionOK(identity?.version)||!/^[a-f0-9]{40}$/.test(identity?.sourceRevision??'')||!Buffer.isBuffer(vsixBytes)||contentHash(vsixBytes)!==identity.vsixSha256) throw Error('Admitted release bytes required');
  if(config?.enabled!==true||!segment(config.publisher)||!segment(config.extension)||!journal?.intent||!github?.inspect||!github?.publish) throw Error('Reviewed channel activation/configuration required');
  identity={...identity};config=structuredClone(config);vsixBytes=Buffer.from(vsixBytes);
  if(docs)docs={...docs,files:Object.fromEntries(Object.entries(docs.files??{}).map(([name,bytes])=>[name,Buffer.from(bytes)]))};
  const publisher=config.publisher,name=config.extension,version=identity.version;
  const receipt=extra=>({...identity,verified:true,...extra});
  const env={};for(const key of ['PATH','TMPDIR','TMP','TEMP','SystemRoot'])if(baseEnvironment[key])env[key]=baseEnvironment[key];
  const run=(channel,args,extraEnv,cwd)=>{
    const tool=tools[channel];
    if(!Array.isArray(tool)||tool.length!==2||tool[0]!==process.execPath||!path.isAbsolute(tool[1])) throw Error('Locked local release CLI required');
    const home=path.join(cwd,'cli-home');mkdirSync(home,{recursive:true,mode:0o700});
    try{return command(tool[0],[tool[1],...args],{cwd,env:{...env,HOME:home,USERPROFILE:home,XDG_CONFIG_HOME:path.join(home,'.config'),...extraEnv},encoding:'utf8',stdio:['ignore','pipe','pipe'],timeout:300000,maxBuffer:4*1024*1024});}
    catch{throw Error(`${channel} submission failed or timed out; inspect publication status before any retry`);}
  };
  const hosts=Array.from(new Set(['marketplace.visualstudio.com','open-vsx.org',...(config.allowedDownloadHosts??[])]));
  const inspectPackage=async(channel)=>{
    let url;
    if(channel==='vscode-marketplace')url=`https://marketplace.visualstudio.com/_apis/public/gallery/publishers/${publisher}/vsextensions/${name}/${version}/vspackage`;
    else {
      const metadata=await get(`https://open-vsx.org/api/${publisher}/${name}/${version}`,{request,hosts,maxBytes:1024*1024});
      if(!metadata)return null;
      const data=JSON.parse(metadata);
      if(data.namespace!==publisher||data.name!==name||data.version!==version||typeof data.files?.download!=='string')throw Error('Open VSX identity mismatch');
      url=data.files.download;
    }
    const bytes=await get(url,{request,hosts});if(!bytes)return null;
    if(contentHash(bytes)!==identity.vsixSha256)throw Error('Registry package differs from admitted immutable VSIX; never overwrite or normalize');
    return receipt({url,vsixSha256:contentHash(bytes)});
  };
  const publishPackage=async(channel)=>{
    // Recheck immediately before submission, including conflict detection.
    if(await inspectPackage(channel))return;
    const secretName=channel==='vscode-marketplace'?'VSCE_PAT':'OVSX_PAT';
    if(typeof credentials[secretName]!=='string'||!credentials[secretName].trim())throw Error(`${secretName} required for approved publisher`);
    if(!tools[channel])throw Error('Locked local release CLI required');
    const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-registry-'));
    try {
      const file=path.join(tmp,`tensorviz-${version}.vsix`);await fs.writeFile(file,vsixBytes,{flag:'wx',mode:0o600});
      await journal.intent(channel,identity);
      run(channel,channel==='vscode-marketplace'?['publish','--packagePath',file,'--skip-duplicate']:['publish',file,'--registryUrl','https://open-vsx.org'],{[secretName]:credentials[secretName]},tmp);
    } finally {await fs.rm(tmp,{recursive:true,force:true});}
  };
  const docsAdapter={
    async inspect(){
      validateDocs();
      const base=`https://${config.docs.site}.web.app`;
      const proof=await get(`${base}/release-info.json`,{request,hosts:[`${config.docs.site}.web.app`],maxBytes:1024*1024});
      if(!proof)return null;
      const published=JSON.parse(proof);
      // Older docs mean not deployed yet; newer/different same-version docs are conflicts.
      if(published.version!==version) {
        if(!versionOK(published.version)||compareVersions(published.version,version)>=0)throw Error('Docs already advertise a conflicting/newer release');
        return null;
      }
      if(!proof.equals(docs.files['release-info.json']))throw Error('Published docs provenance differs');
      for(const [file,bytes] of Object.entries(docs.files)) {
        const actual=await get(`${base}/${file.split('/').map(encodeURIComponent).join('/')}`,{request,hosts:[`${config.docs.site}.web.app`]});
        if(!actual||contentHash(actual)!==contentHash(bytes))throw Error(`Published docs differ: ${file}`);
      }
      return receipt({url:base,docsSha256:docs.sha256});
    },
    async publish(){
      validateDocs();if(await this.inspect())return;
      if(!credentials.GOOGLE_APPLICATION_CREDENTIALS)throw Error('Approved isolated-project workload credentials required');
      let adc;
      try {adc=JSON.parse(await fs.readFile(credentials.GOOGLE_APPLICATION_CREDENTIALS,'utf8'));}
      catch {throw Error('Approved workload credential file could not be read or parsed');}
      if(adc.type!=='external_account'||adc.audience!==config.docs.workloadIdentityAudience
        ||adc.service_account_impersonation_url!==`https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/${config.docs.serviceAccount}:generateAccessToken`
        ||adc.token_url!=='https://sts.googleapis.com/v1/token'||adc.subject_token_type!=='urn:ietf:params:oauth:token-type:jwt')throw Error('Docs credentials do not match the isolated reviewed workload identity');
      // Accept only the job's exact GitHub OIDC source, never attacker-selected
      // files, executable providers, token endpoints or arbitrary credential URLs.
      const expected=new URL(credentials.githubOidcRequestUrl??'https://invalid.invalid');
      if(expected.protocol!=='https:'||!expected.hostname.endsWith('.actions.githubusercontent.com')||expected.username||expected.password||!credentials.githubOidcRequestToken)throw Error('Verified GitHub OIDC request context required');
      expected.searchParams.set('audience',config.docs.oidcAudience);
      const cs=adc.credential_source;
      if(cs?.url!==expected.href||cs.format?.type!=='json'||cs.format?.subject_token_field_name!=='value'||Object.keys(cs.format??{}).sort().join(',')!=='subject_token_field_name,type'
        ||Object.keys(cs).sort().join(',')!=='format,headers,url'||Object.keys(cs.headers??{}).join(',')!=='Authorization'
        ||cs.headers.Authorization!==`Bearer ${credentials.githubOidcRequestToken}`)throw Error('Unapproved workload credential source');
      const sanitized={type:'external_account',audience:adc.audience,subject_token_type:adc.subject_token_type,token_url:adc.token_url,
        service_account_impersonation_url:adc.service_account_impersonation_url,credential_source:{url:expected.href,headers:{Authorization:cs.headers.Authorization},format:{type:'json',subject_token_field_name:'value'}}};
      const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-docs-release-'));
      try {
        const credentialSnapshot=path.join(tmp,'workload-identity.json');await fs.writeFile(credentialSnapshot,JSON.stringify(sanitized),{flag:'wx',mode:0o600});
        const root=path.join(tmp,'public');await fs.mkdir(root);
        for(const [file,bytes] of Object.entries(docs.files)){const target=path.join(root,file);await fs.mkdir(path.dirname(target),{recursive:true});await fs.writeFile(target,bytes,{flag:'wx'});}
        const firebase={hosting:{site:config.docs.site,public:'public',ignore:[],headers:[{source:'**',headers:[{key:'Cache-Control',value:'no-cache'}]}]}};
        await fs.writeFile(path.join(tmp,'firebase.json'),JSON.stringify(firebase),{flag:'wx'});
        await journal.intent('docs',identity);
        run('docs',['deploy','--only','hosting','--project','tensorviz-docs','--config',path.join(tmp,'firebase.json'),'--non-interactive'],{GOOGLE_APPLICATION_CREDENTIALS:credentialSnapshot},tmp);
      } finally {await fs.rm(tmp,{recursive:true,force:true});}
    },
  };
  function validateDocs(){
    if(config.docs?.project!=='tensorviz-docs'||!segment(config.docs.site)||!/^[-a-z0-9]+@tensorviz-docs\.iam\.gserviceaccount\.com$/.test(config.docs.serviceAccount??'')||!/^\/\/iam\.googleapis\.com\/projects\/\d+\/locations\/global\/workloadIdentityPools\/[^/]+\/providers\/[^/]+$/.test(config.docs.workloadIdentityAudience??'')||config.docs.oidcAudience!==`https:${config.docs.workloadIdentityAudience}`||!docs?.files||!Buffer.isBuffer(docs.files['release-info.json']))throw Error('Exact isolated documentation artifact/site required');
    const proof=JSON.parse(docs.files['release-info.json']);
    if(docs.sha256!==identity.docsSha256||!/^[a-f0-9]{64}$/.test(docs.sha256??''))throw Error('Documentation archive is not bound to admitted release');
    if(proof.version!==version||proof.sourceRevision!==identity.sourceRevision||proof.source!==source||proof.base!=='/')throw Error('Documentation source/version mismatch');
    if(!proof.files||proof.contentSha256!==inventoryHash(proof.files)||JSON.stringify([...Object.keys(proof.files),'release-info.json'].sort())!==JSON.stringify(Object.keys(docs.files).sort()))throw Error('Documentation inventory differs');
    for(const [file,bytes] of Object.entries(docs.files))if(!safeFile(file)||!Buffer.isBuffer(bytes)||(file!=='release-info.json'&&proof.files[file]!==contentHash(bytes)))throw Error('Unsafe or changed documentation artifact');
  }
  return {github,'vscode-marketplace':{inspect:()=>inspectPackage('vscode-marketplace'),publish:()=>publishPackage('vscode-marketplace')},'open-vsx':{inspect:()=>inspectPackage('open-vsx'),publish:()=>publishPackage('open-vsx')},docs:docsAdapter};
}
function compareVersions(a,b){for(let i=0;i<3;i++){const d=Number(a.split('.')[i])-Number(b.split('.')[i]);if(d)return d;}return 0;}
