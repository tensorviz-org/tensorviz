// Verify, never extract, the signed docs artifact before isolated hosting deployment.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {contentHash,inventoryHash} from './source-release-admission.mjs';
const safe=name=>typeof name==='string'&&/^[A-Za-z0-9_.\/-]+$/.test(name)&&!name.split('/').some(p=>!p||p==='.'||p==='..'||p.startsWith('.')||p.startsWith('-'));
export async function verifyDocsArtifact(bytes,{version,sourceRevision,sha256}) {
  if(!Buffer.isBuffer(bytes)||contentHash(bytes)!==sha256||!/^\d+\.\d+\.\d+$/.test(version)||!/^[a-f0-9]{40}$/.test(sourceRevision))throw Error('Documentation artifact checksum/provenance required');
  const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-docs-audit-'));
  try {
    const archive=path.join(temporary,'docs.zip');await fs.writeFile(archive,bytes,{flag:'wx',mode:0o600});
    const read=name=>execFileSync('unzip',['-p',archive,name],{maxBuffer:16*1024*1024});
    const names=execFileSync('unzip',['-Z1',archive],{encoding:'utf8',maxBuffer:4*1024*1024}).trim().split('\n');
    if(names.length>10000||new Set(names).size!==names.length||names.some(n=>!safe(n)))throw Error('Unsafe/duplicate documentation paths');
    if(!names.includes('release-info.json')||!names.includes('index.html')||!names.includes('build-info.json'))throw Error('Documentation entry/provenance missing');
    const proofBytes=read('release-info.json'),proof=JSON.parse(proofBytes);
    if(proof.schemaVersion!==1||proof.source!=='https://github.com/tensorviz-org/tensorviz'||proof.version!==version||proof.sourceRevision!==sourceRevision||proof.base!=='/'||!proof.files)throw Error('Documentation source/version mismatch');
    if(JSON.stringify([...Object.keys(proof.files),'release-info.json'].sort())!==JSON.stringify(names.slice().sort()))throw Error('Documentation inventory differs');
    if(inventoryHash(proof.files)!==proof.contentSha256)throw Error('Documentation inventory checksum differs');
    const files={'release-info.json':proofBytes};let total=proofBytes.length;
    for(const [name,hash] of Object.entries(proof.files)) {
      if(!/\.(?:html|css|js|json|txt|svg|png|jpg|jpeg|webp|ico|woff|woff2|ttf|webmanifest|xml)$/.test(name)||!/^[a-f0-9]{64}$/.test(hash))throw Error('Unapproved documentation file type/hash');
      const content=read(name);total+=content.length;
      if(total>256*1024*1024||contentHash(content)!==hash)throw Error('Documentation content checksum/size differs');
      files[name]=content;
    }
    const build=JSON.parse(files['build-info.json']);
    if(build.source!==proof.source||build.revision!==proof.sourceRevision||build.base!==proof.base)throw Error('Embedded documentation build provenance mismatch');
    return {sha256,contentSha256:proof.contentSha256,files};
  } finally {await fs.rm(temporary,{recursive:true,force:true});}
}

/** Stage exact locally built docs with source provenance; no deploy or tests run. */
export async function prepareDocsArtifact(directory,output,{version,sourceRevision}) {
  if(!/^[1-9]\d*\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(version)||!/^[a-f0-9]{40}$/.test(sourceRevision))throw Error('Exact docs release identity required');
  const buildBytes=await fs.readFile(path.join(directory,'build-info.json'));
  const build=JSON.parse(buildBytes);
  if(build.source!=='https://github.com/tensorviz-org/tensorviz'||build.revision!==sourceRevision||build.base!=='/')throw Error('Local docs build provenance mismatch');
  const files={};let total=0;
  async function walk(relative='') {
    for(const entry of await fs.readdir(path.join(directory,relative),{withFileTypes:true})) {
      const name=relative?`${relative}/${entry.name}`:entry.name;
      if(!safe(name)||name==='release-info.json')throw Error('Unsafe or pre-stamped docs source');
      if(entry.isDirectory())await walk(name);
      else if(entry.isFile()){const b=await fs.readFile(path.join(directory,name));total+=b.length;if(total>256*1024*1024||Object.keys(files).length>=9999)throw Error('Docs build too large');files[name]=b;}
      else throw Error('Linked or special documentation input');
    }
  }
  await walk();
  if(!files['build-info.json']?.equals(buildBytes))throw Error('Documentation build provenance changed while packaging');
  const hashes=Object.fromEntries(Object.entries(files).map(([n,b])=>[n,contentHash(b)]));
  files['release-info.json']=Buffer.from(JSON.stringify({schemaVersion:1,source:'https://github.com/tensorviz-org/tensorviz',sourceRevision,version,base:'/',files:hashes,contentSha256:inventoryHash(hashes)},null,2)+'\n');
  const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'tensorviz-docs-package-'));
  try {
    const root=path.join(temporary,'site');await fs.mkdir(root);
    for(const [name,b] of Object.entries(files)){const file=path.join(root,name);await fs.mkdir(path.dirname(file),{recursive:true});await fs.writeFile(file,b,{flag:'wx'});}
    const archive=path.join(temporary,'docs.zip');
    execFileSync('zip',['-q',archive,'-@'],{cwd:root,input:Object.keys(files).sort().join('\n')+'\n'});
    const bytes=await fs.readFile(archive),sha256=contentHash(bytes);
    await verifyDocsArtifact(bytes,{version,sourceRevision,sha256});
    await fs.writeFile(output,bytes,{flag:'wx'});
    return {version,sourceRevision,sha256,bytes:bytes.length};
  } finally {await fs.rm(temporary,{recursive:true,force:true});}
}
