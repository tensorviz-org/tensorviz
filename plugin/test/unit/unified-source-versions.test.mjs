import test from 'node:test';
import assert from 'node:assert/strict';
import {unifiedVersionFiles,validateUnifiedSourceVersions} from '../../scripts/unified-source-versions.mjs';
const fixture=version=>Object.fromEntries(unifiedVersionFiles.map(name=>[name,{version,...(name.endsWith('package-lock.json')?{packages:{'':{version}}}:{})}]));
for(const version of ['0.1.46','1.0.0']) test(`read-only alignment accepts ${version} without changing identities`,()=>{const f=fixture(version),before=structuredClone(f);assert.deepEqual(validateUnifiedSourceVersions(f,version),{version,files:9});assert.deepEqual(f,before);});
for(const name of unifiedVersionFiles) test(`rejects stale ${name}`,()=>{const f=fixture('1.0.0');f[name].version='0.1.46';assert.throws(()=>validateUnifiedSourceVersions(f,'1.0.0'),/Source version/);});
for(const name of unifiedVersionFiles.filter(n=>n.endsWith('package-lock.json'))) test(`rejects stale lock root ${name}`,()=>{const f=fixture('1.0.0');f[name].packages[''].version='0.1.46';assert.throws(()=>validateUnifiedSourceVersions(f,'1.0.0'),/Lockfile root/);});
test('missing source and ambiguous expected version fail closed',()=>{for(const version of [undefined,'latest','1.0','01.0.0','1.0.0-rc.1'])assert.throws(()=>validateUnifiedSourceVersions(fixture('1.0.0'),version));assert.throws(()=>validateUnifiedSourceVersions({},'1.0.0'));});
