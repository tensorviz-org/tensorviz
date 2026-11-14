import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {renderPublicDocument} from '../../scripts/canonical-public-documents.mjs';
const files=['README.md','BUILD.md','PRIVACY.md','TERMS.md','SUPPORT.md','docs/installation.md'];
for(const name of files) test(`${name}: current source templates preserve0.x and render canonical1.x`,async()=>{
  const template=await fs.readFile(new URL(`../../packaging/public/${name}`,import.meta.url),'utf8');
  assert.equal(renderPublicDocument(template,'0.1.46'),template.replaceAll('{{VERSION}}','0.1.46'));
  const rendered=renderPublicDocument(template,'1.0.0');
  assert.ok(!rendered.includes('{{VERSION}}'));
  assert.ok(!rendered.includes('tensorviz-org/tensorviz-plugins'));
  assert.ok(!rendered.includes('reviewed private source'));
  assert.ok(!rendered.includes('Never register the private source'));
  assert.ok(!rendered.includes('private `tensorviz` source'));
  assert.ok(!rendered.includes('Tags identify audited export commits'));
  if(template.includes('/releases/download/'))assert.ok(rendered.includes('https://github.com/tensorviz-org/tensorviz/releases/download/v1.0.0/'));
});
test('renderer rejects ambiguous versions and preserves explicit historic URLs outside repository migration',()=>{
  for(const value of ['latest','1.0','01.0.0'])assert.throws(()=>renderPublicDocument('x',value));
  assert.equal(renderPublicDocument('v0.3.104 remains Darwin-only','1.0.0'),'v0.3.104 remains Darwin-only');
  const old='https://github.com/tensorviz-org/tensorviz-plugins/releases/download/v0.1.46/archive.zip';
  assert.equal(renderPublicDocument(old,'1.0.0'),old);
});
