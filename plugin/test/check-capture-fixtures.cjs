const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

for (const name of ['TENSORVIZ_TEST_PYTHON', 'TENSORVIZ_TEST_SERVICE_PYTHON',
  'TENSORVIZ_TEST_CAPTURE_PYTHON', 'TENSORVIZ_TEST_OTHER_PYTHON', 'TENSORVIZ_TEST_MISSING_PYTHON']) {
  const interpreter = process.env[name];
  assert.ok(interpreter && path.isAbsolute(interpreter) && fs.existsSync(interpreter),
    `Set ${name} to an existing absolute interpreter path; see docs/plugin-build.md#capture-role-tests.`);
}
const inventory = executable => JSON.parse(execFileSync(executable, ['-I', '-c',
  'import json,importlib.metadata as m;print(json.dumps({d.metadata["Name"].lower():d.version for d in m.distributions()}))'], { encoding: 'utf8' }));
for (const name of ['TENSORVIZ_TEST_CAPTURE_PYTHON', 'TENSORVIZ_TEST_OTHER_PYTHON', 'TENSORVIZ_TEST_MISSING_PYTHON']) {
  const installed = inventory(process.env[name]);
  assert.ok(installed.torch && installed.pydantic && !installed.pip && !installed.fastapi && !installed.uvicorn && !installed.libcst,
    `${name} must be a disposable pipless Torch/Pydantic capture environment without tool-server libraries.`);
  if (name === 'TENSORVIZ_TEST_MISSING_PYTHON') {
    assert.ok(!installed.scipy && !installed.torchmetrics && !installed['opencv-python'],
      'TENSORVIZ_TEST_MISSING_PYTHON must be fresh: the recovery test installs its missing model libraries.');
  }
}
assert.equal(new Set(['TENSORVIZ_TEST_CAPTURE_PYTHON', 'TENSORVIZ_TEST_OTHER_PYTHON', 'TENSORVIZ_TEST_MISSING_PYTHON']
  .map(name => fs.realpathSync(path.dirname(path.dirname(process.env[name]))))).size, 3,
  'Environment switching and additive repair require three distinct disposable capture prefixes.');
execFileSync(process.env.TENSORVIZ_TEST_SERVICE_PYTHON, ['-I', '-c', 'import pip,torch,fastapi,uvicorn,libcst,pydantic,dotenv,packaging,numpy,httpx,requests,websockets'], { stdio: 'pipe' });
console.log('Explicit capture-role and prepared-service fixtures are ready.');
