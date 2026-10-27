'use strict';
if (Number(process.versions.node.split('.')[0]) < 22) {
  process.stderr.write('TensorViz requires Node.js 22 or newer. Update Node.js and restart the coding client.\n');
  process.exitCode = 1;
} else if (!require('node:fs').existsSync(require('node:path').join(__dirname, '../runtime/host.cjs'))) {
  process.stderr.write('TensorViz runtime is missing. Install the complete built plugin release; a source checkout alone cannot launch it.\n');
  process.exitCode = 1;
} else {
  require('../runtime/host.cjs');
}
