const fs = require('node:fs');
const path = require('node:path');
const dotenv = require('dotenv');

// Read only the public renderer key; never populate process.env from local files.
module.exports = function pluginLicense(env = process.env, file = path.resolve(__dirname, '../.env.local')) {
  if (env.TENSORVIZ_BROWSER_LICENSE_KEY !== undefined) return env.TENSORVIZ_BROWSER_LICENSE_KEY;
  if (env.REACT_APP_TLDRAW_LICENSE_KEY !== undefined) return env.REACT_APP_TLDRAW_LICENSE_KEY;
  try {
    return dotenv.parse(fs.readFileSync(file)).REACT_APP_TLDRAW_LICENSE_KEY ?? '';
  } catch (error) {
    if (error.code === 'ENOENT') return '';
    throw error;
  }
};
