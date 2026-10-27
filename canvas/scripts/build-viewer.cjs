#!/usr/bin/env node
// Use the existing production toolchain with a separate entry and output.
const path = require('path');
const fs = require('fs');
process.chdir(path.resolve(__dirname, '..'));
Object.assign(process.env, { NODE_ENV: 'production', BABEL_ENV: 'production', PUBLIC_URL: '.', GENERATE_SOURCEMAP: 'false', REACT_APP_AUTH_DISABLED: 'true' });
const paths = require('react-scripts/config/paths');
paths.appIndexJs = path.resolve('src/viewer/index.tsx');
paths.appHtml = path.resolve('src/viewer/index.html');
paths.appBuild = path.resolve('viewer-dist');
const config = require('react-scripts/config/webpack.config')('production');
// Do not invalidate the live app's webpack cache when building this entry.
config.cache.name = 'graph-viewer-production';
const webpack = require('webpack');
fs.rmSync(paths.appBuild, { recursive: true, force: true });
webpack(config, (error, stats) => {
  if (error || stats.hasErrors() || stats.hasWarnings()) {
    console.error(error || stats.toString({ all: false, errors: true, warnings: true }));
    process.exitCode = 1;
  } else {
    const assets = path.dirname(require.resolve('@tldraw/assets/package.json'));
    for (const name of ['fonts', 'icons', 'translations', 'embed-icons', 'LICENSE.md']) {
      fs.cpSync(path.join(assets, name), path.join(paths.appBuild, 'assets', name), { recursive: true });
    }
    // This entry drives webpack directly, so CRA's build.js never runs and public/
    // is not copied. The viewer's index.html asks for ./favicon.svg, so copy the
    // one asset it needs or the tab falls back to the browser's default mark.
    fs.cpSync(path.join(paths.appPublic, 'favicon.svg'), path.join(paths.appBuild, 'favicon.svg'));
    fs.cpSync(path.join(paths.appPublic, 'font-licenses'), path.join(paths.appBuild, 'font-licenses'), { recursive: true });
    console.log('Graph viewer built in viewer-dist (static files; no backend).');
  }
});
