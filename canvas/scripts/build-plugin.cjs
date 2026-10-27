#!/usr/bin/env node
const path = require('path');
const fs = require('fs');
const { gzipSync } = require('zlib');
const panel = process.argv.includes('--panel');
process.chdir(path.resolve(__dirname, '..'));
// Select the renderer license before clearing inherited frontend configuration.
const licenseKey = require('./plugin-license.cjs')();
// Never import local auth/analytics settings into the plugin build.
for (const key of Object.keys(process.env)) if (key.startsWith('REACT_APP_')) delete process.env[key];
Object.assign(process.env, { NODE_ENV: 'production', BABEL_ENV: 'production', PUBLIC_URL: '.', GENERATE_SOURCEMAP: 'false',
  INLINE_RUNTIME_CHUNK: 'false', REACT_APP_AUTH_DISABLED: 'true', REACT_APP_IDE: 'true',
  REACT_APP_TLDRAW_LICENSE_KEY: licenseKey,
  REACT_APP_TENSORVIZ_VERSION: process.env.TENSORVIZ_RELEASE_VERSION || require('../../plugin/package.json').version });
const paths = require('react-scripts/config/paths');
paths.dotenv = path.resolve('../plugin/out/no-local-env');
paths.appIndexJs = path.resolve(panel ? 'src/plugin/panel.tsx' : 'src/plugin/index.tsx');
paths.appHtml = path.resolve(panel ? 'src/plugin/panel.html' : 'src/plugin/index.html');
paths.appBuild = path.resolve(panel ? '../plugin/out/panel' : '../plugins/tensorviz/browser');
const config = require('react-scripts/config/webpack.config')('production');
config.cache.name = panel ? 'plugin-panel-production' : 'plugin-browser-production';
const webpack = require('webpack');
if (panel) {
  // The MCP resource has no asset origin. Inline only our bundled UI faces.
  const oneOf = config.module.rules.find(rule => rule.oneOf).oneOf;
  oneOf.unshift({ test: /\.woff2$/, include: path.resolve('src/assets/fonts'), type: 'asset/inline' });
  config.optimization.splitChunks = false;
  config.optimization.runtimeChunk = false;
  config.plugins.push(new webpack.optimize.LimitChunkCountPlugin({ maxChunks: 1 }));
}
fs.rmSync(paths.appBuild, { recursive: true, force: true });
webpack(config, (error, stats) => {
  if (error || stats.hasErrors() || stats.hasWarnings()) {
    console.error(error || stats.toString({ all: false, errors: true, warnings: true }));
    process.exitCode = 1;
    return;
  }
  const assets = path.dirname(require.resolve('@tldraw/assets/package.json'));
  if (panel) {
    try {
      const assetFiles = {};
      const walk = folder => {
        for (const entry of fs.readdirSync(path.join(assets, folder), { withFileTypes: true })) {
          const name = `${folder}/${entry.name}`;
          if (entry.isDirectory()) walk(name);
          else {
            const mime = name.endsWith('.woff2') ? 'font/woff2' : name.endsWith('.svg') ? 'image/svg+xml' : 'application/json';
            assetFiles[name] = `data:${mime};base64,${fs.readFileSync(path.join(assets, name)).toString('base64')}`;
          }
        }
      };
      for (const folder of ['fonts', 'icons', 'translations', 'embed-icons']) walk(folder);
      const output = stats.toJson({ all: false, assets: true }).assets.map(asset => asset.name);
      if (output.some(name => name !== 'asset-manifest.json' && !/\.(?:js|css|html|txt)$/.test(name))) throw new Error('The panel includes an external asset; inline it before packaging.');
      const fontLicenses = fs.readdirSync(path.join(paths.appPublic, 'font-licenses')).map(name =>
        fs.readFileSync(path.join(paths.appPublic, 'font-licenses', name), 'utf8')).join('\n');
      const css = `/* ${fontLicenses} */\n` + output.filter(name => name.endsWith('.css')).map(name => fs.readFileSync(path.join(paths.appBuild, name), 'utf8')).join('\n');
      const js = output.filter(name => name.endsWith('.js')).map(name => fs.readFileSync(path.join(paths.appBuild, name), 'utf8')).join('\n');
      // Stay below common MCP stdio message limits without network-loaded assets.
      const payload = JSON.stringify({ css, js, assets: assetFiles });
      if (Buffer.byteLength(payload) > 32 * 1024 * 1024) throw new Error('Decoded panel exceeds its resource budget.');
      const compressed = gzipSync(payload, { level: 9 }).toString('base64');
      const loader = `(async()=>{try{
        const bytes=Uint8Array.from(atob('${compressed}'),c=>c.charCodeAt(0));
        const data=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
        window.__TENSORVIZ_PANEL_FILES__=data.assets;
        const style=document.createElement('style');style.textContent=data.css;document.head.append(style);
        const script=document.createElement('script');script.textContent=data.js;document.body.append(script);
      }catch{document.getElementById('root').textContent='This client could not load Model graph. Ask your agent to open the browser graph.';}})();`;
      const html = fs.readFileSync(paths.appHtml, 'utf8').replace('</body>', `<script>${loader}</script></body>`);
      if (Buffer.byteLength(html) > 5 * 1024 * 1024) throw new Error('Panel exceeds the T3 Code 5 MiB HTML resource limit.');
      if (Buffer.byteLength(JSON.stringify({ text: html })) > 8 * 1024 * 1024) throw new Error('Panel exceeds its MCP resource budget.');
      const destination = path.resolve('../plugins/tensorviz/browser/panel.html');
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      fs.writeFileSync(destination, html);
      console.log(`Native graph panel built with inline Canvas assets (${Buffer.byteLength(html)} bytes).`);
    } catch (error) { console.error(error); process.exitCode = 1; return; }
  }
  if (!panel) {
  for (const name of ['fonts', 'icons', 'translations', 'embed-icons', 'LICENSE.md']) {
    fs.cpSync(path.join(assets, name), path.join(paths.appBuild, 'assets', name), { recursive: true });
  }
  fs.cpSync(path.join(paths.appPublic, 'favicon.svg'), path.join(paths.appBuild, 'favicon.svg'));
  fs.cpSync(path.join(paths.appPublic, 'font-licenses'), path.join(paths.appBuild, 'font-licenses'), { recursive: true });
  }
  fs.mkdirSync(path.resolve('../plugin/out'), { recursive: true });
  fs.writeFileSync(path.resolve(`../plugin/out/${panel ? 'panel' : 'browser'}-modules.json`), JSON.stringify(stats.toJson({ all: false,
    modules: true, cachedModules: true, nestedModules: true, modulesSpace: Infinity, nestedModulesSpace: Infinity,
    groupModulesByType: false, groupModulesByPath: false, groupModulesByCacheStatus: false }), null, 2));
  if (!panel) console.log('Plugin browser built from the shared live Canvas.');
});
