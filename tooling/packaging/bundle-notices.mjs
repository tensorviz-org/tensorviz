import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

/** Retain license files for packages actually included by esbuild. */
export function writeBundleNotices(root, inputs, output, options = {}) {
  const packages = new Set();
  for (const input of inputs) {
    const parts = input.replaceAll('\\', '/').split('/');
    const index = parts.lastIndexOf('node_modules');
    if (index < 0) continue;
    const end = index + (parts[index + 1].startsWith('@') ? 3 : 2);
    packages.add(resolve(root, parts.slice(0, end).join('/')));
  }
  if (!packages.size) throw new Error('No bundled dependency packages found for license notices.');
  const sections = [...packages].map(directory => {
    const pkg = JSON.parse(readFileSync(join(directory, 'package.json'), 'utf8'));
    const files = readdirSync(directory).filter(name => /^(licen[sc]e|copying|notice)([.-].*)?$/i.test(name)).sort();
    const supplement = options.missing?.[`${pkg.name}@${pkg.version}`];
    if (!files.length && (!supplement || supplement.license !== pkg.license || !supplement.text || !supplement.source)) {
      throw new Error(`Missing license/notice file for bundled ${pkg.name}@${pkg.version}`);
    }
    return { name: `${pkg.name}@${pkg.version}`, text: files.map(name =>
      `${name}\n\n${readFileSync(join(directory, name), 'utf8')}`).join('\n\n') || `Upstream notice: ${supplement.source}\n\n${supplement.text}` };
  }).sort((a, b) => a.name.localeCompare(b.name) || a.text.localeCompare(b.text));
  writeFileSync(output, `# ${options.title ?? 'Bundled extension-host dependency notices'}\n\n` +
    sections.map(({ name, text }) => `## ${name}\n\n${text}`).join('\n\n') + '\n');
}
