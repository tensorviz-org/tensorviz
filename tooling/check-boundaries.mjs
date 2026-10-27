import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const root = fileURLToPath(new URL('..', import.meta.url));

/** Include type-only edges: a package needing editor types is not independent. */
export function importsOf(source) {
  const imports = [];
  const visit = node => {
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier
      && ts.isStringLiteral(node.moduleSpecifier)) imports.push(node.moduleSpecifier.text);
    if (ts.isCallExpression(node) && (node.expression.kind === ts.SyntaxKind.ImportKeyword
      || ts.isIdentifier(node.expression) && node.expression.text === 'require')
      && node.arguments.length === 1 && ts.isStringLiteral(node.arguments[0])) imports.push(node.arguments[0].text);
    if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument)
      && ts.isStringLiteral(node.argument.literal)) imports.push(node.argument.literal.text);
    ts.forEachChild(node, visit);
  };
  visit(ts.createSourceFile('source.ts', source, ts.ScriptTarget.Latest, true));
  return imports;
}

export function contractViolations(filename, source, directory) {
  return importsOf(source).filter(specifier => {
    if (!specifier.startsWith('.')) return true;
    const relative = path.relative(directory, path.resolve(path.dirname(filename), specifier));
    return relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative);
  });
}

export function hostViolations(filename, source, directory, dependencies) {
  return importsOf(source).filter(specifier => {
    if (specifier.startsWith('.')) {
      const relative = path.relative(directory, path.resolve(path.dirname(filename), specifier));
      return relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative);
    }
    const name = specifier.startsWith('@') ? specifier.split('/').slice(0, 2).join('/') : specifier.split('/')[0];
    return !specifier.startsWith('node:') && !dependencies.includes(name);
  });
}

function sources(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const filename = path.join(directory, entry.name);
    return entry.isDirectory() ? sources(filename) : /\.(?:tsx?|[cm]?js)$/.test(entry.name) ? [filename] : [];
  });
}

/** Follow relative source edges, so hiding cloud imports behind a helper still fails. */
export function reachableSources(entries, read = filename => fs.readFileSync(filename, 'utf8'),
  exists = filename => fs.existsSync(filename) && fs.statSync(filename).isFile()) {
  const visited = new Set();
  const visit = filename => {
    if (visited.has(filename)) return;
    visited.add(filename);
    for (const specifier of importsOf(read(filename))) {
      if (!specifier.startsWith('.')) continue;
      const base = path.resolve(path.dirname(filename), specifier);
      const next = [base, `${base}.ts`, `${base}.tsx`, `${base}/index.ts`, `${base}/index.tsx`]
        .find(candidate => /\.tsx?$/.test(candidate) && exists(candidate));
      if (next) visit(next);
    }
  };
  entries.forEach(visit);
  return [...visited];
}

export function checkBoundaries() {
  const directory = path.join(root, 'packages/contracts/src');
  const errors = sources(directory).flatMap(filename => contractViolations(filename, fs.readFileSync(filename, 'utf8'), directory)
    .map(specifier => `${path.relative(root, filename)} imports ${specifier}: contracts must be independent of hosts and dependencies`));
  const host = path.join(root, 'packages/graph-host/src');
  const dependencies = Object.keys(JSON.parse(fs.readFileSync(path.join(host, '../package.json'), 'utf8')).dependencies);
  for (const filename of sources(host)) {
    errors.push(...hostViolations(filename, fs.readFileSync(filename, 'utf8'), host, dependencies)
      .map(specifier => `${path.relative(root, filename)} imports ${specifier}: shared host must use only its declared dependencies`));
  }
  for (const filename of [...sources(path.join(root, 'plugin/src')), ...sources(path.join(root, 'plugin/scripts'))]) {
    for (const specifier of importsOf(fs.readFileSync(filename, 'utf8'))) {
      if (specifier.startsWith('.') && /^(?:extension\/|canvas\/src\/)/.test(path.relative(root,
        path.resolve(path.dirname(filename), specifier)).split(path.sep).join('/'))) {
        errors.push(`${path.relative(root, filename)} imports ${specifier}: plugin must use shared package APIs`);
      }
    }
  }
  const frontend = path.join(root, 'canvas/src');
  for (const filename of reachableSources(['index.tsx', 'plugin/index.tsx',
    'plugin/panel.tsx', 'viewer/index.tsx'].map(entry => path.join(frontend, entry)))) {
    const name = path.relative(frontend, filename).split(path.sep).join('/');
    if (/^(?:hosted\/|components\/auth\/|contexts\/AuthContext\.|lib\/firebase\.|services\/(?:authService|graphStorage|projectsApi|verificationStore)\.)/.test(name)) {
      errors.push(`${name}: local Canvas entries must not depend on hosted accounts or project storage`);
    }
    if (importsOf(fs.readFileSync(filename, 'utf8')).includes('react-router-dom')) {
      errors.push(`${name}: local Canvas entries must mount without a hosted router`);
    }
  }
  if (errors.length) throw new Error(errors.join('\n'));
  console.log('Shared packages and local Canvas entries respect their dependency boundaries.');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) checkBoundaries();
