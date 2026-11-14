import { existsSync } from 'node:fs';
import path from 'node:path';

export function nativeBackend(directory: string): boolean {
  return existsSync(path.join(directory, 'native-backend.json'));
}

// This loader is intentionally small and contains no backend implementation.
// The selected interpreter checks its own ABI before loading any native code.
const loader = `import json, pathlib, platform, sys, sysconfig
root, entry, *arguments = sys.argv[1:]
manifest = json.loads((pathlib.Path(root) / 'native-backend.json').read_text())
abi = sysconfig.get_config_var('SOABI')
if abi is None and sys.platform == 'win32':
    suffix = sysconfig.get_config_var('EXT_SUFFIX') or ''
    if suffix.startswith('.cp') and suffix.endswith('.pyd'):
        abi = suffix[1:-4]
if sys.implementation.name != 'cpython' or manifest['abi'] != abi or manifest.get('architecture') != platform.machine():
    print(json.dumps({'ok': False, 'code': 'UNSUPPORTED_PYTHON_ABI', 'message': 'This TensorViz runtime requires ' + manifest['abi'] + '. Select a compatible environment or install a runtime for your Python version.'}))
    sys.exit(1)
sys.path.insert(0, root)
import _tensorviz_backend
sys.path.remove(root)
sys.exit(_tensorviz_backend.run(entry, arguments))
`;

/** Both hosts keep source development support; only native releases use -c. */
export function backendPythonArgs(directory: string, entry: string, args: string[] = [], isolated = true): string[] {
  const flags = isolated ? ['-I', '-B'] : ['-B'];
  if (nativeBackend(directory)) return [...flags, '-c', loader, directory, entry.replaceAll('\\', '/'), ...args];
  return [...flags, path.join(directory, entry), ...args];
}

