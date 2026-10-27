"""Compile the private backend; never copy generated C or Python into a release."""
import ast
import json
from importlib.metadata import version
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig


NAMESPACE = "_tensorviz_backend"
OWNED = {"api", "core", "middleware", "models", "research", "services", "utils"}
ENTRIES = {
    "standalone_server.py": "standalone_server",
    "services/validation/host_snapshot.py": "services.validation.host_snapshot",
    "services/ide/environment_setup.py": "services.ide.environment_setup",
    "services/ingest/bootstrap.py": "services.ingest.worker",
    "services.ingest.worker": "services.ingest.worker",
}


class PrivateImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if not node.level and node.module and node.module.split(".")[0] in OWNED:
            node.module = f"{NAMESPACE}.{node.module}"
        return node

    def visit_Import(self, node):
        imports = []
        for alias in node.names:
            if alias.name.split(".")[0] in OWNED:
                # Dotted imports without aliases bind their root, not the leaf.
                # Fail the build if one is added rather than changing that binding.
                if "." in alias.name and not alias.asname:
                    raise ValueError(f"Unsupported private import: {alias.name}")
                imports.append(ast.alias(f"{NAMESPACE}.{alias.name}", alias.asname or alias.name))
            else:
                imports.append(alias)
        node.names = imports
        return node


def compile_backend(source, build, destination):
    if sys.platform != 'darwin' or platform.machine() != 'arm64' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('The verified native target requires macOS Apple Silicon and CPython 3.12.')
    if version('nuitka') != '2.8.10':
        raise RuntimeError('Use the pinned Nuitka 2.8.10 compiler; 4.2.2 fails on a Python 3.12 finally/list-comprehension case.')
    source, build, destination = map(Path, (source, build, destination))
    package = build / NAMESPACE
    shutil.rmtree(package, ignore_errors=True)
    shutil.copytree(source, package)
    for file in package.rglob("*.py"):
        tree = PrivateImports().visit(ast.parse(file.read_text(), filename=str(file)))
        file.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    # Only this private package enters sys.modules. User packages retain their names.
    (package / "__init__.py").write_text(
        "import importlib\nimport json\nimport sys\n"
        f"ENTRIES = {ENTRIES!r}\n"
        "def run(entry, arguments):\n"
        "    sys.argv = [entry, *arguments]\n"
        "    if entry == 'services/ide/environment_setup.py' and arguments == ['probe-capture']:\n"
        "        import torch\n"
        "        if torch.nn.Module.__call__ is torch.nn.Module._call_impl:\n"
        "            print(json.dumps(dict(ok=False, code='CAPTURE_RUNTIME_UNSUPPORTED', message='This PyTorch build uses an unsupported legacy module dispatcher. Choose a compatible environment.')))\n"
        "            return 1\n"
        "        importlib.import_module(__name__ + '.services.ingest.worker')\n"
        "        print(json.dumps(dict(ok=True, torch=torch.__version__, cuda=torch.version.cuda, executable=sys.executable)))\n"
        "        return 0\n"
        "    module = importlib.import_module(__name__ + '.' + ENTRIES[entry])\n"
        "    return module.main()\n"
    )
    subprocess.run([sys.executable, "-m", "nuitka", "--module", str(package),
                    f"--include-package={NAMESPACE}", "--nofollow-imports",
                    "--python-flag=no_docstrings", "--assume-yes-for-downloads",
                    "--output-dir=" + str(build / "compiled"), "--jobs=4"],
                   cwd=build, check=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    binary = build / "compiled" / (NAMESPACE + suffix)
    if not binary.is_file():
        raise RuntimeError("Compiler did not produce the expected extension module")
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    shutil.copy2(binary, destination / binary.name)
    # Mach-O's default install name includes the private build directory.
    subprocess.run(['/usr/bin/install_name_tool', '-id', '@rpath/' + binary.name,
                    str(destination / binary.name)], check=True)
    subprocess.run(['/usr/bin/codesign', '--force', '--sign', '-', str(destination / binary.name)], check=True)
    for file in source.rglob("*"):
        if file.is_file() and file.suffix != ".py":
            target = destination / NAMESPACE / file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            # Host provisioning reads the same requirements at the backend root.
            public_data = destination / file.relative_to(source)
            public_data.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, public_data)
    (destination / "native-backend.json").write_text(json.dumps({
        "formatVersion": 1, "namespace": NAMESPACE,
        "abi": sysconfig.get_config_var("SOABI"), "architecture": platform.machine(), "binary": binary.name,
        "entries": list(ENTRIES),
    }, indent=2) + "\n")


if __name__ == "__main__":
    compile_backend(*map(lambda value: Path(value).resolve(), sys.argv[1:]))
