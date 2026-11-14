"""Plan reviewed dependency installs; model probing is an explicit worker action.

The host owns consent. A plan freezes the environment and wheel hashes; installation
refuses changed plans/environments. Version replacement is limited to marked copies,
or to the selected environment itself when the host's review explicitly approves it.
"""
from __future__ import annotations

import hashlib
import ast
from email.parser import BytesParser
import importlib.util
import importlib.metadata as metadata
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from contextlib import contextmanager
from urllib.parse import unquote, urlparse

class SetupError(ValueError):
    """A safe, user-facing setup failure."""
    def __init__(self, message: str, code: str = "DEPENDENCY_CONFLICT", requirement: str | None = None):
        super().__init__(message)
        self.code = code
        # The unavailable requirement pip named, when it named one; never pip output.
        self.requirement = requirement


REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements-project.txt"
CAPTURE_REQUIREMENTS = REQUIREMENTS.parent / "requirements-capture.txt"
SERVICE_REQUIREMENTS = REQUIREMENTS.parent / "requirements-standalone.txt"
# One helper process serves one explicit target. The manager owns pip/packaging;
# all environment facts and installed packages must come from the target.
TARGET_PYTHON: str | None = None

TARGET_INSPECTION = r'''
import hashlib, importlib.metadata as metadata, importlib.util, json, os, pathlib, platform, re, sys, sysconfig
abi = sysconfig.get_config_var('SOABI')
if abi is None and sys.platform == 'win32':
    suffix = sysconfig.get_config_var('EXT_SUFFIX') or ''
    if suffix.startswith('.cp') and suffix.endswith('.pyd'):
        abi = suffix[1:-4]
packages = {}
for dist in metadata.distributions():
    name = dist.metadata.get('Name')
    if name:
        key = re.sub(r'[-_.]+', '-', name).lower()
        if key in packages and packages[key] != dist.version:
            raise ValueError('Multiple installed versions')
        packages[key] = dist.version
stat = pathlib.Path(sys.executable).stat()
config = pathlib.Path(sys.prefix) / 'pyvenv.cfg'
identity = dict(executable=sys.executable, prefix=sys.prefix, python=sys.version.split()[0],
    implementation=sys.implementation.name, abi=abi,
    platform=sys.platform, architecture=platform.machine(),
    executableStat=[stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
    venvConfig=hashlib.sha256(config.read_bytes()).hexdigest() if config.is_file() else None,
    packages=dict(sorted(packages.items())))
modules = {}
for name in json.loads(sys.argv[1]):
    try:
        modules[name] = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        modules[name] = False
print(json.dumps(dict(identity=identity, modules=modules, stdlib=list(sys.stdlib_module_names),
    basePrefix=sys.base_prefix, conda=(pathlib.Path(sys.prefix) / 'conda-meta').is_dir(),
    externallyManaged=(pathlib.Path(sysconfig.get_path('stdlib')) / 'EXTERNALLY-MANAGED').is_file(),
    writable=os.access(sys.prefix, os.W_OK))))
'''


#: Set only while a read-only inspection runs (see ``one_snapshot``): the target
#: snapshots taken so far in that inspection, as (modules, snapshot) pairs.
_SNAPSHOTS: list[tuple[frozenset[str], dict]] | None = None


@contextmanager
def one_snapshot():
    """Read the target environment once for one read-only inspection.

    ``inspect-capture`` asked the target for its inventory, its environment facts
    and its identity separately: three Python processes returning the same
    snapshot. Inside this scope a later call reuses an earlier snapshot that
    already answered for at least the requested modules, so one inspection sees one
    consistent view of the environment. Actions that change the environment
    (plans, installs) compare snapshots taken before and after a change and never
    run inside this scope."""
    global _SNAPSHOTS
    _SNAPSHOTS = []
    try:
        yield
    finally:
        _SNAPSHOTS = None


def target_snapshot(modules: list[str] | None = None) -> dict:
    wanted = frozenset(modules or [])
    if _SNAPSHOTS is not None:
        for answered, snapshot in _SNAPSHOTS:
            if wanted <= answered:
                return snapshot
    value = _target_snapshot(sorted(wanted) if modules is not None else modules)
    if _SNAPSHOTS is not None:
        _SNAPSHOTS.append((wanted, value))
    return value


def _target_snapshot(modules: list[str] | None = None) -> dict:
    try:
        result = subprocess.run([TARGET_PYTHON, '-I', '-B', '-c', TARGET_INSPECTION,
                                 json.dumps(modules or [])], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        if result.returncode or len(result.stdout) > 512 * 1024:
            raise ValueError()
        value = json.loads(result.stdout)
        if not isinstance(value.get('identity', {}).get('packages'), dict):
            raise ValueError()
        return value
    except (OSError, ValueError, subprocess.TimeoutExpired):
        raise SetupError('The selected Python environment could not be inspected. Choose another environment.',
                         'ENVIRONMENT_UNAVAILABLE') from None


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def inventory() -> dict[str, str]:
    if TARGET_PYTHON:
        return target_snapshot()['identity']['packages']
    result = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            key = canonical(name)
            if key in result and result[key] != dist.version:
                raise SetupError(f"Multiple installed versions of {key}; choose another environment.")
            result[key] = dist.version
    return dict(sorted(result.items()))


def identity() -> dict:
    if TARGET_PYTHON:
        return target_snapshot()['identity']
    return {"executable": sys.executable, "prefix": sys.prefix,
            "python": sys.version.split()[0], "packages": inventory()}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def interpreter_abi():
    abi = sysconfig.get_config_var('SOABI')
    if abi is None and sys.platform == 'win32':
        suffix = sysconfig.get_config_var('EXT_SUFFIX') or ''
        if suffix.startswith('.cp') and suffix.endswith('.pyd'):
            abi = suffix[1:-4]
    return abi


def environment_info(installed: dict[str, str] | None = None) -> dict:
    if TARGET_PYTHON:
        target = target_snapshot()['identity']
        return {key: target[key] for key in ('executable', 'prefix', 'python', 'platform', 'architecture', 'implementation', 'abi')} | {
            'torch': target['packages'].get('torch'), 'pip': target['packages'].get('pip')}
    if installed is None:
        installed = {}
        for name in ("torch", "pip"):
            try:
                installed[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
    return {"executable": sys.executable, "prefix": sys.prefix, "python": sys.version.split()[0],
            "platform": sys.platform, "architecture": platform.machine(),
            "implementation": sys.implementation.name, "abi": interpreter_abi(),
            "torch": installed.get("torch"), "pip": installed.get("pip")}


def inspect_environment(requirements: Path = REQUIREMENTS) -> dict:
    installed = inventory()
    names = [line.strip() for line in requirements.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    incompatible = []
    try:
        from packaging.requirements import Requirement
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement
        except ImportError:
            Requirement = None
    environment = environment_info(installed)
    if Requirement is not None:
        markers = {'python_version': '.'.join(environment['python'].split('.')[:2]),
                   'python_full_version': environment['python']}
        names = [text for text in names if not Requirement(text).marker or Requirement(text).marker.evaluate(markers)]
    missing = [requirement for requirement in names
               if canonical(re.match(r"[\w.-]+", requirement)[0]) not in installed]
    if Requirement is not None:
        for text in names:
            requirement = Requirement(text)
            version = installed.get(canonical(requirement.name))
            if version and not requirement.specifier.contains(version, prereleases=True):
                incompatible.append(f"{requirement.name} {version} (needs {requirement.specifier})")
    minimum = (3, 10) if requirements == CAPTURE_REQUIREMENTS else (3, 11)
    if tuple(map(int, environment['python'].split('.')[:2])) < minimum:
        incompatible.insert(0, f"Python {environment['python']} (needs {minimum[0]}.{minimum[1]} or newer)")
    return {"python": environment['python'], "executable": environment['executable'],
            "prefix": environment['prefix'], "pip": "pip" in installed,
            "torch": installed.get("torch"), "missing": missing, "incompatible": incompatible,
            "environment": environment, "fingerprint": digest(json.dumps(identity(), sort_keys=True).encode())}


def torch_unavailable(stderr: str) -> str:
    """PyPI has never published a PyTorch sdist, so a source build cannot supply a missing wheel."""
    info = environment_info()
    system = {"darwin": "macOS", "win32": "Windows", "linux": "Linux"}.get(info["platform"], info["platform"])
    # Treat installer output as untrusted. Only validated PEP 440 versions may
    # reach the UI; never echo arbitrary text from the index or rely on list order.
    found = re.search(r"\(from versions: ([^\r\n)]{1,4000})\)", stderr)
    newest = ""
    if found and found.group(1).strip() == "none":
        newest = "; it has none for this platform"
    elif found:
        try:
            try:
                from packaging.version import Version
            except ImportError:
                from pip._vendor.packaging.version import Version
            versions = [Version(value.strip()) for value in found.group(1).split(",")]
            if versions:
                newest = f"; the newest available there is {max(versions)}"
        except (ImportError, ValueError):
            pass  # Missing parser or any malformed entry: omit index detail entirely.
    return (f"The package index has no PyTorch build matching this requirement for {system} {info['architecture']} "
            f"with Python {info['python']}{newest}. PyTorch is published only as prebuilt wheels, so no source build "
            "is offered. Select an environment that already has a compatible PyTorch.")


def pip_run(arguments: list[str]) -> None:
    result = subprocess.run([sys.executable, "-m", "pip", *(['--python', TARGET_PYTHON] if TARGET_PYTHON else []), "--disable-pip-version-check",
                             "--isolated", "--no-input", *arguments], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            env={**os.environ, "PIP_CONFIG_FILE": os.devnull})
    if result.returncode:
        # pip output can contain private index credentials. Do not forward it to UI/logs.
        detail = result.stderr.lower()
        if 'no module named pip' in detail:
            raise SetupError("The package installer is unavailable. Repair the tensorViz tool runtime or choose an environment with pip.", "INSTALLER_UNAVAILABLE")
        if any(text in detail for text in ("connectionerror", "connection refused", "temporary failure in name resolution", "certificate_verify_failed", "proxyerror")):
            raise SetupError("pip could not reach the package index. Check network access for this environment and retry.", "PACKAGE_INDEX_UNREACHABLE")
        if "resolutionimpossible" in detail or "conflicting dependencies" in detail:
            raise SetupError("The requested dependencies conflict with installed versions. Adjust the requirements; installed packages were not replaced.")
        if "no such option" in detail:
            raise SetupError("This environment's pip does not support the required planning options.", "PIP_UNSUPPORTED")
        if "no matching distribution found" in detail or "could not find a version that satisfies" in detail:
            named = re.search(r"no matching distribution found for ([A-Za-z0-9][A-Za-z0-9_.,<>=!~*\[\] -]{0,120})", result.stderr, re.IGNORECASE)
            requirement = named.group(1).strip() if named else None
            if requirement and canonical(re.match(r"[\w.-]+", requirement)[0]) == "torch":
                raise SetupError(torch_unavailable(result.stderr), "PACKAGE_UNAVAILABLE", requirement)
            if "--only-binary=:all:" in arguments:
                raise SetupError("No compatible wheel satisfies these requirements in this environment. A reviewed source build may be required.", "NO_COMPATIBLE_WHEEL", requirement)
            raise SetupError("No distribution satisfies these requirements for this Python/platform. Check the package names and supported versions.", "PACKAGE_UNAVAILABLE", requirement)
        if any(text in detail for text in ("failed building wheel", "subprocess-exited-with-error", "metadata-generation-failed")):
            raise SetupError("Package build or metadata preparation failed in this environment. Check its Python version and required build tools.", "SOURCE_BUILD_FAILED")
        raise SetupError("pip could not complete dependency setup in this environment. No different interpreter was selected.", "DEPENDENCY_SETUP_FAILED")


def check_install_target() -> None:
    if not TARGET_PYTHON:
        return
    target = target_snapshot()
    if (not target['writable'] or target['identity']['prefix'] == target['basePrefix'] and
            (target['externallyManaged'] or not target['conda'])):
        raise SetupError('Choose a writable project virtual environment or Conda environment before installing libraries.',
                         'ENVIRONMENT_NOT_WRITABLE')


def archive_sha256(download: dict) -> str:
    archive = download.get("archive_info", {})
    # pip 23.0 (bundled with Python 3.10) reports the older singular hash field.
    legacy = archive.get("hash", "")
    legacy = legacy.removeprefix("sha256=") if legacy.startswith("sha256=") else ""
    checksum = archive.get("hashes", {}).get("sha256", legacy)
    if legacy and checksum.lower() != legacy.lower():
        return ""  # Conflicting evidence must not become an approved artifact.
    return checksum


def validate_wheels(report: dict, baseline: dict[str, str]) -> list[dict]:
    wheels = []
    seen = set()
    for item in report.get("install", []):
        name = canonical(item["metadata"]["name"])
        if name in baseline or name in seen:
            raise SetupError(f"Refusing to replace installed package {name}.")
        seen.add(name)
        url = item["download_info"]["url"]
        checksum = archive_sha256(item["download_info"])
        if (urlparse(url).scheme not in {"https", "file"}
                or not unquote(urlparse(url).path).endswith(".whl")
                or not re.fullmatch(r"[a-fA-F0-9]{64}", checksum)
                or any(c.isspace() for c in url)):
            raise SetupError(f"No verified wheel is available for {name}.")
        wheels.append({"name": name, "version": item["metadata"]["version"],
                       "url": url, "sha256": checksum})
    return wheels


def build_source_wheels(report: dict, directory: Path, baseline: dict[str, str]) -> list[str]:
    """Called only after host approval to execute package build code."""
    built_sources = []
    for item in report.get("install", []):
        name, version = canonical(item["metadata"]["name"]), item["metadata"]["version"]
        download = item["download_info"]
        url = download["url"]
        if unquote(urlparse(url).path).endswith(".whl"):
            continue
        checksum = archive_sha256(download)
        if (name in baseline or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", name)
                or urlparse(url).scheme not in {"https", "file"} or any(c.isspace() for c in url)
                or not re.fullmatch(r"[a-fA-F0-9]{64}", checksum)):
            raise SetupError("The source distribution could not be verified; no installation was planned.")
        output = directory / "built" / name
        output.mkdir(parents=True, exist_ok=False)
        locked = output / "source.txt"
        locked.write_text(f"{url} --hash=sha256:{checksum}\n")
        pip_run(["wheel", "--no-deps", "--no-cache-dir", "--require-hashes", "--wheel-dir", str(output), "-r", str(locked)])
        wheels = list(output.glob("*.whl"))
        if len(wheels) != 1:
            raise SetupError(f"The build for {name} did not produce exactly one wheel.", "SOURCE_BUILD_FAILED")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            entries = [entry for entry in archive.namelist() if entry.endswith('.dist-info/METADATA')]
            if len(entries) != 1:
                raise SetupError(f"The wheel for {name} has invalid metadata.", "SOURCE_BUILD_FAILED")
            built = BytesParser().parsebytes(archive.read(entries[0]))
        if canonical(built.get("Name", "")) != name or built.get("Version") != version:
            raise SetupError(f"The wheel for {name} does not match the resolved package version.", "SOURCE_BUILD_FAILED")
        try:
            from packaging.requirements import Requirement
        except ImportError:
            from pip._vendor.packaging.requirements import Requirement
        expected = sorted(str(Requirement(value)) for value in item["metadata"].get("requires_dist", []))
        actual = sorted(str(Requirement(value)) for value in built.get_all("Requires-Dist", []))
        if expected != actual:
            raise SetupError(f"The dependencies for {name} changed during its build. Resolve a new plan.", "SOURCE_BUILD_FAILED")
        with wheel.open('rb') as stream:
            checksum = hashlib.sha256()
            for chunk in iter(lambda: stream.read(65536), b""):
                checksum.update(chunk)
            checksum = checksum.hexdigest()
        item["download_info"] = {"url": wheel.resolve().as_uri(), "archive_info": {"hashes": {"sha256": checksum}}}
        built_sources.append(name)
    return built_sources


def plan(directory: Path, requirements: Path = REQUIREMENTS, *, allow_source: bool = False,
         replace: set[str] = frozenset(), project: Path | None = None) -> dict:
    check_install_target()
    baseline = identity()
    requested_digest = digest(requirements.read_bytes())
    if replace and (not project or replace != compatibility_replacements()):
        raise SetupError("Version changes are only allowed in a disposable compatibility environment.")
    # Only changing libraries need new project constraints. Unrelated/demo
    # requirements must not turn this repair into a whole-project installation.
    declarations = {name: value for name, value in project_requirements(project).items() if name in replace} if project else {}
    if any(name in declarations and declarations[name] is None for name in replace):
        raise SetupError("A model library uses an unsupported project requirement. Set up that environment manually.")
    preserved = {name: version for name, version in baseline["packages"].items() if name not in replace}
    constraints = directory / "existing.txt"
    # Freezing every installed distribution also protects NumPy, torchvision,
    # torchaudio, triton and nvidia-* against transitive dependency upgrades.
    constraints.write_text("".join(f"{name}==={version}\n"
                                   for name, version in preserved.items())
                           + "".join(re.sub(r"\[[^]]+\]", "", value) + "\n" for value in declarations.values() if value))
    report_path = directory / "pip-report.json"
    pip_run(["install", "--dry-run", *([] if allow_source else ["--only-binary=:all:"]), "--report", str(report_path),
             "--constraint", str(constraints), "-r", str(requirements)])
    report = json.loads(report_path.read_text())
    built_sources = build_source_wheels(report, directory, baseline["packages"]) if allow_source else []
    wheels = validate_wheels(report, preserved)
    if identity() != baseline:
        raise SetupError("The environment changed during inspection. Check it again.")
    if digest(requirements.read_bytes()) != requested_digest:
        raise SetupError('The requested requirements changed during planning. Prepare a new installation plan.')
    data = json.dumps({"baseline": baseline, "wheels": wheels,
                       "requirements": {"path": str(requirements.resolve()), "digest": requested_digest},
                       **({"replace": sorted(replace), "project": str(project), "declarations": declarations} if replace else {})}, sort_keys=True).encode()
    (directory / "plan.json").write_bytes(data)
    return {"digest": digest(data), "packages": [{"name": w["name"], "version": w["version"]}
                                                 for w in wheels], "builtFromSource": built_sources}


def install(directory: Path, approved_digest: str) -> dict:
    check_install_target()
    data = (directory / "plan.json").read_bytes()
    if digest(data) != approved_digest:
        raise SetupError("The installation plan changed. Review a new plan before installing.")
    saved = json.loads(data)
    if saved.get('modelProject') and project_manifest_digest(Path(saved['modelProject'])) != saved['manifestDigest']:
        raise SetupError('Project requirements changed after review. Prepare the model environment again.')
    if saved.get('requirements') and digest(Path(saved['requirements']['path']).read_bytes()) != saved['requirements']['digest']:
        raise SetupError('The requested requirements changed after review. Prepare a new installation plan.')
    baseline = saved["baseline"]
    replace = set(saved.get("replace", []))
    if replace and not saved.get("inPlace") and replace != compatibility_replacements():
        raise SetupError("The compatibility environment no longer matches the reviewed plan.")
    if replace and {name: value for name, value in project_requirements(Path(saved['project'])).items() if name in replace} != saved['declarations']:
        raise SetupError("Project requirements changed after review. Check the model dependencies again.")
    if identity() != baseline:
        raise SetupError("The environment changed after review. Check it again before installing.")
    # Revalidate the frozen plan; only the reviewed compatibility set may change.
    wheels = saved["wheels"]
    if any(w["name"] in baseline["packages"] and w["name"] not in replace for w in wheels):
        raise SetupError("The plan would replace an installed package.")
    locked = directory / "approved.txt"
    locked.write_text("".join(f'{w["url"]} --hash=sha256:{w["sha256"]}\n' for w in wheels))
    if wheels:
        pip_run(["install", "--no-deps", "--no-index", "--require-hashes",
                 "--only-binary=:all:", "--no-user", "--prefix", baseline['prefix'], "-r", str(locked)])
    after = inventory()
    if any(after.get(name) != version for name, version in baseline["packages"].items() if name not in replace):
        raise SetupError("Installed versions changed during setup. Stop and inspect this environment.")
    if any(after.get(w["name"]) != w["version"] for w in wheels):
        raise SetupError("Setup did not install every approved library. Check the environment again.")
    return {"installed": len(wheels)}


def update_managed(directory: Path, requirements: Path = SERVICE_REQUIREMENTS) -> dict:
    """Maintain an already-consented tensorViz runtime without replacing its packages."""
    resolved = plan(directory, requirements)
    return install(directory, resolved["digest"])


def package_requirements(values: object) -> list[str]:
    """Accept PyPI names/version constraints, never pip options, paths or URLs."""
    if not isinstance(values, list) or not 1 <= len(values) <= 32:
        raise SetupError("Provide between 1 and 32 package requirements.")
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    result = []
    for value in values:
        if (not isinstance(value, str) or len(value) > 200
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?(?:[<>=!~][A-Za-z0-9.*+<>=!~, -]*)?", value)):
            raise SetupError("Use package names with optional version constraints, not paths, URLs or pip options.")
        try:
            requirement = Requirement(value)
        except ValueError:
            raise SetupError("Invalid package requirement.") from None
        if requirement.url or requirement.marker:
            raise SetupError("URLs and environment markers are not supported by automatic setup.")
        result.append(str(requirement))
    return result


def plan_packages(directory: Path, requested: object, *, allow_source: bool = False, managed_cpu: bool = False) -> dict:
    requirements = directory / "requested.txt"
    packages = package_requirements(requested)
    if managed_cpu:
        # Only new TensorViz-owned environments opt in. User environments retain
        # their selected Torch/CUDA builds and the ordinary additive planner.
        packages = ["--extra-index-url https://download.pytorch.org/whl/cpu", *packages]
    requirements.write_text("\n".join(packages) + "\n")
    result = plan(directory, requirements, allow_source=allow_source)
    if managed_cpu and any(item['name'].startswith('nvidia-') or item['name'] == 'triton'
                           for item in result.get('packages', [])):
        raise SetupError("The managed CPU plan contains GPU dependencies. No installation was approved.")
    return result


# Import names are not generally distribution names. Only these known mappings
# may supplement a project's explicit requirements during guided graph opening.
MODEL_PACKAGES = {name: name for name in (
    "torch", "torchvision", "torchaudio", "transformers", "diffusers", "accelerate",
    "safetensors", "huggingface_hub", "tokenizers", "sentencepiece", "peft", "timm",
    "einops", "numpy", "scipy", "pandas", "matplotlib", "lightning", "pytorch_lightning",
    "torch_geometric", "mmengine", "omegaconf", "tqdm", "packaging", "torchmetrics",
    "jax", "jaxlib", "flax", "optax",
)}
MODEL_PACKAGES.update({"PIL": "pillow", "cv2": "opencv-python", "sklearn": "scikit-learn", "yaml": "pyyaml"})

# Match the real-model acceptance environment. Project constraints always win.
# A failed import may try this profile in a copy, never by replacing user packages.
MODEL_COMPATIBILITY = {"transformers": ("4.57.6", {"transformers", "tokenizers", "huggingface-hub"})}
COMPATIBILITY_MARKER = "tensorviz-compatibility.json"


def compatibility_replacements() -> set[str]:
    prefix = environment_info()['prefix'] if TARGET_PYTHON else sys.prefix
    marker = Path(prefix) / COMPATIBILITY_MARKER
    if not marker.is_file():
        return set()
    saved = json.loads(marker.read_text())
    if saved.get("prefix") != str(Path(prefix).resolve()):
        return set()
    return set().union(*(MODEL_COMPATIBILITY[name][1] for name in saved.get("packages", []) if name in MODEL_COMPATIBILITY))


def clone_compatibility_environment(destination: Path, requested: object) -> dict:
    packages = package_requirements(requested)
    names = [name for name, (version, _) in MODEL_COMPATIBILITY.items() if f"{name}=={version}" in packages]
    info = target_snapshot() if TARGET_PYTHON else {'identity': {'prefix': sys.prefix, 'executable': sys.executable}, 'basePrefix': sys.base_prefix}
    prefix = info['identity']['prefix']
    if len(names) != len(packages) or prefix == info['basePrefix']:
        raise SetupError("Use a separate tensorViz environment before repairing model library versions.")
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(Path(prefix).resolve()):
        raise SetupError("A compatibility environment needs a new, separate directory.")
    # Dereference links: package paths in the copy must never write through to
    # the source environment.
    shutil.copytree(prefix, destination, symlinks=False, ignore=shutil.ignore_patterns('__pycache__', COMPATIBILITY_MARKER))
    config = destination / 'pyvenv.cfg'
    base_python = Path(info['identity']['executable']).resolve()
    if config.is_file() and not base_python.is_relative_to(Path(prefix).resolve()):
        # A nested Python 3.10 venv may name its parent venv as home. The copied
        # executable needs the real base location to keep finding the stdlib.
        config.write_text(re.sub(r'^home\s*=.*$', lambda _: f'home = {base_python.parent}',
                                 config.read_text(), flags=re.MULTILINE))
    # Python locates pyvenv.cfg beside the copied interpreter. Console entry
    # points also need the new prefix; never rewrite binaries or symlink targets.
    scripts = destination / ('Scripts' if os.name == 'nt' else 'bin')
    for script in scripts.iterdir():
        if script.is_file() and not script.is_symlink() and script.stat().st_size < 65536:
            data = script.read_bytes()
            if data.startswith(b'#!'):
                script.write_bytes(data.replace(prefix.encode(), str(destination).encode()))
    (destination / COMPATIBILITY_MARKER).write_text(json.dumps({"prefix": str(destination), "packages": names}))
    return {"interpreter": str(scripts / ('python.exe' if os.name == 'nt' else 'python'))}


def plan_compatibility(directory: Path, requested: object, project: Path) -> dict:
    replace = compatibility_replacements()
    if not replace:
        raise SetupError("Compatibility repair requires a disposable environment.")
    requirements = directory / 'requested.txt'
    # Include retained distributions in the solve so their requirements cannot
    # become broken dependents of a downgraded model library.
    requirements.write_text('\n'.join(package_requirements(requested) + [
        f'{name}==={version}' for name, version in inventory().items() if name not in replace]) + '\n')
    return plan(directory, requirements, replace=replace, project=project.resolve())


# Torch, its companions and CUDA wheels change the device build; hosts confirm them separately.
TORCH_BUILD = re.compile(r"torch|torchvision|torchaudio|triton|xformers|nvidia-.+")

TARGET_REQUIRES = r'''
import json, os, platform, sys
from importlib import metadata
requires = {}
for dist in metadata.distributions():
    name = dist.metadata.get('Name')
    if name:
        requires[name] = list(dist.requires or [])
info = sys.implementation.version
version = f'{info.major}.{info.minor}.{info.micro}' + ('' if info.releaselevel == 'final' else info.releaselevel[0] + str(info.serial))
print(json.dumps(dict(requires=requires, markers=dict(
    implementation_name=sys.implementation.name, implementation_version=version, os_name=os.name,
    platform_machine=platform.machine(), platform_python_implementation=platform.python_implementation(),
    platform_release=platform.release(), platform_system=platform.system(), platform_version=platform.version(),
    python_full_version=platform.python_version(), python_version='.'.join(platform.python_version_tuple()[:2]),
    sys_platform=sys.platform, extra=''))))
'''


def installed_requirements() -> dict:
    """Each installed distribution's declared requirements and the target's marker values."""
    try:
        result = subprocess.run([TARGET_PYTHON or sys.executable, '-I', '-B', '-c', TARGET_REQUIRES],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        if result.returncode or len(result.stdout) > 4 * 1024 * 1024:
            raise ValueError()
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        raise SetupError('The selected Python environment could not be inspected. Choose another environment.',
                         'ENVIRONMENT_UNAVAILABLE') from None


def broken_dependents(installed: dict, before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Installed distributions whose requirements the planned versions would no longer satisfy."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    changed = {name for name, version in after.items() if before.get(name) != version}
    broken = set()
    for distribution, texts in installed['requires'].items():
        name = canonical(distribution)
        if name in changed:
            continue  # pip resolved the requirements of every distribution it installs.
        for text in texts:
            try:
                requirement = Requirement(text)
            except ValueError:
                continue
            dependency = canonical(requirement.name)
            if dependency not in changed or requirement.marker and not requirement.marker.evaluate(installed['markers']):
                continue
            if not requirement.specifier.contains(after[dependency], prereleases=True):
                broken.add(name)
    return sorted(broken)


def unavailable_build(requirement: str | None, pins: list[str]) -> str:
    environment = environment_info()
    where = (f"Python {'.'.join(environment['python'].split('.')[:2])} on "
             f"{ {'darwin': 'macOS', 'linux': 'Linux', 'win32': 'Windows'}.get(environment['platform'], environment['platform'])} {environment['architecture']}")
    exact = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)\s*==\s*([A-Za-z0-9.+!-]+)", requirement or '')
    what = f"{exact.group(1)} {exact.group(2)}" if exact else requirement or ', '.join(pins)
    return f"{what} has no build for {where}. Choose an environment with a supported Python, or try the installed versions."


def plan_replacement(directory: Path, requested: object, project: Path) -> dict:
    """Plan an explicitly approved version change in the selected environment itself.

    Only the project's own declarations may be requested. Installed packages that the
    new versions would break are resolved again, constrained by any project
    declaration, and listed for review; nothing else may change."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    project = project.resolve()
    pins = package_requirements(requested)
    declared = project_requirements(project)
    names = {canonical(Requirement(pin).name) for pin in pins}
    if any(declared.get(canonical(Requirement(pin).name)) != pin for pin in pins):
        raise SetupError("Only the project's own requirements can replace installed versions. Check the model dependencies again.")
    check_install_target()
    baseline = identity()
    installed = baseline["packages"]
    requires = installed_requirements()
    requirements, report_path = directory / "requested.txt", directory / "pip-report.json"
    request = list(pins)
    for _ in range(4):
        requirements.write_text("\n".join(request) + "\n")
        try:
            pip_run(["install", "--dry-run", "--only-binary=:all:", "--report", str(report_path), "-r", str(requirements)])
        except SetupError as error:
            if error.code in ("NO_COMPATIBLE_WHEEL", "PACKAGE_UNAVAILABLE"):
                raise SetupError(unavailable_build(error.requirement, pins), "NO_MATCHING_BUILD") from None
            if error.code == "DEPENDENCY_CONFLICT":
                raise SetupError(f"No versions satisfy {', '.join(pins)} together with the installed "
                                 f"{', '.join(sorted({canonical(Requirement(item).name) for item in request} - names)) or 'libraries'}. "
                                 "Choose another environment, or try the installed versions.") from None
            raise
        report = json.loads(report_path.read_text())
        after = {**installed, **{canonical(item["metadata"]["name"]): item["metadata"]["version"] for item in report.get("install", [])}}
        broken = broken_dependents(requires, installed, after)
        if not broken:
            break
        requested_names = {canonical(Requirement(item).name) for item in request}
        if any(name in declared and declared[name] is None for name in broken) or set(broken) <= requested_names:
            raise SetupError(f"Replacing these versions would break the installed {', '.join(broken)}. "
                             "Choose another environment, or try the installed versions.")
        request += [declared.get(name) or name for name in broken if name not in requested_names]
    else:
        raise SetupError("The replacement did not settle on a consistent set of versions. Choose another environment.")
    replace = {canonical(item["metadata"]["name"]) for item in report.get("install", [])} & set(installed)
    wheels = validate_wheels(report, {name: version for name, version in installed.items() if name not in replace})
    if identity() != baseline:
        raise SetupError("The environment changed during inspection. Check it again.")
    data = json.dumps({"baseline": baseline, "wheels": wheels, "replace": sorted(replace), "inPlace": True,
                       "project": str(project), "declarations": {name: value for name, value in declared.items() if name in replace}},
                      sort_keys=True).encode()
    (directory / "plan.json").write_bytes(data)
    return {"digest": digest(data), "replacesTorch": any(TORCH_BUILD.fullmatch(w["name"]) for w in wheels),
            "packages": [{"name": w["name"], "version": w["version"], **({"previous": installed[w["name"]]} if w["name"] in installed else {})}
                         for w in wheels]}


def model_import_diagnostic(error: BaseException) -> dict:
    """Keep actionable import evidence without source lines, locals or environment dumps.

    This runs only inside the already-authorized model import worker. Device facts
    come from Torch already loaded by that import; missing observations stay null.
    """
    truncated = False

    def text(value, limit: int) -> str:
        nonlocal truncated
        try:
            value = str(value)
        except Exception:
            value = type(value).__name__
        # Linker/import failures may contain repository URLs. Do not expose URL
        # credentials, common query tokens or authorization values in diagnostics.
        value = re.sub(r"(?i)(https?://)[^/\s@]+@", r"\1[redacted]@", value)
        value = re.sub(r"(?i)([?&](?:token|access_token|api_key|key|password|secret)=)[^&#\s]*", r"\1[redacted]", value)
        value = re.sub(r"(?i)(\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", value)
        value = re.sub(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", value)
        value = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{16,})\b", "[redacted]", value)
        value = "".join(character for character in value if ord(character) >= 32 or character in "\n\t")
        encoded = value.encode('utf-16-le', errors='replace')
        if len(encoded) > limit * 2:
            truncated = True
            return encoded[:limit * 2].decode('utf-16-le', errors='ignore')
        return value

    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and len(chain) < 8 and all(current is not previous for previous in chain):
        chain.append(current)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    if current is not None:
        truncated = True
    causes = []
    for cause in chain:
        entry = {"exceptionType": text(type(cause).__name__, 128), "message": text(cause, 2048)}
        module = getattr(cause, "name", None)
        if isinstance(module, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,255}", module):
            entry["module"] = module
        causes.append(entry)

    lines, frame_count = [], 0
    for cause, entry in reversed(list(zip(chain, causes))):
        if lines:
            lines.append("Caused the following exception:")
        frame = cause.__traceback__
        while frame is not None:
            if frame_count >= 64:
                truncated = True
                break
            code = frame.tb_frame.f_code
            lines.append(f'  File "{text(code.co_filename, 1024)}", line {frame.tb_lineno}, in {text(code.co_name, 128)}')
            frame_count += 1
            frame = frame.tb_next
        lines.append(f'{entry["exceptionType"]}: {entry["message"]}')
    traceback = text("\n".join(lines), 12000)

    torch = sys.modules.get("torch")
    def observed(getter):
        try:
            return getter() if torch is not None else None
        except Exception:
            return None
    try:
        torch_version = text(metadata.version("torch"), 128)
    except Exception:
        torch_version = None
    cuda = observed(lambda: torch.version.cuda)
    cuda_available = observed(lambda: torch.cuda.is_available())
    mps_available = observed(lambda: torch.backends.mps.is_available())
    runtime = {"platform": text(sys.platform, 128), "architecture": text(platform.machine(), 128),
               "python": text(sys.version.split()[0], 128), "torch": torch_version,
               "cuda": text(cuda, 128) if cuda is not None else None,
               "cudaAvailable": cuda_available if isinstance(cuda_available, bool) else None,
               "mpsAvailable": mps_available if isinstance(mps_available, bool) else None}
    primary = causes[-1]
    diagnostic = {"exceptionType": primary["exceptionType"], "message": primary["message"],
                  "traceback": traceback, "causes": causes, "runtime": runtime, "truncated": truncated}
    # The helper transport retains 128k of stdout. JSON escapes can expand each
    # Unicode character sixfold, so per-string limits alone do not bound output.
    while len(json.dumps(diagnostic)) > 64000:
        diagnostic["truncated"] = True
        diagnostic["traceback"] = diagnostic["traceback"][:len(diagnostic["traceback"]) // 2]
        for entry in causes:
            entry["message"] = entry["message"][:max(128, len(entry["message"]) // 2)]
        diagnostic["message"] = causes[-1]["message"]
    return diagnostic


def probe_model_import(root: Path, file: Path) -> dict:
    if TARGET_PYTHON:
        result = subprocess.run([TARGET_PYTHON, '-I', '-B', str(REQUIREMENTS.parent / 'services' / 'ide' / 'environment_setup.py'),
                                 'probe-model', str(root), str(file)], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
        return json.loads(result.stdout.strip().splitlines()[-1])
    root, file = root.resolve(), file.resolve()
    if not file.is_relative_to(root):
        raise SetupError("The model must belong to its workspace.")
    try:
        if __package__:
            from ..validation import setup
        else:
            spec = importlib.util.spec_from_file_location('_tensorviz_environment_model_setup',
                REQUIREMENTS.parent / 'services' / 'validation' / 'setup.py')
            setup = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(setup)
        os.chdir(root)
        setup.load_module(file, root=root)
    except Exception as error:
        diagnostic = model_import_diagnostic(error)
        return {"ok": False, "code": "MODEL_IMPORT_FAILED",
                "message": f'Model import failed: {diagnostic["exceptionType"]}: {diagnostic["message"]}'[:2048],
                "diagnostic": diagnostic}
    return {}


def probe_capture() -> dict:
    # Import the exact private worker package, without importing the tool server
    # or any user model. The selected target owns Torch and its native libraries.
    script = '''
import importlib.util, json, sys
try:
    spec = importlib.util.spec_from_file_location('_tensorviz_capture_probe', sys.argv[1])
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    bootstrap.load_worker()
    import torch
    print(json.dumps(dict(ok=True, torch=torch.__version__, cuda=torch.version.cuda)))
except ModuleNotFoundError as error:
    print(json.dumps(dict(ok=False, code='MISSING_DEPENDENCIES', message=f'Missing capture library: {error.name}.')))
except Exception as error:
    message = str(error) if type(error).__name__ == 'CaptureRuntimeUnsupported' else f'The capture worker could not load ({type(error).__name__}). Choose a compatible environment.'
    print(json.dumps(dict(ok=False, code='CAPTURE_RUNTIME_UNSUPPORTED', message=message)))
'''
    result = subprocess.run([TARGET_PYTHON or sys.executable, '-I', '-B', '-c', script,
                             str(REQUIREMENTS.parent / 'services' / 'ingest' / 'bootstrap.py')],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise SetupError('The selected Python could not load the capture worker. Choose another environment.',
                         'CAPTURE_RUNTIME_UNSUPPORTED') from None


def project_declarations(root: Path) -> dict[str, str]:
    """Read bounded declarations without replacing explicit sources with PyPI guesses."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    declared = {}
    for manifest in (root / "requirements.txt", root / "pyproject.toml"):
        if not manifest.is_file() or manifest.stat().st_size > 262144 or not manifest.resolve().is_relative_to(root):
            continue
        try:
            if manifest.suffix == ".toml":
                try:
                    import tomllib
                except ModuleNotFoundError:
                    import tomli as tomllib
                values = tomllib.loads(manifest.read_text()).get("project", {}).get("dependencies", [])
            else:
                values = manifest.read_text().splitlines()
            for text in values:
                try:
                    parsed = Requirement(text)
                    # Unsupported explicit sources/markers must not fall back to
                    # an unconstrained public package with the same name.
                    declared[canonical(parsed.name)] = str(parsed)
                except (ValueError, TypeError):
                    continue  # Includes, editable installs and URLs need the agent's judgment.
        except (ValueError, OSError):
            continue
    return declared


def project_requirements(root: Path) -> dict[str, str | None]:
    declared = {}
    for name, text in project_declarations(root).items():
        try:
            declared[name] = package_requirements([text])[0]
        except SetupError:
            declared[name] = None
    return declared


def project_manifest_digest(root: Path) -> str:
    return digest(json.dumps(project_declarations(root), sort_keys=True).encode())


def model_environment_requirements(root: Path, file: Path) -> dict:
    """Only required model imports and capture libraries, in a fresh target venv."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    declarations = project_declarations(root.resolve())
    manifest_digest = digest(json.dumps(declarations, sort_keys=True).encode())
    facts = model_dependencies(root, file)
    requested = []
    for item in facts['missingModules']:
        if item.get('optional'):
            continue
        name = canonical(MODEL_PACKAGES.get(item['module'], item['module']))
        value = declarations.get(name, item['requirement'])
        if not value:
            raise SetupError(f"Declare the distribution for required import {item['module']} in the project's requirements before creating its environment.", 'UNKNOWN_MODEL_DEPENDENCY')
        requested.append(value)
    requested.extend(line for line in CAPTURE_REQUIREMENTS.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#'))
    sources, requirements = [], []
    for text in requested:
        parsed = Requirement(text)
        if parsed.marker and not parsed.marker.evaluate():
            continue
        parsed.marker = None
        if parsed.url:
            if not re.fullmatch(r'git\+https://[A-Za-z0-9.-]+/[A-Za-z0-9_./-]+@[a-fA-F0-9]{40}', parsed.url):
                raise SetupError('Automatic model setup supports public HTTPS Git dependencies pinned to a full commit. Local, editable and unpinned sources need a project environment.', 'UNSUPPORTED_MODEL_SOURCE')
            sources.append(str(parsed))
        elif len(str(parsed)) > 200:
            raise SetupError('A model requirement exceeds the setup limit.')
        requirements.append(str(parsed))
    if project_manifest_digest(root.resolve()) != manifest_digest:
        raise SetupError('Project requirements changed during inspection. Prepare the environment again.')
    return {'requirements': list(dict.fromkeys(requirements)), 'sourcePackages': list(dict.fromkeys(sources)), 'manifestDigest': manifest_digest}


def check_package_availability(requirement: str) -> dict:
    """Read index metadata before offering a source build; never run package code."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    parsed = Requirement(requirement)
    if parsed.url or len(requirement) > 200:
        raise SetupError('Only an indexed package can be checked for source-build availability.')
    try:
        pip_run(['index', 'versions', parsed.name])
    except SetupError as error:
        if error.code == 'PACKAGE_UNAVAILABLE':
            raise SetupError('No distribution is available for this Python/platform. Choose a compatible environment, or inspect source without running the model.',
                             error.code, str(parsed)) from None
        raise
    return {}


def plan_model_environment(directory: Path, root: Path, file: Path, approved_manifest: str, *, allow_source: bool = False, environment_name: str = 'environment') -> dict:
    """Resolve reviewed sources only into a newly created, host-owned venv."""
    baseline = identity()
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', environment_name)
            or re.fullmatch(r'con|prn|aux|nul|com[1-9]|lpt[1-9]', environment_name, re.IGNORECASE)
            or Path(baseline['prefix']).resolve() != (directory / environment_name).resolve()
            or set(baseline['packages']) - {'pip', 'setuptools', 'wheel'}):
        raise SetupError('Create a fresh model environment before planning its libraries.')
    before = project_manifest_digest(root)
    if before != approved_manifest:
        raise SetupError('Project requirements changed after source review. Prepare the environment again.')
    recipe = model_environment_requirements(root, file)
    if recipe['sourcePackages'] and not allow_source:
        raise SetupError('The project requires building pinned Git packages. Review source execution first.', 'NO_COMPATIBLE_WHEEL')
    requested = []
    built_sources = []
    for text in recipe['requirements']:
        if text not in recipe['sourcePackages']:
            requested.append(text)
            continue
        try:
            from packaging.requirements import Requirement
        except ImportError:
            from pip._vendor.packaging.requirements import Requirement
        parsed = Requirement(text)
        name = canonical(parsed.name)
        output = directory / 'git-wheels' / name
        output.mkdir(parents=True, exist_ok=False)
        pip_run(['wheel', '--no-deps', '--wheel-dir', str(output), text])
        wheels = list(output.glob('*.whl'))
        if len(wheels) != 1:
            raise SetupError(f'The build for {name} did not produce exactly one wheel.', 'SOURCE_BUILD_FAILED')
        with zipfile.ZipFile(wheels[0]) as archive:
            entries = [entry for entry in archive.namelist() if entry.endswith('.dist-info/METADATA')]
            if len(entries) != 1 or canonical(BytesParser().parsebytes(archive.read(entries[0])).get('Name', '')) != name:
                raise SetupError(f'The built wheel does not match {name}.', 'SOURCE_BUILD_FAILED')
        extras = '[' + ','.join(sorted(parsed.extras)) + ']' if parsed.extras else ''
        requested.append(f'{parsed.name}{extras} @ {wheels[0].resolve().as_uri()}')
        built_sources.append(name)
    requirements = directory / 'model-requirements.txt'
    requirements.write_text('\n'.join(requested) + '\n')
    result = plan(directory, requirements, allow_source=allow_source)
    if project_manifest_digest(root) != before:
        raise SetupError('Project requirements changed during planning. Prepare the environment again.')
    saved = json.loads((directory / 'plan.json').read_bytes())
    saved.update(modelProject=str(root.resolve()), manifestDigest=before, modelSources=recipe['sourcePackages'])
    data = json.dumps(saved, sort_keys=True).encode()
    (directory / 'plan.json').write_bytes(data)
    return {**result, 'digest': digest(data), 'builtFromSource': built_sources + result['builtFromSource']}


def model_dependencies(root: Path, file: Path, required_modules: list[str] | None = None) -> dict:
    """Read bounded local imports and manifests. Never import the user's model."""
    root = root.resolve()
    entry = file.resolve()
    if not entry.is_relative_to(root):
        raise SetupError("The model must belong to its workspace.")
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    declared = project_requirements(root)
    if required_modules is not None and (not isinstance(required_modules, list) or len(required_modules) > 32
            or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) for name in required_modules)):
        raise SetupError("Invalid required module names.")
    pending, visited, external = [(entry, True)], {}, set(required_modules or [])
    required = set(required_modules or [])
    total = 0
    truncated = False
    while pending and len(visited) < 128:
        current, needed = pending.pop()
        current = current.resolve()
        if (current in visited and (visited[current] or not needed)) or not current.is_relative_to(root) or not current.is_file():
            continue
        visited[current] = needed
        # Importing the selected module executes every parent initializer, even
        # when the file itself imports only Torch (for example Janus VQModel).
        for parent in current.parents:
            if parent == root:
                break
            initializer = parent / '__init__.py'
            if initializer.is_file():
                pending.append((initializer, needed))
        size = current.stat().st_size
        if size > 262144 or total + size > 2 * 1024 * 1024:
            truncated = True
            continue
        total += size
        try:
            tree = ast.parse(current.read_text())
        except (SyntaxError, UnicodeError, OSError):
            continue
        # Conditional, try/except and function-body imports are suggestions, not
        # prerequisites. A runtime missing-module diagnostic can confirm them.
        direct = {id(node) for node in tree.body}
        for node in ast.walk(tree):
            required_import = needed and id(node) in direct
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else
                [alias.name for alias in node.names if alias.name != '*'] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                relative = isinstance(node, ast.ImportFrom) and node.level
                base = current.parent if relative else root
                if relative:
                    for _ in range(node.level - 1):
                        base = base.parent
                local_import = False
                for base in ([base] if relative else [root, root / 'src', current.parent]):
                    local = base.joinpath(*name.split('.'))
                    candidates = [local.with_suffix('.py'), local / '__init__.py']
                    if isinstance(node, ast.ImportFrom) and node.module:
                        candidates.extend(local / f'{alias.name}.py' for alias in node.names if alias.name != '*')
                        candidates.extend(local / alias.name / '__init__.py' for alias in node.names if alias.name != '*')
                    # Importing a dotted module also runs its parent packages.
                    candidates.extend(base.joinpath(*name.split('.')[:i], '__init__.py') for i in range(1, len(name.split('.'))))
                    found = [candidate for candidate in candidates if candidate.is_file() and candidate.resolve().is_relative_to(root)]
                    if found or (base / name.split('.')[0]).is_dir():
                        pending.extend((candidate, required_import) for candidate in found)
                        local_import = True
                        break
                if not local_import and not relative:
                    external.add(name.split('.')[0])
                    if required_import:
                        required.add(name.split('.')[0])
    missing, compatibility, incompatible, pinned, unsupported_declared = [], [], [], [], []
    target = target_snapshot(sorted(external)) if TARGET_PYTHON else None
    stdlib = set(target['stdlib']) if target else sys.stdlib_module_names
    installed_packages = target['identity']['packages'] if target else None
    for module in sorted(external - stdlib):
        try:
            present = target['modules'][module] if target else importlib.util.find_spec(module) is not None
        except (ValueError, ImportError):
            present = False
        package = MODEL_PACKAGES.get(module)
        requirement = declared.get(canonical(package or module), package)
        if requirement:
            parsed = Requirement(requirement)
            profile = MODEL_COMPATIBILITY.get(canonical(parsed.name))
            if profile and not parsed.extras and parsed.specifier.contains(profile[0]):
                preferred = f'{canonical(parsed.name)}=={profile[0]}'
                if not present:
                    requirement = preferred
                elif module in required:
                    try:
                        if (installed_packages.get(canonical(parsed.name)) if target else metadata.version(parsed.name)) != profile[0]:
                            compatibility.append(preferred)
                    except metadata.PackageNotFoundError:
                        pass
            if present and module in required:
                try:
                    installed = installed_packages.get(canonical(parsed.name)) if target else metadata.version(parsed.name)
                    if installed and not parsed.specifier.contains(installed):
                        incompatible.append(f'{parsed.name} {installed} (needs {parsed.specifier})')
                        # The exact project declaration, for an explicit override or in-place replacement.
                        pinned.append(str(parsed))
                except metadata.PackageNotFoundError:
                    pass
        if not present:
            missing.append({"module": module, "requirement": requirement,
                            **({"optional": True} if module not in required else {})})
            if canonical(package or module) in declared and requirement is None:
                unsupported_declared.append(module)
    return {"environment": environment_info(), "missingModules": missing, "compatibilityPackages": compatibility, "incompatible": incompatible,
            "incompatibleRequirements": pinned, "unsupportedDeclaredModules": unsupported_declared,
            "filesChecked": len(visited), "truncated": truncated or bool(pending),
            "detail": "Static import check only. Optional and dynamic imports need review; a module name is not necessarily a PyPI package name."}


def inspect_candidate(root: Path, file: Path, required_modules: list[str] | None = None) -> dict:
    """``inspect-capture`` plus the static model check for one discovered environment,
    so ranking every candidate costs one target process each. The model check runs
    first: its snapshot answers the model's modules, and the capture check's
    identity-only read reuses it. A model that cannot be checked leaves the
    environment's own facts intact; only an unavailable environment fails."""
    try:
        model = model_dependencies(root, file, required_modules)
    except SetupError as error:
        if error.code == 'ENVIRONMENT_UNAVAILABLE':
            raise
        model = None
    capture = inspect_environment(CAPTURE_REQUIREMENTS)
    return {**capture, **({"model": {"missingModules": model["missingModules"], "incompatible": model["incompatible"]}} if model else {})}


def main(probe_backend=None, *, requirements=REQUIREMENTS, minimum_python=(3, 10)) -> None:
    global TARGET_PYTHON
    TARGET_PYTHON = None
    if os.environ.get('TENSORVIZ_PARENT_PIPE') == '1':
        sys.path.insert(0, str(REQUIREMENTS.parent))
        from services.ingest.worker_lifetime import guard_parent_pipe
        guard_parent_pipe()
        sys.path.pop(0)
    try:
        if '--target-python' in sys.argv:
            index = sys.argv.index('--target-python')
            TARGET_PYTHON = sys.argv[index + 1]
            if not Path(TARGET_PYTHON).is_absolute():
                raise SetupError('Select an absolute target Python executable.', 'ENVIRONMENT_UNAVAILABLE')
            del sys.argv[index:index + 2]
        if sys.version_info[:2] < minimum_python:
            raise SetupError(f"This tensorViz runtime needs Python {minimum_python[0]}.{minimum_python[1]} or newer; "
                             f"the selected environment uses {sys.version.split()[0]}. Choose another environment.",
                             "UNSUPPORTED_PYTHON")
        action = sys.argv[1]
        if action == "inspect":
            with one_snapshot():
                result = inspect_environment(requirements)
        elif action == "inspect-capture":
            with one_snapshot():
                result = inspect_environment(CAPTURE_REQUIREMENTS)
        elif action == "inspect-service":
            with one_snapshot():
                result = inspect_environment(SERVICE_REQUIREMENTS)
        elif action == "probe-capture":
            result = probe_capture()
        elif action == "inspect-runtime":
            result = {"environment": environment_info()}
        elif action == "create-environment":
            import venv
            # Setup children must not share the lifetime watcher's blocking stdin.
            class SetupEnvBuilder(venv.EnvBuilder):
                def _call_new_python(self, context, *args, **kwargs):
                    kwargs["stdin"] = subprocess.DEVNULL
                    return super()._call_new_python(context, *args, **kwargs)

            SetupEnvBuilder(with_pip=True).create(Path(sys.argv[2]))
            result = {}
        elif action == "probe" and probe_backend is not None:
            result = probe_backend()
        elif action == "plan":
            result = plan(Path(sys.argv[2]), requirements)
        elif action == "plan-capture":
            result = plan(Path(sys.argv[2]), CAPTURE_REQUIREMENTS)
        elif action == "plan-packages-managed-cpu":
            result = plan_packages(Path(sys.argv[2]), json.loads(sys.argv[3]), managed_cpu=True)
        elif action == "plan-packages":
            result = plan_packages(Path(sys.argv[2]), json.loads(sys.argv[3]))
        elif action == "plan-packages-source":
            result = plan_packages(Path(sys.argv[2]), json.loads(sys.argv[3]), allow_source=True)
        elif action == "inspect-model":
            with one_snapshot():
                result = model_dependencies(Path(sys.argv[2]), Path(sys.argv[3]), json.loads(sys.argv[4]) if len(sys.argv) > 4 else None)
        elif action == "inspect-candidate":
            with one_snapshot():
                result = inspect_candidate(Path(sys.argv[2]), Path(sys.argv[3]), json.loads(sys.argv[4]) if len(sys.argv) > 4 else None)
        elif action == "inspect-model-setup":
            result = model_environment_requirements(Path(sys.argv[2]), Path(sys.argv[3]))
        elif action == "check-package-availability":
            result = check_package_availability(sys.argv[2])
        elif action in {"plan-model-environment", "plan-model-environment-source"}:
            result = plan_model_environment(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5],
                allow_source=action.endswith('-source'), environment_name=sys.argv[6] if len(sys.argv) > 6 else 'environment')
        elif action == "clone-compatibility":
            result = clone_compatibility_environment(Path(sys.argv[2]), json.loads(sys.argv[3]))
        elif action == "plan-compatibility":
            result = plan_compatibility(Path(sys.argv[2]), json.loads(sys.argv[3]), Path(sys.argv[4]))
        elif action == "plan-replacement":
            result = plan_replacement(Path(sys.argv[2]), json.loads(sys.argv[3]), Path(sys.argv[4]))
        elif action == "probe-model":
            result = probe_model_import(Path(sys.argv[2]), Path(sys.argv[3]))
        elif action == "update-managed":
            result = update_managed(Path(sys.argv[2]))
        elif action == "install":
            result = install(Path(sys.argv[2]), sys.argv[3])
        else:
            raise SetupError("Unknown environment setup action.")
        print(json.dumps({"ok": True, **result}))
    except Exception as error:
        # Errors from our own validation carry no subprocess output or index credentials.
        message = (f"Missing Python module: {error.name}." if isinstance(error, ModuleNotFoundError)
                   else str(error) if isinstance(error, SetupError)
                   else f"Backend compatibility check failed ({type(error).__name__}). Choose another environment or repair its dependencies.")
        print(json.dumps({"ok": False, "message": message,
                          "code": error.code if isinstance(error, SetupError) else "DEPENDENCY_SETUP_FAILED",
                          **({"requirement": error.requirement} if isinstance(error, SetupError) and error.requirement else {})}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()



