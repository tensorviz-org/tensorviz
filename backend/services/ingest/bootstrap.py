"""Load capture internals privately, leaving common workspace package names free."""
import importlib.util
from pathlib import Path
import sys


class CaptureRuntimeUnsupported(RuntimeError):
    pass


PACKAGE = "_tensorviz_capture_worker"
#: Set on the package object this loader created, so a pre-loaded package (the
#: standby's) is recognised and reused, while anything else that happens to sit in
#: ``sys.modules`` under the same name is replaced exactly as before.
_LOADED_BY = "__tensorviz_loaded_from__"


def _private_package(directory):
    init = str(directory / "__init__.py")
    existing = sys.modules.get(PACKAGE)
    if existing is not None and getattr(existing, _LOADED_BY, None) == init:
        return existing
    spec = importlib.util.spec_from_file_location(PACKAGE, directory / "__init__.py",
                                                submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    setattr(module, _LOADED_BY, init)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


def load_internals():
    """Load (or reuse) the private capture package and return its worker module.

    The standby calls this before its request so a run finds everything imported;
    ``load_worker`` calls it again at the start of the run. Both paths perform the
    dispatcher check and the ``sys.path`` filter, so a run is set up identically
    whether or not anything was pre-loaded."""
    import torch
    if torch.nn.Module.__call__ is torch.nn.Module._call_impl:
        raise CaptureRuntimeUnsupported(
            "This PyTorch build uses a legacy module call dispatcher unsupported by the capture worker. "
            "Choose another environment; the existing PyTorch build was preserved.")
    directory = Path(__file__).resolve().parents[1]
    # Internals load through the private package below. The host also puts the
    # backend on PYTHONPATH; its regular `models` package would otherwise shadow
    # a workspace's namespace package even when the workspace is first on sys.path.
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != directory.parent]
    _private_package(directory)
    return importlib.import_module(f"{PACKAGE}.ingest.worker")


def load_worker():
    return load_internals()


def main():
    return load_worker().main()


if __name__ == "__main__":
    sys.exit(main())
