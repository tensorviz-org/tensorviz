"""Both modes load helpers privately, without claiming the user's services package."""
import importlib.util
import os
from pathlib import Path
import sys


def main():
    script = sys.argv[1] if len(sys.argv) > 1 else None
    # Match `python /absolute/script.py ...`, retaining the inherited PYTHONPATH.
    sys.path[0] = str(Path(script).resolve().parent) if script else os.getcwd()
    package = "_tensorviz_validation_worker"
    # Load the services parent privately so sibling semantic/setup helpers use
    # ordinary relative imports without claiming the user's public services name.
    services = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        package, services / "__init__.py",
        submodule_search_locations=[str(services)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    worker = importlib.import_module(f"{package}.validation.worker")
    return worker.main(script)


if __name__ == "__main__":
    sys.exit(main())
