"""Inspect the model interpreter without importing the web server or workspace."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.ide.environment_setup import main


def probe_capture():
    import torch
    # Load the real execution and preflight paths, including their deferred imports.
    from services.ingest import service, execution_setup, graph_example, candidate_worker
    from services.validation import host_snapshot, worker
    return {"torch": torch.__version__, "cuda": torch.version.cuda}


if __name__ == "__main__":
    main(probe_backend=probe_capture, requirements=Path(__file__).with_name("requirements-capture.txt"),
         minimum_python=(3, 10))
