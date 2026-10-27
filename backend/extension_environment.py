"""VS Code backend compatibility probe layered over the shared environment helper."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.ide.environment_setup import main


def probe_backend():
    # Import the exact application the extension starts, never a user's model.
    import server_v2
    import uvicorn
    import websockets
    import torch
    return {"torch": torch.__version__, "cuda": torch.version.cuda}


if __name__ == "__main__":
    main(probe_backend=probe_backend, minimum_python=(3, 11))
