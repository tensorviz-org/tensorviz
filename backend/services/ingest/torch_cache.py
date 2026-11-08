"""Move optional Torch cache initialization outside the reuse audit window."""
import importlib
import importlib.util


def prepare_cache():
    # Older model interpreters (including Torch 2.7) have no package module.
    # Discover absence only; a broken existing module must retain its real error.
    if (importlib.util.find_spec("torch._dynamo") is not None
            and importlib.util.find_spec("torch._dynamo.package") is not None):
        importlib.import_module("torch._dynamo.package")
