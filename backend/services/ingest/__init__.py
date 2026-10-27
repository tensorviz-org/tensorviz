"""Model ingestion: turn an installed library architecture or a repository into a graph.

The reframe that makes this tractable: do not parse source, *instantiate the model and
interrogate the object*. Cross-file imports, config classes, registries and ``**kwargs``
plumbing are statically undecidable but trivially observable once the object exists — and
building it costs nothing, because every model is materialised under ``FakeTensorMode``
with no weights, no network and no GPU.

Public surface::

    from services.ingest import ingest, scan_only, catalog, available_libraries
    from services.ingest import ModelRef, IngestResult
"""

from importlib import import_module


def __getattr__(name):
    # python -m services.ingest.worker imports this package before main(). Keep
    # Torch/Pydantic deferred so argument errors still produce a worker envelope.
    if name in {"available_libraries", "catalog"}:
        module = ".resolvers"
    elif name in {"IngestError", "ingest", "scan_only"}:
        module = ".service"
    elif name in {"CatalogEntry", "EntrypointCandidate", "IngestDiagnostics", "IngestResult", "ModelRef", "RepeatGroup"}:
        module = ".spec"
    else:
        raise AttributeError(name)
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "ingest",
    "scan_only",
    "catalog",
    "available_libraries",
    "IngestError",
    "ModelRef",
    "IngestResult",
    "IngestDiagnostics",
    "EntrypointCandidate",
    "CatalogEntry",
    "RepeatGroup",
]
