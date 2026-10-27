"""The extension backend exposes deterministic graph services without AI routes."""

from server_v2 import app


def _route_paths(routes):
    """FastAPI 0.162 wraps included routers in ``_IncludedRouter`` (no ``path``).

    Paths stay on the original router. A prefix that is not already part of the
    route path is applied from the include context.
    """
    paths = set()
    for route in routes:
        path = getattr(route, "path", None)
        if path:
            paths.add(path)
        original = getattr(route, "original_router", None)
        if original is None:
            continue
        prefix = getattr(getattr(route, "include_context", None), "prefix", "") or ""
        for nested in _route_paths(original.routes):
            if prefix and not nested.startswith(prefix.rstrip("/") + "/") and nested != prefix:
                nested = prefix.rstrip("/") + "/" + nested.lstrip("/")
            paths.add(nested)
    return paths


def test_graph_services_remain_and_provider_routes_are_absent():
    paths = _route_paths(app.routes)
    assert "/api/ide/project" in paths
    assert "/api/ide/apply" in paths
    assert "/health" in paths
    assert not any(path.startswith(("/api/ai/", "/api/agent/")) for path in paths)
    assert {
        "/api/graphs/sessions",
        "/api/graphs/sessions/{session_id}/apply",
    }.isdisjoint(paths)
