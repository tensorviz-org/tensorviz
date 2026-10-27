"""Sweep guard: EVERY template and popular model must instantiate, pass the
verify_model shape check + dry forward, and pass canonical shape validation
(the engine behind the canvas error badges) with no errors.

This is the regression net for the "templates show input/output-missing or
shape errors" class of bugs: most CV/NLP/generative templates used to omit
Input/Output nodes and had channel/rank/dtype mismatches. If a template
regresses, the parametrized case names it precisely.
"""


import pytest


from api.routers.validate import ValidationRequest, validate_shapes  # noqa: E402
from services.template_service import TemplateService  # noqa: E402
from services.verification_service import verify_model  # noqa: E402

_SVC = TemplateService()
_POPULAR = sorted(_SVC.get_all_popular_models().keys())
_TEMPLATES = {}
for _cat in (_SVC.starter_templates, _SVC.transformer_templates, _SVC.cv_templates, _SVC.nlp_templates, _SVC.generative_templates):
    _TEMPLATES.update(_cat.get_all_templates())
_TEMPLATE_IDS = sorted(_TEMPLATES.keys())


def _input_shape(graph_dict):
    inp = next((n for n in graph_dict.get("root_nodes", []) if n.get("type") == "inputNode"), None)
    if inp:
        raw = inp["data"]["parameters"].get("shape")
        if isinstance(raw, str) and raw:
            try:
                return [int(x) for x in raw.split(",")]
            except ValueError:
                pass
    return [1, 3, 224, 224]


def _canvas_payload(graph_dict, shape):
    """Replicate the frontend flatten + module-boundary rewire (hierarchicalToFlat
    + buildInferablePayload) so we validate exactly what the canvas sends to
    /validate_shapes."""
    mods = graph_dict.get("modules", {}) or {}
    nodes = [dict(n) for n in graph_dict.get("root_nodes", [])]
    for nd in nodes:
        if nd["id"] in mods:
            m = mods[nd["id"]]
            nd["data"] = dict(nd["data"])
            nd["data"]["isGroup"] = True
            nd["data"]["moduleEntryNodeId"] = m.get("input_node_id")
            nd["data"]["moduleExitNodeId"] = m.get("output_node_id")
    for mid, m in mods.items():
        for c in m.get("internal_nodes", []):
            cd = dict(c)
            cd["data"] = dict(cd["data"])
            cd["data"]["parentId"] = mid
            nodes.append(cd)
    edges = [{"id": e["id"], "source": e["source"], "target": e["target"]}
             for e in graph_dict.get("root_edges", [])]
    for m in mods.values():
        for e in m.get("internal_edges", []):
            edges.append({"id": e.get("id"), "source": e["source"], "target": e["target"]})

    def is_wrap(n):
        return n["type"] == "moduleNode" or n["data"].get("isGroup")

    boundary = {}
    for node in nodes:
        if not is_wrap(node):
            continue
        children = [n for n in nodes if n["data"].get("parentId") == node["id"]]
        if not children:
            continue
        cids = {c["id"] for c in children}
        has_in, has_out = set(), set()
        for e in edges:
            if e["source"] in cids and e["target"] in cids:
                has_out.add(e["source"])
                has_in.add(e["target"])
        entry = node["data"].get("moduleEntryNodeId") or next((c["id"] for c in children if c["id"] not in has_in), children[0]["id"])
        exit_ = node["data"].get("moduleExitNodeId") or next((c["id"] for c in children if c["id"] not in has_out), children[-1]["id"])
        boundary[node["id"]] = (entry, exit_)

    keep = [n for n in nodes if not is_wrap(n)]
    ids = {n["id"] for n in keep}
    out_edges = []
    for e in edges:
        s, t = e["source"], e["target"]
        if s in boundary:
            s = boundary[s][1]
        if t in boundary:
            t = boundary[t][0]
        if s in ids and t in ids:
            out_edges.append({"id": e.get("id") or f"{s}->{t}", "source": s, "target": t})
    rn = [{"id": n["id"], "type": n["type"], "position": n.get("position", {"x": 0, "y": 0}),
           "data": {"layerType": n["data"].get("layerType"), "parameters": n["data"].get("parameters", {}),
                    "parentId": n["data"].get("parentId")}} for n in keep]
    return ValidationRequest(nodes=rn, edges=out_edges, input_shape=shape)


@pytest.mark.parametrize("model_id", _POPULAR)
def test_popular_model_verifies_and_validates(model_id):
    graph = _SVC.instantiate_popular_model(model_id, {})
    gd = graph.model_dump()
    shape = _input_shape(gd)

    result = verify_model(gd, shape)
    assert result["ok"] is True, f"{model_id} verify failed: {result.get('errors')}"

    validation = validate_shapes(_canvas_payload(gd, shape))
    assert validation["status"] == "valid", f"{model_id} validation errors: {validation['errors']}"


@pytest.mark.parametrize("template_id", _TEMPLATE_IDS)
def test_template_verifies_and_validates(template_id):
    graph = _SVC.instantiate_template(template_id, _TEMPLATES[template_id].base_config)
    gd = graph.model_dump()
    shape = _input_shape(gd)

    result = verify_model(gd, shape)
    assert result["ok"] is True, f"{template_id} verify failed: {result.get('errors')}"

    validation = validate_shapes(_canvas_payload(gd, shape))
    assert validation["status"] == "valid", f"{template_id} validation errors: {validation['errors']}"
