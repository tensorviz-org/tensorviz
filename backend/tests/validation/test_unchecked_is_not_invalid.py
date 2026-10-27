"""An opaque block starves what follows it — say so once, and do not call it broken (#196).

A placeholder nobody has measured has no output shape, correctly: an opaque statement is
not shape-preserving. But the shape chain then breaks there, and every node after it used
to be reported "input shape could not be resolved" — one error each, on layers that are
perfectly correct and whose only fault is standing downstream of something unknown. On a
transformer that was fifty errors about one unknown block, and because none of them
carried a repair, the editor showed none of them.

Two claims are under test, and they pull against each other on purpose:
  * unchecked is NOT invalid — a graph nobody could fully check has not failed, so the
    innocent layers must stop being errors;
  * unchecked is NOT clean either — it must still be reported, or fixing the false red
    just restores the silence that hid this bug for so long.
"""


from api.routers.validate import ValidationRequest, validate_shapes  # noqa: E402


def _chain(n_after=3, out_shape=None):
    """input -> blk(opaque) -> l0 -> l1 -> ... -> output."""
    nodes = [
        {"id": "in", "type": "inputNode", "position": {"x": 0, "y": 0},
         "data": {"layerType": {"id": "input", "name": "Input", "parameters": []},
                  "parameters": {"shape": "1x16"}}},
        {"id": "blk", "type": "layerNode", "position": {"x": 0, "y": 0},
         "data": {"layerType": {"id": "custom_code", "name": "custom_code", "parameters": []},
                  "parameters": ({"_out_shape": out_shape} if out_shape else {})}},
    ]
    edges = [{"id": "e_in", "source": "in", "target": "blk"}]
    prev = "blk"
    for i in range(n_after):
        nid = f"l{i}"
        nodes.append({"id": nid, "type": "layerNode", "position": {"x": 0, "y": 0},
                      "data": {"layerType": {"id": "relu", "name": "ReLU", "parameters": []},
                               "parameters": {}}})
        edges.append({"id": f"e{i}", "source": prev, "target": nid})
        prev = nid
    nodes.append({"id": "out", "type": "outputNode", "position": {"x": 0, "y": 0},
                  "data": {"layerType": {"id": "output", "name": "Output", "parameters": []},
                           "parameters": {}}})
    edges.append({"id": "e_out", "source": prev, "target": "out"})
    return {"nodes": nodes, "edges": edges}


def _run(graph):
    result = validate_shapes(ValidationRequest(**graph))
    return result if isinstance(result, dict) else result.dict()


def test_the_innocent_layers_after_an_opaque_block_are_not_errors():
    result = _run(_chain(n_after=3))
    assert result["errors"] == [], result["errors"]
    assert result["status"] == "valid"


def test_the_block_is_named_once_however_many_nodes_it_starves():
    """N -> 1 must be a property of the response, not of how a message was worded."""
    for n_after in (1, 3, 12):
        result = _run(_chain(n_after=n_after))
        unchecked = result["unchecked"]
        assert len(unchecked) == 1, (n_after, unchecked)
        assert unchecked[0]["nodeId"] == "blk"
        # every layer plus the output node
        assert len(unchecked[0]["blocked"]) == n_after + 1, unchecked[0]["blocked"]


def test_the_count_in_the_message_matches_the_nodes_reported():
    """The message said 2 while the wire said 1 at one point, because output nodes were
    handled before the blocked branch. A number the user can check must be checkable."""
    result = _run(_chain(n_after=4))
    entry = result["unchecked"][0]
    assert f"{len(entry['blocked'])} node(s) after it were not shape-checked" in entry["message"], entry


def test_it_is_not_silent():
    """Fixing the false red must not restore the silence. A graph the validator could not
    fully check has to SAY so — this is the whole complaint that started #196."""
    result = _run(_chain(n_after=3))
    assert result["unchecked"], "a graph with an unchecked block reported nothing at all"


def test_a_measured_block_reports_nothing_because_nothing_is_unchecked():
    result = _run(_chain(n_after=3, out_shape="1x16"))
    assert result["errors"] == []
    assert result["unchecked"] == [], result["unchecked"]


def test_two_opaque_blocks_are_two_entries_not_one_merged_blob():
    graph = _chain(n_after=2)
    for node in graph["nodes"]:
        if node["id"] == "l0":
            node["data"]["layerType"] = {"id": "custom_code", "name": "custom_code", "parameters": []}
    result = _run(graph)
    causes = sorted(entry["nodeId"] for entry in result["unchecked"])
    assert causes == ["blk", "l0"], result["unchecked"]
    # nearest cause wins: l1 is blamed on l0, not on blk
    by_cause = {e["nodeId"]: e["blocked"] for e in result["unchecked"]}
    assert "l1" in by_cause["l0"], by_cause
    assert "l1" not in by_cause["blk"], by_cause
