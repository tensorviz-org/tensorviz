"""Tests for graph session undo/redo history behavior."""



from models.architecture_models import ArchitectureNode  # noqa: E402
from services.graph_session_service import GraphSessionService  # noqa: E402
from services.graph_session_store import GraphSessionStore  # noqa: E402


def test_undo_redo_round_trip():
    store = GraphSessionStore()
    service = GraphSessionService(store=store)
    session = service.create_session(name="hist_model", input_shape=[1, 3, 32, 32], session_id="hist1")

    before = session.graph.model_copy(deep=True)
    session.graph.nodes.append(
        ArchitectureNode(node_id="conv1", kind="layer", op="nn.Conv2d", params={"in_channels": 3, "out_channels": 8, "kernel_size": 3})
    )
    session.graph.revision += 1

    assert store.push_undo_snapshot(session_id=session.session_id, graph_snapshot=before) is True
    store.save_session(session)

    undone = store.undo(session.session_id)
    assert undone is not None
    assert not any(node.node_id == "conv1" for node in undone.graph.nodes)

    redone = store.redo(session.session_id)
    assert redone is not None
    assert any(node.node_id == "conv1" for node in redone.graph.nodes)


def test_new_mutation_clears_redo_stack():
    store = GraphSessionStore()
    service = GraphSessionService(store=store)
    session = service.create_session(name="redo_clear", input_shape=[1, 3, 32, 32], session_id="hist2")

    before = session.graph.model_copy(deep=True)
    session.graph.revision += 1
    store.push_undo_snapshot(session_id=session.session_id, graph_snapshot=before)
    store.save_session(session)

    assert store.undo(session.session_id) is not None
    state_after_undo = store.history_state(session.session_id)
    assert state_after_undo is not None
    assert state_after_undo["redo_count"] == 1

    # Simulate a new mutation after undo.
    current = store.get_session(session.session_id)
    assert current is not None
    store.push_undo_snapshot(session_id=session.session_id, graph_snapshot=current.graph.model_copy(deep=True))

    state_after_new = store.history_state(session.session_id)
    assert state_after_new is not None
    assert state_after_new["redo_count"] == 0
