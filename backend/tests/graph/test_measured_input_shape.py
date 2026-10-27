"""The shape a PREVIOUS trace measured, fed back to the tier that DRAWS the graph.

Unknown input dimensions stay empty in the source-only preview. After a real
trace measures ``1,8``, the Input node must agree with the measured cards. ``measured_input_shape`` is the extension handing
back what the last successful trace actually ran on, so the two tiers agree
without the user writing anything into their own source file.

Its RUNG is the whole point of this file. It sits:

* below ``input_shape`` — a caller that asked for something specific is asking
  now, and a remembered measurement is not an answer to that;
* below ``# tensorviz: input=`` — the one line where the user states what their
  model is fed. Putting a remembered cache above it would make the sidecar
  silently beat the line the user just typed, which is exactly the regression
  commit 6cd50c3 fixed and ``test_declared_input_shape`` pins;
* above AST discovery — a measurement from a real forward pass beats a shape
  inferred from a literal, because the forward pass is what actually ran.
"""

from __future__ import annotations



from models.graph_models import CodeParseRequest  # noqa: E402
from utils.code_parser import CodeParserService  # noqa: E402

#: The same unguessable model ``test_declared_input_shape`` uses: the layer widths
#: are a parameter the file never gives a value to, so nothing in the source can
#: be read as an input shape.
UNGUESSABLE = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.l1 = nn.Linear(width, width)

    def forward(self, x):
        return self.l1(x)
'''

DECLARED = f"{UNGUESSABLE}\n# tensorviz: input=2,8\n"


def _input_shape(code: str, **kwargs) -> str:
    parser = CodeParserService()
    result = parser.parse_pytorch_code(CodeParseRequest(code=code, **kwargs))
    assert result.get("success"), result
    graph = next(e["graph"] for e in result["graphs"] if e.get("graph"))
    node = next(n for n in graph.nodes if n.id == "input")
    return node.data.parameters["shape"]


def test_a_measured_shape_is_drawn_when_the_file_says_nothing():
    # Without it this file draws the image default and covers a correct Linear
    # model in shape errors — the state the declaration exists to escape, except
    # here the user never had to type the declaration.
    assert _input_shape(UNGUESSABLE) == ""
    assert _input_shape(UNGUESSABLE, measured_input_shape=[1, 8]) == "1,8"


def test_the_file_declaration_still_beats_a_measured_shape():
    """The regression guard. A remembered measurement must never outrank the
    line the user just typed: the sidecar is a cache of what happened last time,
    the comment is a sentence addressed to tensorViz about what should happen."""
    assert _input_shape(DECLARED, measured_input_shape=[1, 3, 224, 224]) == "2,8"


def test_an_explicit_request_still_beats_a_measured_shape():
    # Same reason the declaration loses to `input_shape`: somebody is asking now.
    assert _input_shape(UNGUESSABLE, input_shape=[4, 8], measured_input_shape=[1, 8]) == "4,8"


def test_a_measured_shape_beats_what_the_source_can_be_read_to_imply():
    """The rung below. A ``Linear`` whose first argument is a literal gives the
    AST inference something to work with, and it infers a batch of one. The
    measurement is a shape a forward pass really ran on, so it wins — and this is
    the case that proves the two sources are genuinely different, because a file
    the inference cannot read gives the same answer either way."""
    literal = UNGUESSABLE.replace("nn.Linear(width, width)", "nn.Linear(16, 16)")
    assert _input_shape(literal) == "1,16"
    assert _input_shape(literal, measured_input_shape=[8, 16]) == "8,16"


def test_an_empty_measurement_is_not_a_measurement():
    # The extension sends what the envelope carried. A trace that produced no
    # tensor carries an empty list, and an empty list must fall through to the
    # tiers below. With no source evidence, dimensions remain unknown.
    assert _input_shape(UNGUESSABLE, measured_input_shape=[]) == ""


# ── the two routes the extension actually posts to ───────────────────────────
#
# The rung above is decided in `code_parser`, but the extension never calls it:
# it POSTs a buffer to `/api/ide/parse` (the embedded canvas) or to
# `/api/ide/multifile/project` (the local browser page). A field that stops at
# the router is a field that does nothing, and pydantic drops unknown keys
# silently — so both hops are pinned here rather than assumed.

import hashlib  # noqa: E402

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.routers import ide  # noqa: E402
from auth import CurrentUser, get_current_user  # noqa: E402
from services.ide import multifile  # noqa: E402
from services.project_store import ProjectStore  # noqa: E402


@pytest.fixture()
def client(tmp_path) -> TestClient:
    ide._store = ProjectStore(str(tmp_path / "ide.sqlite3"))
    app = FastAPI()
    app.include_router(ide.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(uid="local-dev-user")
    return TestClient(app)


def _input_node(document: dict) -> str:
    node = next(n for n in document["semantic"]["nodes"] if n["node_id"] == "input")
    return node["params"]["shape"]


def _parse(client: TestClient, code: str, **extra) -> str:
    response = client.post(
        "/api/ide/parse", json={"file": "models/net.py", "source": code, "cls": "Net", **extra}
    )
    assert response.status_code == 200, response.text
    return _input_node(response.json()["document"])


def _multifile(client: TestClient, code: str, **root_extra) -> str:
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    response = client.post(
        "/api/ide/multifile/project",
        json={
            "protocol_version": multifile.IDE_PROJECTION_PROTOCOL_VERSION,
            "root": {
                "file": "models/net.py",
                "cls": "Net",
                "document_version": 1,
                "source_digest": digest,
                **root_extra,
            },
            "sources": [{"file": "models/net.py", "source": code, "digest": digest}],
            "truncated": False,
        },
    )
    assert response.status_code == 200, response.text
    return _input_node(response.json()["document"])


def test_the_parse_route_carries_the_measured_shape_to_the_drawn_input(client: TestClient):
    assert _parse(client, UNGUESSABLE) == ""
    assert _parse(client, UNGUESSABLE, measured_input_shape=[1, 8]) == "1,8"


def test_the_parse_route_still_lets_the_file_declaration_win(client: TestClient):
    assert _parse(client, DECLARED, measured_input_shape=[1, 3, 224, 224]) == "2,8"


def test_the_multifile_route_carries_the_measured_shape_to_the_drawn_input(client: TestClient):
    # The local browser page's endpoint. It reaches the parser through the
    # cross-file expander rather than directly, so "the router accepted the
    # field" and "the root class was drawn with it" are two different facts.
    assert _multifile(client, UNGUESSABLE) == ""
    assert _multifile(client, UNGUESSABLE, measured_input_shape=[1, 8]) == "1,8"


def test_the_multifile_route_still_lets_the_file_declaration_win(client: TestClient):
    assert _multifile(client, DECLARED, measured_input_shape=[1, 3, 224, 224]) == "2,8"
