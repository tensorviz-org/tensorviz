"""The harness itself: if these fail, nothing else in the suite means anything."""


def test_the_backend_answers_over_http(api):
    body = api.health().json()
    assert body["status"] == "healthy"


def test_a_class_projects_to_a_graph(api):
    source = "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.l1 = nn.Linear(8, 8)\n\n    def forward(self, x):\n        return self.l1(x)\n"
    response = api.parse(source, "Net")
    assert response.status_code == 200, response.text
    parsed = response.json()
    ids = [n["node_id"] for n in parsed["document"]["semantic"]["nodes"]]
    assert "l1" in ids


def test_an_edit_round_trips_back_into_the_source(edited):
    source = "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.l1 = nn.Linear(8, 8)\n\n    def forward(self, x):\n        return self.l1(x)\n"
    new_source, refused = edited(
        source, "Net",
        [{"op": "set_param", "op_id": "o1", "node_id": "l1", "param": "out_features", "value": 32}],
    )
    assert refused == []
    assert "nn.Linear(8, 32)" in new_source or "out_features=32" in new_source
