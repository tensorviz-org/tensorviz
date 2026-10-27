from services.validation.contracts import Binding
from services.validation.signatures import check_bindings, check_factory


def test_syntax_errors_are_source_problems_not_invalid_input_bindings():
    inputs = Binding(args=[], kwargs={})
    model = check_bindings(b"class Net(:", "Net", "qa", inputs, inputs)
    factory = check_factory(b"def make(:", "make", "qa", {})
    assert model[0].code == factory[0].code == "invalid_source"
    assert model[0].field == ["profiles", "qa", "model"]
    assert factory[0].field == ["profiles", "qa", "factory", "file"]


def test_required_arguments_still_require_setup():
    issues = check_bindings(b"class Net:\n def forward(self, x, mask): pass", "Net", "qa",
                            Binding(args=[], kwargs={}), Binding(args=[], kwargs={"x": 1}))
    assert issues and all(issue.code != "invalid_source" for issue in issues)
    assert any("mask" in issue.field for issue in issues)
