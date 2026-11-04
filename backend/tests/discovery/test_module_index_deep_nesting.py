"""Export facts must not recurse once per AST level (long generated operator chains)."""
import ast

from services.ide.module_index import ModuleIndex


def test_export_facts_handles_deeply_nested_expression():
    chain = " + ".join(["1"] * 3000)
    source = f"import torch\nVALUE = {chain}\nclass Net(torch.nn.Module):\n    pass\n"
    tree = ast.parse(source)
    facts = ModuleIndex._export_facts(tree)
    assert facts is not None


def test_export_facts_order_unchanged_for_scope_rules():
    source = (
        "import os\n"
        "def f(x=os.environ):\n"
        "    import sys\n"
        "if True:\n"
        "    from math import pi\n"
    )
    facts = ModuleIndex._export_facts(ast.parse(source))
    assert facts is not None


def test_scan_skips_a_pathologically_nested_file_instead_of_failing(tmp_path):
    from services.ingest.repo_scan import scan_repo
    (tmp_path / "model.py").write_text(
        "import torch\nclass Net(torch.nn.Module):\n    def forward(self, x):\n        return x\n")
    (tmp_path / "generated.py").write_text("VALUE = " + " + ".join(["1"] * 3000) + "\n")
    result = scan_repo(str(tmp_path))
    assert result is not None


def test_module_index_marks_too_deep_file_as_unparseable():
    record = ModuleIndex.build({"generated.py": "VALUE = " + " + ".join(["1"] * 3000) + "\n",
                                "ok.py": "X = 1\n"}).modules
    assert record["generated"].syntax_error is not None
    assert "too deep" in record["generated"].syntax_error["message"]
    assert record["ok"].syntax_error is None
