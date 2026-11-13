"""Documentation is derived from exact supplied declarations, never execution."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from services.semantic.source_context import SourceContextRefused, query_source_context, _SOURCE_CACHE


def query(files, symbol, line=None, file="model.py", **options):
    return query_source_context(dict(sources=[dict(file=name, source=text, digest=hashlib.sha256(text.encode()).hexdigest())
                                              for name, text in files.items()],
        query=dict(file=file, symbol=symbol, line=line, depth=0, maxEntities=8, limit=8) | options))


def docs(source, symbol="helper", line=None):
    return query({"model.py": source}, symbol, line).get("documentation")


def test_exact_receiver_method_has_its_declaration_and_no_invented_docstring():
    source = "class Net:\n    def _score_hidden(self, samples, candidates, queries, score_mask):\n        return self.head(candidates, queries)\n    def forward(self, samples, candidates, queries, score_mask):\n        return self._score_hidden(samples, candidates, queries, score_mask)\n"
    result = docs(source, "self._score_hidden", 5)
    assert result["status"] == "resolved"
    assert result["qualifiedName"] == "Net._score_hidden"
    assert result["signature"] == "def _score_hidden(self, samples, candidates, queries, score_mask)"
    assert result["kind"] == "method" and result["truncated"] is False
    assert "docstring" not in result
    assert (result["declaration"]["startLine"], result["declaration"]["endLine"]) == (2, 3)
    assert result["declaration"]["digest"] == hashlib.sha256(source.encode()).hexdigest()


def test_signature_preserves_declaration_semantics_without_executing_defaults(tmp_path):
    marker = tmp_path / "must-not-exist"
    source = f'''from pathlib import Path
async def helper(a: int, /, b=Path({str(marker)!r}).write_text('bad'), *items: str, c: int = 3, **options: bool) -> tuple[int, str]:
    """Short summary.

        Detailed source documentation.
    """
    return a, 'value'
helper(1)
'''
    result = docs(source, "helper", 8)
    assert result["signature"].startswith("async def helper(a: int, /, b=Path(")
    assert "*items: str, c: int=3, **options: bool) -> tuple[int, str]" in result["signature"]
    assert result["docstring"] == "Short summary.\n\nDetailed source documentation."
    assert not marker.exists()


def test_class_documentation_does_not_substitute_init_or_forward():
    source = '''class Block(Base):
    """Source class documentation."""
    def __init__(self, width=4):
        """Constructor only."""
    def forward(self, x):
        """Forward only."""
        return x
Block(12)
'''
    result = docs(source, "Block", 8)
    assert result["kind"] == "class" and result["signature"] == "class Block(Base)"
    assert result["docstring"] == "Source class documentation."


@pytest.mark.parametrize("statement,call", [
    ("from pkg import Block as Alias", "Alias"),
    ("from pkg.blocks import Block as Alias", "Alias"),
    ("import pkg.blocks as blocks", "blocks.Block"),
    ("import pkg.blocks", "pkg.blocks.Block"),
    ("from . import blocks", "blocks.Block"),
])
def test_explicit_aliases_relative_imports_and_reexports_resolve_supplied_declarations(statement, call):
    files = {"pkg/blocks.py": 'class Block:\n    """A supplied custom block."""\n',
             "pkg/__init__.py": "from .blocks import Block\n", "pkg/model.py": f"{statement}\nvalue = {call}()\n",
             "unrelated.py": 'class Block:\n    """Wrong same-name class."""\n'}
    result = query(files, call, 2, file="pkg/model.py")["documentation"]
    assert result["status"] == "resolved" and result["docstring"] == "A supplied custom block."
    assert result["declaration"]["file"] == "pkg/blocks.py"
    assert query(dict(reversed(list(files.items()))), call, 2, file="pkg/model.py")["documentation"] == result


def test_unique_constructed_instance_attributes_and_single_base_inheritance():
    source = '''class Base:
    def helper(self, x):
        """Base implementation."""
        return x
class Block(Base):
    """The actual block."""
class Net:
    def __init__(self):
        self.block = Block()
    def forward(self, x):
        return self.block.helper(x)
'''
    result = docs(source, "self.block.helper", 11)
    assert result["qualifiedName"] == "Base.helper"
    assert result["docstring"] == "Base implementation."
    source = source.replace("self.block.helper(x)", "self.block(x)")
    result = docs(source, "self.block", 11)
    assert result["kind"] == "class" and result["docstring"] == "The actual block."


def test_override_takes_precedence_over_base_implementation():
    source = '''class Base:
    def helper(self):
        """Base."""
class Derived(Base):
    def helper(self):
        """Override."""
    def forward(self):
        return self.helper()
'''
    assert docs(source, "self.helper", 8)["docstring"] == "Override."


@pytest.mark.parametrize("source,symbol,line,reason", [
    ("def helper(): pass\nhelper = unknown\nhelper()\n", "helper", 3, "ambiguous"),
    ("def helper(): pass\ndef helper(): pass\nhelper()\n", "helper", 3, "ambiguous"),
    ("if flag:\n    def helper(): pass\nhelper()\n", "helper", 3, "dynamic_binding"),
    ("@decorate\ndef helper(): pass\nhelper()\n", "helper", 3, "decorated"),
    ("from missing import helper\nhelper()\n", "helper", 2, "unresolved_import"),
    ("from unknown import *\ndef helper(): pass\nhelper()\n", "helper", 3, "dynamic_binding"),
    ("def helper(): pass\nexec('anything')\nhelper()\n", "helper", 3, "dynamic_binding"),
    ("def helper(): pass\nhelper(); helper()\n", "helper", 2, "ambiguous"),
    ("class Net(A, B):\n    def forward(self):\n        return self.helper()\n", "self.helper", 3, "ambiguous"),
    ("class Net:\n    @property\n    def helper(self): pass\n    def forward(self):\n        return self.helper()\n", "self.helper", 5, "decorated"),
    ("class Net:\n    def __init__(self):\n        self.block = factory()\n    def forward(self):\n        return self.block()\n", "self.block", 5, "unresolved_symbol"),
    ("class Block: pass\nclass Net:\n    def __init__(self):\n        if flag:\n            self.block = Block()\n    def forward(self):\n        return self.block()\n", "self.block", 7, "dynamic_binding"),
    ("class Block: pass\nclass Net:\n    def __init__(self):\n        self.block = Block()\n        self.block = Block()\n    def forward(self):\n        return self.block()\n", "self.block", 7, "ambiguous"),
    ("class Net:\n    def helper(self): pass\n    def forward(self):\n        self = unknown\n        return self.helper()\n", "self.helper", 5, "ambiguous"),
    ("class Net:\n    def helper(self): pass\n    def forward(self):\n        return self.helper()\nNet.helper = unknown\n", "self.helper", 4, "ambiguous"),
    ("class Net:\n    def helper(self): pass\n    def forward(self):\n        setattr(self, name, other)\n        return self.helper()\n", "self.helper", 5, "dynamic_binding"),
])
def test_uncertain_bindings_are_unavailable(source, symbol, line, reason):
    assert docs(source, symbol, line) == dict(status="unavailable", reason=reason)


def test_unknown_receiver_does_not_select_a_same_named_class():
    files = {"model.py": "def forward(block):\n    return block.helper()\n",
             "other.py": "class block:\n    def helper(self):\n        'Wrong target.'\n"}
    assert query(files, "block.helper", 2)["documentation"]["status"] == "unavailable"


@pytest.mark.parametrize("mutation,reason", [("helpers.score = other", "ambiguous"),
    ("setattr(helpers, 'score', other)", "dynamic_binding"), ("del helpers.score", "ambiguous")])
@pytest.mark.parametrize("statement,call", [("import helpers", "helpers.score"),
    ("import helpers\nfrom helpers import score", "score")])
def test_visible_module_attribute_mutations_are_not_ignored(mutation, reason, statement, call):
    source = f"{statement}\n{mutation}\n{call}()\n"
    result = query({"model.py": source, "helpers.py": 'def score():\n    """Original."""\n'}, call,
                   len(source.splitlines()))["documentation"]
    assert result == dict(status="unavailable", reason=reason)


def test_unrelated_attribute_introspection_does_not_poison_exact_lexical_method():
    source = '''class Other:
    def copy(self, other):
        for name, value in vars(other).items():
            setattr(other, name, value)
class Net:
    def helper(self):
        """Exact declaration."""
    def forward(self):
        return self.helper()
'''
    assert docs(source, "self.helper", 9)["docstring"] == "Exact declaration."


def test_module_qualified_declaration_requires_exact_file_scope_and_line():
    files = {"models/blocks.py": 'class UncommonBlock:\n    """Exact custom declaration."""\n'}
    result = query(files, "models.blocks.UncommonBlock", 1, file="models/blocks.py")["documentation"]
    assert result["qualifiedName"] == "UncommonBlock"
    for wrong in ("other.blocks.UncommonBlock", "models.blocks.Other.UncommonBlock"):
        with pytest.raises(SourceContextRefused, match="unknown_source_symbol"):
            query(files, wrong, 1, file="models/blocks.py")


def test_a_call_and_declaration_with_the_same_line_and_name_are_ambiguous():
    assert docs("def helper(x=helper()): pass\n", "helper", 1) == dict(status="unavailable", reason="ambiguous")


def test_keyword_only_separator_and_nested_qualified_declaration_are_preserved():
    source = 'def outer():\n    def helper(x, *, training=False):\n        """Nested helper."""\n    return helper(1)\n'
    result = docs(source, "helper", 4)
    assert result["signature"] == "def helper(x, *, training=False)"
    assert result["qualifiedName"] == "outer.<locals>.helper"


def test_alias_cycles_and_long_chains_have_finite_unavailable_results():
    assert docs("a = b\nb = a\na()\n", "a", 3) == dict(status="unavailable", reason="dynamic_binding")
    source = "def helper(): pass\n" + "\n".join(f"alias{i} = " + ("helper" if i == 0 else f"alias{i-1}") for i in range(80)) + "\nalias79()\n"
    assert docs(source, "alias79", 82) == dict(status="unavailable", reason="analysis_limit")


def test_signature_limit_is_unavailable_and_docstrings_are_bounded_utf16():
    result = docs("def helper(value='" + "x" * 2100 + "'): pass\n")
    assert result == dict(status="unavailable", reason="analysis_limit")
    result = docs('def helper():\n    """' + "😀" * 3000 + '"""\n')
    assert result["truncated"] is True
    assert len(result["docstring"].encode("utf-16-le")) // 2 <= 4096
    assert result["docstring"].endswith("…")
    assert len(json.dumps(result).encode()) < 48 * 1024


def test_control_characters_are_not_forwarded_from_docstrings():
    result = docs('def helper():\n    """\\ufeff  Summary\\x01\\x7f\n\n        Details. \\ufeff """\n')
    assert result["docstring"] == "Summary\n\nDetails."
    assert result["truncated"] is True


def test_escaped_lone_surrogates_are_sanitized_without_breaking_the_page():
    result = docs('def helper():\n    """Summary\\ud800\\udfff details."""\n')
    assert result["docstring"] == "Summary details." and result["truncated"] is True
    json.dumps(result, ensure_ascii=False).encode("utf-8")


def test_unindexed_dynamic_annotations_are_refused_without_importing_or_crashing():
    result = docs('def helper(x: eval("anything")): pass\nhelper()\n', "helper", 2)
    assert result == dict(status="unavailable", reason="dynamic_binding")


@pytest.mark.parametrize("hook", ["__getattr__", "__getattribute__", "__setattr__"])
def test_supplied_inherited_dynamic_attribute_hooks_are_not_treated_as_exact_calls(hook):
    source = f'''class Base:
    def {hook}(self, *args):
        return unknown
class Derived(Base):
    def helper(self):
        """Cannot prove this is called."""
    def forward(self):
        return self.helper()
'''
    assert docs(source, "self.helper", 8) == dict(status="unavailable", reason="dynamic_binding")


def test_non_callable_configuration_query_omits_documentation():
    assert "documentation" not in query({"model.py": "WIDTH = 12\nvalue = other(WIDTH)\n"}, "WIDTH", 1)


def test_cached_ast_reuse_does_not_mutate_headers_or_results(monkeypatch):
    import services.semantic.source_context as context
    _SOURCE_CACHE.clear()
    source = 'def helper(a: int = 3) -> int:\n    """Original documentation."""\n    return a\nhelper()\n'
    first = docs(source, "helper", 4)
    unchanged = deepcopy(first)
    first["declaration"]["file"] = "changed.py"
    def forbidden_parse(*args, **kwargs):
        raise AssertionError("cached documentation must reuse its parsed AST")
    monkeypatch.setattr(context.ast, "parse", forbidden_parse)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: docs(source, "helper", 4), range(12)))
    assert all(result == unchanged for result in results)
