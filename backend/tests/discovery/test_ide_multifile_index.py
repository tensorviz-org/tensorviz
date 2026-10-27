"""Static module index — Step 2 of "P0 Reliable Multi-File Model Graphs".

Step 1 froze the contract; step 2 builds the deterministic, AST-only answer to
"which class definition does this name refer to". The guarantees under test:

  * every supported import form (absolute, aliased, relative at any level,
    module-as-value, unambiguous package re-export) resolves to the CANONICAL
    definition ``<module>::<name>`` — so the two ``Block`` classes in the
    fixture (models/blocks.py vs models/heads.py) never contaminate each other;
  * resolution never guesses: unknown wildcards, computed imports, ambiguous bindings,
    cycles and missing modules all produce coded diagnostics from the closed
    set the contract defines, and the offending name stays unresolved;
  * the index is total and isolated: it is built from every supplied file
    (never from the disk), one unparsable file degrades only itself, and two
    runs over the same input are byte-for-byte identical;
  * building the index never executes the sources.

The projection wiring that CONSUMES these answers (qualified instance ids,
cross-file spans, cycle boundaries in the graph) is step 3; these tests pin the
index as a pure function of the source set so step 3 can treat it as trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.ide import multifile
from services.ide.module_index import (
    ClassInfo,
    ModuleIndex,
    module_of_file,
    package_of_file,
)

FIXTURE_MODELS = Path(__file__).parent.resolve().parent / "fixtures" / "ide_multifile_repo" / "models"


def _fixture_sources() -> dict:
    return {
        f"models/{path.name}": path.read_text(encoding="utf-8")
        for path in sorted(FIXTURE_MODELS.glob("*.py"))
    }


@pytest.fixture(scope="module")
def index() -> ModuleIndex:
    return ModuleIndex.build(_fixture_sources())


def _definition(index: ModuleIndex, from_module: str, expr: str) -> ClassInfo:
    result = index.resolve(from_module, expr)
    assert result.diagnostic is None, f"{expr}: expected a definition, got {result.diagnostic}"
    assert isinstance(result.definition, ClassInfo)
    return result.definition


def _diagnostic(index: ModuleIndex, from_module: str, expr: str, code: str, severity: str | None = None) -> dict:
    result = index.resolve(from_module, expr)
    assert result.definition is None, f"{expr}: expected a diagnostic, got {result.definition}"
    assert result.diagnostic is not None
    assert result.diagnostic["code"] == code, result.diagnostic
    if severity is not None:
        assert result.diagnostic["severity"] == severity
    return result.diagnostic


# ── canonical addressing ─────────────────────────────────────────────────────


def test_module_and_package_of_file() -> None:
    assert module_of_file("models/__init__.py") == "models"
    assert module_of_file("models/net.py") == "models.net"
    assert module_of_file("net.py") == "net"
    assert package_of_file("models/net.py") == "models"
    assert package_of_file("net.py") == ""


def test_indexes_every_supplied_module(index: ModuleIndex) -> None:
    expected = {
        "models",
        "models.blocks",
        "models.cycle_a",
        "models.cycle_b",
        "models.dynamic",
        "models.enc",
        "models.heads",
        "models.mixer",
        "models.net",
    }
    assert set(index.modules) == expected
    for module, record in index.modules.items():
        assert record.module == module
        assert module_of_file(record.file) == module
        assert record.syntax_error is None


# ── the supported import forms resolve canonically ───────────────────────────


def test_reexport_through_init_prefers_blocks_over_heads(index: ModuleIndex) -> None:
    # `from models import CoreBlock` → models/__init__ re-exports `.blocks.Block`.
    resolved = _definition(index, "models.net", "CoreBlock")
    assert resolved.key == "models.blocks::Block"
    assert resolved.file == "models/blocks.py"


def test_aliased_absolute_import_prefers_heads(index: ModuleIndex) -> None:
    # `from models.heads import Block as HeadBlock` — the OTHER Block.
    resolved = _definition(index, "models.net", "HeadBlock")
    assert resolved.key == "models.heads::Block"
    assert resolved.file == "models/heads.py"


def test_local_definition_wins_inside_own_module(index: ModuleIndex) -> None:
    # WideBlock(Block) inside heads.py must bind to the local Block.
    base = _definition(index, "models.heads", "Block")
    assert base.key == "models.heads::Block"
    wide = _definition(index, "models.heads", "WideBlock")
    assert wide.bases == ("Block",)


def test_relative_from_import(index: ModuleIndex) -> None:
    resolved = _definition(index, "models.mixer", "Block")
    assert resolved.key == "models.blocks::Block"


def test_module_as_value_attribute_chain(index: ModuleIndex) -> None:
    # `import models.blocks as blocks` then `blocks.Block(...)`.
    resolved = _definition(index, "models.enc", "blocks.Block")
    assert resolved.key == "models.blocks::Block"
    encoder = _definition(index, "models.enc", "Encoder")
    assert encoder.bases == ("blocks.Block",)


def test_root_imports_resolve_across_files(index: ModuleIndex) -> None:
    assert _definition(index, "models.net", "Encoder").key == "models.enc::Encoder"
    assert _definition(index, "models.net", "Mixer").key == "models.mixer::Mixer"
    assert _definition(index, "models.net", "WideBlock").key == "models.heads::WideBlock"
    assert _definition(index, "models.net", "CycleA").key == "models.cycle_a::CycleA"
    assert _definition(index, "models.net", "DynamicStage").key == "models.dynamic::DynamicStage"


def test_cycle_pair_still_resolves_across_files(index: ModuleIndex) -> None:
    # The cycle is a PROJECTION-level boundary (step 3 renders it opaque); at
    # the index level each side's import is an ordinary resolvable binding and
    # resolution must terminate.
    assert _definition(index, "models.cycle_a", "CycleB").key == "models.cycle_b::CycleB"
    assert _definition(index, "models.cycle_b", "CycleA").key == "models.cycle_a::CycleA"


def test_duplicated_class_names_never_merge(index: ModuleIndex) -> None:
    blocks_block = index.modules["models.blocks"].classes["Block"]
    heads_block = index.modules["models.heads"].classes["Block"]
    assert blocks_block.key != heads_block.key
    assert blocks_block.file != heads_block.file
    # ...and the two net.py references land on their own sides.
    assert _definition(index, "models.net", "CoreBlock").file == "models/blocks.py"
    assert _definition(index, "models.net", "HeadBlock").file == "models/heads.py"


def test_class_spans_track_the_definition(index: ModuleIndex) -> None:
    source = _fixture_sources()["models/blocks.py"]
    line = source[: source.index("class Block")].count("\n") + 1
    resolved = _definition(index, "models.net", "CoreBlock")
    assert resolved.line == line
    assert resolved.end_line > line
    assert resolved.end_line <= source.count("\n") + 1
    assert resolved.as_dict()["key"] == "models.blocks::Block"


# ── diagnostics: never guess, always say why ─────────────────────────────────


def test_computed_import_flagged_never_guessed(index: ModuleIndex) -> None:
    record = index.modules["models.dynamic"]
    codes = [d["code"] for d in record.dynamic_diagnostics]
    assert codes == [multifile.UNRESOLVED_DYNAMIC]
    assert "importlib.import_module" in record.dynamic_diagnostics[0]["detail"]
    # DynamicStage itself is a plain class — it indexes fine.
    assert "DynamicStage" in record.classes


def test_external_package_is_info_not_error(index: ModuleIndex) -> None:
    diagnostic = _diagnostic(index, "models.blocks", "nn", multifile.EXTERNAL_PACKAGE, "info")
    assert "torch" in diagnostic["detail"]


def test_local_wildcard_resolves_declared_public_exports_without_execution() -> None:
    index = ModuleIndex.build(
        {
            "base.py": "raise RuntimeError('must not import')\nclass Hidden: ...\nclass _Private: ...\n",
            "w.py": "from base import *\nclass User: ...\n",
        }
    )
    record = index.modules["w"]
    assert not record.import_diagnostics
    assert _definition(index, "w", "User").key == "w::User"
    visited = set()
    assert index.resolve("w", "Hidden", visited=visited).definition.key == "base::Hidden"
    assert visited == {"w.py", "base.py"}
    _diagnostic(index, "w", "_Private", multifile.UNRESOLVED_IMPORT, "warning")


def test_literal_all_and_wildcard_reexports_preserve_canonical_identity_and_dependencies():
    index = ModuleIndex.build({
        "pkg/__init__.py": "from .bridge import *\n__all__ = ['_Hidden']\n",
        "pkg/bridge.py": "from .base import *\n__all__ = ('_Hidden',)\n",
        "pkg/base.py": "__all__ = ['_Hidden', 'Other']\nclass _Hidden: ...\nclass Other: ...\n",
        "use.py": "from pkg import *\n",
    })
    visited = set()
    assert index.resolve("use", "_Hidden", visited=visited).definition.key == "pkg.base::_Hidden"
    assert visited == {"use.py", "pkg/__init__.py", "pkg/bridge.py", "pkg/base.py"}
    _diagnostic(index, "use", "Other", multifile.UNRESOLVED_IMPORT)
    assert not any(record.import_diagnostics for record in index.modules.values())


@pytest.mark.parametrize("exports", [
    "__all__ = make_names()", "__all__ = ['Block']\n__all__.append('Other')",
    "__all__ = ['Block']\n__all__ += ['Other']", "__all__ = ['Block']\nnames = __all__",
    "if flag:\n    __all__ = ['Block']", "__all__ = ['Block']\n__all__ = ['Other']",
    "if flag:\n    Block = Other", "try:\n    from other import Block\nexcept ImportError:\n    pass",
    "__all__ = ['Missing']", "__all__ = [name for name in names]",
    "__all__ = ['Block']\ndef helper(x=__all__.clear()): pass",
    "globals()['Block'] = Other", "exec('Block = Other')",
    "def __getattr__(name): return Other", "__all__ = ['Block']\ndel __all__",
    "from other import __all__", "from other import names as __all__",
])
def test_dynamic_or_conditional_exports_stay_explicitly_unresolved(exports):
    index = ModuleIndex.build({"base.py": "class Block: ...\n" + exports + "\n", "use.py": "from base import *\n"})
    assert [d['code'] for d in index.modules['use'].import_diagnostics] == [multifile.UNSUPPORTED_IMPORT_FORM]
    _diagnostic(index, 'use', 'Block', multifile.AMBIGUOUS_BINDING)
    assert index.modules['use'].wildcard_dependencies == ('base.py',)


@pytest.mark.parametrize("suffix", ["Block = factory()", "class Block: ...", "del Block", "if flag:\n    Block = Other"])
def test_rebound_class_is_not_certified_by_wildcard(suffix):
    index = ModuleIndex.build({"base.py": "class Block: ...\n" + suffix + "\n", "use.py": "from base import *\n"})
    _diagnostic(index, 'use', 'Block', multifile.AMBIGUOUS_BINDING)


def test_wildcards_track_collisions_empty_exports_helpers_and_untouched_private_names():
    index = ModuleIndex.build({
        "left.py": "class Block: ...\n",
        "right.py": "class Block: ...\n",
        "use.py": "from left import *\nfrom right import *\n",
        "none.py": "__all__ = []\nclass Hidden: ...\n",
        "empty.py": "from none import *\nclass User: ...\n",
        "helper.py": "def helper():\n    from external import Helper\nclass Good: ...\n",
        "helpers.py": "from helper import *\n",
    })
    _diagnostic(index, 'use', 'Block', multifile.AMBIGUOUS_BINDING)
    assert not index.modules['empty'].import_diagnostics
    assert _definition(index, 'empty', 'User').key == 'empty::User'
    _diagnostic(index, 'empty', 'Hidden', multifile.UNRESOLVED_IMPORT)
    assert _definition(index, 'helpers', 'Good').key == 'helper::Good'
    assert index.resolve('helpers', 'helper').non_class


def test_unknown_wildcard_cannot_silently_override_an_explicit_binding():
    index = ModuleIndex.build({"base.py": "class Block: ...\n", "use.py": "from base import Block\nfrom missing import *\n"})
    _diagnostic(index, 'use', 'Block', multifile.AMBIGUOUS_BINDING)
    assert index.modules['use'].unresolved_wildcard


def test_wildcard_that_reexports_all_does_not_invent_the_importers_effective_exports():
    index = ModuleIndex.build({'base.py': "__all__ = ['__all__', 'Block']\nclass Block: ...\n",
                               'bridge.py': 'from base import *\n', 'use.py': 'from bridge import *\n'})
    _diagnostic(index, 'use', 'Block', multifile.AMBIGUOUS_BINDING)


def test_cyclic_and_missing_wildcard_exports_terminate_without_guessing():
    index = ModuleIndex.build({"a.py": "from b import *\nclass A: ...\n", "b.py": "from a import *\nclass B: ...\n",
                               "use.py": "from a import *\n", "external.py": "from nonexistent import *\n"})
    _diagnostic(index, 'use', 'A', multifile.AMBIGUOUS_BINDING)
    _diagnostic(index, 'external', 'Block', multifile.AMBIGUOUS_BINDING)
    assert set(index.modules['use'].wildcard_dependencies) == {'a.py', 'b.py'}


def test_every_wildcard_cycle_member_keeps_the_entire_refresh_dependency_set():
    sources = {'a.py': 'from b import *\n', 'b.py': 'from c import *\n', 'c.py': 'from a import *\n',
               'use.py': 'from c import *\n'}
    index = ModuleIndex.build(sources)
    for module in ('a', 'b', 'c', 'use'):
        visited = set()
        assert not index.resolve(module, 'Block', visited=visited).ok
        assert {'a.py', 'b.py', 'c.py'} <= visited
    repaired = ModuleIndex.build({**sources, 'b.py': 'class Block: ...\n'})
    assert _definition(repaired, 'use', 'Block').key == 'b::Block'


def test_literal_package_all_can_name_a_supplied_submodule():
    index = ModuleIndex.build({"pkg/__init__.py": "__all__ = ['blocks']\n", "pkg/blocks.py": "class Block: ...\n",
                               "use.py": "from pkg import *\n"})
    visited = set()
    assert index.resolve('use', 'blocks.Block', visited=visited).definition.key == 'pkg.blocks::Block'
    assert visited == {'use.py', 'pkg/__init__.py', 'pkg/blocks.py'}


def test_wildcard_expansion_has_independent_name_and_binding_limits():
    from services.ide.module_index import _MAX_WILDCARD_EXPORTS, _MAX_WILDCARD_BINDINGS
    classes = ''.join(f'class Block{i}: ...\n' for i in range(_MAX_WILDCARD_EXPORTS + 1))
    index = ModuleIndex.build({'base.py': classes, 'use.py': 'from base import *\n'})
    assert index.modules['use'].unresolved_wildcard
    assert not index.modules['use'].bindings
    classes = classes[:classes.rindex('class Block')]
    index = ModuleIndex.build({'base.py': classes, 'use.py': 'from base import *\n' * 5})
    assert index.modules['use'].unresolved_wildcard
    assert sum(map(len, index.modules['use'].bindings.values())) <= _MAX_WILDCARD_BINDINGS


def test_conditional_import_noted_but_not_bound() -> None:
    index = ModuleIndex.build(
        {
            "base.py": "class Thing: ...\n",
            "c.py": "try:\n    from base import Thing\nexcept ImportError:\n    Thing = None\n",
        }
    )
    assert index.modules["c"].has_conditional_imports
    diagnostic = _diagnostic(index, "c", "Thing", multifile.UNRESOLVED_IMPORT)
    assert "conditional" in diagnostic["detail"]


def test_reexport_cycle_cut_at_boundary() -> None:
    index = ModuleIndex.build(
        {
            "a.py": "from b import X\n",
            "b.py": "from a import X\n",
        }
    )
    _diagnostic(index, "a", "X", multifile.CYCLE_BOUNDARY, "warning")
    _diagnostic(index, "b", "X", multifile.CYCLE_BOUNDARY, "warning")


def test_same_name_two_modules_is_ambiguous() -> None:
    index = ModuleIndex.build(
        {
            "l.py": "class Thing: ...\n",
            "r.py": "class Thing: ...\n",
            "use.py": "from l import Thing\nfrom r import Thing\n",
        }
    )
    diagnostic = _diagnostic(index, "use", "Thing", multifile.AMBIGUOUS_BINDING, "warning")
    assert "l.Thing" in diagnostic["detail"]
    assert "r.Thing" in diagnostic["detail"]


def test_local_definition_plus_import_is_ambiguous() -> None:
    index = ModuleIndex.build(
        {
            "base.py": "class Thing: ...\n",
            "use.py": "from base import Thing\n\nclass Thing: ...\n",
        }
    )
    _diagnostic(index, "use", "Thing", multifile.AMBIGUOUS_BINDING, "warning")


def test_missing_workspace_module_is_unresolved_import() -> None:
    index = ModuleIndex.build(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "from pkg.missing import Thing\n",
        }
    )
    diagnostic = _diagnostic(index, "pkg.a", "Thing", multifile.UNRESOLVED_IMPORT, "warning")
    assert "not in the supplied source set" in diagnostic["detail"]


def test_parent_relative_import() -> None:
    index = ModuleIndex.build(
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/shared.py": "class Util: ...\n",
            "pkg/sub/inner.py": "from ..shared import Util\n",
        }
    )
    assert _definition(index, "pkg.sub.inner", "Util").key == "pkg.shared::Util"
    # A single-dot relative import from the subpackage resolves the same way.
    index2 = ModuleIndex.build(
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/leaf.py": "class Leaf: ...\n",
            "pkg/sub/inner.py": "from .leaf import Leaf\n",
        }
    )
    assert _definition(index2, "pkg.sub.inner", "Leaf").key == "pkg.sub.leaf::Leaf"


def test_relative_import_escaping_root_refused_at_index_time() -> None:
    index = ModuleIndex.build({"top.py": "from ..x import Y\n"})
    record = index.modules["top"]
    assert [d["code"] for d in record.import_diagnostics] == [multifile.UNRESOLVED_IMPORT]
    assert "escapes the workspace root" in record.import_diagnostics[0]["detail"]
    _diagnostic(index, "top", "Y", multifile.UNRESOLVED_IMPORT)


def test_syntax_error_isolated_to_its_own_module() -> None:
    index = ModuleIndex.build(
        {
            "good.py": "class Fine: ...\n",
            "bad.py": "def broken(:\n",
            "use.py": "from bad import Thing\nfrom good import Fine\n",
        }
    )
    broken = index.modules["bad"]
    assert broken.syntax_error is not None
    assert [d["code"] for d in broken.import_diagnostics] == [multifile.DEPENDENCY_SYNTAX_ERROR]
    # One bad file never erases the rest of the index.
    assert _definition(index, "use", "Fine").key == "good::Fine"
    # …but anything pointing INTO it is told exactly which file broke.
    diagnostic = _diagnostic(index, "use", "Thing", multifile.DEPENDENCY_SYNTAX_ERROR, "error")
    assert diagnostic["file"] == "bad.py"
    _diagnostic(index, "bad", "Anything", multifile.DEPENDENCY_SYNTAX_ERROR, "error")


def test_paths_revalidated_at_build() -> None:
    with pytest.raises(ValueError):
        ModuleIndex.build({"../evil.py": "pass\n"})
    with pytest.raises(ValueError):
        ModuleIndex.build({"pkg\\evil.py": "pass\n"})
    assert issubclass(multifile.MultifileError, ValueError)


def test_eval_and_computed_getattr_are_dynamic() -> None:
    index = ModuleIndex.build(
        {"e.py": "X = eval('1')\nY = getattr(mod, name)\nZ = getattr(mod, 'literal')\n"}
    )
    diagnostics = index.modules["e"].dynamic_diagnostics
    assert [d["code"] for d in diagnostics] == [
        multifile.UNRESOLVED_DYNAMIC,
        multifile.UNRESOLVED_DYNAMIC,
    ]
    details = " ".join(d["detail"] for d in diagnostics)
    assert "eval()" in details
    assert "getattr() with a computed name" in details
    assert "literal" not in details  # constant-name getattr is not dynamic


def test_duplicate_spellings_of_one_module_conflict() -> None:
    index = ModuleIndex.build(
        {
            "dup/__init__.py": "class Real: ...\n",
            "dup.py": "class Impostor: ...\n",
            "use.py": "from dup import Real\n",
        }
    )
    record = index.modules["dup"]
    assert [d["code"] for d in record.import_diagnostics] == [multifile.AMBIGUOUS_BINDING]
    # Resolution into the conflicted module surfaces the conflict, not a winner.
    _diagnostic(index, "use", "Real", multifile.AMBIGUOUS_BINDING)


# ── attribute chains and shape guarantees ────────────────────────────────────


@pytest.mark.parametrize(
    "sources, code, severity, file, line, import_text",
    [
        pytest.param(
            {"blocks.py": "class Present:\n    pass\n"},
            multifile.UNRESOLVED_IMPORT, "warning", "blocks.py", 0, "",
            id="missing-class",
        ),
        pytest.param(
            {"blocks.py": "from left import Missing\nfrom right import Missing\n"},
            multifile.AMBIGUOUS_BINDING, "warning", "blocks.py", 1,
            "from left import Missing", id="ambiguous-binding",
        ),
        pytest.param(
            {"blocks.py": "from other import Missing\n", "other.py": "from blocks import Missing\n"},
            multifile.CYCLE_BOUNDARY, "warning", "blocks.py", 1,
            "from other import Missing", id="reexport-cycle",
        ),
        pytest.param(
            {"blocks.py": "class Missing(:\n"},
            multifile.DEPENDENCY_SYNTAX_ERROR, "error", "blocks.py", 1, "",
            id="dependency-syntax",
        ),
        pytest.param(
            {"blocks.py": "from broken import Missing\n", "broken.py": "class Missing(:\n"},
            multifile.DEPENDENCY_SYNTAX_ERROR, "error", "broken.py", 1,
            "from broken import Missing", id="reexport-syntax",
        ),
        pytest.param(
            {"blocks.py": "class Missing: ...\n", "blocks/__init__.py": "class Missing: ...\n"},
            multifile.AMBIGUOUS_BINDING, "error", "blocks.py", 1, "",
            id="conflicting-module-paths",
        ),
    ],
)
def test_dotted_failure_preserves_diagnostic_and_consulted_files(
    sources: dict, code: str, severity: str, file: str, line: int, import_text: str
) -> None:
    # A failure behind a module attribute must be actionable, not an assertion
    # or a guessed class. Its source must remain a freshness dependency.
    index = ModuleIndex.build({
        "net.py": "import blocks\nclass Net(blocks.Missing):\n    pass\n", **sources,
    })
    visited = set()
    result = index.resolve("net", "blocks.Missing", visited=visited)
    assert not result.ok
    assert result.definition is None
    assert result.diagnostic is not None
    assert result.diagnostic["code"] == code
    assert result.diagnostic["severity"] == severity
    assert result.diagnostic["file"] == file
    assert result.diagnostic["line"] == line
    assert result.diagnostic["import_text"] == import_text
    assert result.diagnostic["detail"]
    # A module-path conflict has one placeholder record naming blocks.py.
    assert visited == {"net.py"} | (set(sources) - {"blocks/__init__.py"})


def test_dotted_reexport_resolves_canonical_class_and_tracks_each_hop() -> None:
    index = ModuleIndex.build({
        "net.py": "import pkg\n",
        "pkg/__init__.py": "from .bridge import Public\n",
        "pkg/bridge.py": "from .blocks import Block as Public\n",
        "pkg/blocks.py": "raise RuntimeError('must not execute')\nclass Block: ...\n",
    })
    visited = set()
    result = index.resolve("net", "pkg.Public", visited=visited)
    assert result.ok and result.diagnostic is None
    assert result.definition.key == "pkg.blocks::Block"
    assert visited == {"net.py", "pkg/__init__.py", "pkg/bridge.py", "pkg/blocks.py"}


def test_repeated_segment_does_not_confuse_position() -> None:
    # pkg/blocks.py deliberately names its class `blocks`: the dotted path
    # `blocks.blocks` repeats the segment, and the LAST one is the class.
    index = ModuleIndex.build(
        {
            "pkg/__init__.py": "",
            "pkg/blocks.py": "class blocks: ...\n",
            "use.py": "from pkg import blocks\n",
        }
    )
    assert _definition(index, "use", "blocks.blocks").key == "pkg.blocks::blocks"
    bare = _diagnostic(index, "use", "blocks", multifile.UNRESOLVED_IMPORT, "warning")
    assert "not a class" in bare["detail"]


def test_from_dot_import_binds_submodule() -> None:
    index = ModuleIndex.build(
        {
            "v/__init__.py": "",
            "v/leaf.py": "class Leaf: ...\n",
            "v/view.py": "from . import leaf\n",
        }
    )
    assert _definition(index, "v.view", "leaf.Leaf").key == "v.leaf::Leaf"


def test_deep_absolute_chain() -> None:
    index = ModuleIndex.build(
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/leaf.py": "class Leaf: ...\n",
            "use.py": "import pkg\n",
        }
    )
    assert _definition(index, "use", "pkg.sub.leaf.Leaf").key == "pkg.sub.leaf::Leaf"


def test_class_beneath_class_is_not_indexed() -> None:
    index = ModuleIndex.build({"n.py": "class Outer:\n    class Inner: ...\n"})
    assert set(index.modules["n"].classes) == {"Outer"}
    _diagnostic(index, "n", "Inner", multifile.UNRESOLVED_IMPORT)


def test_building_never_executes_sources() -> None:
    # Top-level code that would end the interpreter or raise if run: the index
    # must parse it and swallow nothing but SyntaxError.
    index = ModuleIndex.build(
        {
            "explode.py": (
                "import sys\nsys.exit(3)\n_ = 1 / 0\nraise RuntimeError('boom')\n"
                "class Survived: ...\n"
            )
        }
    )
    assert "Survived" in index.modules["explode"].classes
    assert index.modules["explode"].syntax_error is None


def test_determinism() -> None:
    sources = _fixture_sources()
    first = ModuleIndex.build(sources)
    second = ModuleIndex.build(sources)
    assert list(first.modules) == list(second.modules)
    probes = [
        ("models.net", "CoreBlock"),
        ("models.net", "HeadBlock"),
        ("models.net", "Encoder"),
        ("models.mixer", "Block"),
        ("models.enc", "blocks.Block"),
        ("models.blocks", "nn"),
        ("models.dynamic", "DynamicStage"),
    ]
    for from_module, expr in probes:
        one = first.resolve(from_module, expr)
        two = second.resolve(from_module, expr)
        assert (one.definition.key if one.definition else None) == (
            two.definition.key if two.definition else None
        )
        assert one.diagnostic == two.diagnostic
    for module in first.modules:
        assert (
            first.modules[module].dynamic_diagnostics
            == second.modules[module].dynamic_diagnostics
        )
