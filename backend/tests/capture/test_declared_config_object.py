"""``# tensorviz: init=Net(cfg=ModelConfig())`` — building the config, not guessing it.

``init=`` handled scalars. The shape it did not handle is the one most real models
have: ``def __init__(self, cfg: ModelConfig)``. Such a user got a refusal that named
the argument and handed them ``# tensorviz: init=Configured(cfg=)`` to copy — a line
with a blank in it that no literal could fill. Writing the obvious thing in that blank
was refused as "not a literal value", and because a refused declaration yields no
overrides, the worker then failed with the SAME "missing 1 required positional
argument: 'cfg'" as before, advising the line already on screen. Measured on
``examples/tensorviz-demo/models/cfg.py`` before this existed: exit 3, both times.

So one constructor call is now a legal argument value, and the worker builds it out of
the module it has just imported. That is not a new security boundary — the next thing
that module does is construct the user's model out of the same namespace — but it is
deliberately not a general expression evaluator, and ``test_ide_init_comment`` pins
what stays refused.

Run as a subprocess, like ``test_ingest_worker``: the exit code is half the contract.
"""

from __future__ import annotations

import textwrap
from pathlib import Path


from tests.capture.test_ingest_worker import run_worker  # noqa: E402

DATACLASS_CONFIG = '''
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ModelConfig:
    width: int = 8
    hidden: int = 16


# tensorviz: init=Configured(cfg=ModelConfig())
# tensorviz: input=2,8
class Configured(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.width, cfg.hidden)
        self.out = nn.Linear(cfg.hidden, cfg.width)

    def forward(self, x):
        return self.out(self.proj(x))
'''


def _module(tmp_path: Path, source: str, name: str = "configured.py") -> str:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def test_a_model_that_takes_a_dataclass_config_is_built_and_traced(tmp_path: Path):
    """The whole point: exit 0 and a graph, where exit 3 was the only outcome.

    The observed shapes are asserted too, because a construction that succeeded with
    the WRONG config object would still exit 0 — ``8 -> 16 -> 8`` is what
    ``ModelConfig()``'s own defaults say, and nothing else produces it.
    """
    code, envelope, stderr = run_worker(
        "--file", _module(tmp_path, DATACLASS_CONFIG), "--cls", "Configured", "--json"
    )
    assert code == 0, (envelope["error"], stderr)
    shapes = {
        node["id"]: (node["data"]["parameters"] or {}).get("_out_shape")
        for node in envelope["result"]["graph"]["nodes"]
    }
    assert shapes["proj"] == "2x16"
    assert shapes["out"] == "2x8"


def test_the_config_may_be_a_plain_class_or_a_named_tuple(tmp_path: Path):
    """Nothing here knows what a dataclass is; it calls a name the module binds.

    Worth pinning because the obvious wrong implementation — reaching for
    ``dataclasses.fields`` to synthesise defaults — passes the test above and fails
    every config that is a ``NamedTuple``, a ``pydantic`` model or an ordinary class,
    which between them are most of them.
    """
    source = DATACLASS_CONFIG.replace(
        "@dataclass\nclass ModelConfig:\n    width: int = 8\n    hidden: int = 16",
        "class ModelConfig:\n"
        "    def __init__(self):\n"
        "        self.width, self.hidden = 8, 16",
    ).replace("from dataclasses import dataclass\n", "")
    code, envelope, stderr = run_worker(
        "--file", _module(tmp_path, source), "--cls", "Configured", "--json"
    )
    assert code == 0, (envelope["error"], stderr)


def test_a_config_imported_from_another_file_is_resolved_where_it_is_bound(tmp_path: Path):
    """``from .config import ModelConfig`` is where configs actually live.

    The name is looked up on the MODEL's module, which is why an import at the top of
    the model's file is enough and no second file has to be found or parsed.
    """
    (tmp_path / "conf.py").write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass ModelConfig:\n    width: int = 8\n    hidden: int = 16\n",
        encoding="utf-8",
    )
    source = DATACLASS_CONFIG.replace(
        "from dataclasses import dataclass", "from conf import ModelConfig"
    ).replace(
        "@dataclass\nclass ModelConfig:\n    width: int = 8\n    hidden: int = 16\n\n\n", ""
    )
    code, envelope, stderr = run_worker(
        "--file", _module(tmp_path, source), "--cls", "Configured", "--json"
    )
    assert code == 0, (envelope["error"], stderr)


def test_a_config_that_itself_needs_an_argument_says_which_and_where_to_put_it(
    tmp_path: Path,
):
    """The failure this feature creates, and the one it must not fumble.

    "constructing Configured failed: missing 1 required positional argument: 'width'"
    is true and useless: the user's declaration is nearly right, and the argument
    belongs to the CONFIG. The advice therefore names ``ModelConfig``, and the line it
    hands back is the user's own declaration with the blank filled in — not a new
    ``# tensorviz: init=ModelConfig(width=)``, which ``declared_init`` would attribute
    to ModelConfig and never read for Configured.
    """
    source = DATACLASS_CONFIG.replace("width: int = 8", "width: int")
    code, envelope, _stderr = run_worker(
        "--file", _module(tmp_path, source), "--cls", "Configured", "--json"
    )
    assert code == 3
    assert envelope["result"] is None
    message = envelope["error"]["message"]
    assert "ModelConfig needs 'width'" in message
    assert "# tensorviz: init=Configured(cfg=ModelConfig(width=))" in message


def test_the_line_that_advice_hands_back_actually_works(tmp_path: Path):
    """Advice that does not round-trip is a second dead end, which is what this
    feature exists to remove. Filling the blank in the line above must construct."""
    source = DATACLASS_CONFIG.replace("width: int = 8", "width: int").replace(
        "init=Configured(cfg=ModelConfig())", "init=Configured(cfg=ModelConfig(width=8))"
    )
    code, envelope, stderr = run_worker(
        "--file", _module(tmp_path, source), "--cls", "Configured", "--json"
    )
    assert code == 0, (envelope["error"], stderr)


def test_a_config_the_module_does_not_bind_is_named_rather_than_guessed_at(
    tmp_path: Path,
):
    """A typo, or a class the user forgot to import. Falling back to a no-argument
    construction would report the ORIGINAL missing-argument error and hide the typo."""
    source = DATACLASS_CONFIG.replace("cfg=ModelConfig()", "cfg=MdoelConfig()")
    code, envelope, _stderr = run_worker(
        "--file", _module(tmp_path, source), "--cls", "Configured", "--json"
    )
    assert code == 3
    message = envelope["error"]["message"]
    assert "MdoelConfig" in message
    assert "not defined or imported" in message


def test_a_declaration_naming_another_class_still_leaves_this_one_unbuilt(
    tmp_path: Path,
):
    """The config call must not widen the attribution rule.

    A file with two models carries two lines; reading the neighbour's would build a
    model the user did not ask for, and a config object makes that failure quieter
    still because the wrong config often constructs.
    """
    source = DATACLASS_CONFIG.replace(
        "# tensorviz: init=Configured(cfg=ModelConfig())",
        "# tensorviz: init=Neighbour(cfg=ModelConfig())",
    )
    code, envelope, _stderr = run_worker(
        "--file", _module(tmp_path, source), "--cls", "Configured", "--json"
    )
    assert code == 3
    assert envelope["result"] is None
    diagnostic = envelope["error"]["diagnostic"]
    assert diagnostic["phase"] == "binding"
    assert diagnostic["code"] == "missing_argument"
    assert diagnostic["field"] == ["init"] and diagnostic["names"] == ["cfg"]
