"""Whether a node on the project canvas can be validated, and what it needs.

Step 5.9. The refusal is the feature, not the error path: a Validate button that
can be pressed and then explains itself afterwards has already wasted the press,
and "The model could not be constructed. Supply its constructor arguments with
# tensorviz: init= and retry" is a sentence that arrives too late to be help.

Read statically, from what the index already knows. Nothing here imports, builds
or runs anything — the whole project plane is forbidden from touching a file, and
a verdict that needed a running interpreter could not be shown on a card.

The three answers, and why there are exactly three:

``ready``       nothing is missing; pressing Validate runs something.
``needs``       something is missing that the USER can supply, and the card says
                which declaration would supply it. This is the interesting state
                and the reason the module exists.
``unsupported`` this kind of node is not a thing you validate at all — a file, a
                dataset, a configuration. Silence would be read as "broken"; the
                honest answer is that the question does not apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

#: Kinds a forward pass is a meaningful question about. Everything else gets
#: `unsupported` rather than a button that would do nothing.
RUNNABLE_KINDS = frozenset({"model", "module", "layer"})


@dataclass(frozen=True)
class Verdict:
    """What the card shows where the Validate button goes."""

    state: str
    #: One sentence, in the user's vocabulary. Empty when `ready`.
    detail: str = ""
    #: The declaration that would resolve it, ready to be written by 4.2's
    #: confirm path. Empty when there is nothing to offer.
    remedy: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {"state": self.state, "detail": self.detail, "remedy": self.remedy}


def _quote(value: Any) -> str:
    return f'"{value}"' if isinstance(value, str) else repr(value)


def _call(label: str, kwargs: Dict[str, Any]) -> str:
    inner = ", ".join(f"{name}={_quote(value)}" for name, value in kwargs.items())
    return f'tensorviz.init("{label}"{", " if inner else ""}{inner})'


def verdict_for(
    kind: str,
    label: str,
    params: Dict[str, Any],
    *,
    configurations: Optional[Sequence[Dict[str, Any]]] = None,
) -> Verdict:
    """Can this node be validated, and if not, what would make it so.

    ``configurations`` are the configuration nodes of the same experiment, each a
    ``{"label": ..., "settings": {...}}``. They are what turns a dead end into an
    offer: the index already read `GPTConfig`'s seven defaults for its card, so a
    module asking for a `config` can be told exactly which one to name — without
    evaluating a line of the user's code to find out.
    """
    if kind not in RUNNABLE_KINDS:
        return Verdict("unsupported", f"A {kind} has no forward pass to check.")

    declared: Dict[str, Any] = dict(params.get("_init_args") or {})
    required: List[str] = [str(name) for name in (params.get("_ctor_required") or [])]
    missing = [name for name in required if name not in declared]

    if missing:
        return _needs_construction(label, missing, configurations or [])

    if not _has_input(params):
        return Verdict(
            "needs",
            f"No example input. {label} builds, but nothing here says what shape to "
            f"run through it — no layer states a width and no input is declared.",
            # `?`, not a plausible-looking shape. Offering `1x3x224x224` here would
            # repeat the defect this module exists to end: a guess printed with the
            # authority of a fact, and an image shape offered to a language model.
            f'tensorviz.input("{label}", shape="?")',
        )

    return Verdict("ready")


def _needs_construction(
    label: str, missing: List[str], configurations: Sequence[Dict[str, Any]]
) -> Verdict:
    named = ", ".join(f"`{name}`" for name in missing)
    plural = "s" if len(missing) > 1 else ""

    # The case that makes this worth building. Four of nanoGPT's five modules take
    # a `config`, and the index has already read that config's defaults — so the
    # answer is a name the user recognises, not a demand for values.
    if len(missing) == 1 and missing[0] in {"config", "cfg", "args", "opts", "hparams"}:
        candidates = [str(entry.get("label")) for entry in configurations if entry.get("label")]
        if len(candidates) == 1:
            only = candidates[0]
            return Verdict(
                "needs",
                f"{label} needs {named}, and this project declares one: {only}.",
                _call(label, {missing[0]: only}),
            )
        if candidates:
            listed = ", ".join(candidates)
            return Verdict(
                "needs",
                f"{label} needs {named}. This project declares {len(candidates)}: {listed}.",
                _call(label, {missing[0]: candidates[0]}),
            )

    return Verdict(
        "needs",
        f"{label} cannot be built: its __init__ requires {named}, and nothing here "
        f"says what value{plural} to use.",
        _call(label, {name: "?" for name in missing}),
    )


def _has_input(params: Dict[str, Any]) -> bool:
    """An example input exists when it was declared, or when a layer states a width.

    The second half is what keeps the common case quiet: a `Conv2d(3, ...)` states
    its own input channels, and asking the user to declare what the code already
    says would be a refusal nobody deserves.
    """
    if params.get("_input_declared"):
        return True
    # The comment form, which is what a user writes and what shipped first. Read as
    # a yes/no rather than as a shape: the expression form needs torch to evaluate,
    # and "is an input declared" does not.
    if params.get("_input_comment"):
        return True
    return bool(params.get("_in_shape") or params.get("_input_shape"))
