"""A construction failure must say what to DO about THAT failure.

Every one of them used to end with "Pass constructor arguments via `overrides`.",
which is correct for exactly one cause. The multi-file cycle fixture is an
`ImportError` no argument can fix, and telling its user to pass one sends them off
to write a `# tensorviz: init=` line that cannot help — the refusal was accurate and
useless at the same time.

The reason codes are asserted rather than the prose: the extension picks its own
sentence from the code (`extension/src/worker.ts`), so the code is the contract and
the wording is free to improve.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from services.ingest.service import _construction_advice, construction_reason


class _Cfg:
    pass


def _missing_argument() -> TypeError:
    """The real exception, raised by Python, not a hand-written lookalike."""

    class Configured:
        def __init__(self, cfg):  # noqa: ANN001 - the point is the missing argument
            self.cfg = cfg

    try:
        Configured()  # type: ignore[call-arg]
    except TypeError as exc:
        return exc
    raise AssertionError("Configured() must fail")


@pytest.mark.parametrize(
    "exc, reason",
    [
        (_missing_argument(), "missing_argument"),
        (
            ImportError(
                "cannot import name 'CycleA' from partially initialized module "
                "'models.cycle_a' (most likely due to a circular import)"
            ),
            "circular_import",
        ),
        (ImportError("No module named 'timm'"), "import_failed"),
        (RecursionError("maximum recursion depth exceeded"), "recursive_construction"),
        (ValueError("width must be positive"), "construct_failed"),
    ],
)
def test_each_cause_gets_its_own_reason_code(exc: BaseException, reason: str) -> None:
    assert construction_reason(exc) == reason


def test_a_circular_import_is_not_told_to_pass_an_argument() -> None:
    """The specific wrong advice this exists to remove."""
    advice = _construction_advice(
        "CycleA",
        ImportError(
            "cannot import name 'CycleA' from partially initialized module "
            "'models.cycle_a' (most likely due to a circular import)"
        ),
    )
    assert "overrides" not in advice
    assert "No constructor argument can fix this." in advice
    assert "inside the function that uses it" in advice


def test_a_missing_argument_is_given_a_line_the_user_can_copy() -> None:
    """Naming the argument is the difference between a refusal and an instruction."""
    advice = _construction_advice("Configured", _missing_argument())
    assert "'cfg'" in advice
    assert "# tensorviz: init=Configured(cfg=)" in advice


def test_a_config_object_is_fixed_in_the_model_s_own_declaration() -> None:
    """``inside`` is what keeps the copyable line from being a second dead end.

    When the missing argument belongs to the CONFIG rather than to the model, the
    default sentence hands back `# tensorviz: init=ModelConfig(width=)`. That line is
    attributed to ModelConfig by ``declared_init`` and is never read when building
    Configured, so following the advice changes nothing — the failure the user sees
    is identical. The line has to be their existing declaration with the blank filled.
    """
    @dataclass
    class ModelConfig:
        width: int

    try:
        ModelConfig()  # type: ignore[call-arg]
    except TypeError as exc:
        advice = _construction_advice("ModelConfig", exc, inside=("Configured", "cfg"))
    else:
        raise AssertionError("ModelConfig() must fail")

    assert "ModelConfig needs 'width'" in advice
    assert "# tensorviz: init=Configured(cfg=ModelConfig(width=))" in advice


def test_recursive_construction_says_so_rather_than_blaming_arguments() -> None:
    advice = _construction_advice("CycleA", RecursionError("maximum recursion depth exceeded"))
    assert "constructs itself" in advice
    assert "overrides" not in advice


def test_an_unclassified_failure_points_to_the_profile_and_underlying_exception() -> None:
    """An unknown failure must not claim constructor overrides are its remedy."""
    advice = _construction_advice("Net", ValueError("width must be positive"))
    assert "profile" in advice and "underlying exception" in advice
    assert "overrides" not in advice
