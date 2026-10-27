"""The project identity's validator, and proof it did not weaken the class one.

The risk this guards: `sameTargetIdentity` is an all-six-fields equality and is the
staleness refusal behind `revealNode`, `focusNode` and every Ask. Introducing a
project target by making three of those six optional would silently turn that check
partial. So the project identity is a separate type with its own validator, and these
tests assert both halves of that: the new one is strict, and the old one is untouched.
"""


import pytest

# abspath, not a bare join: `services/ide/local_browser_protocol.py` locates the
# shared fixture corpus with `Path(__file__).parents[3]`, and an unnormalised
# `.../tests/..` entry on sys.path leaves a `..` segment inside `__file__` that
# `parents` does not collapse — so the corpus is then looked for under tests/.

from services.ide import local_browser_protocol as protocol  # noqa: E402

DIGEST = "a" * 64


def identity(**overrides):
    base = {
        "workspaceFolder": "/w",
        "projectRoot": "packages/alpha",
        "experimentId": "exp:1234",
        "indexDigest": DIGEST,
        "indexRevision": 3,
    }
    base.update(overrides)
    return base


def test_a_well_formed_project_identity_is_accepted():
    assert protocol.project_identity_is_valid(identity())


def test_an_unfocused_experiment_is_a_real_state():
    """The canvas can legitimately show a project with nothing focused."""
    assert protocol.project_identity_is_valid(identity(experimentId=""))


def test_the_workspace_root_itself_has_an_empty_project_root():
    assert protocol.project_identity_is_valid(identity(projectRoot=""))


@pytest.mark.parametrize(
    "field",
    ["workspaceFolder", "projectRoot", "experimentId", "indexDigest", "indexRevision"],
)
def test_every_field_is_required(field):
    """Absent is malformed, never a default — the rule the class identity follows."""
    payload = identity()
    del payload[field]
    assert not protocol.project_identity_is_valid(payload)


@pytest.mark.parametrize(
    "field",
    ["workspaceFolder", "projectRoot", "experimentId", "indexDigest", "indexRevision"],
)
def test_explicit_null_is_refused_like_every_other_optional_here(field):
    assert not protocol.project_identity_is_valid(identity(**{field: None}))


def test_an_unknown_key_is_refused():
    assert not protocol.project_identity_is_valid(identity(extra="x"))


def test_the_index_digest_must_look_like_one():
    for bad in ("", "nothex", "A" * 64, "a" * 63, "a" * 65):
        assert not protocol.project_identity_is_valid(identity(indexDigest=bad)), bad


def test_a_negative_or_fractional_revision_is_refused():
    assert not protocol.project_identity_is_valid(identity(indexRevision=-1))
    assert not protocol.project_identity_is_valid(identity(indexRevision=1.5))


def test_a_boolean_is_not_an_integer_revision():
    """`True == 1` in Python; a validator that forgets this accepts nonsense."""
    assert not protocol.project_identity_is_valid(identity(indexRevision=True))


def test_a_workspace_folder_alone_is_not_a_project_identity():
    assert not protocol.project_identity_is_valid({"workspaceFolder": "/w"})


def test_non_dict_values_are_refused():
    for bad in (None, [], "", 3, True):
        assert not protocol.project_identity_is_valid(bad)


# ── the class identity must be exactly as it was ────────────────────────────────


def class_target(**overrides):
    base = {
        "workspaceFolder": "/w",
        "file": "models/net.py",
        "cls": "Net",
        "documentVersion": 2,
        "sourceDigest": DIGEST,
        "backendRevision": 4,
    }
    base.update(overrides)
    return base


def test_the_class_identity_still_requires_all_six_fields():
    """If a project target had been bolted onto this type, these would now pass."""
    assert protocol._target_is_valid(class_target())
    for field in ("file", "cls", "documentVersion", "sourceDigest", "backendRevision"):
        payload = class_target()
        del payload[field]
        assert not protocol._target_is_valid(payload), f"{field} became optional"


def test_a_project_identity_is_not_accepted_where_a_class_target_belongs():
    assert not protocol._target_is_valid(identity())


def test_a_class_target_is_not_accepted_as_a_project_identity():
    assert not protocol.project_identity_is_valid(class_target())
