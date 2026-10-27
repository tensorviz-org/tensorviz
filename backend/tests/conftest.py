"""Shared pytest configuration.

Endpoints now depend on ``get_current_user``, so the suite must run with auth
disabled by default (returns the fixed dev principal). ``setdefault`` means any
explicit ``AUTH_DISABLED`` already in the environment wins, and tests that toggle
it per-test via ``monkeypatch.setenv`` (e.g. test_auth.py) still override this.

``backend`` goes on ``sys.path`` here before any domain is collected. Test
modules import shared helpers through ``tests.<domain>`` and need no per-file
bootstrap. This also keeps individual-domain runs independent of collection order.
"""

import os
import sys

os.environ.setdefault("AUTH_DISABLED", "true")

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "roundtrip: code -> graph -> code stability. Parses PyTorch source, "
        "regenerates it from the parsed graph, and parses the result again, "
        "asserting the two graphs describe the same model. Run on its own with "
        '-m roundtrip; see docs/tensorviz-plan.md §9.',
    )
