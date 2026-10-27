"""Marks the fixture's ``custom/`` as a real package.

It is a sibling of ``models/`` rather than another file inside it because
``/api/ide/workspace/classes`` is pinned on the exact set of classes ``models/``
holds (``tests/api/test_ide_router.py``). The M6 fixtures are about a model that
imports a block from ANOTHER module anyway, so a second package is what the case
under test actually looks like.
"""
