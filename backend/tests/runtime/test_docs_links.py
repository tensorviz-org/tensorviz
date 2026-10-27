"""Every link in README.md and docs/ must resolve.

A retired issues page was deleted while six documentation pages and two README
passages still pointed at it. Nothing caught that for two days, because nothing
read the docs for link validity — so the one place the project kept its honest
warnings answered 404 to every reader the docs sent there.

The implementation lives in docs/check_links.py so it is also runnable by hand
(`make docs-links`) and from a docs-only CI job; this is the wrapper that puts it
in the suite developers already run.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "docs"))

from check_links import check  # noqa: E402


def test_every_documentation_link_resolves():
    problems = check()
    assert not problems, "broken documentation links:\n  " + "\n  ".join(problems)


def test_an_explicit_html_anchor_in_markdown_counts_as_an_id(tmp_path):
    """Markdown may define an anchor as `<a id="...">`; GitHub renders and links it.

    `_ids_in` collected ONLY heading slugs for a .md file and never looked at
    `id=`, although this module's own docstring promises that a `#fragment`
    matches "an ``id=`` in that file". One research document uses 22 such anchors,
    so the gate reported 97 broken links that were all real, resolvable targets —
    noise that trains people to ignore the one check standing between a reader and
    a 404.
    """
    from check_links import _ids_in

    page = tmp_path / "page.md"
    page.write_text(
        '# Some Title\n\n<a id="s01"></a>\n\n## Next\n\n[jump](#s01)\n',
        encoding="utf-8",
    )
    ids = _ids_in(page)
    assert "s01" in ids, "an explicit anchor is a real target"
    assert {"some-title", "next"} <= ids, "heading slugs must keep working"
