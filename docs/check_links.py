#!/usr/bin/env python3
"""Docs must resolve, and the names they cite must exist in the tree.

Run from anywhere; no third-party dependencies:

    python3 docs/check_links.py

Checks root guides (including ``SECURITY.md`` and ``CODE_OF_CONDUCT.md``),
coding-assistant context, example READMEs, ``packages/*/README.md``,
``plugins/**/README.md``, extension notices and walkthroughs, and every
HTML/Markdown file under ``docs/``:

  * relative targets exist on disk;
  * same-page ``#fragment`` targets match an ``id=`` or heading slug;
  * cross-page fragments match an id in that page;
  * a bare ``See X.`` with no link fails;
  * ``npm run`` / ``npm test`` / ``npm start`` scripts exist in the named package;
  * ``make <target>`` exists in the Makefile;
  * backticked or written ``tensorviz.*`` ids exist as settings, commands, views,
    or the extension id;
  * backticked MCP tool names exist as ``registerTool`` / ``registerAppTool`` calls;
  * ``docs/reference.md`` matches ``docs/generate_reference.py``;
  * ``docs/local-setup.md`` names the extension and plugin versions from the manifests.

External ``http(s)://`` and ``mailto:`` links are not fetched.

Exit 0 when everything resolves, 1 with a report otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

from generate_reference import known_tensorviz_ids, render, tool_names  # noqa: E402

HREF = re.compile(r"""\bhref\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
SRC = re.compile(r"""\bsrc\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_ID = re.compile(r"""\bid\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
MD_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
FENCE = re.compile(r"```.*?```", re.DOTALL)
BARE_SEE = re.compile(r"See\s+(?!\[)(?!https?://)([^\n\[\]]{2,160}?)\.")
NPM_RUN = re.compile(r"npm run ([A-Za-z0-9:._-]+)(?:\s+--prefix\s+([A-Za-z0-9_./-]+))?")
NPM_LIFE = re.compile(r"npm (test|start)(?:\s+--prefix\s+([A-Za-z0-9_./-]+))?")
MAKE_TICK = re.compile(r"^make\s+([A-Za-z0-9_.-]+)\s*$")
FENCED = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
TOOLISH = re.compile(
    r"^(?:open_graph(?:_[a-z0-9_]+)?|get_(?:graph|run|selection|proposal|execution)(?:_[a-z0-9_]+)?|"
    r"list_[a-z0-9_]+|inspect_[a-z0-9_]+|configure_[a-z0-9_]+|suggest_[a-z0-9_]+|"
    r"validate_model|compare_runs|cancel_[a-z0-9_]+|propose_[a-z0-9_]+|focus_view|"
    r"run_proposal_[a-z0-9_]+|read_model_[a-z0-9_]+|graph_panel_[a-z0-9_]+|"
    r"detach_graph_[a-z0-9_]+|close_graph_[a-z0-9_]+)$"
)
TV_ID = re.compile(r"(?<![\w./-])tensorviz\.([A-Za-z][A-Za-z0-9.]*)")
TICK = re.compile(r"`([^`\n]+)`")
FILE_SUFFIXES = {"toml", "json", "md", "py", "html", "svg", "txt", "yml", "yaml"}
SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:", "javascript:", "//")


def _ids_in(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    if path.suffix == ".md":
        slugs = set()
        for heading in MD_HEADING.findall(text):
            slug = re.sub(r"[^a-z0-9\s-]", "", heading.strip().lower())
            slugs.add(re.sub(r"\s+", "-", slug))
        # Markdown may also define an anchor explicitly, as `<a id="s01"></a>`.
        # GitHub renders that and links to it, and the docstring above already
        # promises a fragment matches "an ``id=`` in that file" — but only the
        # heading slugs were ever collected here, so 22 real anchors in one
        # research document reported as 97 broken links. A gate whose failures
        # are noise is a gate people learn to skip.
        return slugs | set(HTML_ID.findall(text))
    return set(HTML_ID.findall(text))


def _links_in(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".md":
        return MD_LINK.findall(text)
    return HREF.findall(text) + SRC.findall(text)


def _sources() -> list[Path]:
    files = [REPO_ROOT / name for name in (
        "README.md", "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md",
        "AGENTS.md", "CLAUDE.md", ".github/copilot-instructions.md",
    )]
    files += sorted(DOCS.rglob("*.html"))
    files += sorted(DOCS.rglob("*.md"))
    files += sorted((REPO_ROOT / "examples").rglob("*.md"))
    files += sorted((REPO_ROOT / "extension").glob("*.md"))
    files += sorted((REPO_ROOT / "extension" / "walkthrough").glob("*.md"))
    files += sorted((REPO_ROOT / "packages").glob("*/README.md"))
    files += sorted((REPO_ROOT / "plugins").rglob("README.md"))
    files += sorted((REPO_ROOT / "plugins").rglob("SKILL.md"))
    # rglob walks the same README twice when both globs match; keep one path.
    seen = set()
    unique = []
    for f in files:
        if f.is_file() and f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def _unfenced(text: str) -> str:
    return FENCE.sub("", text)


def _link_problems() -> list[str]:
    problems: list[str] = []
    ids_cache: dict[Path, set[str]] = {}

    for source in _sources():
        rel_source = source.relative_to(REPO_ROOT)
        for raw in _links_in(source):
            link = raw.strip()
            if not link or link.startswith(SKIP_PREFIXES):
                continue

            target, _, fragment = link.partition("#")

            if not target:  # same-page anchor
                if fragment and fragment not in ids_cache.setdefault(source, _ids_in(source)):
                    problems.append(f"{rel_source}: anchor #{fragment} matches no id in this file")
                continue

            resolved = (source.parent / target).resolve()
            if not resolved.exists():
                problems.append(f"{rel_source}: -> {target} does not exist")
                continue

            if fragment and resolved.is_file():
                ids = ids_cache.setdefault(resolved, _ids_in(resolved))
                if ids and fragment not in ids:
                    problems.append(
                        f"{rel_source}: -> {target}#{fragment} — that file has no id '{fragment}'"
                    )

    return problems


def _see_problems() -> list[str]:
    problems = []
    for source in _sources():
        text = _unfenced(source.read_text(encoding="utf-8", errors="replace"))
        rel = source.relative_to(REPO_ROOT)
        for match in BARE_SEE.finditer(text):
            label = " ".join(match.group(1).split())
            problems.append(f"{rel}: bare 'See {label}.' has no link")
    return problems


def _package_scripts(prefix: str | None) -> set[str] | None:
    if prefix in (None, ".", "./"):
        path = REPO_ROOT / "package.json"
    else:
        cleaned = prefix.rstrip("/")
        path = REPO_ROOT / cleaned / "package.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("scripts", {}))


def _make_targets() -> set[str]:
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"^([A-Za-z0-9_.-]+)\s*:", text, re.MULTILINE))


def _symbol_problems() -> list[str]:
    problems = []
    scripts_cache: dict[str | None, set[str] | None] = {}
    targets = _make_targets()
    ids = known_tensorviz_ids()
    tools = tool_names()
    for source in _sources():
        rel = source.relative_to(REPO_ROOT)
        if rel.as_posix() == "docs/reference.md":
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        for script, prefix in NPM_RUN.findall(text):
            key = prefix or None
            if key not in scripts_cache:
                scripts_cache[key] = _package_scripts(key)
            found = scripts_cache[key]
            where = prefix or "the repository root"
            if found is None:
                problems.append(f"{rel}: npm prefix '{prefix}' has no package.json")
            elif script not in found:
                problems.append(f"{rel}: npm script '{script}' is not in {where}")
        for script, prefix in NPM_LIFE.findall(text):
            key = prefix or None
            if key not in scripts_cache:
                scripts_cache[key] = _package_scripts(key)
            found = scripts_cache[key]
            where = prefix or "the repository root"
            if found is None:
                problems.append(f"{rel}: npm prefix '{prefix}' has no package.json")
            elif script not in found:
                problems.append(f"{rel}: npm script '{script}' is not in {where}")
        make_calls = [MAKE_TICK.match(tick.strip()) for tick in TICK.findall(text)]
        for block in FENCED.findall(text):
            for line in block.splitlines():
                make_calls.append(MAKE_TICK.match(line.strip()))
        for match in make_calls:
            if match and match.group(1) not in targets:
                problems.append(f"{rel}: make target '{match.group(1)}' is not in the Makefile")
        for suffix in TV_ID.findall(text):
            if suffix.split(".")[0] in FILE_SUFFIXES or suffix in {"dev", "org"}:
                continue
            if not any(char.isupper() for char in suffix):
                continue
            token = f"tensorviz.{suffix}"
            if not _known_tensorviz(token, ids):
                problems.append(f"{rel}: '{token}' is not a setting, command, view, or extension id")
        for tick in TICK.findall(text):
            name = tick.strip()
            if TOOLISH.fullmatch(name) and name not in tools:
                problems.append(f"{rel}: MCP tool '{name}' is not registered")
    return problems


def _known_tensorviz(token: str, ids: set[str]) -> bool:
    if token in ids:
        return True
    parent = token
    while "." in parent:
        parent = parent.rsplit(".", 1)[0]
        if parent in ids and parent != "tensorviz":
            return True
    return False


def _version_problems() -> list[str]:
    setup = (DOCS / "local-setup.md").read_text(encoding="utf-8")
    extension = json.loads((REPO_ROOT / "extension" / "package.json").read_text(encoding="utf-8"))["version"]
    plugin = json.loads((REPO_ROOT / "plugin" / "package.json").read_text(encoding="utf-8"))["version"]
    problems = []
    if extension not in setup:
        problems.append(f"docs/local-setup.md: versions table is missing extension {extension}")
    if plugin not in setup:
        problems.append(f"docs/local-setup.md: versions table is missing plugin {plugin}")
    for needle in ("Node.js 22", "Python 3.12", "3.11", "3.10"):
        if needle not in setup:
            problems.append(f"docs/local-setup.md: versions table is missing {needle}")
    return problems


def _reference_problems() -> list[str]:
    path = DOCS / "reference.md"
    if not path.is_file():
        return ["docs/reference.md is missing. Run: python3 docs/generate_reference.py"]
    expected = render()
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        return ["docs/reference.md is stale. Run: python3 docs/generate_reference.py"]
    return []


def check() -> list[str]:
    return (
        _link_problems()
        + _see_problems()
        + _symbol_problems()
        + _version_problems()
        + _reference_problems()
    )


def main() -> int:
    problems = check()
    checked = len(_sources())
    if problems:
        print(f"{len(problems)} documentation problem(s) across {checked} file(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"All links resolve across {checked} documentation file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
