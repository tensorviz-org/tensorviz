#!/usr/bin/env python3
"""Write docs/reference.md from the extension manifest and MCP registrations.

Run from anywhere:

    python3 docs/generate_reference.py

``docs/check_links.py`` compares the committed page with ``render()`` and fails
when they differ. Edit the sources, then re-run this script.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXT_PKG = ROOT / "extension" / "package.json"
PLUGIN_PKG = ROOT / "plugin" / "package.json"
CONTRACTS = ROOT / "backend" / "services" / "validation" / "contracts.py"

TOOL_RE = re.compile(
    r"""register(?:App)?Tool\(\s*(?:[A-Za-z0-9_$.]+\s*,\s*)?['"]([a-z][a-z0-9_]*)['"]"""
)
DESC_RE = re.compile(r"""description:\s*['"]((?:\\.|[^'"])*)['"]""")
SCAN_ROOTS = (
    ROOT / "extension" / "src",
    ROOT / "packages" / "graph-host" / "src",
    ROOT / "plugin" / "src",
)

# Portable graph-file format id (canvas/src/services/graphFile.ts), not a setting.
FORMAT_ID = "tensorviz.graph"

SOURCE_DIRECTIVES = """
## Source directives

Comments in model source. A saved `tensorviz.toml` profile supplies an explicit
recipe and is used first. These comments apply when that recipe does not.

| Directive | Where it is read | What it does |
| --- | --- | --- |
| `# tensorviz: input=` | `backend/services/input_comment.py` (`comment_for_class`), used by `backend/services/ingest/service.py` and `backend/utils/code_parser.py` | Example forward input for one class. A comment inside a class body belongs to that class. A comment that sits before a class belongs to the next top-level class, and to no other class. A comment with no following class is the file-wide default for classes that have no comment of their own. The comma form `2,4` is a shape. `2,8;2,8` is two arguments; the torch-free reader takes the first argument's comma shape. The worker also accepts an expression such as `torch.randint(0, 50257, (1, 16))` through `parse_input_spec`. |
| `# tensorviz: init=` | `backend/utils/code_parser.py`, `backend/services/ingest/service.py`, `extension/src/initComment.ts` | How to construct the class when it cannot be called with no arguments. One call, keyword arguments only, literal values. A config object is one level deep: `cfg=ModelConfig(width=8)`. Place it on or above the class line. |
| `# tensorviz: out_shape=` | `backend/services/ingest/introspect.py` | Declared output of a class the trace did not measure. It must sit in that class body. `(2, 16)` is a numeric shape. `(B, D)` is recorded and is not turned into numbers. |

`input=` is used when the opening has no explicit execution recipe, tensor
inputs, or example input (`_input_from_comment` in `backend/services/ingest/service.py`).
During graph setup, a saved `tensorviz.toml` profile never reaches the resolver,
and one complete in-file example call (constructor and inputs, such as
`model(torch.ones(...))`) is applied before the comment
(`backend/services/ide/graph_setup_resolution.py`). An explicit `input_shape`
still wins over the comment.
""".strip()


def _package() -> dict:
    return json.loads(EXT_PKG.read_text(encoding="utf-8"))


def _plugin_version() -> str:
    return json.loads(PLUGIN_PKG.read_text(encoding="utf-8"))["version"]


def setting_keys() -> list[str]:
    props = _package()["contributes"]["configuration"]["properties"]
    return sorted(props)


def command_rows() -> list[tuple[str, str, str]]:
    rows = []
    for item in _package()["contributes"]["commands"]:
        rows.append((item["command"], item.get("title", ""), item.get("category", "")))
    return rows


def view_ids() -> list[str]:
    ids = []
    for views in _package()["contributes"]["views"].values():
        for view in views:
            ids.append(view["id"])
    return ids


def walkthrough_ids() -> list[str]:
    return [item["id"] for item in _package()["contributes"].get("walkthroughs", [])]


def known_tensorviz_ids() -> set[str]:
    pkg = _package()
    ids = {f"{pkg['publisher']}.{pkg['name']}", FORMAT_ID}
    ids.update(setting_keys())
    ids.update(command for command, _, _ in command_rows())
    ids.update(view_ids())
    ids.update(walkthrough_ids())
    # Context keys such as tensorviz.canvasVisible are set in extension source,
    # not only in contributes.configuration.
    for path in (ROOT / "extension" / "src").rglob("*.ts"):
        if "/test/" in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        ids.update(re.findall(r"""['"](tensorviz\.[A-Za-z][A-Za-z0-9.]*)['"]""", text))
    return ids


def _audience(path: Path, index: int, text: str) -> str:
    before = text.rfind("export function", 0, index)
    signature = text[before:before + 90] if before != -1 else ""
    rel = path.as_posix()
    if "registerCanvasCodeTools" in signature:
        return "canvas agent"
    if rel.endswith("plugin/src/graphPanel.ts"):
        return "plugin panel"
    if "/plugin/src/" in rel:
        return "plugin"
    return "extension"


def tool_records() -> list[dict]:
    found: dict[str, dict] = {}
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.ts")):
            if "/test/" in path.as_posix() or path.name.endswith(".test.ts"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in TOOL_RE.finditer(text):
                name = match.group(1)
                window = text[match.end():match.end() + 700]
                desc_match = DESC_RE.search(window)
                description = ""
                if desc_match:
                    description = bytes(desc_match.group(1), "utf-8").decode("unicode_escape")
                    description = re.sub(r"\s+", " ", description).strip()
                record = found.setdefault(name, {"name": name, "audiences": [], "files": [], "description": ""})
                audience = _audience(path, match.start(), text)
                if audience not in record["audiences"]:
                    record["audiences"].append(audience)
                rel = str(path.relative_to(ROOT))
                if rel not in record["files"]:
                    record["files"].append(rel)
                if description and not record["description"]:
                    record["description"] = description
    return [found[name] for name in sorted(found)]


def tool_names() -> set[str]:
    return {record["name"] for record in tool_records()}


def _cell(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text or "—"


def _default(value) -> str:
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, separators=(",", ":")) + "`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, str):
        return f"`{value}`" if value else "`\"\"`"
    return f"`{value}`"


def _settings_table() -> str:
    props = _package()["contributes"]["configuration"]["properties"]
    lines = [
        "| Setting | Type | Default | Scope | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in sorted(props):
        spec = props[key]
        note = spec.get("markdownDescription") or spec.get("description") or ""
        if spec.get("deprecationMessage"):
            note = (note + " Manifest deprecation: " + spec["deprecationMessage"]).strip()
        if spec.get("enum"):
            note = "enum: " + ", ".join(f"`{item}`" for item in spec["enum"]) + ". " + note
        lines.append(
            f"| `{key}` | {spec.get('type', '—')} | {_default(spec.get('default', ''))} | {spec.get('scope', '—')} | {_cell(note)} |"
        )
    return "\n".join(lines)


def _commands_table() -> str:
    lines = [
        "| Command | Title | Category |",
        "| --- | --- | --- |",
    ]
    seen = set()
    for command, title, category in command_rows():
        if command in seen:
            continue
        seen.add(command)
        lines.append(f"| `{command}` | {_cell(title, 80)} | {_cell(category, 40)} |")
    return "\n".join(lines)


def _tools_table() -> str:
    lines = [
        "| Tool | Registered for | Source |",
        "| --- | --- | --- |",
    ]
    for record in tool_records():
        lines.append(
            f"| `{record['name']}` | {', '.join(record['audiences'])} | {', '.join(f'`{item}`' for item in record['files'])} |"
        )
    return "\n".join(lines)


def _assert_sources() -> None:
    contracts = CONTRACTS.read_text(encoding="utf-8")
    required = [
        "class ValidationConfig",
        "class ModelProfile",
        "class ScriptProfile",
        "class Limits",
        "timeout_ms: Annotated[StrictInt, Field(ge=100, le=86400000)] = 60000",
        "class ProfileBase",
    ]
    missing = [item for item in required if item not in contracts]
    if missing:
        raise SystemExit("tensorviz.toml keys drifted from backend/services/validation/contracts.py: " + ", ".join(missing))
    parser = (ROOT / "backend" / "utils" / "code_parser.py").read_text(encoding="utf-8")
    ingest = (ROOT / "backend" / "services" / "ingest" / "service.py").read_text(encoding="utf-8")
    introspect = (ROOT / "backend" / "services" / "ingest" / "introspect.py").read_text(encoding="utf-8")
    if "tensorviz:\\s*input\\s*=" not in parser or "tensorviz:\\s*init\\s*=" not in parser:
        raise SystemExit("input=/init= directives drifted from backend/utils/code_parser.py")
    if "tensorviz:\\s*input\\s*=" not in ingest:
        raise SystemExit("input= directive drifted from backend/services/ingest/service.py")
    if "tensorviz:\\s*out_shape\\s*=" not in introspect:
        raise SystemExit("out_shape= directive drifted from backend/services/ingest/introspect.py")
    if len(setting_keys()) != 15:
        raise SystemExit(f"expected 15 extension settings, found {len(setting_keys())}")


def render() -> str:
    _assert_sources()
    pkg = _package()
    version = pkg["version"]
    plugin_version = _plugin_version()
    engine = pkg["engines"]["vscode"]
    body = f"""# Reference

Generated from `extension/package.json`, MCP `registerTool` / `registerAppTool` calls, and the validation contracts. Do not edit this file by hand. Run `python3 docs/generate_reference.py`.

Product versions in this tree: extension `{version}`, plugin host `{plugin_version}`, VS Code engine `{engine}`. Development targets and Python minima: [versions](local-setup.md#versions).

## Contents

- [Extension settings](#extension-settings)
- [Commands](#commands)
- [MCP tools](#mcp-tools)
- [tensorviz.toml](#tensorviztoml)
- [Source directives](#source-directives)

## Extension settings

Fifteen keys under `contributes.configuration.properties`. `tensorviz.graphSurface` and `tensorviz.traceOnSave` are deprecated; the descriptions in the manifest say what they do now.

{_settings_table()}

`tensorviz.mcpPort` allows 0 through 65535. Zero chooses an available port. `tensorviz.codeFocus.opacity` allows 10 through 60.

`tensorviz.traceOnSave` stays in the manifest so an old value still loads. `browserTraceSettings` in `extension/src/traceSettings.ts` always returns `onSave: false`, so a true value does not schedule a capture. Saving refreshes the source projection. Opening a model captures it. **Run model** reviews another capture. `scheduleTrace` in `extension/src/localBrowserSync.ts` still exists, and it returns immediately because that reader never opts in.

## Commands

Palette titles use the category `tensorViz` plus the title, for example **tensorViz: Open as Graph**. Two commands can share a title and differ by id.

{_commands_table()}

`tensorviz.openProjectInBrowser` is hidden from the command palette (`when: false`). `tensorviz.graphDiffWorking` and `tensorviz.graphDiffStaged` are shown from Source Control, not the palette.

## MCP tools

Names below are every `registerTool` and `registerAppTool` registration under `extension/src`, `packages/graph-host/src`, and `plugin/src`, excluding tests. **canvas agent** tools are registered only by `registerCanvasCodeTools` for the in-canvas agent. **plugin panel** tools with app visibility are the panel transport; the model-workflow skill tells the agent not to call those transport tools. Behavior and arguments: [MCP tools](mcp-tools.md).

{_tools_table()}

## tensorviz.toml

`ValidationConfig` in `backend/services/validation/contracts.py` is the file. `version` is `1`. `profiles` holds 1 to 64 named profiles. Each profile is a `model` or a `script`.

| Key | Applies to | Constraint |
| --- | --- | --- |
| `version` | file | `1` |
| `profiles` | file | 1 to 64 entries |
| `kind` | profile | `model` or `script` |
| `cwd` | profile | `"."` or a path. Default `"."` |
| `tracked_files` | profile | at most 256 paths |
| `limits.timeout_ms` | profile | 100 through 86,400,000. Default 60,000 |
| `limits.max_events` | profile | 1 through 100,000. Default 10,000 |
| `limits.max_tensors` | profile | 1 through 100,000. Default 10,000 |
| `limits.max_record_bytes` | profile | at least 1,024, at most the record cap. Default is the cap |
| `limits.max_capture_bytes` | profile | optional; 1,024 through 16,777,216 |
| `model` | model | `file:symbol` |
| `framework` | model | omit for PyTorch, or `jax` |
| `mode` | model | `train` or `eval` |
| `device` | model | execution device |
| `grad_enabled` | model | boolean |
| `deterministic` | model | boolean. Default true |
| `seed` | model | optional natural number |
| `init` | model | constructor binding. Exactly one of `init` or `factory` |
| `factory` | model | `file`, `callable`, optional `kwargs` |
| `inputs` | model | a binding, or a factory |
| `script` | script | path |
| `args` | script | at most 256 strings |
| `capture` | script | `model` symbol and positive `call` |

An example that matches `backend/tests/fixtures/model_profiles/valid.toml` is in [saved profiles](model-validation-profiles.md).

{SOURCE_DIRECTIVES}
"""
    return body if body.endswith("\n") else body + "\n"


def main() -> None:
    path = DOCS / "reference.md"
    path.write_text(render(), encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
