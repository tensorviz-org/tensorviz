#!/usr/bin/env python3
"""Validate and hash explicitly allowlisted report files; optionally stage them.

Installs nothing, runs no tests, commits nothing, and never pushes. Add a new
receipt to report-allowlist.json only after human/agent content review.
"""

import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import xml.etree.ElementTree as ET

RULES = {
    "personal_home": r"/(?:Users|home)/[^/\s<>]+/",
    "private_temp": r"/var/" r"folders/",
    "private_key_material": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "github_secret": r"(?:ghp_|github_pat_)[A-Za-z0-9_]{12,}",
    "signed_url": r"[?&](?:jwt|sig|token|access_token)=",
    "authorization_header": r"Authorization:\s*(?:Bearer|Basic)\s+[A-Za-z0-9]",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    names = json.loads((root / "report-allowlist.json").read_text())["files"]
    if names != sorted(set(names)):
        raise SystemExit("Allowlist must be sorted and contain unique names")
    entries = []
    problems = []
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or any(
                part in {".git", ".github", "private", "node_modules", ".venv", "venv"}
                for part in path.parts):
            raise SystemExit("Refused unsafe allowlist path")
        local = root.joinpath(*path.parts)
        if not local.is_file() or local.is_symlink() or not local.resolve().is_relative_to(root):
            raise SystemExit("Missing or unsafe report file: " + name)
        raw = local.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raise SystemExit("Report file exceeds review size limit: " + name)
        content = raw.decode("utf-8")
        if path.suffix == ".json":
            json.loads(content)
        if path.suffix == ".py":
            ast.parse(content)
        if path.suffix == ".svg":
            svg = ET.fromstring(content)
            if svg.tag != "{http://www.w3.org/2000/svg}svg" or "<!DOCTYPE" in content:
                raise SystemExit("Refused unexpected SVG document: " + name)
            for element in svg.iter():
                if element.tag.rsplit("}", 1)[-1] in {"script", "foreignObject", "iframe"}:
                    raise SystemExit("Refused active SVG element: " + name)
                for attribute, value in element.attrib.items():
                    if attribute.lower().startswith("on"):
                        raise SystemExit("Refused SVG event handler: " + name)
                    if attribute.rsplit("}", 1)[-1] == "href" and not value.startswith("#"):
                        raise SystemExit("Refused external SVG reference: " + name)
        for label, pattern in RULES.items():
            if re.search(pattern, content):
                problems.append({"file": name, "kind": label})
        if name == "README.md":
            for link in re.findall(r"\]\(([^)]+)\)", content):
                if link.startswith(("https://", "http://", "#")):
                    continue
                if not (root / link.split("#")[0]).is_file():
                    problems.append({"file": name, "kind": "missing local link", "link": link})
        entries.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    if problems:
        print(json.dumps({"problems": problems}, indent=2))
        return 1
    manifest = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Explicitly reviewed text-file allowlist; no source artifacts, raw profiles, caches, environments, or workflow files",
        "validation": {"json_and_python_syntax": "passed", "privacy_pattern_scan": "passed",
                       "readme_local_links": "passed", "file_count": len(entries)},
        "files": entries,
        "self_exclusion": "This manifest does not hash itself",
    }
    (root / "report-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.stage:
        subprocess.run(["git", "-C", str(root), "add", "--", *names, "report-manifest.json"], check=True)
    print(json.dumps({"validated_files": len(entries), "staged": args.stage,
                      "committed": False, "pushed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
