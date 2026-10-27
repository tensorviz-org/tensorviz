#!/usr/bin/env python3
"""Read every real PR's file diff and verify it against the pinned local Git pair.

Resumes from durable receipts while publication continues. This is a read-only
GitHub audit; it does not submit approvals or comments on self-authored PRs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from graph_review_stress_publish import api
from graph_review_stress import git


def audit(repository, repo, state):
    manifest = {row["id"]: row for row in json.loads((state / "manifest.json").read_text())}
    output = state / "github-file-audits.jsonl"
    done = {row["id"] for row in map(json.loads, output.read_text().splitlines())} if output.exists() else set()
    while len(done) < len(manifest):
        receipts = [json.loads(line) for line in (state / "github-prs.jsonl").read_text().splitlines()]
        for pr in receipts:
            if pr["id"] in done:
                continue
            item = manifest[pr["id"]]
            assert pr["base"] == item["base"] and pr["head"] == item["head"]
            files = api(f"repos/{repository}/pulls/{pr['number']}/files?per_page=100")
            expected = {}
            diff = git(repo, "diff", "--numstat", item["base"], item["head"])
            for line in diff.splitlines():
                added, removed, name = line.split("\t")
                sha = git(repo, "rev-parse", item["head"] + ":" + name)
                expected[name] = {"additions": int(added), "deletions": int(removed), "sha": sha}
            assert {file["filename"] for file in files} == set(expected), (item["id"], "Changed file mismatch")
            for file in files:
                assert file["status"] == "modified" and file.get("patch"), (item["id"], "Missing real source diff")
                assert all(file[key] == value for key, value in expected[file["filename"]].items()), (item["id"], "Diff metadata mismatch")
            evidence = {**pr, "auditedAt": datetime.now(timezone.utc).isoformat(), "files": files, "passed": True}
            with output.open("a") as stream:
                stream.write(json.dumps(evidence) + "\n")
                stream.flush()
            done.add(item["id"])
            if len(done) % 25 == 0: print(f"Audited real GitHub file diffs: {len(done)}/1000", flush=True)
        if len(done) < len(manifest): time.sleep(15)
    # One final GitHub snapshot establishes the complete set and current heads.
    pulls = []
    for page in range(1, 12):
        batch = api(f"repos/{repository}/pulls?state=all&per_page=100&page={page}")
        pulls.extend(batch)
        if len(batch) < 100: break
    assert len(pulls) == 1000
    by_branch = {pr["head"]["ref"]: pr for pr in pulls}
    assert len(by_branch) == 1000
    for item in manifest.values():
        pr = by_branch[item["branch"]]
        assert pr["base"]["sha"] == item["base"] and pr["head"]["sha"] == item["head"]
    (state / "github-final-audit.json").write_text(json.dumps({"repository": repository,
        "auditedAt": datetime.now(timezone.utc).isoformat(), "pullRequests": len(pulls),
        "fileDiffs": len(done), "uniqueBranches": len(by_branch), "allRevisionShasMatch": True}, indent=2) + "\n")
    print("COMPLETE: 1000 actual PRs and their GitHub file diffs verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    audit(args.repository, args.repo.resolve(), args.state.resolve())
