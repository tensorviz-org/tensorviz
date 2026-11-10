# plugin-mac-audit — Published plugin Mac isolation and compatibility audit

Status: **Blocked** · 5/7 checkpoints done.

Generated from `progress.csv`; update the CSV and run `python3 tooling/feature_docs.py render`.

- [x] **plugin-mac-audit-01** · plan · Record owner scope and exact published baseline
  Status: done; owner: Codex; updated: 2026-11-11.
  Evidence: plan.md

- [x] **plugin-mac-audit-02** · verification · Verify published assets and fresh installs
  Status: done; owner: Codex; updated: 2026-11-11.
  Evidence: report.md

- [x] **plugin-mac-audit-03** · verification · Capture and render published synthetic graph; test recovery
  Status: done; owner: Codex; updated: 2026-11-11.
  Evidence: report.md

- [x] **plugin-mac-audit-04** · implementation · Fix reproduced defects and add regression coverage
  Status: done; owner: Codex; updated: 2026-11-11.
  Evidence: report.md

- [x] **plugin-mac-audit-05** · verification · Recheck candidate in affected real clients; source/package checks
  Status: done; owner: Codex; updated: 2026-11-11.
  Evidence: report.md
  Notes: Scoped evidence; signed-in GUI remains blocked.

- [ ] **plugin-mac-audit-06** · verification · Signed-in isolated desktop GUI acceptance
  Status: blocked; owner: Codex; updated: 2026-11-11.
  Evidence: Computer Use denies com.openai.codex; no verified isolated profile

- [ ] **plugin-mac-audit-07** · review · Sanitize evidence, render report, push branch and draft PR
  Status: in_progress; owner: Codex; updated: 2026-11-11.
  Evidence: report.md
