# TensorViz merge protection design

Proposed only. No GitHub settings or permissions changed.

## Verified state

- `tensorviz-org/tensorviz` is private (ID 1435323397), default branch `main`; main reports `protected: false` at SHA `c7ae9e2b4c91e5098ea24bc8617463fa151c2624`.
- Connected-user metadata reports admin, but the integration cannot read classic branch protection: HTTP 403, `Resource not accessible by integration`.
- Ruleset listing returns HTTP 403: `Upgrade to GitHub Pro or make this repository public to enable this feature.` No ruleset inventory was retrieved.
- Main has no CODEOWNERS file. Org members/teams, outside collaborators, apps/deploy keys and exact billing plan are unverified. No team name or ID should be invented.
- Public Free organization repos support branch/tag rulesets. Private organization repos need an eligible paid organization plan; a personal Pro upgrade is not a substitute.

## Proposed settings, pending approval

Use a verified `tensorviz-org` team with explicit write access and at least two human members. Reuse a suitable team; otherwise approve its creation and members. Teams cannot contain outside collaborators. New org members are not automatically added to a team. Converting a member to an outside collaborator removes them from teams, even if repository access remains.

These templates use `VERIFIED_TEAM_ID` as an intentionally invalid JSON placeholder. Verify IDs and scope before applying. Preserve unrelated protections.

### Separate merge-actor gate

```json
{
  "name": "Only organization maintainers merge",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [
    {"actor_id": VERIFIED_TEAM_ID, "actor_type": "Team", "bypass_mode": "pull_request"}
  ],
  "conditions": {"ref_name": {"include": ["~ALL"], "exclude": []}},
  "rules": [
    {"type": "update", "parameters": {"update_allows_fetch_and_merge": false}}
  ]
}
```

The only bypass is the verified org team, limited to pull requests. Do not add generic write/admin roles, individuals, bots, apps or deploy keys.

### Independent review rule, with no bypass

```json
{
  "name": "Require organization review",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {"ref_name": {"include": ["~ALL"], "exclude": []}},
  "rules": [
    {
      "type": "pull_request",
      "parameters": {
        "allowed_merge_methods": ["merge", "squash", "rebase"],
        "dismiss_stale_reviews_on_push": true,
        "dismissal_restriction": {
          "enabled": true,
          "allowed_actors": [{"id": VERIFIED_TEAM_ID, "type": "Team"}]
        },
        "require_code_owner_review": false,
        "require_last_push_approval": true,
        "required_approving_review_count": 1,
        "required_review_thread_resolution": true,
        "required_reviewers": [{
          "reviewer": {"id": VERIFIED_TEAM_ID, "type": "Team"},
          "minimum_approvals": 1,
          "file_patterns": ["*", "**/*"]
        }]
      }
    },
    {"type": "non_fast_forward"}
  ]
}
```

Current GitHub supports team reviewers directly in rulesets. CODEOWNERS is unnecessary for this design. Keeping merge permission separate means the team cannot bypass the mandatory review.

**Scope decision:** `~ALL` implements all PR targets, but blocks direct upstream feature-branch updates. Use forks for development. Targeting only `~DEFAULT_BRANCH` and selected release branches is less disruptive but does not meet the literal all-PR requirement. Protect main/release branch deletion separately; do not block ordinary feature-branch cleanup. Rule administrators can edit/delete policy, so restrict that privilege to trusted org members.

### Immutable release tags

```json
{
  "name": "Keep release tags immutable",
  "target": "tag",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {"ref_name": {"include": ["refs/tags/v*", "refs/tags/plugin-v*", "refs/tags/latest"], "exclude": []}},
  "rules": [
    {"type": "update", "parameters": {"update_allows_fetch_and_merge": false}},
    {"type": "deletion"}
  ]
}
```

The migration worker confirmed `GITHUB_TOKEN` creates both tag namespaces and does not update branches. This rule allows initial tag creation but blocks subsequent movement/deletion, including by Actions. Do not restrict creation to humans unless a separate approved automation exception is designed. The publisher must verify source SHA and refuse asset/tag overwrite.

### Release environment and activation order

Configure `release` with the verified team as required reviewer, explicit selected deployment **branch** `main`, and admin bypass disabled. Consider preventing self-review. Store publishing secrets in that environment. A missing environment referenced in YAML is automatically created **without protection**. “Protected branches only” allows all branches if there are no branch protections, so use explicit main.

Required environment reviewers work on public Free/Team repos. Private repos need Enterprise for reviewer gates; Team alone does not unlock them.

The worker's proposed YAML checks exact repository/main and `RELEASES_ENABLED == 'true'`. Activate in order: verify protections and reviewed release environment, disable legacy release workflows, then set `RELEASES_ENABLED` last. This is a proposal; do not claim it is active from the YAML alone.

## Verification and limitations

1. Obtain action-time approval for exact security settings, team/access changes and branch scope. Do not publish the repo or upgrade billing merely to unblock configuration.
2. Using authorized admin access, inventory member/team grants, outside collaborators, app writers/deploy keys and all effective repo/org rules. Audit outside write/admin grants; do not remove them without approval.
3. Verify at least two eligible humans: a PR author cannot approve their own PR, so a one-person team blocks that member's PRs.
4. Apply independent rulesets, retrieve by ID and read effective rules for main, a nested feature branch and each release branch. Check active enforcement, exact targets and bypass lists.
5. In approved disposable test PRs, verify zero approvals and outside-only approval remain blocked, eligible approval satisfies review, a new commit invalidates it, and an outsider/admin outside the team cannot merge. Check root, nested and dotfile changes. Do not claim behavior tests passed based only on setting reads.
6. Release workflows must not commit to main, replace catalog branches, merge PRs or publish fork input. Fork PR jobs stay read-only, secret-free and isolated. Never execute PR-head code in privileged `pull_request_target` jobs.
7. Verify environment reviewer/branch/bypass settings and tag behavior before calling complete. State plan/permission/test-actor blockers explicitly. Current tool access does not permit applying or verifying this configuration.

## Official sources

- https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets
- https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets#required-reviewers
- https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset
- https://docs.github.com/en/organizations/organizing-members-into-teams/about-teams
- https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/managing-outside-collaborators/converting-an-organization-member-to-an-outside-collaborator
- https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments
- https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments
