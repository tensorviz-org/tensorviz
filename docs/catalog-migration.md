# Move an existing plugin to the public catalog

The public catalog is now [tensorviz-org/tensorviz](https://github.com/tensorviz-org/tensorviz).
Its marketplace name remains `tensorviz-standalone`. If you registered the old
`tensorviz-org/tensorviz-plugins` source, replace that registration before updating;
adding a second source with the same marketplace name can fail.

Use the same execution environment and client profile as your existing install.
Save the marketplace listing, back up your client configuration, and retain your
current plugin/catalog folder for rollback. Keep TensorViz permission settings,
model recipes, service state, and runtime caches. These steps do not call the
plugin uninstall command or ask you to delete those files.

## Codex

Inspect the source first:

```sh
codex plugin marketplace list --json
```

If `tensorviz-standalone` points to the old repository:

```sh
codex plugin marketplace remove tensorviz-standalone --json
codex plugin marketplace add tensorviz-org/tensorviz --json
codex plugin add tensorviz@tensorviz-standalone --json
```

Check the returned `version` and `installedPath`. If the source already points
to the public repository, use the normal [update commands](agent-plugin.md#update-an-existing-codex-installation).

## Claude Code

Inspect the source and settings scope first:

```sh
claude plugin marketplace list --json
```

For an existing **user-scope** registration and plugin:

```sh
claude plugin marketplace remove tensorviz-standalone --scope user --json
claude plugin marketplace add tensorviz-org/tensorviz --scope user --json
claude plugin update tensorviz@tensorviz-standalone --scope user --json
```

If your existing registration and plugin use `project` or `local`, use that scope
consistently instead of `user`. Keep `--scope` on removal: omitting it removes the
marketplace declaration from every scope. If the plugin is not installed in that
scope, use `claude plugin install` in place of `update`.

## Check and roll back

Start a new chat or restart the client and verify its installed version and path.
Changing the catalog alone does not establish runtime or graph-panel acceptance.
During the 1.0 launch, the catalog can still serve verified 0.1.48 packages until
1.0 publication completes; use the reported package version as the authority.

For rollback, retain your saved source and complete previous catalog folder.
Replace only the `tensorviz-standalone` marketplace registration with that saved
source (both clients accept a local catalog folder), using the same Claude scope.
Reinstall/select the retained package, restore saved client configuration if
needed, and restart. Keep the old folder separate; do not merge releases or edit
cached launchers. The [historical 0.1.48 release](https://github.com/tensorviz-org/tensorviz-plugins/releases/tag/v0.1.48)
remains available at its original immutable location.

Options checked with Codex CLI 0.165.0 and Claude Code 2.1.325. If your client does
not recognize them, check its local `--help` before changing configuration.
