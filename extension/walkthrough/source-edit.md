### Propose a change, review the diff

Zoom into a layer until its source controls appear, then choose the top-right
**Edit** control. The form opens on that node. Parameter, add, delete and group
forms work without an agent or a provider key.

Every proposal opens a complete native Python diff in VS Code. Choose **Apply source
change** or **Discard proposal**.

- Apply makes one unsaved `WorkspaceEdit`; saving is yours to decide.
- Native Undo is the restoration path.
- Shared definitions affect every instance, and grouping changes parameter and
  checkpoint keys.
- Stale source, changed dependencies or an untrusted workspace refuse the write.

Supported edits stay within one owning class and the current release boundary.
