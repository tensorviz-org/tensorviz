### Validate and read the measurements

**Open as Graph** captures one forward call once setup is resolved. After source
changes, save and choose **Refresh graph** to capture again. **Change inputs**
edits the saved setup; generated values are examples you can replace.

- Observed nodes flash green, a mapped failure red, and unreached nodes stay
  unverified.
- Cards distinguish measured, static, declared, unobserved and stale evidence.
- **Output tensor bytes** is logical footprint, not peak memory.
- A failed, refused or cancelled run keeps earlier measurements explicitly retained.

Saving refreshes the source projection and does not capture. `tensorviz.traceOnSave`
stays in settings and is ignored, including when it is true. Choose **Refresh graph**
to capture again.
