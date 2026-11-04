# Input and output names

Captured Input and Output cards, module boundary rows and port tooltips display
recorded dictionary keys and keyword argument names. A returned dictionary such
as `{"patch_tokens": tensor, "pooled": tensor}` shows `patch_tokens` and `pooled`,
with `Out patch_tokens` and `Out pooled` on the module. A keyword input such as
`attention_mask` keeps that name. Shapes and dtypes remain beside each value.

Unnamed positional inputs and tuple/list outputs use one-based labels such as
`Input 1` and `Output 3` (`In 1` / `Out 3` on multi-value module boundaries).
Nested containers preserve their structure, for example `states[0]`; punctuation
and non-string dictionary keys use bracket notation. Repeated input aliases remain
listed together. Tensor roles and positional Python parameter names are not guessed.

The optional `DisplayPort.displayName` is derived from the recorded argument tree.
Canonical paths, port IDs, values, edges, shapes and dtypes stay unchanged. The raw
capture paths remain available in captured details. The shared renderer covers
the extension, local browser, plugin panel and static graph-file viewer.

Older version-4 graph files recover names from their retained capture when opened;
saved display names are recomputed from that evidence. Live documents without names
use keyword names where present and concise numbered fallbacks for dictionaries.
They never show serializer paths such as `output.entries[2].value` on the cards.
Refreshing with the updated backend supplies the actual dictionary keys.
