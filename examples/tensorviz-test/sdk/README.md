# tensorviz SDK

Three static-first declarations the tensorViz project indexer reads from source:

```python
import tensorviz

tensorviz.experiment("wide-net")
tensorviz.node("dataset", "raw-logs")
tensorviz.node("transform", "tokenise")
tensorviz.edge("raw-logs", "tokenise", "feeds")
```

Install in the interpreter your project uses: `python -m pip install -e sdk/` from `examples/tensorviz-test/`.
The calls are no-ops at runtime beyond recording in-process. Contract and rationale:
[example project](../README.md).
