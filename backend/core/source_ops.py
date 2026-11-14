"""Static graph semantics shared without importing any execution framework."""

# These operations produce tensors without consuming one. They form independent
# roots, not steps in the model's input chain.
SOURCE_OPS = frozenset({"parameter", "buffer"})
