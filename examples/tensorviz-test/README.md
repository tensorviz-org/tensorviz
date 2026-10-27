# examples/tensorviz-test

A small copy-task transformer and a smaller CNN, with a local `tensorviz.toml`.

From this directory:

```bash
pip install -e sdk
python train.py
```

`train.py` reads `config.json`, trains `models/tiny_transformer.py` on CPU, and
writes `tiny_transformer.pt`. `sdk/` is a tiny local package: `tensorviz.experiment`,
`tensorviz.node`, and `tensorviz.edge` record string labels in-process and do
nothing else. Install it into the interpreter you use for this folder.

Open this folder in the tensorViz extension and use **Open as Graph** on
`TinyTransformer` or `Simple`. The saved profiles are `transformer` and `simple`
in `tensorviz.toml`.
