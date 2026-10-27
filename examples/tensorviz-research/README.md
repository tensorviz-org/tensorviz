# tensorviz-research

A small but complete character-level language-model project: a transformer built
from its own parts, a corpus that needs no download, a training loop, held-out
evaluation, checkpointing, and two alternative ways of running it.

It exists to be **looked at**. The structure is the kind tensorViz has to draw
correctly, so it is deliberately not flattened into one file.

## Run it

`train.py` and `finetune_lora.py` import the local `tensorviz` package that lives in `examples/tensorviz-test/sdk`. Install that package into the interpreter you use here, then run from this directory:

```bash
pip install -e ../tensorviz-test/sdk
python train.py          # baseline, CPU, well under a minute
python sweep.py          # depth sweep, plain top-to-bottom script
python finetune_lora.py  # also needs transformers, datasets, peft, trl
```

The SDK records `tensorviz.experiment`, `tensorviz.node`, and `tensorviz.edge` labels. It is not the TensorViz extension. See [examples/tensorviz-test](../tensorviz-test/README.md).

## What is where

| Path | |
| --- | --- |
| `models/attention.py` | `MultiHeadAttention`, `FeedForward` |
| `models/blocks.py` | `TransformerBlock`, composed from both of the above |
| `models/tiny_lm.py` | `TinyLM`, composed from `TransformerBlock` — three levels deep |
| `data/char_dataset.py` | `CharCorpus`, a `Dataset` over an embedded corpus |
| `data/loaders.py` | batching, kept separate so a sweep can vary it alone |
| `training/trainer.py` | `LanguageModelTrainer` — the loop, the optimizer, the clipping |
| `training/metrics.py` | perplexity and token accuracy |
| `training/checkpoint.py` | `torch.save` / `torch.load` |
| `evaluation/evaluate.py` | one no-grad pass over the held-out split |
| `configs/base.json` | the baseline hyperparameters |
| `notes.py` | scratch; nothing imports it |

## The three entrypoints are deliberately different

`train.py` declares itself with `tensorviz.experiment(...)` and has a `__main__`
guard. `finetune_lora.py` declares itself too, but assembles its pipeline entirely
out of library classes — nothing is subclassed. `sweep.py` does neither: it is a
plain script that runs top to bottom, which is how a great deal of research code is
actually written.

`train.py` and `sweep.py` share `TinyLM` and the trainer, so the same definition is
used by more than one experiment.
