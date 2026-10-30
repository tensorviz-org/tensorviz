# Canvas block families

The Canvas keeps three signals apart so each stays readable as features grow:

| Signal | Meaning | Appearance |
| --- | --- | --- |
| Block family | What an operation does | The card's line glyph in the family colour, on a lightly tinted tile in Ink |
| Selection or tracing | What you are inspecting | A pale sage outline or connection highlight |
| Execution status | What happened during a run | An explicit icon, label and outline |

Selecting a convolution keeps its convolution glyph. A failed normalization layer
keeps its normalization glyph and shows the error as its own icon, label and red
outline. A sage normalization glyph never means "passed", and a clay loss glyph
never means "failed". The glyph tile is the only coloured part of a card.

## Families

| Family | Ink colour | Glyph | Examples |
| --- | --- | --- | --- |
| Convolution | `#A8B9C7` | grid | `Conv1d/2d/3d`, `ConvTranspose2d`, `aten.convolution` |
| Linear | `#B5BACB` | network | `Linear`, `Bilinear`, `addmm`, `matmul` |
| Normalization | `#AABDAE` | scale | `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm` |
| Activation | `#C6BDA4` | waveform | `ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Softmax` |
| Pooling | `#BAAFC5` | shrink | `MaxPool2d`, `AdaptiveAvgPool2d` |
| Dropout | `#C2ADB5` | dashed circle | `Dropout`, `Dropout2d` |
| Recurrent | `#B5A5C1` | repeat | `LSTM`, `GRU` and cells |
| Attention / Transformer | `#ABA7C7` | focus | `MultiheadAttention`, `TransformerEncoderLayer`, scaled attention |
| Embedding / sparse | `#BEAF96` | hash | `Embedding`, `EmbeddingBag` |
| Padding | `#B6B8A5` | frame | `ReflectionPad2d`, `ZeroPad2d`, `pad` |
| Resampling | `#A5BBB6` | scaling | `Upsample`, interpolate, bilinear resize |
| Shape / rearrangement | `#AAB8BE` | shuffle | `Flatten`, view/reshape, permute, slice, `PixelShuffle` |
| Distance / similarity | `#B9A9AE` | ruler | `CosineSimilarity`, `PairwiseDistance` |
| Loss | `#C2AA9F` | target | `CrossEntropyLoss`, `MSELoss` |
| Arithmetic / merge | `#BFB1A4` | merge | add, multiply, concatenate, reductions |

In Ink the glyph's tile also takes 16% of the family colour so the hue reads at a
glance; the glyph keeps at least 3.7:1 on it. Paper uses darker variants of the same
hues on untinted tiles, since a tint would take the strokes below 3:1 on white. These
colors are a tensorViz palette, not official PyTorch colors. Several are deliberately
close: the glyph shape and the block's name identify it exactly.

Neutral blocks keep a neutral tile and a gray glyph (`#848D85` in Ink, `#83888F` in
Paper; input and output `#C9CFC8` / `#555A66`): **Input** and **Output** (arrows),
**Container** (`Sequential`, `ModuleList`, `ModuleDict`, and parallel wrappers
labelled as such), **Composite module** (your own classes, custom code, fused
modules) and **Unclassified** operations, which show their type name.

## How a block is classified

Classification follows the block's identity, never its display name:

1. Input and output boundaries are neutral.
2. A captured module call uses its fully qualified class, such as
   `torch.nn.modules.conv.Conv2d`. Only `torch.nn` classes map to a family; a class
   of your own stays a composite even when it is called `Linear` or `LayerNorm`.
3. A captured operation uses its ATen operator, such as `aten.convolution.default`.
4. A source graph uses its catalog id (`conv2d`) or its `torch` /
   `torch.nn.functional` target.
5. Anything else is unclassified.

Equivalent module, functional and ATen forms therefore share a family, while
`nn.Bilinear` (Linear) and a bilinear resize (Resampling) stay apart. Lazy,
quantized and QAT variants keep their family and add a modifier label. The same
result drives cards, collapsed modules, expanded frames, the overview/export SVG,
the inspector and the legend.

## Inspecting a block

Selecting a block shows, in the selection panel (and the graph viewer's detail panel):

- the family glyph and name, with any modifier;
- the exact class or operator it was classified from;
- for a composite module or container, the families of its children recorded in the
  graph, in data-flow order, for example **Convolution → Normalization → Activation**.
  Nothing is inferred when no child is classified.

Expand a composite module to see its individually coloured children.

## Legend, connections and readability

**Blocks** in the Canvas overlay is a collapsed legend that lists only the families
present in the displayed graph, as coloured glyphs. The overview and exported SVG
draw the same glyph tiles, so the family colour stays visible when cards are small.

Connections stay neutral: 1.4px gray at rest, slightly stronger for the selected
node's own inputs and outputs, and pale sage for a traced path, with unrelated
connections quieter. Family colors never reach connections, and nothing glows or
animates. Rows with several tensors are numbered from zero (`IN 0`, `IN 1`, `OUT`);
the full argument path is the row's tooltip and appears in captured details.

## Accessibility

Every family colour is carried by a distinct glyph shape and named in the inspector
and legend, so the family survives grayscale and color-vision differences. In
forced-colors mode the glyphs use the system text color and selection uses the system
highlight. Keyboard
focus has its own ring, separate from the selection outline.

Themes are in [Canvas](reference/canvas.md#canvas-appearance).
