"""Deterministic, small executable architecture specimens for Git review QA.

These are representative computation patterns, not pretrained implementations or
claims of fidelity to a named production checkpoint. Only this generated code is
executed by the test runner; graph review itself stays source-only.
"""
from __future__ import annotations

import ast
import hashlib
import sys
import types


FAMILIES = {
    "feedforward": "mlp residual_mlp highway gated_mlp swiglu maxout dense_concat cross_network bilinear_fusion low_rank",
    "vision": "conv1d conv2d conv3d depthwise separable grouped dilated pointwise residual_cnn bottleneck inverted_residual squeeze_excitation inception densenet unet transposed_cnn pixel_shuffle patch_embed mixer convnext",
    "sequence": "embedding position_embedding attention self_attention cross_attention causal_attention multiquery_attention linear_attention transformer_encoder transformer_decoder encoder_decoder lstm gru rnn lstm_cell gru_cell bidirectional_lstm temporal_conv wavenet state_space",
    "generative": "autoencoder conv_autoencoder variational_encoder vae_decoder gan_generator gan_discriminator diffusion_denoiser time_conditioned affine_flow additive_coupling",
    "graph": "gcn graphsage graph_attention gin message_passing edge_network graph_readout pointnet pointnet_local point_transformer",
    "tabular": "wide_deep deepfm neural_cf two_tower dcn tabnet gated_feature deep_sets embedding_bag hashed_embedding",
    "multimodal_control": "clip_dual multimodal_concat gated_fusion crossmodal_attention actor_critic dueling_q distributional_q world_model mixture_experts soft_moe",
    "scientific_audio_containers": "fourier_operator siren rbf neural_ode_step spectral_conv audio_encoder speech_conv_recurrent module_list module_dict multi_file",
}
SIZES = [(4, 1), (8, 2), (12, 3), (16, 4), (20, 6), (24, 8), (28, 12), (32, 16), (48, 24), (64, 32)]
MUTATIONS = ["comments", "import_alias", "keyword_equivalence", "explicit_default", "constructor_parameter",
             "activation_replacement", "insert_layer", "remove_layer", "output_rewire", "scalar_attribute",
             "scalar_literal", "operand_order", "duplicate_call", "functional_activation", "helper_body",
             "constructor_reorder", "unused_attribute", "module_rename", "branch_condition", "child_parameter"]


def architecture(family: str, width: int, depth: int, cls: str):
    w = width
    init, body, extras = [], [], {}
    shape, dtype = [2, w], "float32"
    linear = lambda name, i=w, o=w: init.append(f"self.{name} = nn.Linear({i}, {o})")
    conv = lambda name, dim=2, i=w, o=w, k=3, args="padding=1": init.append(f"self.{name} = nn.Conv{dim}d({i}, {o}, {k}, {args})")
    for stage in range(depth):
        a, b, c = (f"{letter}{stage}" for letter in "abc")
        if family in {"mlp", "residual_mlp", "highway", "gated_mlp", "swiglu", "maxout", "dense_concat", "cross_network", "bilinear_fusion", "low_rank"}:
            if family == "bilinear_fusion":
                init.append(f"self.{a} = nn.Bilinear({w}, {w}, {w})")
                body.append(f"x = self.{a}(x, x)")
            elif family == "low_rank":
                linear(a, w, w // 2); linear(b, w // 2, w)
                body.append(f"x = self.{b}(self.{a}(x))")
            else:
                linear(a)
                if family in {"highway", "gated_mlp", "swiglu", "maxout"}: linear(b)
                if family == "dense_concat": linear(b, 2 * w, w)
                body += {
                    "mlp": [f"x = torch.relu(self.{a}(x))"],
                    "residual_mlp": [f"x = x + torch.relu(self.{a}(x))"],
                    "highway": [f"gate = torch.sigmoid(self.{b}(x))", f"x = gate * self.{a}(x) + (1 - gate) * x"],
                    "gated_mlp": [f"x = self.{a}(x) * torch.sigmoid(self.{b}(x))"],
                    "swiglu": [f"x = F.silu(self.{a}(x)) * self.{b}(x)"],
                    "maxout": [f"x = torch.maximum(self.{a}(x), self.{b}(x))"],
                    "dense_concat": [f"x = self.{b}(torch.cat([x, self.{a}(x)], dim=-1))"],
                    "cross_network": [f"x = x + x * torch.tanh(self.{a}(x))"],
                }[family]
        elif family in FAMILIES["vision"].split():
            dim = 1 if family == "conv1d" else 3 if family == "conv3d" else 2
            shape = [1, w, *([4] * dim)]
            if family in {"depthwise", "separable", "grouped", "dilated", "pointwise"}:
                args = f"padding=1, groups={w if family != 'grouped' else 2}" if family in {"depthwise", "separable", "grouped"} else "padding=2, dilation=2" if family == "dilated" else "padding=0"
                conv(a, k=1 if family == "pointwise" else 3, args=args)
                body.append(f"x = self.{a}(x)")
                if family == "separable": conv(b, k=1, args="padding=0"); body.append(f"x = self.{b}(x)")
            elif family in {"bottleneck", "inverted_residual"}:
                hidden = w // 2 if family == "bottleneck" else w * 2
                conv(a, o=hidden, k=1, args="padding=0"); conv(b, i=hidden, o=hidden); conv(c, i=hidden, k=1, args="padding=0")
                body.append(f"x = x + self.{c}(torch.relu(self.{b}(torch.relu(self.{a}(x)))))")
            elif family == "squeeze_excitation":
                conv(a); conv(b, k=1, args="padding=0")
                body.append(f"x = self.{a}(x) * torch.sigmoid(self.{b}(x.mean(dim=(-2, -1), keepdim=True)))")
            elif family in {"inception", "densenet"}:
                conv(a, k=1, args="padding=0"); conv(b); conv(c, i=w * (2 if family == "inception" else 3), k=1, args="padding=0")
                parts = f"[self.{a}(x), self.{b}(x)]" if family == "inception" else f"[x, self.{a}(x), self.{b}(x)]"
                body.append(f"x = self.{c}(torch.cat({parts}, dim=1))")
            elif family == "unet":
                conv(a); init.extend([f"self.{b} = nn.AvgPool2d(2)", f"self.{c} = nn.Upsample(scale_factor=2, mode='nearest')"])
                body.append(f"x = self.{a}(x) + self.{c}(self.{b}(x))")
            elif family == "transposed_cnn":
                init.append(f"self.{a} = nn.ConvTranspose2d({w}, {w}, 3, padding=1)"); body.append(f"x = self.{a}(x)")
            elif family == "pixel_shuffle":
                conv(a, o=4*w, k=1, args="padding=0"); init.extend([f"self.{b} = nn.PixelShuffle(2)", f"self.{c} = nn.AvgPool2d(2)"])
                body.append(f"x = self.{c}(self.{b}(self.{a}(x)))")
            elif family in {"patch_embed", "mixer"}:
                shape = [2, 4, w]; linear(a); linear(b, 4, 4)
                body.append(f"x = self.{b}(self.{a}(x).transpose(1, 2)).transpose(1, 2)")
                if family == "mixer": body.append("x = x + torch.tanh(x)")
            elif family == "convnext":
                conv(a, args=f"padding=1, groups={w}"); conv(b, o=2*w, k=1, args="padding=0"); conv(c, i=2*w, k=1, args="padding=0")
                body.append(f"x = x + self.{c}(F.gelu(self.{b}(self.{a}(x))))")
            else:
                conv(a, dim=dim); body.append(f"x = self.{a}(x)" + (" + x" if family == "residual_cnn" else ""))
        elif family in FAMILIES["sequence"].split():
            shape = [2, 4, w]
            if family in {"embedding", "position_embedding"}:
                shape, dtype = [2, 4], "int64"
                if stage == 0:
                    init.append(f"self.tokens = nn.Embedding({w*8}, {w})"); body.append("x = self.tokens(x)")
                    if family == "position_embedding":
                        init.append(f"self.positions = nn.Embedding(4, {w})"); body.append("x = x + self.positions(torch.arange(4))")
                linear(a); body.append(f"x = self.{a}(x)")
            elif family in {"attention", "self_attention", "cross_attention", "causal_attention", "multiquery_attention", "linear_attention", "crossmodal_attention"}:
                if family == "attention":
                    init.append(f"self.{a} = nn.MultiheadAttention({w}, 2, batch_first=True)"); body.append(f"x = self.{a}(x, x, x)[0]")
                else:
                    linear(a); linear(b); linear(c)
                    body.extend([f"q = self.{a}(x)", f"k = self.{b}(x)", f"v = self.{c}(x)"])
                    if family == "cross_attention": body.append("k = k.flip(1)")
                    if family == "multiquery_attention": body.append("k = k.mean(dim=-1, keepdim=True).expand_as(k)")
                    if family == "linear_attention": body.append("x = torch.softmax(q, dim=-1) @ (torch.softmax(k, dim=1).transpose(-1, -2) @ v)")
                    else:
                        body.append(f"scores = (q @ k.transpose(-1, -2)) / {w**0.5!r}")
                        if family == "causal_attention": body.append("scores = scores.masked_fill(torch.ones(4, 4, dtype=torch.bool).triu(1), -10000.0)")
                        body.append("x = torch.softmax(scores, dim=-1) @ v")
            elif family in {"transformer_encoder", "transformer_decoder", "encoder_decoder"}:
                kind = "TransformerEncoderLayer" if family == "transformer_encoder" else "TransformerDecoderLayer"
                init.append(f"self.{a} = nn.{kind}({w}, 2, dim_feedforward={w*2}, batch_first=True, dropout=0.0)")
                if family == "encoder_decoder":
                    init.append(f"self.{b} = nn.TransformerEncoderLayer({w}, 2, dim_feedforward={w*2}, batch_first=True, dropout=0.0)"); body.append(f"x = self.{a}(x, self.{b}(x))")
                else: body.append(f"x = self.{a}(x{', x' if kind == 'TransformerDecoderLayer' else ''})")
            elif family in {"lstm", "gru", "rnn", "bidirectional_lstm"}:
                op = "LSTM" if family == "bidirectional_lstm" else family.upper()
                init.append(f"self.{a} = nn.{op}({w}, {w//2 if family == 'bidirectional_lstm' else w}, batch_first=True{', bidirectional=True' if family == 'bidirectional_lstm' else ''})")
                body.append(f"x = self.{a}(x)[0]")
            elif family in {"lstm_cell", "gru_cell"}:
                shape = [2, w]; init.append(f"self.{a} = nn.{'LSTMCell' if family == 'lstm_cell' else 'GRUCell'}({w}, {w})")
                body.append(f"x = self.{a}(x){'[0]' if family == 'lstm_cell' else ''}")
            elif family in {"temporal_conv", "wavenet"}:
                shape = [2, w, 8]; conv(a, dim=1, args="padding=2, dilation=2")
                if family == "wavenet": conv(b, dim=1); body.append(f"x = torch.tanh(self.{a}(x)) * torch.sigmoid(self.{b}(x))")
                else: body.append(f"x = x + self.{a}(x)")
            else:
                linear(a); linear(b); body.append(f"x = torch.tanh(self.{a}(x) + self.{b}(x.cumsum(dim=1)))")
        elif family in FAMILIES["generative"].split():
            if family == "conv_autoencoder":
                shape = [1, w, 4, 4]; conv(a, o=w//2); init.append(f"self.{b} = nn.ConvTranspose2d({w//2}, {w}, 3, padding=1)")
                body.append(f"x = self.{b}(torch.relu(self.{a}(x)))")
            elif family in {"autoencoder", "vae_decoder", "gan_generator"}:
                hidden = w//2 if family == "autoencoder" else w*2
                linear(a, w, hidden); linear(b, hidden, w)
                body.append(f"x = {'torch.tanh' if family == 'gan_generator' else 'torch.sigmoid'}(self.{b}(torch.relu(self.{a}(x))))")
            elif family == "variational_encoder":
                linear(a); linear(b); body.append(f"x = self.{a}(x) + torch.exp(0.5 * self.{b}(x).clamp(-4, 4))")
            elif family == "gan_discriminator":
                linear(a); body.append(f"x = torch.sigmoid(self.{a}(x))")
            elif family in {"diffusion_denoiser", "time_conditioned"}:
                linear(a); linear(b); body.append(f"x = x + self.{a}(torch.sin(self.{b}(x)))" if family == "diffusion_denoiser" else f"x = self.{a}(x) + self.{b}(torch.ones_like(x) * 0.25)")
            else:
                linear(a, w//2, w//2); linear(b, w//2, w//2)
                body.extend(["left, right = x.chunk(2, dim=-1)", f"right = right {'* torch.exp(torch.tanh(self.' + a + '(left))) +' if family == 'affine_flow' else '+'} self.{b}(left)", "x = torch.cat([left, right], dim=-1)"])
        elif family in FAMILIES["graph"].split():
            shape = [2, 4, w]; linear(a)
            if family in {"gcn", "graphsage", "gin", "message_passing"}:
                if family == "graphsage": linear(b, 2*w, w)
                expr = {"gcn": f"self.{a}(x.mean(dim=1, keepdim=True).expand_as(x))", "graphsage": f"self.{b}(torch.cat([x, self.{a}(x.mean(dim=1, keepdim=True).expand_as(x))], dim=-1))", "gin": f"self.{a}(1.1 * x + x.sum(dim=1, keepdim=True))", "message_passing": f"self.{a}(x + x.roll(1, dims=1))"}[family]
                body.append(f"x = {expr}")
            elif family in {"graph_attention", "point_transformer"}:
                linear(b); body.append(f"x = torch.softmax(self.{a}(x) @ self.{a}(x).transpose(-1, -2), dim=-1) @ self.{b}(x)")
                if family == "point_transformer": body.append("x = x - x.mean(dim=1, keepdim=True)")
            elif family == "edge_network":
                linear(b, 2*w, w); body.append(f"x = self.{b}(torch.cat([self.{a}(x), x.roll(1, dims=1) - x], dim=-1))")
            elif family == "pointnet_local": body.append(f"x = self.{a}(x - x.mean(dim=1, keepdim=True))")
            else:
                body.append(f"x = self.{a}(x)")
                if stage == depth - 1: body.append("x = x.max(dim=1).values" if family == "pointnet" else "x = x.mean(dim=1)")
        elif family in FAMILIES["tabular"].split():
            if family in {"embedding_bag", "hashed_embedding"}:
                shape, dtype = [2, 4], "int64"
                if stage == 0:
                    init.append(f"self.tokens = nn.{'EmbeddingBag' if family == 'embedding_bag' else 'Embedding'}({w*8}, {w})")
                    body.append("x = self.tokens(x)" if family == "embedding_bag" else f"x = self.tokens(x.remainder({w*8})).sum(dim=1)")
                linear(a); body.append(f"x = self.{a}(x)")
            else:
                linear(a); linear(b)
                expr = {"wide_deep": f"self.{a}(x) + torch.relu(self.{b}(x))", "deepfm": f"self.{a}(x) + 0.5 * (self.{b}(x).square() - self.{b}(x.square()))", "neural_cf": f"self.{a}(x) * self.{b}(x)", "two_tower": f"F.normalize(self.{a}(x), dim=-1) + F.normalize(self.{b}(x), dim=-1)", "dcn": f"x + x * torch.tanh(self.{a}(x)) + self.{b}(x)", "tabnet": f"self.{a}(x * torch.softmax(self.{b}(x), dim=-1))", "gated_feature": f"self.{a}(x) * torch.sigmoid(self.{b}(x))", "deep_sets": f"self.{b}(torch.relu(self.{a}(x)).mean(dim=0, keepdim=True).expand_as(x))"}[family]
                body.append(f"x = {expr}")
        elif family in FAMILIES["multimodal_control"].split():
            linear(a); linear(b)
            if family in {"multimodal_concat", "world_model"}: linear(c, 2*w, w)
            if family in {"mixture_experts", "soft_moe"}: linear(c)
            expr = {"clip_dual": f"F.normalize(self.{a}(x), dim=-1) * F.normalize(self.{b}(x), dim=-1)", "multimodal_concat": f"self.{c}(torch.cat([self.{a}(x), self.{b}(x)], dim=-1))", "gated_fusion": f"torch.sigmoid(self.{a}(x)) * self.{b}(x) + x", "crossmodal_attention": f"torch.softmax(self.{a}(x) @ self.{b}(x).transpose(-1, -2), dim=-1) @ x", "actor_critic": f"torch.softmax(self.{a}(x), dim=-1) + self.{b}(x).mean(dim=-1, keepdim=True)", "dueling_q": f"self.{a}(x) + self.{b}(x) - self.{b}(x).mean(dim=-1, keepdim=True)", "distributional_q": f"torch.softmax(self.{a}(x) + self.{b}(x), dim=-1)", "world_model": f"self.{c}(torch.cat([x, torch.tanh(self.{a}(x)) + self.{b}(x)], dim=-1))", "mixture_experts": f"torch.where(self.{c}(x) > 0, self.{a}(x), self.{b}(x))", "soft_moe": f"torch.sigmoid(self.{c}(x)) * self.{a}(x) + (1 - torch.sigmoid(self.{c}(x))) * self.{b}(x)"}[family]
            body.append(f"x = {expr}")
        else:
            if family in {"module_list", "module_dict"}:
                if stage == 0:
                    members = ", ".join(f"nn.Linear({w}, {w})" if family == "module_list" else f"'layer{i}': nn.Linear({w}, {w})" for i in range(depth))
                    init.append(f"self.blocks = nn.{ 'ModuleList([' + members + '])' if family == 'module_list' else 'ModuleDict({' + members + '})' }")
                    body.extend([f"for block in self.blocks{'.values()' if family == 'module_dict' else ''}:", "    x = torch.tanh(block(x))"])
            elif family == "multi_file":
                extras["blocks.py"] = "from torch import nn\nclass Block(nn.Module):\n    def __init__(self, width):\n        super().__init__()\n        self.fc = nn.Linear(width, width)\n        self.norm = nn.LayerNorm(width, eps=1e-5)\n    def forward(self, x):\n        return self.norm(self.fc(x))\n"
                init.append(f"self.{a} = Block({w})"); body.append(f"x = self.{a}(x)")
            elif family in {"audio_encoder", "speech_conv_recurrent"}:
                shape = [2, w, 8]; conv(a, dim=1)
                body.append(f"x = self.{a}(x)")
                if family == "speech_conv_recurrent":
                    init.append(f"self.{b} = nn.GRU({w}, {w}, batch_first=True)"); body.append(f"x = self.{b}(x.transpose(1, 2))[0].transpose(1, 2)")
            else:
                linear(a)
                expr = {"fourier_operator": f"self.{a}(torch.fft.irfft(torch.fft.rfft(x, dim=-1), n={w}, dim=-1))", "siren": f"torch.sin(3.0 * self.{a}(x))", "rbf": f"torch.exp(-self.{a}(x).square())", "neural_ode_step": f"x + 0.05 * torch.tanh(self.{a}(x))", "spectral_conv": f"self.{a}(torch.fft.fft(x, dim=-1).real)"}[family]
                body.append(f"x = {expr}")
    init.extend(["self.activation = nn.ReLU()", "self.dropout = nn.Dropout(0.1)", "self.scale = 1.0", "self.unused = 7"])
    body.extend(["x = self.activation(x)", "return self.dropout(x)"])
    imports = "import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n" + ("from blocks import Block\n" if extras else "")
    source = imports + f"\nclass {cls}(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
    source += "\n".join("        " + line for line in init)
    source += "\n\n    def forward(self, x):\n" + "\n".join("        " + line for line in body) + "\n"
    return {"model.py": source, **extras}, {"shape": shape, "dtype": dtype}


def revisions(files, kind):
    before = dict(files)
    s = before["model.py"]
    if kind in {"scalar_attribute", "scalar_literal", "operand_order"}:
        s = s.replace("return self.dropout(x)", "return self.dropout(x) * self.scale" if kind == "scalar_attribute" else "return self.dropout(x) * 1.0" if kind == "scalar_literal" else "return self.dropout(x) - 2.0")
    if kind == "functional_activation": s = s.replace("x = self.activation(x)", "x = torch.relu(x)")
    if kind == "helper_body":
        s = s.replace("return self.dropout(x)", "return self.finish(self.dropout(x))") + "\n    def finish(self, x):\n        return torch.relu(x)\n"
    if kind == "branch_condition":
        s = s.replace("return self.dropout(x)", "if self.training:\n            return self.dropout(x)\n        return x")
    # A child file in every child-parameter case tests an actual imported edit.
    if kind == "child_parameter" and "blocks.py" not in before:
        before["review_child.py"] = "from torch import nn\nclass ReviewChild(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.dropout = nn.Dropout(0.1)\n    def forward(self, x):\n        return self.dropout(x)\n"
        s = "from review_child import ReviewChild\n" + s.replace("self.dropout = nn.Dropout(0.1)", "self.dropout = ReviewChild()")
    before["model.py"] = s
    after = dict(before)
    unchanged = kind in {"comments", "import_alias", "keyword_equivalence", "explicit_default", "constructor_reorder", "unused_attribute"}
    target, parameter, old, new = "dropout", None, None, None
    if kind == "comments": s = "# Review corpus: source commentary only.\n\n" + s.replace("    def forward", "    # The computation is unchanged.\n    def forward")
    elif kind == "import_alias": s = s.replace("from torch import nn", "from torch import nn as layers").replace("nn.", "layers.")
    elif kind == "keyword_equivalence": s = s.replace("nn.Dropout(0.1)", "nn.Dropout(p=0.1)")
    elif kind == "explicit_default": s = s.replace("nn.Dropout(0.1)", "nn.Dropout(0.1, inplace=False)")
    elif kind == "constructor_parameter":
        tree = ast.parse(s)
        call = next((n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr in {"Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose2d", "Bilinear", "LSTM", "GRU", "RNN", "LSTMCell", "GRUCell"}), None)
        if call is None:
            s = s.replace("nn.Dropout(0.1)", "nn.Dropout(0.3)"); parameter, old, new = "p", 0.1, 0.3
        else:
            line = s.splitlines()[call.end_lineno - 1]
            lines = s.splitlines(keepends=True)
            lines[call.end_lineno - 1] = line[:call.end_col_offset-1] + ", bias=False" + line[call.end_col_offset-1:] + "\n"
            s = "".join(lines)
            assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and n.value is call)
            target = assignment.targets[0].attr; parameter, old, new = "bias", True, False
    elif kind == "activation_replacement": s = s.replace("nn.ReLU()", "nn.GELU()"); target = "activation"
    elif kind == "insert_layer": s = s.replace("self.dropout =", "self.adapter = nn.Tanh()\n        self.dropout =").replace("return self.dropout(x)", "return self.dropout(self.adapter(x))"); target = "adapter"
    elif kind == "remove_layer": s = s.replace("        self.dropout = nn.Dropout(0.1)\n", "").replace("self.dropout(x)", "x")
    elif kind == "output_rewire": s = s.replace("return self.dropout(x)", "return x")
    elif kind == "scalar_attribute": s = s.replace("self.scale = 1.0", "self.scale = 2.0"); parameter, old, new = "other", 1, 2
    elif kind == "scalar_literal": s = s.replace("self.dropout(x) * 1.0", "self.dropout(x) * 2.0")
    elif kind == "operand_order": s = s.replace("self.dropout(x) - 2.0", "2.0 - self.dropout(x)")
    elif kind == "duplicate_call": s = s.replace("x = self.activation(x)", "x = self.activation(x) + self.activation(x)")
    elif kind == "functional_activation": s = s.replace("x = torch.relu(x)", "x = torch.sigmoid(x)")
    elif kind == "helper_body": s = s.replace("return torch.relu(x)", "return torch.sigmoid(x)")
    elif kind == "constructor_reorder": s = s.replace("self.activation = nn.ReLU()\n        self.dropout = nn.Dropout(0.1)", "self.dropout = nn.Dropout(0.1)\n        self.activation = nn.ReLU()")
    elif kind == "unused_attribute": s = s.replace("self.unused = 7", "self.unused = 8")
    elif kind == "module_rename": s = s.replace("self.activation", "self.nonlinearity"); target = "nonlinearity"
    elif kind == "branch_condition": s = s.replace("if self.training:", "if not self.training:")
    elif kind == "child_parameter":
        child = "blocks.py" if "blocks.py" in after else "review_child.py"
        after[child] = after[child].replace("eps=1e-5", "eps=1e-3").replace("nn.Dropout(0.1)", "nn.Dropout(0.3)")
    else: raise ValueError(kind)
    after["model.py"] = s
    assert before != after, kind
    return before, after, {"unchanged": unchanged, "target": target, "parameter": parameter, "old": old, "new": new}


def validate(files, cls, inputs):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(7301)
    previous = {}
    try:
        for file, source in files.items():
            if file == "model.py": continue
            name = file.removesuffix(".py")
            previous[name] = sys.modules.get(name)
            module = types.ModuleType(name)
            exec(compile(source, file, "exec"), module.__dict__)
            sys.modules[name] = module
        namespace = {}
        exec(compile(files["model.py"], "model.py", "exec"), namespace)
        net = namespace[cls]().eval()
        dtype = getattr(torch, inputs["dtype"])
        x = torch.ones(inputs["shape"], dtype=dtype) if dtype == torch.int64 else torch.linspace(-0.1, 0.1, int(torch.tensor(inputs["shape"]).prod())).reshape(inputs["shape"])
        with torch.inference_mode(): out = net(x)
        assert isinstance(out, torch.Tensor) and torch.isfinite(out).all(), "Output must be a finite tensor"
        return {"parameters": sum(p.numel() for p in net.parameters()), "outputShape": list(out.shape), "outputDigest": hashlib.sha256(out.contiguous().numpy().tobytes()).hexdigest()}
    finally:
        for name, old in previous.items():
            if old is None: sys.modules.pop(name, None)
            else: sys.modules[name] = old


def cases():
    number = 0
    for domain, families in FAMILIES.items():
        for family in families.split():
            family_index = number // 10
            for size, (width, depth) in enumerate(SIZES):
                number += 1
                cls = f"Review{''.join(part.title() for part in family.split('_'))}{size+1}"
                kind = MUTATIONS[(family_index + 3 * size) % len(MUTATIONS)]
                files, inputs = architecture(family, width, depth, cls)
                before, after, oracle = revisions(files, kind)
                yield {"id": f"{number:04d}-{family}-{size+1}", "family": family, "domain": domain, "size": size+1,
                       "width": width, "depth": depth, "cls": cls, "kind": kind, "inputs": inputs, "oracle": oracle,
                       "before": before, "after": after}
    assert number == 1000, number
