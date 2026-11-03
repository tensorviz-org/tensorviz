"""Which ``# tensorviz: input=`` comments belong to which model class.

Setup, the source-only drawing, the capture worker and the project graph all ask this
question, and while each answered it its own way one class could be drawn, run and
saved with three different inputs. This module is the one answer.

Standard library only, on purpose. The capture worker loads ``services/`` as a private
package with the backend root removed from ``sys.path`` (``bootstrap.py``) and imports
this module relatively, so an absolute ``services``, ``core``, ``utils`` or ``models``
import here would break automatic capture.
"""
from __future__ import annotations

import ast
import functools
import io
import re
import tokenize

#: ``# tensorviz: input=2,8`` (or ``2,8;2,8`` for two arguments). Only what
#: ``applicable`` reads from real comments is a declaration; the worker searches with
#: this pattern only for source Python cannot read.
INPUT_COMMENT = re.compile(r"#\s*tensorviz:\s*input\s*=\s*(?P<value>[^\n]+)")

#: The bases ``module_discovery`` proves are modules: torch.nn's containers, the layers
#: the op registry knows and the documented library bases. A copy, because the worker
#: cannot import that module; ``test_the_worker_counts_the_framework_bases_discovery_counts``
#: keeps the two equal.
MODULE_BASES = frozenset({'torch.nn.modules.module.Module'} | {'torch.nn.' + name for name in """
    AdaptiveAvgPool1d AdaptiveAvgPool2d AdaptiveAvgPool3d AdaptiveMaxPool1d AdaptiveMaxPool2d
    AdaptiveMaxPool3d AlphaDropout AvgPool1d AvgPool2d AvgPool3d BCELoss BCEWithLogitsLoss
    BatchNorm1d BatchNorm2d BatchNorm3d Bilinear CELU CTCLoss ChannelShuffle ConstantPad1d
    ConstantPad2d ConstantPad3d Conv1d Conv2d Conv3d ConvTranspose1d ConvTranspose2d
    ConvTranspose3d CosineEmbeddingLoss CrossEntropyLoss Dropout Dropout1d Dropout2d Dropout3d ELU
    Embedding EmbeddingBag Flatten Fold FractionalMaxPool2d GELU GLU GRU GRUCell GroupNorm
    Hardshrink Hardsigmoid Hardswish Hardtanh HuberLoss Identity InstanceNorm1d InstanceNorm2d
    InstanceNorm3d KLDivLoss L1Loss LPPool1d LPPool2d LSTM LSTMCell LayerNorm LazyLinear LeakyReLU
    Linear LocalResponseNorm LogSoftmax MSELoss MaxPool1d MaxPool2d MaxPool3d Mish Module
    ModuleDict ModuleList MultiheadAttention NLLLoss PReLU PixelShuffle PixelUnshuffle RMSNorm RNN
    RNNCell RReLU ReLU ReLU6 ReflectionPad1d ReflectionPad2d ReflectionPad3d ReplicationPad1d
    ReplicationPad2d ReplicationPad3d SELU Sequential SiLU Sigmoid SmoothL1Loss Softmax Softplus
    Softshrink Softsign Tanh Tanhshrink Threshold Transformer TransformerDecoderLayer
    TransformerEncoderLayer TripletMarginLoss Unflatten Unfold Upsample ZeroPad2d""".split()} | {
    'transformers.PreTrainedModel', 'transformers.modeling_utils.PreTrainedModel',
    'diffusers.ModelMixin', 'diffusers.models.modeling_utils.ModelMixin',
    'diffusers.models.modeling_utils.LegacyModelMixin',
    'lightning.LightningModule', 'lightning.pytorch.LightningModule',
    'lightning.pytorch.core.LightningModule',
    'lightning.pytorch.core.module.LightningModule',
    'pytorch_lightning.LightningModule', 'pytorch_lightning.core.LightningModule',
    'pytorch_lightning.core.module.LightningModule',
    'peft.PeftModel', 'peft.peft_model.PeftModel',
    'peft.PeftModelForCausalLM', 'peft.peft_model.PeftModelForCausalLM',
    'peft.PeftModelForSeq2SeqLM', 'peft.peft_model.PeftModelForSeq2SeqLM',
    'timm.models.vision_transformer.VisionTransformer', 'timm.models.resnet.ResNet',
    'torchvision.models.VisionTransformer', 'torchvision.models.vision_transformer.VisionTransformer',
    'torchvision.models.ResNet', 'torchvision.models.resnet.ResNet',
    'torch_geometric.nn.MessagePassing', 'torch_geometric.nn.conv.MessagePassing',
    'torch_geometric.nn.conv.message_passing.MessagePassing',
    'mmengine.model.BaseModule', 'mmengine.model.base_module.BaseModule',
    'mmengine.model.BaseModel', 'mmengine.model.base_model.BaseModel',
    'mmengine.model.base_model.base_model.BaseModel',
} | {
    f'{module}.{name}'
    for family, name in (
        ('bert', 'BertPreTrainedModel'), ('gpt2', 'GPT2PreTrainedModel'),
        ('llama', 'LlamaPreTrainedModel'), ('qwen2', 'Qwen2PreTrainedModel'),
        ('qwen3', 'Qwen3PreTrainedModel'), ('t5', 'T5PreTrainedModel'),
        ('vit', 'ViTPreTrainedModel'), ('whisper', 'WhisperPreTrainedModel'),
        ('wav2vec2', 'Wav2Vec2PreTrainedModel'), ('dinov2', 'Dinov2PreTrainedModel'),
    )
    for module in ('transformers', f'transformers.models.{family}',
                   f'transformers.models.{family}.modeling_{family}')
} | {
    f'{module}.{name}'
    for implementation, name in (
        ('unets.unet_2d', 'UNet2DModel'), ('unets.unet_2d_condition', 'UNet2DConditionModel'),
        ('autoencoders.autoencoder_kl', 'AutoencoderKL'),
        ('transformers.transformer_2d', 'Transformer2DModel'),
        ('transformers.transformer_flux', 'FluxTransformer2DModel'),
        ('transformers.transformer_sd3', 'SD3Transformer2DModel'),
    )
    for module in ('diffusers', 'diffusers.models', f'diffusers.models.{implementation}')
})


@functools.lru_cache(maxsize=1)
def _comments(source):
    """The ``(value, line)`` declarations, and the column of every comment on a line of its own.

    Cached because the drawing and the project graph ask about every class of one file
    in turn, and tokenizing a large file once per class cost more than drawing it.
    """
    # One pass that keeps only comments. Python 3.12.0-3.12.3 give every token its own
    # copy of its line, so holding all tokens of a long generated line ran out of memory.
    declarations, columns = [], {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        if match := re.fullmatch(r'#\s*tensorviz:\s*input\s*=(.*)', token.string):
            declarations.append((match.group(1).strip(), token.start[0]))
        if not token.line[:token.start[1]].strip():
            columns[token.start[0]] = token.start[1]
    return tuple(declarations), columns


def _file_modules(tree):
    """The ``(line, name)`` classes ``module_discovery`` proves are modules from this file alone.

    Its walk in small, for the worker, which cannot import it: a class counts when a
    base, read through the file's own imports and assignments as they stand at the
    class, is one of ``MODULE_BASES`` or another class that counts. A flax
    ``nn.Module`` does not. A base another file supplies (a relative or wildcard
    import) is not proven here; setup proves it from that file. The reverse needs the
    workspace too: a workspace with its own top-level package named like a framework
    (``torch/``, ``transformers/``) makes discovery skip that framework's bases, which
    this file alone still proves.
    """
    # The bare shorthand discovery keeps for editor snippets, until something rebinds it.
    namespace = {'nn': {'torch.nn'}, 'torch': {'torch'}, 'Module': {'torch.nn.Module'}}
    parents = {}
    # The index records only top-level imports. A wildcard import there names what
    # this file cannot see, so every wildcard import forgets every name; without
    # one, a wildcard import nested in a block binds nothing.
    wildcard = any(isinstance(node, ast.ImportFrom) and any(alias.name == '*' for alias in node.names)
                   for node in tree.body)

    class Walk(ast.NodeVisitor):
        def value(self, node):
            if isinstance(node, ast.Name):
                return namespace.get(node.id, set())
            if isinstance(node, ast.Attribute):
                return {f'{ref}.{node.attr}' for ref in self.value(node.value) if isinstance(ref, str)}
            return self.value(node.value) if isinstance(node, ast.Subscript) else set()

        def visit_ClassDef(self, node):
            parents[node] = set().union(*(self.value(base) for base in node.bases))
            namespace[node.name] = {node}

        def visit_FunctionDef(self, node):
            namespace[node.name] = set()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                namespace[node.id] = set()

        def visit_Attribute(self, node):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                root = node.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    namespace[root.id] = set()

        def visit_Assign(self, node):
            refs = self.value(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = set(refs)
                else:
                    self.visit(target)

        def visit_AnnAssign(self, node):
            if node.value is not None:
                self.visit_Assign(ast.Assign(targets=[node.target], value=node.value))

        def visit_Import(self, node):
            for alias in node.names:
                namespace[alias.asname or alias.name.split('.')[0]] = {alias.name if alias.asname else alias.name.split('.')[0]}

        def visit_ImportFrom(self, node):
            for alias in node.names:
                if alias.name == '*':
                    if wildcard:
                        namespace.update({name: set() for name in namespace})
                else:
                    namespace[alias.asname or alias.name] = set() if node.level else {f'{node.module}.{alias.name}'}

        def branches(self, bodies):
            before, after = dict(namespace), []
            for body in bodies:
                namespace.clear()
                namespace.update(before)
                for statement in body:
                    self.visit(statement)
                after.append(dict(namespace))
            namespace.clear()
            for name in set().union(*after):
                namespace[name] = set().union(*(branch.get(name, set()) for branch in after))

        def visit_If(self, node):
            self.branches([node.body, node.orelse])

        def visit_Try(self, node):
            self.branches([node.body + node.orelse, *(handler.body for handler in node.handlers)])
            for statement in node.finalbody:
                self.visit(statement)

        visit_TryStar = visit_Try

        def visit_For(self, node):
            self.visit(node.target)
            self.branches([node.body, []])
            for statement in node.orelse:
                self.visit(statement)

        visit_AsyncFor = visit_For

        def visit_While(self, node):
            self.branches([node.body, []])
            for statement in node.orelse:
                self.visit(statement)

    Walk().visit(tree)
    found, grown = set(), True
    while grown:
        grown = {node for node, bases in parents.items()
                 if node not in found and any(base in found or base in MODULE_BASES for base in bases)}
        found |= grown
    # A class whose name was rebound later is not the one the file exports.
    return {(node.lineno, node.name) for node in found if node in namespace.get(node.name, ())}


def applicable(source, cls, *, tree=None, proven=None):
    """``(own, stray)`` for the top-level class ``cls``.

    ``own`` is the ``(value, line)`` declarations that apply to ``cls``, in file order.
    ``stray`` is the line of the first declaration that applies to no class, or None.
    In a file with at most one model class, every declaration applies to it. Declaration
    text inside a string or docstring is not a comment, so it never counts.

    ``proven`` returns the ``(line, name)`` classes a module index proved are modules.
    It is called only for a file with a declaration and several classes. Without it
    (the capture worker has no index), the classes this file alone proves are modules
    count, as the index proves them from this file (``_file_modules``), and so does a
    subclass of any model class in this file. A class without forward whose module
    base only another file proves, and which subclasses no model here, does not.

    Raises SyntaxError or tokenize.TokenError for source Python cannot read, and
    RecursionError for source nested too deep for ast to build.
    """
    if 'tensorviz:' not in source:
        return [], None
    if '\r' in source:
        # Python reads "\r\n" and a lone "\r" as one line break, and numbers the tree's
        # lines so. Tokenizing line by line did not, and refused source ast accepts.
        source = source.replace('\r\n', '\n').replace('\r', '\n')
    declarations, comments = _comments(source)
    declarations = list(declarations)
    if not declarations:
        return [], None
    tree = tree if tree is not None else ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) <= 1:
        return declarations, None
    named = [node for node in classes if node.name == cls]
    definition = named[0] if len(named) == 1 else None

    def forward_model(node):
        return (any(ast.unparse(base) != 'object' for base in node.bases)
                and any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == 'forward'
                        and [arg.arg for arg in item.args.args[:1]] == ['self'] for item in node.body))

    if proven is not None:
        proved = proven()
    else:
        try:
            proved = _file_modules(tree)
        except RecursionError:
            # Its walk takes a frame per level of an expression or elif chain, and Python
            # imports one a few hundred levels deep. Prove nothing from such a file: its
            # forward models and their subclasses still count, as before the walk.
            proved = set()
    # Count the classes Open as Graph offers: proven modules, and subclasses of a base
    # the source index cannot prove that define their own forward(self, ...).
    counted = {node for node in classes if node is definition or (node.lineno, node.name) in proved or forward_model(node)}
    grown = proven is None
    while grown:
        # The file alone cannot prove a model whose module base another file supplies,
        # which setup proves, and with it that model's subclasses here. So the worker
        # counts a class that subclasses a model in this file, as the project graph does.
        names = {node.name for node in counted}
        grown = {node for node in classes if node not in counted and any(ast.unparse(base) in names for base in node.bases)}
        counted |= grown
    models = [node for node in classes if node in counted]
    if len(models) <= 1:
        return declarations, None
    # A model owns its body, indented comments after it, and the unindented comment
    # and decorator lines directly above it. Another model's declaration is no conflict.
    lines = source.split('\n')
    extents = []
    for node in models:
        start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        while comments.get(start - 1) == 0:
            start -= 1
        end = following = node.end_lineno
        while following < len(lines) and (comments.get(following + 1, 0) > 0 or not lines[following].strip()):
            following += 1
            end = following if following in comments else end
        extents.append((start, end, node))
    owners = [next((node for start, end, node in extents if start <= line <= end), None) for _, line in declarations]
    own = [item for item, owner in zip(declarations, owners) if definition is not None and owner is definition]
    # Elsewhere in the file, a declaration cannot say which model it is for.
    return own, declarations[owners.index(None)][1] if None in owners else None
