"""Source-only model candidates with declaration identities and scoped bindings.

Conditional alternatives are candidates, never proof of an executed branch. This
index drives discovery only; source projection and capture retain their stricter
resolution and runtime type checks.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from typing import NamedTuple

from core.op_registry import all_specs
from services.ide.module_index import ModuleIndex, module_of_file, package_of_file


class _Ref(NamedTuple):
    module: str
    name: str = ''
    line: int = 0  # A nonzero line identifies a concrete class, not an import.


TORCH_MODULE_BASES = {'torch.nn.Module', 'torch.nn.modules.module.Module',
            'torch.nn.Sequential', 'torch.nn.ModuleList', 'torch.nn.ModuleDict'} | {
            'torch.' + spec.canonical for spec in all_specs() if spec.canonical.startswith('nn.')}

# Documented PyTorch wrapper bases, not arbitrary classes with these short names.
# Discovery must work without importing/installing an optional training library.
# This catalog admits candidates only; execution still checks torch.nn.Module.
# Exact exports and upstream inheritance references: docs/inheritance-discovery.md.
EXTERNAL_MODULE_BASES = {
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
}


def module_declarations(sources: dict[str, str], *, framework_bases=None, index=None) -> set[tuple[str, int, str]]:
    """Return (file, class line, name), without importing any supplied source.

    Pass ``index`` to reuse a ModuleIndex already built from exactly these sources.
    """
    if index is None:
        index = ModuleIndex.build(sources)
    exports, parents, locations = {}, {}, {}

    for file, source in sources.items():
        module = module_of_file(file)
        record = index.modules[module]
        if record.syntax_error or record.file != file or any(
                note['code'] == 'AMBIGUOUS_BINDING' for note in record.import_diagnostics):
            continue
        # Preserve the original bare nn.Module/Module shorthand for editor snippets.
        # An explicit import, assignment or declaration always replaces it.
        namespace = {'nn': {_Ref('torch.nn')}, 'torch': {_Ref('torch')},
                     'Module': {_Ref('torch.nn', 'Module')}}
        declared_names = set()

        class Visitor(ast.NodeVisitor):
            def bind(self, name, refs):
                namespace[name] = set(refs)
                declared_names.add(name)

            def value(self, node):
                if isinstance(node, ast.Name):
                    return namespace.get(node.id, set())
                if isinstance(node, ast.Attribute):
                    return {_Ref(ref.module, '.'.join(filter(None, (ref.name, node.attr))))
                            for ref in self.value(node.value) if not ref.line}
                if isinstance(node, ast.Subscript):
                    return self.value(node.value)
                return set()

            def visit_ClassDef(self, node):
                ref = _Ref(module, node.name, node.lineno)
                parents[ref] = set().union(*(self.value(base) for base in node.bases))
                locations[ref] = (file, node.lineno, node.name)
                self.bind(node.name, {ref})

            def visit_FunctionDef(self, node):
                self.bind(node.name, set())

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Name(self, node):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    self.bind(node.id, set())

            def visit_Attribute(self, node):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    root = node.value
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name):
                        self.bind(root.id, set())

            def visit_Assign(self, node):
                refs = self.value(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.bind(target.id, refs)
                    else:
                        self.visit(target)  # Unsupported unpacking invalidates its targets.

            def visit_AnnAssign(self, node):
                if node.value is not None:
                    self.visit_Assign(ast.Assign(targets=[node.target], value=node.value))

            def visit_Import(self, node):
                for alias in node.names:
                    self.bind(alias.asname or alias.name.split('.')[0],
                              {_Ref(alias.name if alias.asname else alias.name.split('.')[0])})

            def visit_ImportFrom(self, node):
                target = ModuleIndex._resolve_relative(node.module or '', node.level, package_of_file(file))
                for alias in node.names:
                    if alias.name == '*':
                        if record.unresolved_wildcard:
                            for name in list(namespace):
                                self.bind(name, set())
                        else:
                            for name, bindings in record.bindings.items():
                                if name != '*' and len(bindings) == 1:
                                    binding = bindings[0]
                                    if binding.line == node.lineno:
                                        self.bind(name, {_Ref(binding.target_module, binding.target_name or '')})
                    else:
                        self.bind(alias.asname or alias.name, {_Ref(target, alias.name)} if target is not None else set())

            def branches(self, branches):
                before, alternatives = dict(namespace), []
                for body in branches:
                    namespace.clear()
                    namespace.update(before)
                    for statement in body:
                        self.visit(statement)
                    alternatives.append(dict(namespace))
                namespace.clear()
                for name in set().union(*(branch.keys() for branch in alternatives)):
                    namespace[name] = set().union(*(branch.get(name, set()) for branch in alternatives))

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

        Visitor().visit(ast.parse(source))
        exports[module] = {name: namespace[name] for name in declared_names}

    shadowed_packages = {name.split('.')[0] for name in index.modules}
    if framework_bases is None:
        framework_bases = TORCH_MODULE_BASES | EXTERNAL_MODULE_BASES

    def targets(start):
        pending, seen, result = [start], set(), set()
        while pending:
            ref = pending.pop()
            if ref in seen:
                continue
            seen.add(ref)
            if ref.line:
                result.add(ref)
                continue
            qualified = '.'.join(filter(None, (ref.module, ref.name)))
            if qualified in framework_bases and qualified.split('.')[0] not in shadowed_packages:
                result.add(None)
                continue
            head, dot, tail = ref.name.partition('.')
            bindings = exports.get(ref.module, {}).get(head)
            if bindings is not None:
                for binding in bindings:
                    if dot:
                        if binding.line:
                            continue
                        binding = _Ref(binding.module, '.'.join(filter(None, (binding.name, tail))))
                    pending.append(binding)
            else:
                child = ref.module + '.' + head if ref.module else head
                if dot and child in index.modules:
                    pending.append(_Ref(child, tail))
        return result

    # Reverse reachability avoids recursion through long inheritance chains.
    children = defaultdict(set)
    for cls, bases in parents.items():
        for base in bases:
            for parent in targets(base):
                children[parent].add(cls)
    found, pending = set(), list(children[None])
    while pending:
        cls = pending.pop()
        if cls not in found:
            found.add(cls)
            pending.extend(children[cls])
    return {locations[cls] for cls in found if cls in exports[cls.module].get(cls.name, set())}
