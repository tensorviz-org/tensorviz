"""Bounded lexical availability joined to exact calls in a straight-line forward.

This does not execute source or infer availability from canvas order. Unsupported
control flow, unknown calls, mutations or incomplete observations refuse proof.
"""
import ast
from dataclasses import dataclass

from .contracts import ValueRef, IntLiteral, BoolLiteral, NoneLiteral, TextLiteral, FloatLiteral, SequenceArgument
from .source_bindings import source_binding

MAX_STATEMENTS = 128
MAX_CALLS = 128
MAX_CHOICES = 128
MAX_STATES = 8192
BINARY_TARGETS = {ast.Add: "aten.add.Tensor", ast.Sub: "aten.sub.Tensor", ast.Mult: "aten.mul.Tensor", ast.Div: "aten.div.Tensor"}


class FlowRefused(ValueError):
    pass


@dataclass(frozen=True)
class Available:
    value: object
    definition: ast.AST


class StraightLineFlow:
    def __init__(self, resolver, expression, node, *, deferred_argument=None):
        self.resolver, self.expression, self.target_node = resolver, expression, node
        # Candidate checks may compare one selected binary operand after proving
        # its lexical value. Edit authorization always checks every argument.
        self.deferred_argument = deferred_argument
        self.anchor = expression.anchors[0]
        self.index = resolver.indexes[self.anchor.file]
        parents = {child: parent for parent in ast.walk(self.index.tree) for child in ast.iter_child_nodes(parent)}
        function = parents.get(node)
        while function is not None and not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = parents.get(function)
        if (not isinstance(function, ast.FunctionDef) or function.name != "forward" or function.decorator_list
                or not isinstance(parents.get(function), ast.ClassDef) or function.args.vararg or function.args.kwarg
                or not (function.args.posonlyargs + function.args.args)):
            raise FlowRefused("unsupported_source_scope")
        self.function = function
        body = function.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and type(body[0].value.value) is str:
            body = body[1:]
        if not body or len(body) > MAX_STATEMENTS or not isinstance(body[-1], ast.Return):
            raise FlowRefused("unsupported_source_scope")
        for statement in body[:-1]:
            if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
                    or isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.value is not None):
                raise FlowRefused("unsupported_source_scope")
        self.body = body
        definition = self.binding(function, "definition")
        if definition is None or self.index.definition_nodes.get(definition["id"], (None,))[0] != definition:
            raise FlowRefused("source_binding_mismatch")
        definitions = {item.id for item in resolver.capture.definitions if item.sourceBindingIds == [definition["id"]]}
        calls = [call for call in resolver.capture.invocations if resolver.modules[call.moduleId].definitionId in definitions]
        if not calls or len(calls) > MAX_CALLS or resolver.capture.coverage.omissions:
            raise FlowRefused("incomplete_source_flow")
        self.states, self.events = {}, {}
        total = 0
        for call in calls:
            state, event = self.analyse(call)
            total += len(state)
            if len(state) > MAX_CHOICES or total > MAX_STATES:
                raise FlowRefused("source_flow_limit")
            self.states[call.id], self.events[call.id] = state, event

    def binding(self, node, kind="argument"):
        return source_binding(self.resolver.capture.source.id, self.resolver.files[self.anchor.file],
            self.resolver.sources[self.anchor.file].split("\n"), node, kind, self.anchor.scope)

    def analyse(self, call):
        resolver = self.resolver
        if call.status != "completed":
            raise FlowRefused("incomplete_source_flow")
        descendants, pending = set(), [call.id]
        while pending:
            id = pending.pop()
            descendants.add(id)
            pending.extend(child.id for child in resolver.child_calls[id])
        if any(op.effects or op.status != "completed" for id in descendants for op in resolver.by_invocation[id]):
            raise FlowRefused("mutating_source_scope")
        positional = self.function.args.posonlyargs + self.function.args.args
        parameters = positional[1:] + self.function.args.kwonlyargs
        if len(call.arguments.args) > len(positional) - 1:
            raise FlowRefused("ambiguous_source_parameters")
        env, used_keywords = {}, set()
        for index, parameter in enumerate(parameters):
            if index < len(call.arguments.args):
                if parameter.arg in call.arguments.kwargs:
                    raise FlowRefused("ambiguous_source_parameters")
                value = call.arguments.args[index]
            elif parameter.arg in call.arguments.kwargs and parameter not in self.function.args.posonlyargs:
                used_keywords.add(parameter.arg)
                value = call.arguments.kwargs[parameter.arg]
            else:
                # Default objects can be changed after definition; omitted values are not proof.
                raise FlowRefused("ambiguous_source_parameters")
            env[parameter.arg] = Available(value, parameter)
        if used_keywords != set(call.arguments.kwargs):
            raise FlowRefused("ambiguous_source_parameters")
        direct_ops = resolver.by_invocation[call.id]
        direct_calls = resolver.child_calls[call.id]
        used_ops, used_calls = set(), set()
        last_sequence = call.sequence
        selected_state, selected_event = None, None

        def event_for(node):
            binding = self.binding(node, "call")
            if binding is None:
                raise FlowRefused("source_flow_limit")
            ops = [op for op in direct_ops if op.sourceBindingIds == [binding["id"]]]
            children = [child for child in direct_calls if child.sourceBindingIds == [binding["id"]]]
            if len(ops) + len(children) != 1:
                raise FlowRefused("ambiguous_source_flow")
            return ops[0] if ops else children[0], bool(children)

        def evaluate(node, depth=0):
            nonlocal last_sequence, selected_state, selected_event
            if depth > 12:
                raise FlowRefused("source_flow_limit")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in env:
                    raise FlowRefused("unavailable_source_value")
                if node is self.target_node:
                    # A deleted unary call leaves a name use with no dispatcher
                    # event. Its completed invocation and full surviving flow
                    # are the witness; absence of the old operation is not.
                    selected_state, selected_event = dict(env), call
                return env[node.id].value
            if isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) <= MAX_CHOICES:
                return SequenceArgument(kind="tuple" if isinstance(node, ast.Tuple) else "list",
                    items=[evaluate(item, depth + 1) for item in node.elts])
            if isinstance(node, ast.Constant):
                if node.value is None:
                    return NoneLiteral(kind="none")
                if type(node.value) is bool:
                    return BoolLiteral(kind="bool", value=node.value)
                if type(node.value) is str and len(node.value) <= 4096:
                    return TextLiteral(kind="str", value=node.value)
            number = self.resolver.source_number(node)
            if number is not None:
                return FloatLiteral.model_validate(number.model_dump()) if number.kind == "float" else number
            if not isinstance(node, (ast.BinOp, ast.Call)):
                raise FlowRefused("unsupported_source_expression")
            event, child = event_for(node)
            if node is self.target_node:
                selected_state, selected_event = dict(env), event
            if isinstance(node, ast.BinOp):
                # Nested computation has its own event and cannot masquerade as a name use.
                args = [evaluate(node.left, depth + 1), evaluate(node.right, depth + 1)]
                same_arguments = len(event.arguments.args) == len(args) and all(
                    actual == expected or node is self.target_node and index == self.deferred_argument
                    for index, (actual, expected) in enumerate(zip(event.arguments.args, args)))
                if (child or event.target != BINARY_TARGETS.get(type(node.op)) or not same_arguments
                        or event.arguments.kwargs and not (isinstance(node.op, (ast.Add, ast.Sub))
                            and event.arguments.kwargs == {"alpha": IntLiteral(kind="int", value="1")})):
                    raise FlowRefused("ambiguous_source_flow")
            elif child:
                if (not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name)
                        or node.func.value.id != positional[0].arg or any(isinstance(arg, ast.Starred) for arg in node.args)
                        or any(kw.arg is None for kw in node.keywords)):
                    raise FlowRefused("unsupported_source_expression")
                args = [evaluate(arg, depth + 1) for arg in node.args]
                kwargs = {kw.arg: evaluate(kw.value, depth + 1) for kw in node.keywords}
                if event.arguments.args != args or event.arguments.kwargs != kwargs:
                    raise FlowRefused("ambiguous_source_flow")
            else:
                # Existing activation proof includes import identity and source options.
                try:
                    cap = resolver.activation_capability(event.id)
                except ValueError:
                    raise FlowRefused("unsupported_source_expression") from None
                input_node = node.args[0] if node.args else next(kw.value for kw in node.keywords if kw.arg == "input")
                if cap.options.inplace or event.arguments.args != [evaluate(input_node, depth + 1)]:
                    raise FlowRefused("ambiguous_source_flow")
            if not isinstance(event.output, ValueRef) or event.sequence <= last_sequence:
                raise FlowRefused("ambiguous_source_flow")
            last_sequence = event.endSequence if child else event.sequence
            (used_calls if child else used_ops).add(event.id)
            return event.output

        for statement in self.body:
            if isinstance(statement, ast.Return):
                output = NoneLiteral(kind="none") if statement.value is None else evaluate(statement.value)
                if output != call.output:
                    raise FlowRefused("ambiguous_source_flow")
            else:
                value = evaluate(statement.value)
                target = statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
                if target.id == positional[0].arg:
                    raise FlowRefused("ambiguous_source_parameters")
                env[target.id] = Available(value, target)
        if (selected_state is None or selected_event is None or used_ops != {op.id for op in direct_ops}
                or used_calls != {child.id for child in direct_calls}):
            raise FlowRefused("incomplete_source_flow")
        return selected_state, selected_event
