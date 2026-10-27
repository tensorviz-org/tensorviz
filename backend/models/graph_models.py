"""Core data models for graph representation"""
from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LayerType(BaseModel):
    id: str
    name: str
    parameters: List[Dict[str, Any]]


class LayerData(BaseModel):
    layerType: LayerType
    parameters: Dict[str, Any]
    # Hierarchical fields
    parentId: Optional[str] = None
    childNodeIds: List[str] = []
    depth: int = 0
    moduleType: Optional[str] = None  # 'container', 'custom', 'builtin'
    internalGraph: Optional[Dict[str, Any]] = None
    # Whether a moduleNode should render expanded by default. Read by the
    # frontend (hierarchicalToFlat.applyModuleMeta) off node.data.expanded.
    expanded: Optional[bool] = None


class Node(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: LayerData


class Edge(BaseModel):
    id: str
    source: str
    target: str
    data: Optional[Dict[str, Any]] = None


class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


class ModuleDefinition(BaseModel):
    """Internal structure of a module"""
    module_id: str
    parent_id: Optional[str]
    internal_nodes: List[Node] = []
    internal_edges: List[Edge] = []
    input_node_id: Optional[str] = None  # Entry point (single-input convention)
    output_node_id: Optional[str] = None  # Exit point
    parameters: Dict[str, Any] = {}
    # ── Modular-codegen fields (additive; set by services/graph_modularizer) ──
    # When True, `internal_nodes` describe ONE canonical block that the main
    # model instantiates `parameters["repeat_count"]` times as an nn.ModuleList
    # and runs in a forward loop — instead of the legacy behaviour where a
    # module with N materialised layers is emitted once. Existing templates
    # leave this False so their codegen is unchanged.
    is_modular_block: bool = False
    # External source ids that feed the canonical block, in forward-arg order
    # (primary/sequence input first, then conditioning inputs). Used to seed the
    # block forward's var map.
    input_node_ids: List[str] = []
    # Forward-arg names aligned with `input_node_ids` (e.g. ["x", "cond0"]).
    input_arg_names: List[str] = []
    # Root-level source ids that carry the conditioning args (aligned with
    # input_arg_names[1:]); the remaining incoming edge to the module node is the
    # primary input. Lets the main forward route the loop's extra args.
    conditioning_source_ids: List[str] = []
    # Boundary IN edges of the canonical block (external source -> internal
    # target) — used to wire the block forward's args to its internal nodes.
    boundary_edges: List[Edge] = []


class HierarchicalGraph(BaseModel):
    """Complete hierarchical graph structure"""
    root_nodes: List[Node]
    root_edges: List[Edge]
    modules: Dict[str, ModuleDefinition] = {}  # module_id -> definition
    node_hierarchy: Dict[str, str] = {}  # node_id -> parent_module_id


class CodeParseRequest(BaseModel):
    """Request model for code parsing"""
    code: str
    model_name: Optional[str] = None
    input_shape: Optional[List[int]] = None
    # The shape a PREVIOUS trace of this class measured, handed back by the caller
    # that kept it (the extension's `.tensorviz/views/<rel>#<Class>.view.json`).
    # Deliberately NOT `input_shape`: that field OUTRANKS `# tensorviz: input=`,
    # and routing a remembered cache through it would make the sidecar silently
    # beat the line the user just typed. This one sits one rung lower — below the
    # declaration, above AST discovery. See tests/graph/test_measured_input_shape.py.
    measured_input_shape: Optional[List[int]] = None
    # Where `code` came from, when the caller knows (the IDE always does; the web
    # paste box does not). Written onto every node as `_source_file` so a node can
    # be traced back to the line that declares it. The parser never opens it — the
    # source it parses is `code` and nothing else.
    file: Optional[str] = Field(default=None, max_length=4096)


# ── Model-plane request envelopes ────────────────────────────────────────────
# `graph` deliberately stays a raw mapping. Every consumer takes a dict —
# services.verification_service.verify_model, services.onnx_export.export_onnx,
# services.graph_modularizer.modularize_for_codegen and
# server_v2._convert_to_hierarchical_graph all call .get() on it — so handing them
# a parsed pydantic instance would break each one with AttributeError. Typing the
# *envelope* is what these models are for: it turns a non-object `graph`, a
# non-integer `opset` and an unbounded `input_shape` into a 422 with a field name
# instead of a 500 carrying raw exception text.
#
# Extra keys stay ignored (pydantic's default) on purpose: the template gallery
# posts a `model_name` these endpoints never read, and extra="forbid" would 422 it.

# A tensor shape, bounded. 8 axes is two more than any real model uses (NCDHW is
# 5), and a single axis above a million describes a tensor no dry forward pass can
# allocate. Both are rejections, never truncations. No min_length: an empty list
# is how callers say "use the default", handled by `or` at the call sites.
InputShape = Annotated[List[Annotated[int, Field(ge=1, le=1_000_000)]], Field(max_length=8)]

# A generated class name becomes a Python identifier in the emitted file; 128
# characters is already far past anything readable.
_MAX_CLASS_NAME = 128


class GraphPayloadRequest(BaseModel):
    """Body shared by the model-plane endpoints: { graph, input_shape? }."""
    graph: Dict[str, Any]
    input_shape: Optional[InputShape] = None


class GenerateCodeRequest(GraphPayloadRequest):
    class_name: Optional[str] = Field(default=None, max_length=_MAX_CLASS_NAME)


class ExportCodeRequest(GenerateCodeRequest):
    """/api/code/export forwards its whole body to generate_code, so it has to BE
    a GenerateCodeRequest — hence the inheritance rather than a sibling model."""
    format: str = Field(default="pytorch", max_length=32)


class VerifyModelRequest(GraphPayloadRequest):
    run_forward: bool = True


class ExportPackageRequest(GraphPayloadRequest):
    class_name: Optional[str] = Field(default=None, max_length=_MAX_CLASS_NAME)
    model_name: Optional[str] = Field(default=None, max_length=_MAX_CLASS_NAME)


class ExportOnnxRequest(ExportPackageRequest):
    # torch.onnx accepts roughly 7-20; the window is deliberately loose so a new
    # torch can raise the ceiling, but an out-of-range opset is the caller's
    # mistake and must answer 4xx rather than escaping as an unhandled 500.
    opset: int = Field(default=17, ge=7, le=23)


class CustomizeModelRequest(BaseModel):
    base_graph: Dict[str, Any]
    customizations: List[Dict[str, Any]] = []


class InstantiateTemplateRequest(BaseModel):
    config: Dict[str, Any] = {}
    instance_id: Optional[str] = Field(default=None, max_length=128)


class InstantiatePopularModelRequest(BaseModel):
    """`customizations` is the semantic key; `config` is the legacy one the
    template gallery still sends. Whichever is present wins, in that order."""
    customizations: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = {}


class InstantiateModuleTemplateRequest(BaseModel):
    parameters: Dict[str, Any] = {}
    instance_id: Optional[str] = Field(default=None, max_length=128)
