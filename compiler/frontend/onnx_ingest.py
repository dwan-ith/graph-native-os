"""
compiler/frontend/onnx_ingest.py

Converts an ONNX ModelProto → Graph IR.

Responsibilities:
  - Parse all initializers (weights/constants) into Tensor objects with .data set
  - Parse all value_info (activations) into Tensor objects with .data = None
  - Parse all nodes into Op objects with topologically-ordered op_ids
  - Infer shapes for any tensors not covered by the proto's value_info
  - Reject models with dynamic shapes or unsupported dtypes

Does NOT perform any graph optimization — that is the job of the pass pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import onnx
from onnx import numpy_helper, shape_inference, TensorProto

from compiler.frontend.ir import (
    Dtype,
    Graph,
    Op,
    Tensor,
    dtype_from_onnx,
)


# Ops whose semantics require control flow / subgraphs: fundamentally
# incompatible with a fully-static execution image.
_CONTROL_FLOW_OPS = {"If", "Loop", "Scan", "SequenceOps"}

_SUPPORTED_OPSETS = (13, 21)   # inclusive domain-version range we accept


def _validate_model_proto(model) -> None:
    """Reject models this compiler can never lower, with named diagnostics."""
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            if not (_SUPPORTED_OPSETS[0] <= entry.version <= _SUPPORTED_OPSETS[1]):
                raise ValueError(
                    f"Unsupported ONNX opset version {entry.version} "
                    f"(supported: {_SUPPORTED_OPSETS[0]}..{_SUPPORTED_OPSETS[1]})"
                )
        elif entry.domain not in ("ai.onnx.training",):
            raise ValueError(
                f"Unsupported operator domain '{entry.domain}' "
                "(custom/complement domains are not supported)"
            )

    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError(
                f"Initializer '{init.name}' uses external data format; "
                "models must be self-contained single files"
            )

    for node in model.graph.node:
        if node.op_type in _CONTROL_FLOW_OPS or node.domain not in ("", "ai.onnx"):
            raise ValueError(
                f"Unsupported node '{node.name}': op_type={node.op_type!r}, "
                f"domain={node.domain!r}. Control flow and custom domains "
                "cannot be lowered to a static image."
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_shape(type_proto) -> Tuple[int, ...]:
    """Extract shape tuple from an ONNX TypeProto.  Returns -1 for dynamic dims."""
    tensor_type = type_proto.tensor_type
    if not tensor_type.HasField("shape"):
        return (-1,)
    dims = []
    for d in tensor_type.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            dims.append(d.dim_value)
        else:
            dims.append(-1)
    return tuple(dims)


def _parse_dtype(type_proto) -> Dtype:
    elem_type = type_proto.tensor_type.elem_type
    return dtype_from_onnx(elem_type)


def _initializer_to_tensor(init: TensorProto) -> Tensor:
    """Convert an ONNX initializer (weight/bias) into a constant Tensor."""
    arr = numpy_helper.to_array(init)
    dtype = dtype_from_onnx(init.data_type)
    shape = tuple(int(d) for d in init.dims)
    # Store raw bytes; the code generator will emit these into .rodata
    return Tensor(
        name=init.name,
        dtype=dtype,
        shape=shape,
        data=arr.tobytes(),
    )


def _parse_attr(attr) -> object:
    """Convert a single ONNX AttributeProto to a plain Python value."""
    import onnx
    AT = onnx.AttributeProto
    if attr.type == AT.INT:
        return attr.i
    if attr.type == AT.FLOAT:
        return attr.f
    if attr.type == AT.STRING:
        return attr.s.decode("utf-8")
    if attr.type == AT.TENSOR:
        return numpy_helper.to_array(attr.t)
    if attr.type == AT.INTS:
        return list(attr.ints)
    if attr.type == AT.FLOATS:
        return list(attr.floats)
    if attr.type == AT.STRINGS:
        return [s.decode("utf-8") for s in attr.strings]
    return None  # unsupported attr type — silently ignore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(path: Path | str, input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None) -> Graph:
    """Load and parse an ONNX model file into the typed Graph IR.

    Args:
        path: Path to the .onnx file.
        input_shapes: Optional override for symbolic input shapes.
                      Maps tensor name → concrete shape tuple.
                      Required when the model has symbolic batch dimensions.

    Returns:
        A fully populated Graph with all tensors and ops.
        The ops list is in topological order.

    Raises:
        ValueError: on unsupported dtypes, dynamic shapes not resolved,
                    or structural issues in the proto.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    try:
        # load_external_data=False keeps EXTERNAL tensors as declarations;
        # _validate_model_proto then rejects them with a clear diagnostic
        # instead of failing deep inside the loader.
        model = onnx.load(str(path), load_external_data=False)
    except Exception as e:
        raise ValueError(f"Failed to load ONNX model '{path}': {e}") from e
    _validate_model_proto(model)

    # Concretize any symbolic dimensions the user provided
    if input_shapes:
        for inp in model.graph.input:
            if inp.name in input_shapes:
                shape = input_shapes[inp.name]
                t = inp.type.tensor_type
                for i, dim in enumerate(t.shape.dim):
                    dim.dim_value = shape[i]
                    if dim.HasField("dim_param"):
                        dim.ClearField("dim_param")

    # Run ONNX shape inference so all intermediate tensors have known shapes
    try:
        model = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
    except Exception as e:
        raise ValueError(f"ONNX shape inference failed: {e}") from e

    # Non-strict inference silently DROPS intermediates whose shapes could
    # not be inferred.  Surface that here with a named diagnostic instead of
    # a confusing 'unknown tensor' failure deep in the pipeline.
    inferred_names = {vi.name for vi in model.graph.value_info}
    graph_proto_early = model.graph
    known_names = (
        {init.name for init in graph_proto_early.initializer}
        | {inp.name for inp in graph_proto_early.input}
        | inferred_names
        | {out.name for out in graph_proto_early.output}
    )
    missing = [
        out for node in graph_proto_early.node
        for out in node.output
        if out and out not in known_names
    ]
    if missing:
        bad_op = next(
            (f"{n.op_type} (node '{n.name}')" for n in graph_proto_early.node
             for o in n.output if o in missing),
            "unknown op",
        )
        raise ValueError(
            f"Shape inference left intermediate tensor(s) {sorted(set(missing))[:5]} "
            f"unresolved; first failing op: {bad_op}. Check dimension compatibility."
        )

    graph_proto = model.graph
    tensors: Dict[str, Tensor] = {}

    # ---- 1. Initializers → constant tensors --------------------------------
    initializer_names = set()
    for init in graph_proto.initializer:
        t = _initializer_to_tensor(init)
        tensors[init.name] = t
        initializer_names.add(init.name)

    # ---- 2. Graph inputs → activation tensors (if not already an initializer)
    graph_inputs: List[str] = []
    for inp in graph_proto.input:
        if inp.name in initializer_names:
            continue  # already handled above as a constant
        dtype = _parse_dtype(inp.type)
        shape = _parse_shape(inp.type)
        tensors[inp.name] = Tensor(name=inp.name, dtype=dtype, shape=shape)
        graph_inputs.append(inp.name)

    # ---- 3. All intermediate & output value_info → activation tensors ------
    for vi in list(graph_proto.value_info) + list(graph_proto.output):
        if vi.name in tensors:
            continue
        dtype = _parse_dtype(vi.type)
        shape = _parse_shape(vi.type)
        tensors[vi.name] = Tensor(name=vi.name, dtype=dtype, shape=shape)

    graph_outputs: List[str] = [o.name for o in graph_proto.output]

    # ---- 4. Nodes → Ops (explicit topological sort)
    # Build edges: tensor_name -> list of node indices that consume it
    # Build in-degrees: node_idx -> number of inputs not yet produced

    node_list = list(graph_proto.node)
    produced_tensors = set(initializer_names).union(set(graph_inputs))

    ready_nodes = []
    in_degrees = {i: 0 for i in range(len(node_list))}
    consumers = {t_name: [] for t_name in tensors}

    for i, node in enumerate(node_list):
        pending_inputs = 0
        for inp in node.input:
            if inp and inp not in produced_tensors:
                pending_inputs += 1
                if inp not in consumers:
                    consumers[inp] = []
                consumers[inp].append(i)
        in_degrees[i] = pending_inputs
        if pending_inputs == 0:
            ready_nodes.append(i)

    sorted_nodes = []
    while ready_nodes:
        # Pop a ready node
        idx = ready_nodes.pop(0)
        node = node_list[idx]
        sorted_nodes.append(node)

        # Mark its outputs as produced
        for out in node.output:
            produced_tensors.add(out)
            if out in consumers:
                for cons_idx in consumers[out]:
                    in_degrees[cons_idx] -= 1
                    if in_degrees[cons_idx] == 0:
                        ready_nodes.append(cons_idx)

    if len(sorted_nodes) != len(node_list):
        raise ValueError(
            f"Topological sort failed: cycle or unresolved inputs "
            f"({len(sorted_nodes)}/{len(node_list)} resolved)"
        )

    ops: List[Op] = []
    for op_id, node in enumerate(sorted_nodes):
        attrs = {a.name: _parse_attr(a) for a in node.attribute}
        op = Op(
            op_id=op_id,
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attrs=attrs,
        )
        ops.append(op)

    g = Graph(
        name=graph_proto.name or path.stem,
        ops=ops,
        tensors=tensors,
        inputs=graph_inputs,
        outputs=graph_outputs,
    )
    g.validate()
    return g
