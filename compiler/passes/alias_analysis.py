"""
compiler/passes/alias_analysis.py

Memory Aliasing and In-Place Update Pass.

Identify tensors that can safely share the exact physical memory pointer of
another tensor, eliminating duplicate arena allocations.

Two categories:
  1. Views (Zero-copy): Reshape, Flatten, Squeeze, Unsqueeze.
     The output tensor is a reinterpretation of the input tensor's bytes.
     Always safe to alias.

  2. In-place Updates: Element-wise ops like Relu, Clip, Add, Sub, Mul, Div.
     If the input tensor `X` is completely dead after this operation, the
     output tensor `Y` can safely map to the exact same physical memory.
     (This is a destructive alias — `Y` overwrites `X`).

This pass sets the `alias_of` property on tensors. The true physical size of a
tensor network is resolved by traversing `alias_of` chains to the root tensor.
"""

from __future__ import annotations

from typing import Dict

from compiler.frontend.ir import Graph

_VIEW_OPS = {"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity"}
# Unary ops: output element i depends only on input element i -> in-place safe.
_INPLACE_UNARY = {"Relu", "Clip", "Sigmoid", "Tanh", "Softmax", "Erf",
                  "LeakyRelu"}
# Binary ops: in-place is only safe when NO broadcasting happens, i.e. the
# written operand has exactly the output shape (otherwise a broadcast read
# pattern or a smaller written region corrupts the shared buffer).
_INPLACE_BINARY = {"Add", "Sub", "Mul", "Div", "Pow"}

def _get_root_alias(graph: Graph, name: str) -> str:
    """Follow alias chains to find the true root allocation."""
    curr = name
    while True:
        t = graph.tensors.get(curr)
        if t is None or not t.alias_of:
            break
        curr = t.alias_of
    return curr

def _build_last_use_map(graph: Graph) -> Dict[str, int]:
    """Map tensor_name -> op_id of its final consumer."""
    last_use: Dict[str, int] = {}
    for op in graph.ops:
        for inp in op.inputs:
            if inp:
                last_use[inp] = op.op_id
    # Graph outputs are "used" at infinity
    for out in graph.outputs:
        last_use[out] = 9999999
    return last_use

def run(graph: Graph) -> Graph:
    """Annotate tensors with alias_of to maximize memory reuse."""

    last_use = _build_last_use_map(graph)

    for op in graph.ops:
        if len(op.inputs) == 0 or len(op.outputs) == 0:
            continue

        out_name = op.outputs[0]
        out_t = graph.tensors[out_name]

        # Don't alias if the output is already a constant
        if out_t.is_constant:
            continue

        in_name = op.inputs[0]
        in_t = graph.tensors.get(in_name)
        if in_t is None or in_t.is_constant:
            continue

        # Case 1: Views (Zero-copy reshapes)
        # Transpose physically permutes bytes, so it needs its own buffer;
        # contiguous reinterpretations (Reshape/Flatten/Squeeze/Identity)
        # can alias the source directly.
        if op.op_type in _VIEW_OPS:
            out_t.alias_of = in_name
            continue

        # Case 2: In-place Elementwise Updates
        if op.op_type in _INPLACE_UNARY:
            # Check if this operator is the absolute LAST use of in_name
            if last_use.get(in_name, op.op_id) <= op.op_id:
                # Safe to destructively overwrite in_name
                # Only aliasing if byte sizes match to prevent buffer overflow
                if in_t.byte_size >= out_t.byte_size:
                    out_t.alias_of = in_name
        elif op.op_type in _INPLACE_BINARY:
            # Broadcasting makes destructive overwrite unsafe: require the
            # overwritten operand to have exactly the output shape.
            if (
                last_use.get(in_name, op.op_id) <= op.op_id
                and tuple(in_t.shape) == tuple(out_t.shape)
            ):
                out_t.alias_of = in_name

    return graph
