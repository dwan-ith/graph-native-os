"""
compiler/passes/fusion.py

Operator Fusion Pass.

Identifies producer-consumer chains that can be executed as a single fused
kernel, reducing intermediate arena traffic.

Supported fusion patterns (each has a matching kernel in the runtime):
  Conv   -> Relu      => Conv_Relu
  Gemm   -> Relu      => Gemm_Relu
  MatMul -> Add       => MatMul_Add     (only when the Add's other operand
                                         is a rank<=1 constant bias vector;
                                         general adds stay unfused so the
                                         broadcasting kernel handles them)
  Add    -> Relu      => Add_Relu

A fusion is valid if and only if:
  1. The shared intermediate tensor has exactly one consumer.
  2. The shared tensor is NOT a graph-level output (rewiring would
     disconnect the model's external interface).
  3. The fused op type has a kernel implementation (enforced by the rule
     table itself — no rule may name a nonexistent kernel).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from compiler.frontend.ir import Graph, Op


# Each entry: (producer_type, consumer_type) → fused_type
# INVARIANT: every fused_type must have kernel_<lower> in kernels.h.
_FUSION_RULES: Dict[Tuple[str, str], str] = {
    ("Conv",   "Relu"):  "Conv_Relu",
    ("Gemm",   "Relu"):  "Gemm_Relu",
    ("MatMul", "Add"):   "MatMul_Add",
    ("Add",    "Relu"):  "Add_Relu",
}


def _build_consumer_count(graph: Graph) -> Dict[str, int]:
    """Count how many ops consume each tensor."""
    counts: Dict[str, int] = {name: 0 for name in graph.tensors}
    for op in graph.ops:
        for tname in op.inputs:
            if tname and tname in counts:
                counts[tname] += 1
    return counts


def _is_bias_add(graph: Graph, consumer: Op, shared_tensor: str) -> bool:
    """True when every non-shared input of an Add is a small constant vector,
    i.e. the pattern the MatMul_Add epilogue kernel implements exactly."""
    for t in consumer.inputs:
        if t == shared_tensor or not t:
            continue
        tensor = graph.tensors.get(t)
        if tensor is None or not tensor.is_constant:
            return False
        if len(tensor.shape) > 1:
            return False
    return True


def run(graph: Graph) -> Graph:
    """Fuse eligible producer-consumer op pairs.  Mutates graph in-place."""
    changed = True
    while changed:
        changed = False
        consumer_count = _build_consumer_count(graph)
        fused_indices: Set[int] = set()
        new_ops: List[Op] = []

        for idx, op in enumerate(graph.ops):
            if idx in fused_indices:
                continue

            if len(op.outputs) != 1:
                new_ops.append(op)
                continue

            shared_tensor = op.outputs[0]
            if consumer_count.get(shared_tensor, 0) != 1:
                new_ops.append(op)
                continue
            if shared_tensor in graph.outputs:
                # Never dissolve a tensor the outside world reads directly.
                new_ops.append(op)
                continue

            consumer_idx = next(
                (i for i, o in enumerate(graph.ops)
                 if i > idx and shared_tensor in o.inputs),
                None,
            )
            if consumer_idx is None or consumer_idx in fused_indices:
                new_ops.append(op)
                continue

            consumer = graph.ops[consumer_idx]
            rule_key = (op.op_type, consumer.op_type)
            fused_type = _FUSION_RULES.get(rule_key)
            if fused_type is None:
                new_ops.append(op)
                continue
            if fused_type == "MatMul_Add" and not _is_bias_add(graph, consumer, shared_tensor):
                new_ops.append(op)
                continue

            fused_inputs = list(op.inputs) + [
                t for t in consumer.inputs if t != shared_tensor
            ]
            fused_outputs = list(consumer.outputs)
            fused_attrs = {**op.attrs, **{f"consumer_{k}": v for k, v in consumer.attrs.items()}}

            fused_op = Op(
                op_id=op.op_id,
                op_type=fused_type,
                inputs=fused_inputs,
                outputs=fused_outputs,
                attrs=fused_attrs,
                fused_from=[op.op_type, consumer.op_type],
            )

            new_ops.append(fused_op)
            fused_indices.add(consumer_idx)
            changed = True

        graph.ops = new_ops

    return graph
