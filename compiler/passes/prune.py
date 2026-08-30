"""
compiler/passes/prune.py

Dead Tensor Elimination.

Removes every tensor that is unreachable from the live computation:
  - not an input/output of any remaining op,
  - not a graph-level input or output,
  - and not the memory root of a kept aliased (view / in-place) tensor.

Why it matters:
  * Fusion leaves orphan intermediates behind (e.g. T0 after
    MatMul+Add -> MatMul_Add).  Without pruning they are emitted into the
    descriptor table pointing at arena offset 0 — colliding with real
    allocations.  Harmless only by accident.
  * Constant folding leaves orphan weight blobs that would otherwise be
    baked into .rodata forever (e.g. absorbed BatchNorm parameters).
"""

from __future__ import annotations

from typing import Set

from compiler.frontend.ir import Graph


def run(graph: Graph) -> Graph:
    reachable: Set[str] = set()

    def mark(name: str) -> None:
        if name in reachable:
            return
        reachable.add(name)
        t = graph.tensors.get(name)
        if t is not None and t.alias_of:
            mark(t.alias_of)

    for op in graph.ops:
        for n in op.inputs:
            if n:
                mark(n)
        for n in op.outputs:
            if n:
                mark(n)

    for n in graph.inputs + graph.outputs:
        mark(n)

    # Keep alias ROOTS of everything reachable (mark() handles chains, but
    # also make sure intermediate links stay).
    graph.tensors = {name: t for name, t in graph.tensors.items() if name in reachable}
    return graph
