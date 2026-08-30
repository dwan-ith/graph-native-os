"""
compiler/memory/liveness.py

Tensor Liveness Analysis.

For each activation tensor in the graph, determines the half-open interval
[first_use, last_use] in terms of op_id.  Two tensors with overlapping live
ranges CANNOT share arena memory.

'first_use' = the op_id of the op that PRODUCES the tensor (writes it for the
              first time).  For graph-level inputs this is -1 (live from the start).
'last_use'  = the op_id of the LAST op that reads this tensor.  After this point
              the tensor is dead and its arena slot can be reused.

Constants / weights are excluded — they live in ROM, not in the arena.
"""

from __future__ import annotations

import dataclasses
from typing import Dict

from compiler.frontend.ir import Graph


@dataclasses.dataclass(order=True)
class LiveRange:
    """Half-open interval [first_def, last_use] for a single activation tensor."""
    tensor_name: str = dataclasses.field(compare=False)
    first_def:   int = 0   # op_id that produces tensor; -1 = graph input
    last_use:    int = 0   # op_id of last consumer; equal to first_def if unused

    @property
    def size(self) -> int:
        """Span of the live interval (in op-count units)."""
        return self.last_use - self.first_def

    def overlaps(self, other: "LiveRange") -> bool:
        """True if this interval and other are simultaneously live."""
        # Intervals [a, b] and [c, d] overlap iff a <= d and c <= b
        return self.first_def <= other.last_use and other.first_def <= self.last_use

    def __repr__(self) -> str:
        return (
            f"LiveRange({self.tensor_name!r}, "
            f"[{self.first_def}, {self.last_use}])"
        )


def _resolve_root(graph: Graph, name: str) -> str:
    """Follow alias_of chains to the physical root allocation."""
    seen = set()
    curr = name
    while graph.tensors.get(curr) is not None and graph.tensors[curr].alias_of:
        if curr in seen:
            raise RuntimeError(f"Alias cycle detected at '{curr}'")
        seen.add(curr)
        curr = graph.tensors[curr].alias_of
    return curr


def compute_liveness(graph: Graph) -> Dict[str, LiveRange]:
    """Return a mapping from tensor name → LiveRange for all activation tensors.

    Algorithm:
      Forward pass over the topologically-ordered op list:
        - When a tensor first appears as an output of op N → first_def = N
        - For every op M that reads a tensor → last_use = max(last_use, M)
      Graph-level input tensors start with first_def = -1.
      Graph-level output tensors are kept alive until the last op (they
      can't be freed before the caller reads them).

    Complexity: O(|ops| × |avg_inputs_per_op|) — linear in graph size.
    """
    ranges: Dict[str, LiveRange] = {}

    # Initialise graph inputs
    for tname in graph.inputs:
        t = graph.tensors[tname]
        if not t.is_constant:
            ranges[tname] = LiveRange(tensor_name=tname, first_def=-1, last_use=-1)

    last_op_id = len(graph.ops) - 1

    for op in graph.ops:
        # Outputs: record when each tensor is first defined
        for tname in op.outputs:
            t = graph.tensors.get(tname)
            if t is None or t.is_constant:
                continue
            if tname not in ranges:
                ranges[tname] = LiveRange(tensor_name=tname,
                                          first_def=op.op_id,
                                          last_use=op.op_id)

        # Inputs: extend last_use to this op
        for tname in op.inputs:
            if not tname:   # ONNX allows empty input names for optional inputs
                continue
            t = graph.tensors.get(tname)
            if t is None or t.is_constant:
                continue
            if tname in ranges:
                ranges[tname].last_use = max(ranges[tname].last_use, op.op_id)

    # Graph-level outputs must stay alive through the end
    for tname in graph.outputs:
        if tname in ranges:
            ranges[tname].last_use = last_op_id

    # Resolve aliases: Fold the live range of alias tensors back into their root tensor.
    # We do this backwards or by following paths.
    for pass_ in range(len(ranges)): # Max depth is |V|
        changed = False
        for tname, lr in list(ranges.items()):
            t = graph.tensors.get(tname)
            if t and t.alias_of:
                root_name = _resolve_root(graph, tname)
                if root_name not in ranges:
                    # A view of a ROM constant needs no arena slot; anything
                    # else would silently lose its allocation.
                    if graph.tensors[root_name].is_constant:
                        del ranges[tname]
                        changed = True
                        continue
                    raise RuntimeError(
                        f"Liveness bug: alias root '{root_name}' of "
                        f"'{tname}' has no live range"
                    )
                root_lr = ranges[root_name]
                # Update root to span both
                if lr.last_use > root_lr.last_use:
                    root_lr.last_use = lr.last_use
                    changed = True
                if lr.first_def != -1 and (
                    root_lr.first_def == -1 or lr.first_def < root_lr.first_def
                ):
                    root_lr.first_def = lr.first_def
                    changed = True
                # Remove the alias from independent live ranges (it has no allocation)
                del ranges[tname]
                changed = True
        if not changed:
            break

    return ranges
