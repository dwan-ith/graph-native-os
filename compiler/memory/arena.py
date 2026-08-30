"""
compiler/memory/arena.py

Static Tensor Arena Allocator.

Given a set of live ranges (from liveness.py), assigns a byte offset within
a contiguous arena buffer to every activation tensor such that:

  1. No two tensors whose live ranges overlap share any byte.
  2. The total arena size is minimised (best-effort; optimal is NP-hard in general).

Algorithm: Greedy First-Fit by Decreasing Size (GFFD).
  - Sort tensors by byte_size descending.
  - For each tensor (in that order), find the lowest offset within the arena
    that does not conflict with any already-allocated tensor whose live range
    overlaps.
  - Assign that offset; record the allocation.

This is the same class of algorithm used by TFLite Micro's arena allocator,
but here it runs fully offline so there is no search cost at runtime.

All allocations are aligned to ARENA_ALIGNMENT bytes (default 64, matching
typical cache-line and vector-unit requirements).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

from compiler.frontend.ir import Graph
from compiler.memory.liveness import LiveRange

ARENA_ALIGNMENT = 64  # bytes; must be a power of two


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


@dataclasses.dataclass
class Allocation:
    """The resolved arena slot for a single tensor."""
    tensor_name: str
    offset:      int      # byte offset within g_arena[]
    size:        int      # byte count (un-padded)
    aligned_end: int      # offset + padded size (next free byte after this slot)

    def end(self) -> int:
        return self.aligned_end

    def overlaps_address(self, other: "Allocation") -> bool:
        return self.offset < other.aligned_end and other.offset < self.aligned_end


@dataclasses.dataclass
class ArenaLayout:
    """Complete result of the allocation pass."""
    allocations: Dict[str, Allocation]  # tensor name → Allocation
    arena_size:  int                    # total bytes required (aligned)

    def offset_of(self, tensor_name: str) -> int:
        return self.allocations[tensor_name].offset

    def report(self, lower_bound: Optional[int] = None) -> str:
        lines = [f"Arena size: {self.arena_size} bytes ({self.arena_size / 1024:.2f} KiB)"]
        if lower_bound is not None and lower_bound > 0:
            eff = 100.0 * lower_bound / max(self.arena_size, 1)
            lines.append(
                f"Peak-live lower bound: {lower_bound} bytes "
                f"(allocator efficiency: {eff:.1f}%)"
            )
        lines.append(f"{'Tensor':<48} {'Offset':>10} {'Size':>10} {'AlignEnd':>10}")
        lines.append("-" * 82)
        for alloc in sorted(self.allocations.values(), key=lambda a: a.offset):
            lines.append(
                f"{alloc.tensor_name:<48} {alloc.offset:>10} "
                f"{alloc.size:>10} {alloc.aligned_end:>10}"
            )
        return "\n".join(lines)


def allocate(
    graph: Graph,
    live_ranges: Dict[str, LiveRange],
    alignment: int = ARENA_ALIGNMENT,
) -> ArenaLayout:
    """Run both placement heuristics, verify each, and return the smaller.

    Strategies:
      * GFFD        — greedy first-fit by decreasing size
      * LINEAR_SCAN — process intervals by start time (classic interval
                      scheduling; often wins on pipeline-shaped lifetimes)

    For interval graphs, peak simultaneous live bytes is a lower bound on
    any feasible arena size; the report includes how close we land.
    """
    candidates = []
    for tname, lr in live_ranges.items():
        t = graph.tensors[tname]
        if t.is_constant:
            continue
        candidates.append((t.byte_size, tname))

    layouts = []
    for order_key in (_order_gffd(candidates), _order_linear_scan(graph, live_ranges, candidates)):
        layout = _place(order_key, live_ranges, alignment)
        verify(graph, layout, live_ranges)
        layouts.append(layout)

    best = min(layouts, key=lambda lay: lay.arena_size)
    return best


def _order_gffd(candidates):
    """(size desc) — stable on ties via name for determinism."""
    return sorted(candidates, key=lambda x: (-x[0], x[1]))


def _order_linear_scan(graph, live_ranges, candidates):
    """Start-time asc, then end-time desc, then size desc."""
    return sorted(
        candidates,
        key=lambda x: (
            live_ranges[x[1]].first_def,
            -live_ranges[x[1]].last_use,
            -x[0],
            x[1],
        ),
    )


def _place(
    ordered_candidates,
    live_ranges: Dict[str, LiveRange],
    alignment: int,
) -> ArenaLayout:
    allocations: Dict[str, Allocation] = {}
    arena_size = 0

    for byte_size, tname in ordered_candidates:
        lr = live_ranges[tname]

        # Collect allocations that are simultaneously live with this tensor
        conflicting: List[Allocation] = [
            allocations[other]
            for other in allocations
            if live_ranges[other].overlaps(lr)
        ]

        # Find the lowest offset that doesn't overlap any conflicting allocation
        offset = _find_free_offset(byte_size, conflicting, alignment)
        aligned_end = _align_up(offset + byte_size, alignment)

        allocations[tname] = Allocation(
            tensor_name=tname,
            offset=offset,
            size=byte_size,
            aligned_end=aligned_end,
        )
        arena_size = max(arena_size, aligned_end)

    return ArenaLayout(allocations=allocations, arena_size=arena_size)


def peak_live_bytes(graph: Graph, live_ranges: Dict[str, LiveRange]) -> int:
    """Lower bound on arena size: maximum total bytes simultaneously live.

    Computed with a sweep line over def/use events.  Any correct allocator
    must use at least this many bytes.
    """
    events = []  # (time, +bytes at def, -bytes after last_use)
    for tname, lr in live_ranges.items():
        if graph.tensors[tname].is_constant:
            continue
        events.append((lr.first_def, graph.tensors[tname].byte_size))
        events.append((lr.last_use + 1, -graph.tensors[tname].byte_size))
    events.sort(key=lambda e: (e[0], e[1]))   # frees before allocs at same t

    peak = cur = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return _align_up(peak, ARENA_ALIGNMENT)


def verify(
    graph: Graph,
    layout: ArenaLayout,
    live_ranges: Dict[str, LiveRange],
) -> None:
    """Prove the allocation is memory-safe, or raise.

    Checks the invariants that make zero-heap inference sound:
      1. Every live activation tensor has exactly one arena slot.
      2. Tensors whose live ranges overlap occupy disjoint address ranges.
      3. Aliased tensors resolve into their root tensor's slot.
      4. No address exceeds the reported arena size.
    """
    names = set(live_ranges.keys())

    # -- 1. coverage -------------------------------------------------------
    missing = names - set(layout.allocations.keys())
    if missing:
        raise RuntimeError(f"Arena bug: no allocation for live tensors: {sorted(missing)}")
    extra = set(layout.allocations.keys()) - names
    if extra:
        raise RuntimeError(f"Arena bug: allocated dead tensors: {sorted(extra)}")

    # -- 2 + 4. pairwise disjointness & bounds ------------------------------
    entries = list(layout.allocations.values())
    for i, a in enumerate(entries):
        if a.aligned_end > layout.arena_size:
            raise RuntimeError(
                f"Arena bug: {a.tensor_name} ends at {a.aligned_end} "
                f"> arena size {layout.arena_size}"
            )
        for b in entries[i + 1:]:
            if not live_ranges[a.tensor_name].overlaps(live_ranges[b.tensor_name]):
                continue
            if a.overlaps_address(b):
                raise RuntimeError(
                    f"Arena bug: '{a.tensor_name}' {a} and '{b.tensor_name}' "
                    f"{b} overlap while simultaneously live"
                )

    # -- 3. alias containment ----------------------------------------------
    for tname, t in graph.tensors.items():
        if t.is_constant or not t.alias_of:
            continue
        root = tname
        seen = set()
        while graph.tensors[root].alias_of:
            if root in seen:
                raise RuntimeError(f"Arena bug: alias cycle at {root!r}")
            seen.add(root)
            root = graph.tensors[root].alias_of
        if root not in layout.allocations:
            raise RuntimeError(
                f"Arena bug: alias root '{root}' of '{tname}' has no allocation"
            )
        root_alloc = layout.allocations[root]
        need = graph.tensors[tname].byte_size
        avail = root_alloc.aligned_end - root_alloc.offset
        if need > avail:
            raise RuntimeError(
                f"Arena bug: aliased tensor '{tname}' needs {need} bytes but "
                f"root '{root}' slot holds {avail}"
            )


def _find_free_offset(
    byte_size: int,
    conflicting: List[Allocation],
    alignment: int,
) -> int:
    """Return the lowest aligned offset that fits byte_size without overlapping
    any of the conflicting allocations.

    Builds a sorted list of (start, end) occupied ranges and scans for a gap.
    """
    if not conflicting:
        return 0

    # Sort occupied ranges by start address
    occupied = sorted((a.offset, a.aligned_end) for a in conflicting)

    # Try offset = 0 first
    candidate = 0
    for (occ_start, occ_end) in occupied:
        if candidate + byte_size <= occ_start:
            break   # fits in the gap before this occupied block
        if candidate < occ_end:
            # Overlap: push candidate past this block (aligned)
            candidate = _align_up(occ_end, alignment)

    return candidate
