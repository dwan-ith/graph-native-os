"""
compiler/passes/power_analysis.py

Phase-Aware Power Domain Analysis Pass.

This pass analyses the statically-known schedule and annotates each sunit_t
with:
 1. The set of hardware power domains it requires.
 2. The earliest a domain can be put to sleep after the last op that needs it.
 3. The latest a domain must wake before the first op that requires it.

Output: a PowerPlan object which the C code generator uses to emit:
  - `power_domain_state_t g_power_plan[]`  -- ROM table
  - `power_phase_mgr_run(frame_start_us)`  -- evaluated before each inference

Design:
  - Each operation declares its required power domains via `target_device`.
  - The power manager evaluates this ahead of time (offline) and records the
    exact op-indices at which each domain can be gated off, and the op-index
    (minus wakeup_latency_ops) at which it must be re-enabled.
  - At runtime, the manager simply indexes into this ROM table --- zero
    heuristics, zero idle-detection.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Set

from compiler.frontend.ir import Graph


# Map device strings to power domain names
_DEVICE_TO_DOMAIN: Dict[str, str] = {
    "DEVICE_CPU": "PWR_CPU",
    "DEVICE_NPU": "PWR_NPU",
    "DEVICE_DMA": "PWR_DMA",
}


@dataclasses.dataclass
class DomainTiming:
    """Offline-computed power window for a single domain in one inference pass."""
    domain:        str
    first_use_op:  int   # index into g_sunits[] of first op that needs this domain
    last_use_op:   int   # index into g_sunits[] of last  op that needs this domain
    # Derived: which ops to insert an ENABLE before, and DISABLE after
    enable_before: int   # == first_use_op  (wake up just in time)
    disable_after: int   # == last_use_op   (sleep as early as possible)


@dataclasses.dataclass
class PowerPlan:
    """Complete offline power management plan for one graph."""
    timings:        List[DomainTiming]
    n_ops:          int  # total ops (== len(g_sunits))
    # Domains that are ALWAYS on (CPU must always be on to run the scheduler)
    always_on_mask: Set[str] = dataclasses.field(default_factory=set)

    def report(self) -> str:
        lines = ["Power Domain Schedule:"]
        lines.append(f"  {'Domain':<12} {'FirstUse':>8} {'LastUse':>8}")
        lines.append("  " + "-" * 32)
        for t in self.timings:
            always = " [always-on]" if t.domain in self.always_on_mask else ""
            lines.append(f"  {t.domain:<12} {t.first_use_op:>8} {t.last_use_op:>8}{always}")
        return "\n".join(lines)


def analyse(graph: Graph) -> PowerPlan:
    """Compute the power plan from the annotated graph."""
    domain_first: Dict[str, int] = {}
    domain_last:  Dict[str, int] = {}

    for i, op in enumerate(graph.ops):
        d = _DEVICE_TO_DOMAIN.get(op.target_device, "PWR_CPU")
        if d not in domain_first:
            domain_first[d] = i
        domain_last[d] = i

    timings: List[DomainTiming] = []
    for domain, first in sorted(domain_first.items()):
        last = domain_last[domain]
        timings.append(DomainTiming(
            domain=domain,
            first_use_op=first,
            last_use_op=last,
            enable_before=first,
            disable_after=last,
        ))

    # CPU must always be on (it runs the scheduler loop itself).  Guarantee
    # an explicit always-on entry even when no op executes on the CPU.
    always_on = {"PWR_CPU"}
    if not any(t.domain == "PWR_CPU" for t in timings):
        timings.append(DomainTiming(
            domain="PWR_CPU",
            first_use_op=0,
            last_use_op=max(len(graph.ops) - 1, 0),
            enable_before=0,
            disable_after=0,
        ))
    timings.sort(key=lambda t: t.domain)

    # CPU must always be on (it runs the scheduler loop itself)
    always_on = {"PWR_CPU"}

    return PowerPlan(
        timings=timings,
        n_ops=len(graph.ops),
        always_on_mask=always_on,
    )
