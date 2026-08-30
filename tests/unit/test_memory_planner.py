"""
tests/unit/test_memory_planner.py

Property tests for the static memory planner.

The core formal claim of Phase 0 is:

    No two tensors that are simultaneously live ever share arena bytes,
    and zero heap allocation happens at runtime.

These tests attack that claim with randomized operator DAGs, in-place
aliasing chains, views, and adversarial reuse patterns.  The allocator's
own verify() is exercised on every allocation; here we re-check it
independently so a bug in both would be required to pass.
"""

import random

import pytest

from compiler.frontend.ir import Dtype, Graph, Op, Tensor
from compiler.memory import liveness, arena


def _make_dag(rng: random.Random):
    """Build a random valid chain/DAG of elementwise + view ops."""
    n_ops = rng.randint(1, 12)
    ops = []
    tensors = {
        "inp": Tensor(name="inp", dtype=Dtype.FLOAT32, shape=(4, 4)),
    }
    cur = "inp"
    for i in range(n_ops):
        kind = rng.choice(["relu", "add", "view", "branch"])
        out = f"t{i}"
        if kind == "relu":
            tensors[out] = Tensor(name=out, dtype=Dtype.FLOAT32, shape=(4, 4))
            ops.append(Op(op_id=i, op_type="Relu", inputs=[cur], outputs=[out], attrs={}))
        elif kind == "add":
            # second operand is a fresh activation -> forces overlapping lives
            other = f"b{i}"
            tensors[other] = Tensor(name=other, dtype=Dtype.FLOAT32, shape=(4, 4))
            if i > 0:
                ops.append(Op(op_id=len(ops), op_type="Relu",
                              inputs=[cur], outputs=[other], attrs={}))
                tensors[out] = Tensor(name=out, dtype=Dtype.FLOAT32, shape=(4, 4))
                ops.append(Op(op_id=len(ops), op_type="Add",
                              inputs=[cur, other], outputs=[out], attrs={}))
            else:
                tensors[out] = Tensor(name=out, dtype=Dtype.FLOAT32, shape=(4, 4))
                ops.append(Op(op_id=len(ops), op_type="Add",
                              inputs=[cur, other], outputs=[out], attrs={}))
        elif kind == "view":
            tensors[out] = Tensor(name=out, dtype=Dtype.FLOAT32, shape=(16,))
            ops.append(Op(op_id=len(ops), op_type="Reshape", inputs=[cur],
                          outputs=[out], attrs={}))
        else:
            # branch: two consumers of the same tensor (blocks in-place alias)
            o1, o2 = f"t{i}a", f"t{i}b"
            tensors[o1] = Tensor(name=o1, dtype=Dtype.FLOAT32, shape=(4, 4))
            tensors[o2] = Tensor(name=o2, dtype=Dtype.FLOAT32, shape=(4, 4))
            ops.append(Op(op_id=len(ops), op_type="Relu", inputs=[cur],
                          outputs=[o1], attrs={}))
            ops.append(Op(op_id=len(ops), op_type="Sigmoid", inputs=[cur],
                          outputs=[o2], attrs={}))
            out = o1
        cur = out

    outputs = [cur]
    g = Graph(
        name="random",
        ops=ops,
        tensors=tensors,
        inputs=["inp"],
        outputs=outputs,
    )
    return g


def _independent_check(graph, layout, ranges):
    """Re-verify overlap-freedom without calling arena.verify()."""
    entries = list(layout.allocations.values())
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            ra, rb = ranges[a.tensor_name], ranges[b.tensor_name]
            if not ra.overlaps(rb):
                continue
            assert a.offset >= b.aligned_end or b.offset >= a.aligned_end, (
                f"{a.tensor_name}@{a.offset}..{a.aligned_end} overlaps "
                f"{b.tensor_name}@{b.offset}..{b.aligned_end}"
            )


@pytest.mark.parametrize("seed", range(200))
def test_random_dag_never_overlaps(seed):
    rng = random.Random(seed)
    graph = _make_dag(rng)
    ranges = liveness.compute_liveness(graph)
    layout = arena.allocate(graph, ranges)   # runs verify() internally
    _independent_check(graph, layout, ranges)


def test_aliased_chain_shares_one_slot():
    """Relu->Relu->Relu in-place chain: all outputs map onto the input slot."""
    tensors = {
        "x": Tensor(name="x", dtype=Dtype.FLOAT32, shape=(8,)),
    }
    ops = []
    prev = "x"
    names = ["x"]
    for i in range(3):
        out = f"y{i}"
        tensors[out] = Tensor(name=out, dtype=Dtype.FLOAT32, shape=(8,))
        ops.append(Op(op_id=i, op_type="Relu", inputs=[prev], outputs=[out], attrs={}))
        prev = out
        names.append(out)

    g = Graph(name="chain", ops=ops, tensors=tensors,
              inputs=["x"], outputs=["y2"])
    g = __import__("compiler.passes.alias_analysis", fromlist=["run"]).run(g)
    g = __import__("compiler.passes.prune", fromlist=["run"]).run(g)

    ranges = liveness.compute_liveness(g)
    layout = arena.allocate(g, ranges)

    ranges = liveness.compute_liveness(g)
    layout = arena.allocate(g, ranges)

    # All three Relu outputs alias root 'x': only ONE allocation exists and
    # every view resolves into that single slot.
    assert set(layout.allocations.keys()) == {"x"}
    assert layout.arena_size == 64  # 8 floats aligned up to the 64B alignment


def test_graph_output_is_never_recycled():
    """A graph output buffer must stay exclusive until the end of inference."""
    tensors = {
        "x": Tensor(name="x", dtype=Dtype.FLOAT32, shape=(4,)),
        "big": Tensor(name="big", dtype=Dtype.FLOAT32, shape=(64,)),
        "y": Tensor(name="y", dtype=Dtype.FLOAT32, shape=(4,)),
    }
    ops = [
        Op(op_id=0, op_type="Relu", inputs=["x"], outputs=["y"], attrs={}),
        Op(op_id=1, op_type="Sigmoid", inputs=["big"], outputs=["big"], attrs={}),
    ]
    g = Graph(name="g", ops=ops, tensors=tensors, inputs=["x", "big"],
              outputs=["y", "big"])
    ranges = liveness.compute_liveness(g)
    layout = arena.allocate(g, ranges)
    oy = layout.offset_of("y")
    ob = layout.offset_of("big")
    # 'big' is live to the very end; y must not sit inside it.
    assert not (oy < ob + 256 and ob < oy + 16)


def test_verify_rejects_forced_overlap():
    """If someone corrupts a range pair, verify() must raise."""
    tensors = {
        "x": Tensor(name="x", dtype=Dtype.FLOAT32, shape=(4,)),
        "y": Tensor(name="y", dtype=Dtype.FLOAT32, shape=(4,)),
    }
    g = Graph(name="g", ops=[], tensors=tensors, inputs=["x", "y"], outputs=["x", "y"])
    ranges = liveness.compute_liveness(g)
    bad_layout = arena.ArenaLayout(
        allocations={
            "x": arena.Allocation("x", 0, 16, 64),
            "y": arena.Allocation("y", 0, 16, 64),   # deliberate overlap
        },
        arena_size=64,
    )
    with pytest.raises(RuntimeError, match="overlap"):
        arena.verify(g, bad_layout, ranges)


def test_arena_smaller_than_sum():
    """Sequential lifetimes must reuse space: total < naive sum."""
    tensors = {f"t{i}": Tensor(name=f"t{i}", dtype=Dtype.FLOAT32, shape=(1024,))
               for i in range(5)}
    ops = []
    for i in range(4):
        ops.append(Op(op_id=i, op_type="Relu",
                      inputs=[f"t{i}"], outputs=[f"t{i+1}"], attrs={}))
    g = Graph(name="seq", ops=ops, tensors=tensors, inputs=["t0"], outputs=["t4"])
    ranges = liveness.compute_liveness(g)
    layout = arena.allocate(g, ranges)
    naive = 5 * 4096
    assert layout.arena_size <= 4096 * 2
    assert layout.arena_size < naive
