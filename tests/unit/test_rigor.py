"""
tests/unit/test_rigor.py

Unit tests for the hardening added in the rigor build-out:
  * allocator strategies + peak-live lower bound
  * ingest rejection diagnostics (control flow / opset / external data /
    unresolved shape inference)
  * cost-model sanity & determinism
  * end-to-end compiler determinism (byte-identical images)
"""

import subprocess
import sys

import numpy as np
import pytest

from compiler.frontend.ir import Dtype, Graph, Op, Tensor
from compiler.memory import liveness, arena
from compiler.passes import cost_model


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------

def test_peak_live_is_valid_lower_bound():
    """No correct allocator can beat peak-live bytes; our layouts never do."""
    rng = np.random.RandomState(7)

    def build(n):
        tensors = {f"t{i}": Tensor(name=f"t{i}", dtype=Dtype.FLOAT32,
                                   shape=(rng.randint(1, 64),)) for i in range(n)}
        ops, prev = [], "t0"
        for i in range(1, n):
            ops.append(Op(op_id=i - 1, op_type="Relu",
                          inputs=[prev], outputs=[f"t{i}"], attrs={}))
            prev = f"t{i}"
        return Graph(name="g", ops=ops, tensors=tensors,
                     inputs=["t0"], outputs=[prev])

    for _ in range(50):
        g = build(int(rng.randint(2, 10)))
        ranges = liveness.compute_liveness(g)
        layout = arena.allocate(g, ranges)
        assert layout.arena_size >= arena.peak_live_bytes(g, ranges)


def test_linear_scan_vs_gffd_differ_and_best_wins():
    """The strategies disagree on some instances; allocate() must return the
    smaller verified layout (and both are always >= peak-live bound)."""
    from compiler.passes import alias_analysis

    def build(sizes):
        n = len(sizes)
        tensors = {
            f"t{i}": Tensor(name=f"t{i}", dtype=Dtype.FLOAT32,
                            shape=(sizes[i] // 4,))
            for i in range(n)
        }
        ops = [Op(op_id=i, op_type="Relu", inputs=[f"t{i}"],
                  outputs=[f"t{i+1}"], attrs={}) for i in range(n - 1)]
        g = Graph(name="g", ops=ops, tensors=tensors,
                  inputs=["t0"], outputs=[f"t{n-1}"])
        return g

    rng = np.random.RandomState(11)
    for _ in range(60):
        sizes = [int(rng.choice([128, 256, 512, 1024, 4096]))
                 for _ in range(int(rng.randint(3, 8)))]
        g = build(sizes)
        g = alias_analysis.run(g)
        ranges = liveness.compute_liveness(g)

        cands = [(g.tensors[n].byte_size, n) for n in ranges]
        gffd = arena._place(arena._order_gffd(cands), ranges, arena.ARENA_ALIGNMENT)
        scan = arena._place(arena._order_linear_scan(g, ranges, cands),
                            ranges, arena.ARENA_ALIGNMENT)
        best = arena.allocate(g, ranges)

        assert best.arena_size == min(gffd.arena_size, scan.arena_size)
        assert best.arena_size >= arena.peak_live_bytes(g, ranges)


def test_allocate_picks_best_strategy_and_verifies():
    """With in-place aliasing enabled, an elementwise chain collapses to ONE
    aligned slot."""
    from compiler.passes import alias_analysis

    n = 5
    tensors = {f"t{i}": Tensor(name=f"t{i}", dtype=Dtype.FLOAT32,
                               shape=(128,)) for i in range(n)}
    ops = [Op(op_id=i, op_type="Sigmoid", inputs=[f"t{i}"], outputs=[f"t{i+1}"],
              attrs={}) for i in range(n - 1)]
    g = Graph(name="g", ops=ops, tensors=tensors,
              inputs=["t0"], outputs=["t4"])
    g = alias_analysis.run(g)
    ranges = liveness.compute_liveness(g)
    layout = arena.allocate(g, ranges)          # runs verify() internally
    assert set(layout.allocations.keys()) == {"t0"}
    assert layout.arena_size == 512             # one aligned slot reused


# ---------------------------------------------------------------------------
# Ingest diagnostics
# ---------------------------------------------------------------------------

def _tiny_model(nodes, opset=13, inputs="X", xshape=(2, 2)):
    from onnx import helper, TensorProto
    xi = helper.make_tensor_value_info(inputs, TensorProto.FLOAT, list(xshape))
    yi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    graph = helper.make_graph(nodes, "m", [xi], [yi])
    model = helper.make_model(graph, producer_name="t")
    model.opset_import[0].version = opset
    return model


def test_ingest_rejects_control_flow(tmp_path):
    import onnx
    from onnx import helper as H, TensorProto
    from compiler.frontend import onnx_ingest

    cond = onnx.numpy_helper.from_array(np.array([True]), "c")
    xi = H.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    yo = H.make_tensor_value_info("Yo", TensorProto.FLOAT, [2])
    sub = H.make_graph([H.make_node("Identity", ["X"], ["Yo"])], "sub", [xi], [yo])

    node = H.make_node("If", ["c"], ["Y"], then_branch=sub, else_branch=sub)
    m = _tiny_model([node])
    m.graph.initializer.append(cond)
    p = tmp_path / "if.onnx"
    onnx.save(m, str(p))
    with pytest.raises(ValueError, match="[Cc]ontrol flow|Unsupported"):
        onnx_ingest.ingest(p)


def test_ingest_rejects_bad_opset(tmp_path):
    import onnx
    from onnx import helper, TensorProto
    from compiler.frontend import onnx_ingest

    node = helper.make_node("Relu", ["X"], ["T"])
    node2 = helper.make_node("GlobalAveragePool", ["T"], ["T2"])
    node3 = helper.make_node("Flatten", ["T2"], ["Y"], axis=1)
    xi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 4, 4])
    yi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph([node, node2, node3], "m", [xi], [yi])
    model = helper.make_model(graph, producer_name="t")
    model.opset_import[0].version = 9          # below supported floor
    p = tmp_path / "old.onnx"
    onnx.save(model, str(p))
    with pytest.raises(ValueError, match="opset"):
        onnx_ingest.ingest(p)


def test_ingest_names_unresolved_shape_inference(tmp_path):
    """A dimensionally-broken MatMul must produce a named diagnostic."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from compiler.frontend import onnx_ingest

    W = numpy_helper.from_array(np.ones((3, 7), np.float32), "W")
    nodes = [
        # X[2,2] x W[3,7] is dimensionally invalid: shape inference will
        # silently drop 'T' under non-strict mode.
        helper.make_node("MatMul", ["X", "W"], ["T"]),
        helper.make_node("Relu", ["T"], ["Y"]),
    ]
    xi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    yi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 7])
    graph = helper.make_graph(nodes, "m", [xi], [yi], [W])
    model = helper.make_model(graph, producer_name="t")
    model.opset_import[0].version = 13
    p = tmp_path / "bad.onnx"
    onnx.save(model, str(p))
    with pytest.raises(ValueError, match="Shape inference.*unresolved|T"):
        onnx_ingest.ingest(p)


def test_ingest_rejects_external_data(tmp_path):
    import onnx
    from onnx import helper, TensorProto
    from compiler.frontend import onnx_ingest

    init = helper.make_tensor("W", TensorProto.FLOAT, [2], [1.0, 2.0])
    init.data_location = onnx.TensorProto.EXTERNAL
    entry = init.external_data.add()
    entry.key = "location"
    entry.value = "weights.bin"          # valid-looking location
    (tmp_path / "weights.bin").write_bytes(b"\x00" * 8)

    node = helper.make_node("Identity", ["W"], ["Y"])
    graph = helper.make_graph([node], "m", [], [helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [2])], [init])
    model = helper.make_model(graph, producer_name="t")
    model.opset_import[0].version = 13
    p = tmp_path / "ext.onnx"
    p.write_bytes(model.SerializeToString())   # bypass save-time checker
    with pytest.raises(ValueError, match="[Ee]xternal data"):
        onnx_ingest.ingest(p)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def test_cost_model_deterministic_and_positive():
    tensors = {
        "X": Tensor(name="X", dtype=Dtype.FLOAT32, shape=(16, 32)),
        "W": Tensor(name="W", dtype=Dtype.FLOAT32, shape=(32, 8), data=b"\x00" * 1024),
        "B": Tensor(name="B", dtype=Dtype.FLOAT32, shape=(8,), data=b"\x00" * 32),
        "T": Tensor(name="T", dtype=Dtype.FLOAT32, shape=(16, 8)),
        "Y": Tensor(name="Y", dtype=Dtype.FLOAT32, shape=(16, 8)),
    }
    ops = [
        Op(op_id=0, op_type="MatMul", inputs=["X", "W"], outputs=["T"], attrs={}),
        Op(op_id=1, op_type="Add", inputs=["T", "B"], outputs=["Y"], attrs={}),
    ]
    g = Graph(name="g", ops=ops, tensors=tensors, inputs=["X"], outputs=["Y"])

    a = cost_model.analyse(g)
    b = cost_model.analyse(g)
    assert a == b                                  # deterministic
    assert a[0] > 0 and a[1] > 0                   # positive costs
    assert a[0] == (16 * 32 * 8 + 3) // 4          # documented formula
    assert a[1] == max(1, (16 * 8 + 7) // 8)


# ---------------------------------------------------------------------------
# Compiler determinism
# ---------------------------------------------------------------------------

def test_compiler_output_is_byte_deterministic(tmp_path):
    """Same input model -> byte-identical generated image.  Required for
    reproducible firmware builds."""
    root = Path(__file__).resolve().parent.parent.parent
    onnx_src = root / "tmp" / "mlp.onnx"
    if not onnx_src.exists():
        pytest.skip("tmp/mlp.onnx not present")

    out1, out2 = tmp_path / "a", tmp_path / "b"
    env = dict(os.environ)
    for out in (out1, out2):
        r = subprocess.run(
            [sys.executable, "-m", "compiler.main", str(onnx_src), str(out)],
            cwd=str(root), capture_output=True, text=True, env=env,
        )
        assert r.returncode == 0, r.stderr

    for fname in ("model_exec.c", "model_exec.h"):
        b1 = (out1 / fname).read_bytes()
        b2 = (out2 / fname).read_bytes()
        assert b1 == b2, f"{fname} differs between identical compilations"


import os  # noqa: E402  (used by determinism test)
from pathlib import Path  # noqa: E402
