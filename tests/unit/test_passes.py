"""
tests/unit/test_passes.py

Behavioural contracts for individual compiler passes.
"""

import numpy as np

from compiler.frontend.ir import Dtype, Graph, Op, Tensor
from compiler.passes import constant_fold, fusion, bn_absorb, prune, alias_analysis


def _const(name, arr):
    return Tensor(name=name, dtype=Dtype.FLOAT32,
                  shape=tuple(arr.shape), data=arr.astype(np.float32).tobytes())


def _act(name, shape):
    return Tensor(name=name, dtype=Dtype.FLOAT32, shape=tuple(shape))


def test_constant_fold_folds_chain():
    tensors = {
        "a": _const("a", np.array([[2.0]])),
        "b": _const("b", np.array([[3.0]])),
        "c": _const("c", np.array([[4.0]])),
        "t1": _act("t1", (1, 1)),
        "t2": _act("t2", (1, 1)),
    }
    ops = [
        Op(op_id=0, op_type="Mul", inputs=["a", "b"], outputs=["t1"], attrs={}),
        Op(op_id=1, op_type="Add", inputs=["t1", "c"], outputs=["t2"], attrs={}),
    ]
    g = Graph(name="g", ops=ops, tensors=tensors, inputs=[], outputs=[])
    g = constant_fold.run(g)
    assert len(g.ops) == 0
    folded = g.tensors["t2"]
    assert folded.is_constant
    val = np.frombuffer(folded.data, dtype=np.float32)
    assert float(val[0]) == 10.0   # (2*3) + 4


def test_fusion_bias_only_matmul_add():
    tensors = {
        "X": _act("X", (2, 3)),
        "W": _const("W", np.ones((3, 2))),
        "B": _const("B", np.zeros(2)),
        "T": _act("T", (2, 2)),
    }
    ops = [
        Op(op_id=0, op_type="MatMul", inputs=["X", "W"], outputs=["T"], attrs={}),
        Op(op_id=1, op_type="Add", inputs=["T", "B"], outputs=["Y"], attrs={}),
    ]
    tensors["Y"] = _act("Y", (2, 2))
    g = Graph(name="g", ops=ops, tensors=tensors, inputs=["X"], outputs=["Y"])
    g = fusion.run(g)
    assert len(g.ops) == 1
    assert g.ops[0].op_type == "MatMul_Add"
    assert g.ops[0].inputs == ["X", "W", "B"]


def test_fusion_refuses_residual_add():
    """MatMul + Add where the addend is an ACTIVATION must NOT fuse into
    MatMul_Add: the fused epilogue only implements a bias vector."""
    tensors = {
        "X": _act("X", (2, 3)),
        "W": _const("W", np.ones((3, 2))),
        "R": _act("R", (2, 2)),          # runtime residual branch
        "T": _act("T", (2, 2)),
    }
    ops = [
        Op(op_id=0, op_type="MatMul", inputs=["X", "W"], outputs=["T"], attrs={}),
        Op(op_id=1, op_type="Add", inputs=["T", "R"], outputs=["Y"], attrs={}),
    ]
    tensors["Y"] = _act("Y", (2, 2))
    g = Graph(name="g", ops=ops, tensors=tensors, inputs=["X", "R"], outputs=["Y"])
    g = fusion.run(g)
    types = [op.op_type for op in g.ops]
    assert "MatMul_Add" not in types
    assert types == ["MatMul", "Add"]


def test_fusion_never_dissolves_graph_output():
    tensors = {
        "X": _act("X", (2, 2)),
        "T": _act("T", (2, 2)),
    }
    # Relu output IS the graph output; a hypothetical consumer chain may not
    # remove it from the interface.
    ops = [Op(op_id=0, op_type="Relu", inputs=["X"], outputs=["T"], attrs={})]
    g = Graph(name="g", ops=ops, tensors=tensors, inputs=["X"], outputs=["T"])
    before = list(g.outputs)
    g = fusion.run(g)
    assert g.outputs == before


def test_bn_absorb_gemm_transb():
    """BN absorption with transB=1 scales W rows (out_features axis)."""
    out_f, in_f = 3, 4
    W = np.random.randn(out_f, in_f).astype(np.float32)
    scale = (np.random.rand(out_f) + 0.5).astype(np.float32)
    bias = np.random.randn(out_f).astype(np.float32)
    mean = np.random.randn(out_f).astype(np.float32)
    var = (np.random.rand(out_f) + 0.5).astype(np.float32)

    X = _act("X", (2, in_f))
    T = _act("T", (2, out_f))
    Y = _act("Y", (2, out_f))
    tensors = {
        "X": X,
        "W": _const("W", W),
        "T": T,
        "s": _const("s", scale.reshape(1, -1) if False else scale),
        "b": _const("b", bias),
        "m": _const("m", mean),
        "v": _const("v", var),
        "T1": _act("T1", (2, out_f)),
        "Y": Y,
    }
    ops = [
        Op(op_id=0, op_type="Gemm", inputs=["X", "W"], outputs=["T"],
           attrs={"transB": 1}),
        Op(op_id=1, op_type="BatchNormalization",
           inputs=["T", "s", "b", "m", "v"], outputs=["T1"],
           attrs={"epsilon": 1e-5}),
    ]
    g = Graph(name="g", ops=ops, tensors=tensors, inputs=["X"], outputs=["Y"])
    g = bn_absorb.run(g)
    assert len(g.ops) == 1

    Wf = np.frombuffer(g.tensors["W"].data, dtype=np.float32).reshape(out_f, in_f)
    gamma = scale / np.sqrt(var + 1e-5)
    # rows of W (out-feature axis under transB=1) scaled by gamma
    np.testing.assert_allclose(Wf, W * gamma[:, None], rtol=1e-6)


def test_prune_removes_orphans_and_keeps_roots():
    tensors = {
        "x": _act("x", (4,)),
        "dead": _act("dead", (4,)),
        "dead_w": _const("dead_w", np.ones(2)),
        "y": _act("y", (8,)),
    }
    tensors["y"].alias_of = "x"          # view root must survive
    ops = [Op(op_id=0, op_type="Reshape", inputs=["x"], outputs=["y"], attrs={})]
    g = Graph(name="g", ops=ops, tensors=tensors, inputs=["x"], outputs=["y"])
    g = prune.run(g)
    assert set(g.tensors.keys()) == {"x", "y"}


def test_alias_binary_requires_exact_shape():
    """Same byte size but different shape: a flat elementwise write over X
    while the kernel reads it under Y's broadcast geometry is unsound."""
    tensors = {
        "X": _act("X", (3, 2)),
        "B": _act("B", (3,)),            # broadcast operand
        "Y": _act("Y", (2, 3)),          # same 6 elements, different layout
    }
    ops = [Op(op_id=0, op_type="Add", inputs=["X", "B"], outputs=["Y"], attrs={})]
    g = Graph(name="g", ops=ops, tensors=tensors, inputs=["X", "B"], outputs=["Y"])
    g = alias_analysis.run(g)
    assert g.tensors["Y"].alias_of is None   # shape mismatch blocks in-place


def test_alias_identity_is_view():
    tensors = {
        "X": _act("X", (2, 2)),
        "Y": _act("Y", (2, 2)),
    }
    ops = [Op(op_id=0, op_type="Identity", inputs=["X"], outputs=["Y"], attrs={})]
    g = Graph(name="g", ops=ops, tensors=tensors, inputs=["X"], outputs=["Y"])
    g = alias_analysis.run(g)
    assert g.tensors["Y"].alias_of == "X"
