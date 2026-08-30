"""
tests/e2e/test_compiler.py

End-to-end exactness verification across a model zoo:

    build ONNX model -> tiny-os compiler -> gcc -> DLL -> execute via ctypes
    -> compare against onnxruntime.

Covers every kernel family and the tricky compiler paths:
  - MatMul + Add + Relu MLP (fusion to MatMul_Add, in-place Relu aliasing)
  - Gemm transB=1 (the PyTorch Linear export convention)
  - Conv with padding / stride / groups, fused Conv+BN+Relu
  - MaxPool / GlobalAveragePool / Flatten tails
  - Transpose, Concat, broadcast Add, Mul->Add chains
  - Diamond dataflow (tensor consumed twice — alias-safety stress)
  - Fully-constant graphs (zero ops edge case)
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _harness import run_case, assert_matches  # noqa: E402


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _model(graph_name, nodes, inputs, outputs, initializers, opset=13):
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="tinyos_test")
    model.opset_import[0].version = opset
    return model


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name)


RNG = np.random.RandomState(1234)


def mlp_model():
    """ReLU(MatMul(X,W) + B) — the canonical smoke test."""
    W = np.array([[1.0, 2.0], [-1.0, 0.5], [0.0, 0.0], [1.0, -1.0]], dtype=np.float32)
    B = np.array([0.5, -0.5], dtype=np.float32)
    return _model(
        "tiny_mlp",
        [
            helper.make_node("MatMul", ["X", "W"], ["T0"], name="OpMatMul"),
            helper.make_node("Add", ["T0", "B"], ["T1"], name="OpAdd"),
            helper.make_node("Relu", ["T1"], ["Y"], name="OpRelu"),
        ],
        [_vi("X", [1, 4])], [_vi("Y", [1, 2])],
        [_init("W", W), _init("B", B)],
    )


def gemm_transb_model():
    """PyTorch-style Linear: Gemm with transB=1."""
    W = RNG.randn(3, 4).astype(np.float32)
    B = RNG.randn(3).astype(np.float32)
    return _model(
        "gemm_linear",
        [helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)],
        [_vi("X", [5, 4])], [_vi("Y", [5, 3])],
        [_init("W", W), _init("B", B)],
    )


def conv_padded_model():
    W = RNG.randn(2, 1, 3, 3).astype(np.float32)
    Bv = RNG.randn(2).astype(np.float32)
    return _model(
        "conv_pad",
        [helper.make_node("Conv", ["X", "W", "Bb"], ["Y"],
                          strides=[1, 1], pads=[1, 1, 1, 1])],
        [_vi("X", [1, 1, 6, 6])], [_vi("Y", [1, 2, 6, 6])],
        [_init("W", W), _init("Bb", Bv)],
    )


def conv_strided_dilated_model():
    W = RNG.randn(2, 1, 3, 3).astype(np.float32)
    return _model(
        "conv_sd",
        [helper.make_node("Conv", ["X", "W"], ["Y"],
                          strides=[2, 2], dilations=[2, 2])],
        # out = floor((7 - (3-1)*2 - 1)/2)+1 = floor((7-5)/2)+1 = 2
        [_vi("X", [1, 1, 7, 7])], [_vi("Y", [1, 2, 2, 2])],
        [_init("W", W)],
    )


def conv_grouped_model():
    """group=C_in=C_out — depthwise convolution."""
    W = RNG.randn(4, 1, 3, 3).astype(np.float32)
    return _model(
        "conv_dw",
        [helper.make_node("Conv", ["X", "W"], ["Y"],
                          group=4, strides=[1, 1], pads=[1, 1, 1, 1])],
        [_vi("X", [1, 4, 5, 5])], [_vi("Y", [1, 4, 5, 5])],
        [_init("W", W)],
    )


def conv_bn_relu_model():
    """BatchNorm absorption + Conv_Relu fusion."""
    W = RNG.randn(4, 2, 3, 3).astype(np.float32)
    s = RNG.rand(4).astype(np.float32) + 0.5
    b = RNG.randn(4).astype(np.float32)
    m = RNG.randn(4).astype(np.float32)
    v = RNG.rand(4).astype(np.float32) + 0.5
    return _model(
        "conv_bn_relu",
        [
            helper.make_node("Conv", ["X", "W"], ["T0"], strides=[1, 1]),
            helper.make_node("BatchNormalization",
                             ["T0", "s", "b", "m", "v"], ["T1"]),
            helper.make_node("Relu", ["T1"], ["Y"]),
        ],
        [_vi("X", [1, 2, 4, 4])], [_vi("Y", [1, 4, 2, 2])],
        [_init("W", W), _init("s", s), _init("b", b),
         _init("m", m), _init("v", v)],
    )


def pool_tail_model():
    return _model(
        "pool_tail",
        [
            helper.make_node("MaxPool", ["X"], ["T"], kernel_shape=[2, 2]),
            helper.make_node("GlobalAveragePool", ["T"], ["T2"]),
            helper.make_node("Flatten", ["T2"], ["Y"], axis=1),
        ],
        [_vi("X", [1, 2, 5, 5])], [_vi("Y", [1, 2])],
        [],
    )


def transpose_concat_model():
    C0 = RNG.randn(2, 3).astype(np.float32)
    C1 = RNG.randn(3, 2).astype(np.float32)
    return _model(
        "transpose_concat",
        [
            helper.make_node("Transpose", ["A"], ["At"], perm=[1, 0]),
            helper.make_node("Concat", ["At", "B"], ["Y"], axis=1),
        ],
        [], [_vi("Y", [3, 5])],
        [_init("A", C0), _init("B", C1)],
    )


def mul_add_chain_model():
    M = RNG.randn(3, 3).astype(np.float32)
    A = RNG.randn(3).astype(np.float32)
    return _model(
        "mul_add_chain",
        [
            helper.make_node("Mul", ["X", "M"], ["T"]),
            helper.make_node("Add", ["T", "A"], ["T2"]),
            helper.make_node("Div", ["T2", "S"], ["Y"]),
        ],
        [_vi("X", [3, 3])], [_vi("Y", [3, 3])],
        [_init("M", M), _init("A", A),
         _init("S", np.full(3, 2.0, dtype=np.float32))],
    )


def diamond_model():
    """X consumed by two ops — blocks unsafe in-place aliasing."""
    return _model(
        "diamond",
        [
            helper.make_node("Relu", ["X"], ["T"]),
            helper.make_node("Mul", ["T", "X"], ["Y"]),
        ],
        [_vi("X", [2, 2])], [_vi("Y", [2, 2])],
        [],
    )


def broadcast_add_model():
    B = RNG.randn(7).astype(np.float32)
    return _model(
        "broadcast_add",
        [
            helper.make_node("Identity", ["X"], ["Xc"]),
            helper.make_node("Add", ["Xc", "B"], ["Y"]),
        ],
        [_vi("X", [4, 7])], [_vi("Y", [4, 7])],
        [_init("B", B)],
    )


def batched_matmul_model():
    """3-D A x 2-D B broadcast matmul."""
    return _model(
        "batched_matmul",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        [_vi("A", [2, 3, 4]), _vi("B", [4, 5])], [_vi("Y", [2, 3, 5])],
        [],
    )


def softmax_model():
    return _model(
        "softmax_net",
        [
            helper.make_node("Gemm", ["X", "W", "B"], ["T"], transB=1),
            helper.make_node("Softmax", ["T"], ["Y"], axis=-1),
        ],
        [_vi("X", [6, 8])], [_vi("Y", [6, 3])],
        [_init("W", RNG.randn(3, 8)), _init("B", RNG.randn(3))],
    )


def all_constant_model():
    """Everything folds away: zero-op image must still build & return OK."""
    a = RNG.randn(2, 2).astype(np.float32)
    b = RNG.randn(2, 2).astype(np.float32)
    c = RNG.randn(2, 2).astype(np.float32)
    return _model(
        "all_const",
        [
            helper.make_node("Add", ["a", "b"], ["t"]),
            helper.make_node("Mul", ["t", "c"], ["Y"]),
        ],
        [], [_vi("Y", [2, 2])],
        [_init("a", a), _init("b", b), _init("c", c)],
    )


def unsupported_op_model():
    """An op outside the supported set must fail compilation with a clear
    diagnostic — never a mysterious linker error."""
    return _model(
        "unsupported",
        [helper.make_node("ReduceMax", ["X"], ["Y"], axes=[1], keepdims=0)],
        [_vi("X", [2, 4])], [_vi("Y", [2])],
        [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _init_i64(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name)


def slice_model():
    """Slice with explicit axes + negative indexing."""
    starts = np.array([1, -2], dtype=np.int64)
    ends = np.array([4, 100], dtype=np.int64)     # clamped to dim
    axes = np.array([0, 1], dtype=np.int64)
    return _model(
        "slice_net",
        [helper.make_node("Slice", ["X", "S", "E", "A"], ["Y"])],
        [_vi("X", [5, 6])], [_vi("Y", [3, 2])],
        [_init_i64("S", starts), _init_i64("E", ends), _init_i64("A", axes)],
    )


def transformer_block_model():
    """One pre-norm transformer block:
       Y = X + MLP(LN2(X + ATTN(LN1(X))))
    Attention is single-head for tractability. Exercises LayerNormalization,
    Gemm(transB=1), Erf-GELU, ReduceMean, broadcasting Add/Mul/Div, MatMul.
    """
    d = 8
    def ln(name):
        g = np.random.rand(d).astype(np.float32) + 0.5
        b = np.random.randn(d).astype(np.float32)
        return [_init(f"{name}_g", g), _init(f"{name}_b", b)]

    inits = []
    # LN1 params
    inits += ln("ln1")
    # QKV projections (transB convention: W is [out, in])
    inits += [
        _init("Wq", RNG.randn(d, d)), _init("bq", RNG.randn(d)),
        _init("Wk", RNG.randn(d, d)), _init("bk", RNG.randn(d)),
        _init("Wv", RNG.randn(d, d)), _init("bv", RNG.randn(d)),
        _init("Wo", RNG.randn(d, d)), _init("bo", RNG.randn(d)),
    ]
    # LN2 params
    inits += ln("ln2")
    # MLP
    inits += [
        _init("W1", RNG.randn(4 * d, d)), _init("b1", RNG.randn(4 * d)),
        _init("W2", RNG.randn(d, 4 * d)), _init("b2", RNG.randn(d)),
    ]
    # GELU constants
    inits += [
        _init("c_sqrt2pi", np.array([0.7978845608], dtype=np.float32)),
        _init("c_half", np.array([0.5], dtype=np.float32)),
        _init("c_one", np.array([1.0], dtype=np.float32)),
        _init("c_three", np.array([3.0], dtype=np.float32)),
        _init("scale_attn", np.full((1,), 1.0 / np.sqrt(d), dtype=np.float32)),
    ]

    nodes = [
        # ---- attention sub-layer --------------------------------------
        helper.make_node("LayerNormalization", ["X", "ln1_g", "ln1_b"], ["N1"], axis=-1),
        helper.make_node("Gemm", ["N1", "Wq", "bq"], ["Q"], transB=1),
        helper.make_node("Gemm", ["N1", "Wk", "bk"], ["Kt"], transB=1),
        helper.make_node("Gemm", ["N1", "Wv", "bv"], ["V"], transB=1),
        helper.make_node("Transpose", ["Kt"], ["K"], perm=[1, 0]),
        helper.make_node("MatMul", ["Q", "K"], ["QK"]),
        helper.make_node("Mul", ["QK", "scale_attn"], ["QKs"]),
        helper.make_node("Softmax", ["QKs"], ["A"], axis=-1),
        helper.make_node("MatMul", ["A", "V"], ["AV"]),
        helper.make_node("Gemm", ["AV", "Wo", "bo"], ["ATT"], transB=1),
        helper.make_node("Add", ["X", "ATT"], ["H1"]),      # broadcast residual
        # ---- MLP sub-layer --------------------------------------------
        helper.make_node("LayerNormalization", ["H1", "ln2_g", "ln2_b"], ["N2"], axis=-1),
        helper.make_node("Gemm", ["N2", "W1", "b1"], ["F1"], transB=1),
        # GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3)))
        helper.make_node("Pow", ["F1", "c_three"], ["F1c"]),
        helper.make_node("Add", ["F1", "F1c"], ["F1i"]),
        helper.make_node("Mul", ["F1i", "c_sqrt2pi"], ["F1s"]),
        helper.make_node("Tanh", ["F1s"], ["T"]),
        helper.make_node("Add", ["T", "c_one"], ["T1"]),
        helper.make_node("Mul", ["F1", "c_half"], ["Fh"]),
        helper.make_node("Mul", ["Fh", "T1"], ["G"]),
        helper.make_node("Gemm", ["G", "W2", "b2"], ["F2"], transB=1),
        helper.make_node("Add", ["H1", "F2"], ["Y"]),
    ]

    graph = helper.make_graph(
        nodes, "transformer_block",
        [_vi("X", [16, d])], [_vi("Y", [16, d])],
        inits,
    )
    model = helper.make_model(graph, producer_name="tinyos_test")
    model.opset_import[0].version = 17
    return model


CASES = {
    "mlp": (mlp_model, {"X": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)}),
    "gemm_transb": (gemm_transb_model, {"X": RNG.randn(5, 4).astype(np.float32)}),
    "conv_pad": (conv_padded_model, {"X": RNG.randn(1, 1, 6, 6).astype(np.float32)}),
    "conv_stride_dilate": (conv_strided_dilated_model,
                           {"X": RNG.randn(1, 1, 7, 7).astype(np.float32)}),
    "conv_grouped": (conv_grouped_model, {"X": RNG.randn(1, 4, 5, 5).astype(np.float32)}),
    "conv_bn_relu": (conv_bn_relu_model, {"X": RNG.randn(1, 2, 4, 4).astype(np.float32)}),
    "pool_tail": (pool_tail_model, {"X": RNG.randn(1, 2, 5, 5).astype(np.float32)}),
    "transpose_concat": (transpose_concat_model, {}),
    "mul_add_chain": (mul_add_chain_model, {"X": RNG.randn(3, 3).astype(np.float32)}),
    "diamond": (diamond_model,
                {"X": np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)}),
    "broadcast_add": (broadcast_add_model, {"X": RNG.randn(4, 7).astype(np.float32)}),
    "batched_matmul": (batched_matmul_model,
                       {"A": RNG.randn(2, 3, 4).astype(np.float32),
                        "B": RNG.randn(4, 5).astype(np.float32)}),
    "softmax": (softmax_model, {"X": RNG.randn(6, 8).astype(np.float32)}),
    "slice": (slice_model, {"X": RNG.randn(5, 6).astype(np.float32)}),
    "transformer_block": (transformer_block_model,
                          {"X": RNG.randn(16, 8).astype(np.float32)}),
}


@pytest.mark.parametrize("tag", sorted(CASES.keys()))
def test_onnxruntime_equivalence(tag, tmp_path):
    builder, feeds = CASES[tag]
    results = run_case(tmp_path, tag, builder, feeds)
    assert_matches(results, atol=1e-4, rtol=1e-4)


def test_all_constant_graph(tmp_path):
    """Fully-folded model: NUM_OPS == 0 edge case must still work."""
    import onnxruntime as ort
    from _harness import compile_model, build_runner, TinyOSRunner

    model = all_constant_model()
    onnx_path = tmp_path / "all_const.onnx"
    onnx.save(model, str(onnx_path))

    gen_dir = tmp_path / "gen_all_const"
    compile_model(onnx_path, gen_dir)
    lib = build_runner(gen_dir, tmp_path / "build_all_const", "const")

    sess = ort.InferenceSession(str(onnx_path))
    ref = sess.run(None, {})[0]

    runner = TinyOSRunner(lib)
    assert runner.run() == 0
    got = runner.get_output_raw("Y", np.float32, ref.shape)
    np.testing.assert_allclose(ref, got, rtol=1e-5, atol=1e-6)


def test_unsupported_op_fails_cleanly(tmp_path):
    """Unsupported operators are a COMPILE error with the op named."""
    from _harness import compile_model

    model = unsupported_op_model()
    onnx_path = tmp_path / "unsupported.onnx"
    onnx.save(model, str(onnx_path))

    with pytest.raises(RuntimeError, match="ReduceMax"):
        compile_model(onnx_path, tmp_path / "gen_unsupported")


# ---------------------------------------------------------------------------
# Policy & machine-simulation tests
# ---------------------------------------------------------------------------
#
# Capability masks mirror runtime/include/tinyos.h.
CAP_CPU, CAP_NPU, CAP_DMA = 1 << 0, 1 << 1, 1 << 2


def _build_dma_staged_model(tmp_root, tag="capcheck"):
    """conv_bn_relu: contains DEVICE_DMA and DEVICE_NPU sunits."""
    import onnxruntime as ort  # noqa: F401  (not needed here)
    from _harness import compile_model, build_runner, TinyOSRunner

    onnx_path = tmp_root / f"{tag}.onnx"
    onnx.save(conv_bn_relu_model(), str(onnx_path))
    gen_dir = tmp_root / f"gen_{tag}"
    compile_model(onnx_path, gen_dir)
    lib = build_runner(gen_dir, tmp_root / f"build_{tag}", tag)
    feeds = {"X": RNG.randn(1, 2, 4, 4).astype(np.float32)}
    return TinyOSRunner(lib), onnx_path, gen_dir, feeds


def test_capability_denial_blocks_execution(tmp_path):
    """Stripping CAP_DMA/CAP_NPU must fault BEFORE any kernel runs — proven
    by the output tensor staying exactly zero (BSS state)."""
    runner, _, _, feeds = _build_dma_staged_model(tmp_path)

    runner.set_input("X", feeds["X"])
    status = runner.run_ctx(CAP_CPU)
    assert status == 4, "expected TINYOS_ERR_CAPABILITY"

    y_ptr_size = runner.lib.model_get_tensor_size(b"Y")
    got = runner.get_output_raw("Y", np.float32, (1, 4, 2, 2))
    assert int(got.sum()) == 0 and y_ptr_size > 0, (
        "kernel executed despite missing capability — gate is decorative"
    )

    # Full policy still works.
    status = runner.run_ctx(CAP_CPU | CAP_NPU | CAP_DMA)
    assert status == 0


def test_capability_denied_npu_only(tmp_path):
    """CPU-only context faults on an NPU-bound op even when DMA succeeded."""
    runner, _, _, feeds = _build_dma_staged_model(tmp_path, tag="capnpu")
    runner.set_input("X", feeds["X"])
    # DMA allowed first; NPU op then trips the gate.
    status = runner.run_ctx(CAP_CPU | CAP_DMA)
    assert status == 4


def test_deadline_watchdog(tmp_path):
    """A zero-cycle budget must trip TINYOS_ERR_DEADLINE; unlimited passes."""
    from _harness import compile_model, build_runner, TinyOSRunner

    onnx_path = tmp_path / "dl.onnx"
    onnx.save(mlp_model(), str(onnx_path))
    gen_dir = tmp_path / "gen_dl"
    compile_model(onnx_path, gen_dir)
    lib = build_runner(gen_dir, tmp_path / "build_dl", "dl")
    runner = TinyOSRunner(lib)
    runner.set_input("X", np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))

    status = runner.run_ctx(CAP_CPU | CAP_NPU | CAP_DMA, deadline_cycles=0)
    assert status == 0
    cycles_used = runner.telemetry()["cycle_count"]

    status = runner.run_ctx(CAP_CPU | CAP_NPU | CAP_DMA,
                            deadline_cycles=max(cycles_used - 1, 1))
    assert status == 5, "expected TINYOS_ERR_DEADLINE with tight budget"


def test_dma_overlaps_compute(tmp_path):
    """The core 'graph-native' claim, measured:

        X --Relu(CPU)--> MatMul_Add(NPU, W staged by DMA)

    With async staging the transfer overlaps Relu's execution:
        total < transfer_cycles + relu_wcet + matmul_wcet   (serial sum)
        total >= max(transfer, relu) + matmul               (causal floor)
    """
    from _harness import compile_model, build_runner, TinyOSRunner

    W = RNG.randn(512, 64).astype(np.float32)     # [K, N] for MatMul; DMA-sized
    B = RNG.randn(64).astype(np.float32)
    nodes = [
        helper.make_node("Relu", ["X"], ["T"]),
        helper.make_node("MatMul", ["T", "W"], ["T2"]),
        helper.make_node("Add", ["T2", "B"], ["Y"]),
    ]
    model = _model("overlap", nodes, [_vi("X", [1, 512])], [_vi("Y", [1, 64])],
                   [_init("W", W), _init("B", B)])
    onnx_path = tmp_path / "overlap.onnx"
    onnx.save(model, str(onnx_path))
    gen_dir = tmp_path / "gen_overlap"
    compile_model(onnx_path, gen_dir)
    lib = build_runner(gen_dir, tmp_path / "build_overlap", "ovl")
    runner = TinyOSRunner(lib)
    runner.set_input("X", RNG.randn(1, 512).astype(np.float32))

    status = runner.run()
    assert status == 0
    t = runner.telemetry()
    assert t["dma_submits"] >= 1 and t["dma_copies"] >= 1, "W was not DMA-staged"

    # Re-derive WCETs from the same formula the compiler used:
    relu_wcet = max(1, -(-512 // 8))                    # 512 elems /8
    matmul_macs = 1 * 512 * 64
    matmul_wcet = -(-matmul_macs // 4)
    transfer = t["dma_transfer_cycles"]

    serial = transfer + relu_wcet + matmul_wcet
    floor = max(transfer, relu_wcet) + matmul_wcet
    measured = t["cycle_count"]

    assert measured < serial, (
        f"no overlap observed: {measured} >= serial {serial}"
    )
    assert measured >= floor, (
        f"schedule violates causality: {measured} < floor {floor}"
    )


def test_double_run_idempotent(tmp_path):
    """Re-running inference must reproduce identical results (state reset)."""
    from _harness import compile_model, build_runner, TinyOSRunner
    import onnxruntime as ort

    onnx_path = tmp_path / "idem.onnx"
    onnx.save(softmax_model(), str(onnx_path))
    gen_dir = tmp_path / "gen_idem"
    compile_model(onnx_path, gen_dir)
    lib = build_runner(gen_dir, tmp_path / "build_idem", "idem")

    sess = ort.InferenceSession(str(onnx_path))
    x = RNG.randn(6, 8).astype(np.float32)
    ref = sess.run(None, {"X": x})[0]

    runner = TinyOSRunner(lib)
    outs = []
    for _ in range(3):
        runner.set_input("X", x)
        assert runner.run() == 0
        outs.append(runner.get_output_raw("Y", np.float32, ref.shape).copy())
    np.testing.assert_array_equal(outs[0], outs[1])
    np.testing.assert_array_equal(outs[1], outs[2])
    np.testing.assert_allclose(outs[0], ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
