"""
tests/e2e/test_fuzz.py

Property-based end-to-end fuzzing.

For each seed we synthesize a random computation graph from the supported
operator subset, then require exact pipeline equivalence with onnxruntime:

    random ONNX -> tiny-os compiler -> gcc -> execute  ==  onnxruntime

Generator failures (invalid shape combinations) are skipped — this suite
tests the COMPILER, not the generator.  Any tiny-os compile/build/mismatch
failure is a real bug and fails the run.

Seed count is controlled by TINYOS_FUZZ_SEEDS (default 24).
"""

import os
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto

sys.path.insert(0, str(Path(__file__).parent))
from _harness import run_case, assert_matches  # noqa: E402

N_SEEDS = int(os.environ.get("TINYOS_FUZZ_SEEDS", "24"))


class ShapeTracker:
    """Builds a valid DAG op-by-op, tracking tensor shapes."""

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.shapes = {}
        self.nodes = []
        self.inits = []
        self.inputs = []
        self.counter = 0

    def fresh(self, prefix):
        name = f"{prefix}_{self.counter}"
        self.counter += 1
        return name

    def add_init(self, arr, name=None):
        name = name or self.fresh("c")
        self.inits.append(
            onnx.numpy_helper.from_array(arr.astype(np.float32), name)
        )
        self.shapes[name] = tuple(arr.shape)
        return name

    # -- candidate generators; each returns output name or None ----------

    def _input(self):
        r = self.rng.randint(0, 2)
        shape = [(8, 8), (6, 10)][r]
        name = "X"
        self.shapes[name] = shape
        self.inputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))
        )
        return name

    def unary(self, src):
        kind = self.rng.choice(["Relu", "Sigmoid", "Tanh", "Erf", "Identity"])
        out = self.fresh(kind.lower())
        self.nodes.append(helper.make_node(kind, [src], [out]))
        self.shapes[out] = self.shapes[src]
        return out

    def binary(self, a, b):
        sa, sb = self.shapes[a], self.shapes[b]
        out = self.fresh("bin")
        kind = str(self.rng.choice(["Add", "Mul", "Sub"]))
        if sa == sb:
            pass                                   # elementwise exact
        elif len(sb) == 1 and sa[-1] == sb[0]:
            pass                                   # broadcast bias-style
        else:
            b = a                                  # fall back to a+a
            sb = sa
        self.nodes.append(helper.make_node(kind, [a, b], [out]))
        # result shape = sa (both supported patterns preserve it)
        self.shapes[out] = sa
        return out

    def matmul_like(self, src):
        s = self.shapes[src]
        if len(s) != 2:
            return None
        k_in = s[1]
        n_out = int(self.rng.choice([4, 8]))
        w = self.rng.randn(n_out, k_in).astype(np.float32) * 0.2
        w_name = self.add_init(w)
        use_bias = self.rng.rand() < 0.7
        inputs = [src, w_name]
        b_shape = (n_out,)
        if use_bias:
            b_name = self.add_init(self.rng.randn(*b_shape).astype(np.float32) * 0.1)
            inputs.append(b_name)
        out = self.fresh("gemm")
        attrs = {"transB": 1}
        self.nodes.append(helper.make_node("Gemm", inputs, [out], **attrs))
        self.shapes[out] = (s[0], n_out)
        return out

    def conv(self, src):
        s = self.shapes[src]
        if len(s) != 4:
            return None
        n, c, h, w = s
        cout = int(self.rng.choice([c, 4]))
        k = 3
        weight = (self.rng.randn(cout, c, k, k) * 0.2).astype(np.float32)
        w_name = self.add_init(weight)
        out = self.fresh("conv")
        self.nodes.append(helper.make_node(
            "Conv", [src, w_name], [out], strides=[1, 1], pads=[1, 1, 1, 1]))
        self.shapes[out] = (n, cout, h, w)
        return out

    def pool(self, src):
        s = self.shapes[src]
        if len(s) != 4 or s[2] % 2 or s[3] % 2 or s[2] < 2 or s[3] < 2:
            return None
        out = self.fresh("pool")
        self.nodes.append(helper.make_node(
            "MaxPool", [src], [out], kernel_shape=[2, 2], strides=[2, 2]))
        self.shapes[out] = (s[0], s[1], s[2] // 2, s[3] // 2)
        return out

    def gap(self, src):
        if len(self.shapes[src]) != 4:
            return None
        out = self.fresh("gap")
        self.nodes.append(helper.make_node("GlobalAveragePool", [src], [out]))
        s = self.shapes[src]
        self.shapes[out] = (s[0], s[1], 1, 1)
        return out

    def softmax(self, src):
        out = self.fresh("softmax")
        self.nodes.append(helper.make_node("Softmax", [src], [out], axis=-1))
        self.shapes[out] = self.shapes[src]
        return out

    def reshape_flatten(self, src):
        total = int(np.prod(self.shapes[src]))
        out = self.fresh("flat")
        self.nodes.append(helper.make_node("Flatten", [src], [out], axis=1))
        first = self.shapes[src][0]
        self.shapes[out] = (first, total // first)
        return out

    def transpose(self, src):
        rank = len(self.shapes[src])
        perm = list(range(rank))
        self.rng.shuffle(perm)
        out = self.fresh("tr")
        self.nodes.append(helper.make_node("Transpose", [src], [out], perm=perm))
        s = self.shapes[src]
        self.shapes[out] = tuple(s[p] for p in perm)
        return out


_GENERATORS = ["unary", "binary", "matmul_like", "conv", "pool", "gap",
               "softmax", "reshape_flatten", "transpose"]
_WEIGHTS = [5, 4, 4, 2, 2, 1, 2, 2, 2]


def generate_model(seed: int):
    rng = np.random.RandomState(seed)
    t = ShapeTracker(rng)

    # Alternate between vector/matrix mode and image mode.
    x = t._input()
    current = [x]

    for _ in range(int(rng.randint(4, 12))):
        src = current[-1]
        gen = t.rng.choice(_GENERATORS, p=np.array(_WEIGHTS) / sum(_WEIGHTS))
        if gen == "binary":
            # second operand: an earlier tensor if shapes allow, else self
            b_src = current[-2] if len(current) >= 2 else src
            result = t.binary(src, b_src)
        else:
            result = getattr(t, gen)(src)
        if result is not None:
            current.append(result)

    used = [n for n in current if n != "X"]
    if not used:                       # degenerate; force one op
        used.append(t.unary(x))
    outputs = [used[-1]]
    out_infos = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, list(t.shapes[n]))
        for n in outputs
    ]
    graph = helper.make_graph(t.nodes, f"fuzz_{seed}", t.inputs, out_infos, t.inits)
    model = helper.make_model(graph, producer_name="fuzz")
    model.opset_import[0].version = 13
    return model, {i.name: None for i in t.inputs}


@pytest.mark.parametrize("seed", range(1000, 1000 + N_SEEDS))
def test_fuzz_seed(seed, tmp_path):
    model, feeds_spec = generate_model(seed)

    try:
        onnx.checker.check_model(model)
    except Exception as e:
        pytest.skip(f"generator produced invalid model for seed {seed}: {e}")

    import onnxruntime as ort
    sess_inputs = {i.name: i for i in model.graph.input}
    feeds = {}
    for name, vi in sess_inputs.items():
        shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        feeds[name] = np.random.RandomState(seed).randn(*shape).astype(np.float32)

    try:
        ort.InferenceSession(model.SerializeToString())
    except Exception as e:
        pytest.skip(f"onnxruntime rejected generated model (seed {seed}): {e}")

    onnx_path = tmp_path / f"fuzz_{seed}.onnx"
    onnx.save(model, str(onnx_path))

    results = run_case(tmp_path, f"fuzz{seed}", lambda: model, feeds,
                       atol=1e-4, rtol=1e-4)
    assert_matches(results, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
