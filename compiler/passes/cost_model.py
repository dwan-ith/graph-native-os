"""
compiler/passes/cost_model.py

Static Cost Model — offline WCET estimation for the simulated machine.

For each op we emit a worst-case cycle estimate (sunit_t.wcet_cycles) used
by:
  * the deadline watchdog (model_exec_run_ctx),
  * the simulated clock that lets DMA transfers overlap compute.

These are ANALYTIC PLACEHOLDERS with a documented, deterministic formula —
the point is a stable mechanism (deadline enforcement + overlap accounting),
not silicon-calibrated performance.  All formulas assume f32 reference
kernels on one core at IPC_CYCLES_PER_ELEMENT.

Formula per family (elements = numel of largest tensor touched):
  elementwise/view : ceil(elements / ELEMS_PER_CYCLE)
  matmul/gemm      : ceil(M*K*N / MACS_PER_CYCLE)
  conv             : ceil(N*Cout*Hout*Wout*Cin*kH*kW / MACS_PER_CYCLE)
  pooling          : ceil(elements_out * kH * kW / ELEMS_PER_CYCLE)
  layernorm        : elements * 2 (mean+var passes dominate)
  reduction        : input elements
  dma transfer     : handled by the DMA engine (bytes / BW), wcet = 0
"""

from __future__ import annotations

from typing import Dict

from compiler.frontend.ir import Graph, Op

ELEMS_PER_CYCLE = 8
MACS_PER_CYCLE = 4


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= max(int(d), 1)
    return n


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def estimate_op(op: Op, graph: Graph) -> int:
    """Deterministic cycle estimate for one op."""
    t = op.op_type
    shapes = [graph.tensors[n].shape for n in op.inputs if n and not graph.tensors[n].is_constant]
    shapes += [graph.tensors[n].shape for n in op.inputs if n and graph.tensors[n].is_constant]

    def in_shape(i=0):
        name = op.inputs[i] if i < len(op.inputs) else None
        if not name:
            return ()
        return graph.tensors[name].shape

    def out_shape(i=0):
        name = op.outputs[i] if i < len(op.outputs) else None
        if not name:
            return ()
        return graph.tensors[name].shape

    if t == "DMA_LOAD":
        return 0  # cost lives on the DMA engine clock

    if t in ("MatMul", "MatMul_Add"):
        A = in_shape(0)
        B = in_shape(1) if t != "MatMul_Add" else in_shape(1)
        M = _numel(A[:-1]) if len(A) >= 2 else 1
        K = A[-1] if A else 1
        N = B[-1] if B else 1
        return _ceil_div(max(M, 1) * max(K, 1) * max(N, 1), MACS_PER_CYCLE)

    if t == "Gemm" or t == "Gemm_Relu":
        A = in_shape(0)
        B = in_shape(1)
        transA = int(op.attrs.get("transA", 0))
        transB = int(op.attrs.get("transB", 0))
        M = A[1] if transA else (A[0] if len(A) >= 1 else 1)
        K = A[0] if transA else (A[1] if len(A) >= 2 else 1)
        N = B[0] if transB else (B[1] if len(B) >= 2 else 1)
        return _ceil_div(max(M, 1) * max(K, 1) * max(N, 1), MACS_PER_CYCLE)

    if t in ("Conv", "Conv_Relu"):
        X = in_shape(0)   # N,C,H,W
        Wt = in_shape(1)  # M,C/g,kH,kW
        Y = out_shape(0)
        if len(X) == 4 and len(Wt) == 4:
            n, _, h, w = Y if len(Y) == 4 else (1, 1, 1, 1)
            macs = n * Wt[0] * h * w * Wt[1] * Wt[2] * Wt[3]
            return _ceil_div(macs, MACS_PER_CYCLE)
        return _ceil_div(_numel(out_shape(0)), MACS_PER_CYCLE)

    if t in ("MaxPool", "AveragePool"):
        k = op.attrs.get("kernel_shape") or [1, 1]
        kh, kw = int(k[0]), int(k[1])
        return _ceil_div(_numel(out_shape(0)) * kh * kw, ELEMS_PER_CYCLE)

    if t == "LayerNormalization":
        return _numel(in_shape(0)) * 2

    if t in ("ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin"):
        return _numel(in_shape(0))

    if t == "Softmax":
        return _numel(in_shape(0)) * 3   # max + exp + normalize passes

    if t == "Slice":
        return _ceil_div(_numel(out_shape(0)), ELEMS_PER_CYCLE)

    # Elementwise & views & everything else: linear in output size.
    return max(1, _ceil_div(_numel(out_shape(0)), ELEMS_PER_CYCLE))


def analyse(graph: Graph) -> Dict[int, int]:
    """Return {op_id: wcet_cycles} for every op in the graph."""
    return {op.op_id: estimate_op(op, graph) for op in graph.ops}
