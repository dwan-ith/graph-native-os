"""
compiler/passes/bn_absorb.py

BatchNorm Absorption Pass.

At inference time, BatchNorm parameters (scale, bias, mean, var) are fixed
constants.  They can be folded mathematically into the weights and biases of
the preceding Conv or Gemm op:

  BN(x) = scale * (x - mean) / sqrt(var + eps) + bias
         = (scale / sqrt(var + eps)) * x
           + (bias - scale * mean / sqrt(var + eps))
         = gamma * x + beta          [fused form]

For a Conv layer:
  W_fused[k] = W[k] * gamma[k]
  b_fused[k] = b[k] * gamma[k] + beta[k]    (where b may be zero if Conv has no Bias)

This eliminates the BatchNorm op entirely and reduces memory by removing
its four weight tensors.

Only fuses patterns of the exact form:
  Conv → BatchNormalization
  Gemm → BatchNormalization
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from compiler.frontend.ir import Dtype, Graph, Tensor


def _get_constant_array(graph: Graph, name: str) -> Optional[np.ndarray]:
    t = graph.tensors.get(name)
    if t is None or not t.is_constant:
        return None
    np_dtype = {
        Dtype.FLOAT32: np.float32, Dtype.FLOAT16: np.float16,
        Dtype.INT8: np.int8, Dtype.UINT8: np.uint8,
        Dtype.INT32: np.int32, Dtype.INT64: np.int64,
    }.get(t.dtype, np.float32)
    return np.frombuffer(t.data, dtype=np_dtype).reshape(t.shape).copy()


def _set_constant_array(graph: Graph, name: str, arr: np.ndarray, dtype: Dtype) -> None:
    graph.tensors[name] = Tensor(
        name=name, dtype=dtype, shape=tuple(arr.shape), data=arr.tobytes()
    )


def _find_single_consumer(graph: Graph, tensor_name: str, after_op_id: int) -> Optional[int]:
    """Return the index in graph.ops of the unique op that consumes tensor_name,
    or None if there are zero or multiple consumers."""
    consumers = [
        i for i, op in enumerate(graph.ops)
        if tensor_name in op.inputs and op.op_id > after_op_id
    ]
    return consumers[0] if len(consumers) == 1 else None


def run(graph: Graph) -> Graph:
    """Absorb all Conv/Gemm → BatchNorm patterns in the graph.

    Mutates graph in place.  Returns the same graph.
    """
    ops_to_remove = set()

    for idx, op in enumerate(graph.ops):
        if op.op_type not in ("Conv", "Gemm"):
            continue
        if len(op.outputs) != 1:
            continue

        conv_out = op.outputs[0]

        # Find the unique BatchNorm consumer of this op's output
        consumer_idx = _find_single_consumer(graph, conv_out, op.op_id)
        if consumer_idx is None:
            continue
        bn_op = graph.ops[consumer_idx]
        if bn_op.op_type != "BatchNormalization":
            continue
        if len(bn_op.inputs) < 5:
            continue

        # BN inputs: X, scale, B, mean, var
        bn_scale = _get_constant_array(graph, bn_op.inputs[1])
        bn_bias  = _get_constant_array(graph, bn_op.inputs[2])
        bn_mean  = _get_constant_array(graph, bn_op.inputs[3])
        bn_var   = _get_constant_array(graph, bn_op.inputs[4])
        if any(x is None for x in (bn_scale, bn_bias, bn_mean, bn_var)):
            continue  # Non-constant BN params: cannot absorb

        eps = float(bn_op.attrs.get("epsilon", 1e-5))

        # gamma[k] = scale[k] / sqrt(var[k] + eps)
        # beta[k]  = bias[k] - mean[k] * gamma[k]
        gamma = bn_scale / np.sqrt(bn_var + eps)
        beta  = bn_bias - bn_mean * gamma

        # Locate Conv weight and bias names
        weight_name = op.inputs[1] if len(op.inputs) > 1 else None
        bias_name   = op.inputs[2] if len(op.inputs) > 2 else None

        W = _get_constant_array(graph, weight_name) if weight_name else None
        if W is None:
            continue  # Non-constant weights; skip

        out_dtype = graph.tensors[weight_name].dtype
        np_dtype = {
            Dtype.FLOAT32: np.float32, Dtype.FLOAT16: np.float16,
            Dtype.INT8: np.int8, Dtype.UINT8: np.uint8,
            Dtype.INT32: np.int32, Dtype.INT64: np.int64,
        }.get(out_dtype, np.float32)
        gamma = gamma.astype(np_dtype)
        beta = beta.astype(np_dtype)

        # Reshape gamma so it scales along the OUTPUT-feature axis of W.
        #
        #   Conv:   W is (C_out, C_in/g, kH, kW)      → scale axis 0
        #   Gemm:   Y = op(A)·op(B) is (M, N); the N (output-feature)
        #           dimension always lives inside B/W:
        #             transB=1 (PyTorch Linear): W is (N, K) → scale axis 0
        #             transB=0:                  W is (K, N) → scale axis 1
        #           (transA only reshapes A and never moves W's feature
        #            axis.)
        if op.op_type == "Conv":
            gamma_r = gamma.reshape(-1, *([1] * (W.ndim - 1)))
        else:
            transB = int(op.attrs.get("transB", 0))
            feat_axis = 0 if transB else 1
            shape = [1] * W.ndim
            shape[feat_axis] = -1
            gamma_r = gamma.reshape(shape)

        W_fused = W * gamma_r
        _set_constant_array(graph, weight_name, W_fused, out_dtype)

        if bias_name:
            b = _get_constant_array(graph, bias_name)
            if b is not None:
                b_fused = b.astype(np_dtype) * gamma + beta
                _set_constant_array(graph, bias_name, b_fused, out_dtype)
        else:
            # No existing bias: synthesize one from beta
            new_bias_name = f"__bn_absorbed_bias_{op.op_id}"
            _set_constant_array(graph, new_bias_name, beta, out_dtype)
            op.inputs = list(op.inputs) + [new_bias_name]

        # Rewire: Conv output now goes directly to BN's output
        bn_out = bn_op.outputs[0]
        op.outputs = [bn_out]
        graph.tensors[bn_out] = graph.tensors.get(conv_out, graph.tensors[bn_out])
        graph.tensors[bn_out].data = None  # Ensure it's treated as activation

        ops_to_remove.add(consumer_idx)

    graph.ops = [op for i, op in enumerate(graph.ops) if i not in ops_to_remove]
    return graph
