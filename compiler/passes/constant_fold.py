"""
compiler/passes/constant_fold.py

Constant Folding Pass.

Evaluates ops whose ALL inputs are constants at compile time, replaces the
op's outputs with new constant tensors, and removes the op from the graph.

Uses numpy for evaluation — this is always safe at float32 precision.
For quantized graphs, constant-foldable ops are typically only reshape/squeeze/
transpose on weights, which are integer-safe.
"""

from __future__ import annotations

import numpy as np
from typing import List

from compiler.frontend.ir import Dtype, Graph, Op, Tensor


_NUMPY_DTYPE = {
    Dtype.FLOAT32: np.float32,
    Dtype.FLOAT16: np.float16,
    Dtype.INT8:    np.int8,
    Dtype.INT16:   np.int16,
    Dtype.INT32:   np.int32,
    Dtype.INT64:   np.int64,
    Dtype.UINT8:   np.uint8,
    Dtype.BOOL:    np.bool_,
}


def _tensor_to_ndarray(t: Tensor) -> np.ndarray:
    np_dtype = _NUMPY_DTYPE[t.dtype]
    return np.frombuffer(t.data, dtype=np_dtype).reshape(t.shape)


def _ndarray_to_tensor(name: str, arr: np.ndarray, dtype: Dtype) -> Tensor:
    return Tensor(name=name, dtype=dtype, shape=tuple(arr.shape), data=arr.tobytes())


def _eval_op(op: Op, inputs: List[np.ndarray]) -> List[np.ndarray] | None:
    """Attempt to evaluate op using numpy.  Returns None if not supported."""
    t = op.op_type

    if t == "Reshape":
        shape_arr = inputs[1].tolist()
        return [inputs[0].reshape(shape_arr)]
    if t == "Flatten":
        axis = op.attrs.get("axis", 1)
        shape = inputs[0].shape
        outer = int(np.prod(shape[:axis])) if axis > 0 else 1
        inner = int(np.prod(shape[axis:]))
        return [inputs[0].reshape(outer, inner)]
    if t == "Transpose":
        perm = op.attrs.get("perm")
        return [np.transpose(inputs[0], axes=perm)]
    if t == "Squeeze":
        axes = op.attrs.get("axes")
        if len(inputs) > 1:
            axes = inputs[1].tolist()
        if axes is None:
            return [np.squeeze(inputs[0])]
        return [np.squeeze(inputs[0], axis=tuple(int(a) for a in axes))]
    if t == "Unsqueeze":
        axes = op.attrs.get("axes")
        if len(inputs) > 1:
            axes = inputs[1].tolist()
        out = inputs[0]
        for ax in sorted(int(a) for a in axes):
            out = np.expand_dims(out, axis=ax)
        return [out]
    if t == "Add" and len(inputs) == 2:
        return [inputs[0] + inputs[1]]
    if t == "Mul" and len(inputs) == 2:
        return [inputs[0] * inputs[1]]
    if t == "Sub" and len(inputs) == 2:
        return [inputs[0] - inputs[1]]
    if t == "Div" and len(inputs) == 2:
        return [inputs[0] / inputs[1]]
    if t == "Sqrt":
        return [np.sqrt(inputs[0])]
    if t == "Cast":
        to_type = op.attrs.get("to", 1)
        np_dtype = {
            1: np.float32, 6: np.int32, 7: np.int64,
            2: np.uint8, 3: np.int8,
        }.get(to_type, np.float32)
        return [inputs[0].astype(np_dtype)]
    if t == "Gather":
        axis = op.attrs.get("axis", 0)
        return [np.take(inputs[0], inputs[1], axis=axis)]
    if t == "Shape":
        return [np.array(inputs[0].shape, dtype=np.int64)]
    if t == "Concat":
        axis = op.attrs.get("axis", 0)
        return [np.concatenate(inputs, axis=axis)]
    return None


def run(graph: Graph) -> Graph:
    """Run constant folding.  Returns the same Graph object (mutated in place).

    Any op whose outputs are all computable from compile-time constants is
    evaluated.  Its output tensors become new constants.  The op is removed
    from graph.ops.
    """
    changed = True
    while changed:
        changed = False
        surviving_ops: List[Op] = []

        for op in graph.ops:
            # Check whether ALL non-empty inputs are constants
            input_tensors = [graph.tensors.get(n) for n in op.inputs if n]
            if not all(t is not None and t.is_constant for t in input_tensors):
                surviving_ops.append(op)
                continue

            input_arrays = [_tensor_to_ndarray(t) for t in input_tensors]
            results = _eval_op(op, input_arrays)
            if results is None:
                surviving_ops.append(op)
                continue

            # Replace output tensors with constants
            for out_name, arr in zip(op.outputs, results):
                out_t = graph.tensors[out_name]
                graph.tensors[out_name] = _ndarray_to_tensor(out_name, arr, out_t.dtype)

            changed = True  # Removed an op; scan again in case of chained folds

        graph.ops = surviving_ops

    return graph
