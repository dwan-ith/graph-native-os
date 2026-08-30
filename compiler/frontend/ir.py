"""
compiler/frontend/ir.py

Typed Intermediate Representation for the graph-native OS compiler.

After ingestion, every operator node and tensor edge in the ONNX graph is
converted to this IR.  Nothing outside this module should reference raw ONNX
proto objects — all subsequent passes work exclusively on IR types.
"""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------

class Dtype(Enum):
    FLOAT32 = auto()
    FLOAT16 = auto()
    INT8    = auto()
    INT16   = auto()
    INT32   = auto()
    INT64   = auto()
    UINT8   = auto()
    BOOL    = auto()

    @property
    def itemsize(self) -> int:
        _sizes = {
            Dtype.FLOAT32: 4,
            Dtype.FLOAT16: 2,
            Dtype.INT8:    1,
            Dtype.INT16:   2,
            Dtype.INT32:   4,
            Dtype.INT64:   8,
            Dtype.UINT8:   1,
            Dtype.BOOL:    1,
        }
        return _sizes[self]


_ONNX_DTYPE_MAP = {
    1:  Dtype.FLOAT32,
    10: Dtype.FLOAT16,
    3:  Dtype.INT8,
    5:  Dtype.INT16,
    6:  Dtype.INT32,
    7:  Dtype.INT64,
    2:  Dtype.UINT8,
    9:  Dtype.BOOL,
}


def dtype_from_onnx(elem_type: int) -> Dtype:
    if elem_type not in _ONNX_DTYPE_MAP:
        raise ValueError(f"Unsupported ONNX element type: {elem_type}")
    return _ONNX_DTYPE_MAP[elem_type]


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Tensor:
    """An edge in the computation graph.  May be an input, output, weight,
    constant, or intermediate activation."""

    name:  str
    dtype: Dtype
    shape: Tuple[int, ...]        # -1 encodes a dynamic dim (error later)
    data:  Optional[bytes] = None # Non-None → this tensor is a constant / weight stored in ROM
    alias_of: Optional[str] = None # Name of root tensor to share memory with (for Reshape views)

    @property
    def is_constant(self) -> bool:
        return self.data is not None

    @property
    def byte_size(self) -> int:
        if any(d < 0 for d in self.shape):
            raise RuntimeError(
                f"Tensor '{self.name}' has dynamic dimension; cannot compute byte size. "
                "All shapes must be fully static for the memory planner."
            )
        n = 1
        for d in self.shape:
            n *= d
        return n * self.dtype.itemsize

    def __repr__(self) -> str:
        kind = "const" if self.is_constant else "activation"
        return f"Tensor({self.name!r}, {self.dtype.name}, shape={self.shape}, {kind})"


# ---------------------------------------------------------------------------
# Op
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Op:
    """A node in the computation graph.

    op_type follows ONNX naming conventions (Conv, Relu, MatMul, etc.).
    After fusion passes, a fused op carries a synthesized type such as
    'Conv_Relu' and fused_from lists the original op names.
    """

    op_id:         int
    op_type:       str
    inputs:        List[str]     # tensor names (ordered)
    outputs:       List[str]     # tensor names (ordered)
    attrs:         Dict           # raw attribute dict (strings, ints, floats, lists)
    fused_from:    List[str] = dataclasses.field(default_factory=list)
    target_device: str = "DEVICE_CPU"  # "DEVICE_CPU", "DEVICE_NPU", "DEVICE_DMA"

    def __repr__(self) -> str:
        return (
            f"Op(id={self.op_id}, type={self.op_type!r}, "
            f"in={self.inputs}, out={self.outputs})"
        )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Graph:
    """The complete typed IR of a computation graph.

    Invariants maintained by the ingestion pass and checked by validate():
    - Every tensor name referenced in op.inputs / op.outputs exists in self.tensors
    - Every weight / constant has Tensor.is_constant == True
    - No tensor has a dynamic dimension (shape component == -1) in activations
    - self.ops is in topological order
    """

    name:    str
    ops:     List[Op]
    tensors: Dict[str, Tensor]  # name → Tensor
    inputs:  List[str]          # graph-level input tensor names
    outputs: List[str]          # graph-level output tensor names

    # Populated by the memory planner — not set during ingestion
    arena_size: int = 0                   # total bytes required for activation arena
    tensor_offsets: Dict[str, int] = dataclasses.field(default_factory=dict)

    def get_tensor(self, name: str) -> Tensor:
        if name not in self.tensors:
            raise KeyError(f"Unknown tensor: {name!r}")
        return self.tensors[name]

    def activation_tensors(self) -> List[Tensor]:
        """Return all non-constant tensors (i.e., tensors that live in the arena)."""
        return [t for t in self.tensors.values() if not t.is_constant]

    def validate(self) -> None:
        """Raise if the graph violates any structural invariant."""
        for op in self.ops:
            for tname in op.inputs + op.outputs:
                if tname not in self.tensors:
                    raise ValueError(
                        f"Op {op.op_id} ({op.op_type!r}) references unknown tensor {tname!r}"
                    )
        for tname in self.inputs + self.outputs:
            if tname not in self.tensors:
                raise ValueError(f"Graph I/O references unknown tensor {tname!r}")
        # All activation shapes must be fully static
        for t in self.activation_tensors():
            if any(d < 0 for d in t.shape):
                raise ValueError(
                    f"Tensor {t.name!r} has dynamic dimension {t.shape}. "
                    "All activation shapes must be statically known."
                )

    def validate_topological(self) -> None:
        """Raise unless ops are in topological order and every activation
        is written exactly once (SSA).  Run after ALL mutating passes —
        liveness and arena soundness depend on both properties."""
        producer: Dict[str, int] = {}
        for i, op in enumerate(self.ops):
            for out in op.outputs:
                if out in producer:
                    raise ValueError(
                        f"Tensor {out!r} written by ops {producer[out]} and {i}: "
                        "single-assignment violated"
                    )
                producer[out] = i
        for i, op in enumerate(self.ops):
            for inp in op.inputs:
                p = producer.get(inp)
                if inp and p is not None and p >= i:
                    raise ValueError(
                        f"Op {i} consumes output of op {p}: not topological"
                    )
        for name in self.inputs + self.outputs:
            if name not in producer and not self.tensors[name].is_constant:
                pass   # graph inputs legitimately have no producer

    def __repr__(self) -> str:
        return (
            f"Graph({self.name!r}, ops={len(self.ops)}, "
            f"tensors={len(self.tensors)}, "
            f"inputs={self.inputs}, outputs={self.outputs})"
        )
