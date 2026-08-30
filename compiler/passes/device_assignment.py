"""
compiler/passes/device_assignment.py

Heterogeneous device assignment and DMA injection pass.

This pass emulates the 'Schedule Synthesizer' cost model from the TinyOS
architecture. It walks the graph and:
  1. Routes heavy tensor contractions (Conv, MatMul) to DEVICE_NPU.
  2. Routes element-wise and activation operations to DEVICE_CPU.
  3. Preemptively inserts explicit "DMA_LOAD" operations into the DAG before
     any NPU node that consumes large constant weights (ROM). This forces the
     Dataflow Scheduler to execute a DMA transfer on DEVICE_DMA, exposing
     overlap opportunities with preceding CPU computations.
"""

from __future__ import annotations
from typing import Dict, List

from compiler.frontend.ir import Graph, Op, Tensor

_NPU_OPS = {"Conv", "MatMul", "MatMul_Add", "Conv_Relu", "Gemm_Relu"}

# Constants smaller than this stay directly readable from ROM: simulating a
# DMA block transfer for an 8-byte bias costs more than the transfer itself.
_DMA_MIN_BYTES = 256

def run(graph: Graph) -> Graph:
    new_ops: List[Op] = []

    # Unique names for DMA SRAM buffers.  A constant consumed by several NPU
    # ops is fetched ONCE and shared — its live range then spans all
    # consumers, which liveness/arena handle correctly.
    sram_for_const: Dict[str, str] = {}
    dma_counter = 0

    for original_op in graph.ops:
        if original_op.op_type in _NPU_OPS:
            original_op.target_device = "DEVICE_NPU"

            # Identify large constant weights that should be staged via DMA
            new_inputs = list(original_op.inputs)
            for i, in_name in enumerate(original_op.inputs):
                if not in_name:
                    continue
                if in_name in sram_for_const:
                    # Reuse the existing SRAM staging buffer + its DMA op.
                    new_inputs[i] = sram_for_const[in_name]
                    continue
                t = graph.tensors[in_name]
                if t.is_constant and t.byte_size >= _DMA_MIN_BYTES:
                    # Insert a DMA_LOAD op
                    sram_name = f"_dma_sram_{in_name}_{dma_counter}"
                    dma_counter += 1

                    # SRAM tensor lives in the arena (data=None)
                    graph.tensors[sram_name] = Tensor(
                        name=sram_name,
                        dtype=t.dtype,
                        shape=t.shape,
                        data=None
                    )

                    dma_op = Op(
                        op_id=len(graph.ops) + dma_counter,  # re-indexed later
                        op_type="DMA_LOAD",
                        inputs=[in_name],     # src is ROM weight
                        outputs=[sram_name],  # dst is SRAM arena buffer
                        attrs={"byte_size": t.byte_size},
                        target_device="DEVICE_DMA"
                    )
                    new_ops.append(dma_op)

                    sram_for_const[in_name] = sram_name
                    new_inputs[i] = sram_name

            original_op.inputs = new_inputs

        else:
            original_op.target_device = "DEVICE_CPU"

        new_ops.append(original_op)

    graph.ops = new_ops
    return graph
