"""
compiler/main.py

Main entrypoint for the graph-native OS compiler.

Usage:
  python3 -m compiler.main path/to/model.onnx out_dir/

Pipeline:
  ingest -> constant-fold -> BN-absorb -> fuse -> device-route (+DMA)
        -> alias -> prune -> liveness -> arena (verified) -> power-plan
        -> C image
"""

import argparse
from pathlib import Path

from compiler.frontend import onnx_ingest
from compiler.passes import constant_fold, bn_absorb, fusion, device_assignment
from compiler.passes import alias_analysis, prune, power_analysis, cost_model
from compiler.memory import liveness, arena
from compiler.codegen import c_generator


def run_pipeline(model_path: Path, out_dir: Path) -> None:
    print("--- TinyOS Compiler ---")
    print(f"Loading ONNX model: {model_path}")
    graph = onnx_ingest.ingest(model_path)

    print(f"Initial ops: {len(graph.ops)}")

    print("Running optimization passes...")
    graph = constant_fold.run(graph)
    print(f" Ops after constant fold: {len(graph.ops)}")

    graph = bn_absorb.run(graph)
    print(f" Ops after BN absorb: {len(graph.ops)}")

    graph = fusion.run(graph)
    print(f" Ops after fusion: {len(graph.ops)}")

    graph = device_assignment.run(graph)
    print(f" Ops after device routing & DMA insertion: {len(graph.ops)}")

    # Re-index operations linearly
    for i, op in enumerate(graph.ops):
        op.op_id = i

    graph = alias_analysis.run(graph)
    graph = prune.run(graph)
    n_consts = sum(1 for t in graph.tensors.values() if t.is_constant)
    print(f" After pruning: {len(graph.tensors)} tensors ({n_consts} constants)")

    # Defense-in-depth: every pass above mutates the graph. Re-verify the
    # invariants that memory planning and scheduling depend on.
    graph.validate()
    graph.validate_topological()

    print("Running memory planning...")
    live_ranges = liveness.compute_liveness(graph)
    layout = arena.allocate(graph, live_ranges)
    lower_bound = arena.peak_live_bytes(graph, live_ranges)

    print(layout.report(lower_bound=lower_bound))

    print("Running power & cost analysis...")
    plan = power_analysis.analyse(graph)
    print(plan.report())
    wcet = cost_model.analyse(graph)

    print(f"Generating C image in {out_dir}/ ...")
    codegen = c_generator.CGenerator(graph, layout, out_dir, power_plan=plan, wcet=wcet)
    codegen.generate()

    print("Done. Zero-malloc kernel generated successfully.")


def main():
    parser = argparse.ArgumentParser(description="Graph-Native edge OS compiler.")
    parser.add_argument("model", type=str, help="Path to input ONNX model")
    parser.add_argument("out_dir", type=str, help="Output directory for generated C code")
    args = parser.parse_args()

    run_pipeline(Path(args.model), Path(args.out_dir))

if __name__ == "__main__":
    main()
