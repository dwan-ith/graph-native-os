# Graph-Native Operating System

An embedded inference runtime whose scheduler, memory subsystem, power
manager, and capability enforcer are all driven by the **full static
computation graph** — compiled offline into a single ROM-able C image with
zero heap allocation at runtime.

## What is actually implemented (Phase 0/1)

The system is a host-side simulation of the target edge SoC: the "NPU" is a
device label plus reference kernels executing on the CPU, and the DMA engine
is a deterministic cycle-accurate simulator (`runtime/dispatch/dma_engine.c`)
that owns both the clock and the payload copies. Within that scope, the
following guarantees hold and are enforced by tests:

1. **Static memory.** Every activation lives at a compile-time offset in one
   arena (`g_arena[]`). The allocator runs two placement strategies (GFFD and
   linear-scan), verifies each with `verify()` — pairwise address-disjointness
   for simultaneously-live tensors, alias-root containment, bounds — and
   returns the smaller. `peak_live_bytes()` provides an interval-graph lower
   bound; reports include allocator efficiency against it.
2. **Zero malloc.** Neither the generated image nor any kernel allocates;
   execution is a pure function of arena + .rodata tables.
3. **Numeric equivalence, fuzzed.** A 24-model zoo plus seeded random-graph
   fuzzing (`tests/e2e/test_fuzz.py`, `TINYOS_FUZZ_SEEDS` to scale) match
   onnxruntime at `rtol/atol = 1e-4` — covering Conv (strided/padded/
   dilated/grouped), Gemm transposes, batched MatMul, LayerNormalization,
   Softmax, Slice, ReduceMean, pooling, views, fused kernels, diamond
   dataflow, and a full pre-norm transformer block.
4. **Fail-loud compilation.** Unsupported operators, control-flow ops,
   external-data models, out-of-range opsets, dimensionally-broken graphs,
   rank > 8, missing allocations, alias cycles, and SSA violations all abort
   compilation with named diagnostics.
5. **Determinism.** Identical input models produce byte-identical C images;
   identical inputs produce cycle-identical simulated schedules.
6. **Enforced policy.** `model_exec_run_ctx(caps, deadline)` gates every op's
   device against caller-supplied capabilities and checks the simulated clock
   after every op against a cycle deadline. Negative tests prove kernels do
   not execute when a capability is absent (output memory stays untouched).
7. **Measured overlap.** DMA staging is asynchronous: transfers occupy
   channels for `bytes/BW` simulated cycles while compute advances the same
   clock by per-op WCET (offline analytic model). Tests assert total time is
   strictly below the serial sum yet above the causal floor.

## Architecture

```
tiny-os/
├── compiler/
│   ├── frontend/       ONNX ingestion → typed IR (topological, shape-inferred)
│   ├── passes/         constant-fold · BN-absorb · fusion · device routing
│   │                   (+ DMA insertion) · alias analysis · dead-tensor
│   │                   pruning · power-domain analysis
│   ├── memory/         tensor liveness → verified static arena allocation
│   └── codegen/        C image generator (descriptors, attr param blobs,
│                       DAG exec table, power plan)
├── runtime/
│   ├── include/        kernel ABI: tensor descriptors, params structs,
│   │                   capability masks, status codes
│   ├── kernels/cpu_ref Reference kernels (broadcasting elementwise,
│   │                   N-D matmul, grouped conv, pools, softmax, ...)
│   └── power/          phase-aware power manager (ROM plan-driven)
├── generated/          compiler output (model_exec.{h,c})
└── tests/
    ├── unit/           planner property tests (200 random DAGs), pass contracts
    └── e2e/            compile → gcc → ctypes vs onnxruntime equivalence zoo
```

### Key design points

* **Attribute freezing.** Operator attributes (Conv strides/pads/dilations/
  group, Gemm transposes, Softmax/LayerNorm axis, BN epsilon, ...) become
  read-only param blobs in `.rodata`, passed to kernels via `sunit_t.params`.
  Kernels never parse attributes at runtime; `NULL` means ONNX defaults.
* **Dataflow scheduling with async DMA.** Ops carry static predecessor/
  successor lists; the dispatch loop submits DEVICE_DMA rows to the simulated
  fabric without blocking, waits only when a consumer's inputs are not yet
  resident, polls completions, and advances a deterministic clock by each
  op's offline WCET. Faults return typed status codes
  (`TINYOS_ERR_CAPABILITY`, `TINYOS_ERR_DEADLINE`, `TINYOS_ERR_DEADLOCK`).
* **In-place aliasing.** Unary elementwise ops reuse their input's buffer
  when it dies at that op; binary ops additionally require exact shape match
  (broadcast writes are never in-place); contiguous reshapes are zero-copy
  views. Liveness folds alias chains so roots span their viewers' ranges.
* **Phase-aware power.** The offline pass computes per-domain
  `[first_use_op, last_use_op]` windows from device assignment; the runtime
  enables a domain just before its first op and gates it off after its last,
  with an inter-frame deep sleep. CPU is always-on by construction.
* **Defense in depth.** After every mutating pass the pipeline re-validates
  graph invariants (SSA single-assignment, topological order) before memory
  planning; codegen independently re-checks exec-table invariants.

## Running

```bash
pip install -r requirements.txt

# compile a model
python -m compiler.main path/to/model.onnx generated/

# full test suite: unit + e2e zoo + fuzz (Windows note: if %TEMP% is broken,
# add --basetemp=<dir>; see conftest.py)
python -m pytest tests -q --basetemp=./tmp/pytest

# more fuzzing
TINYOS_FUZZ_SEEDS=200 python -m pytest tests/e2e/test_fuzz.py -q

# lint
ruff check compiler tests --select E,F,W
```

## Known limitations (honest list)

* Host simulation only: no real NPU hardware (DEVICE_NPU runs a CPU kernel);
  DMA is a cycle-accurate simulator, not real hardware.
* WCET figures are documented analytic placeholders — the deadline/overlap
  mechanisms are real, the calibration is not.
* Reference kernels are unoptimized scalar C (correctness first); float32
  compute only (fp16/int8 storage types parse but kernels compute f32).
* Dynamic shapes rejected by design (fully-static requirement).
* Constant folding supports a fixed numpy-evaluable subset; anything else
  must have a kernel (whitelist-checked).
* Power gating assumes ascending-index execution (guaranteed statically:
  predecessors always precede successors in the exec table).
