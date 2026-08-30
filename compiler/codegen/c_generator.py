"""
compiler/codegen/c_generator.py

C Image Generator.

Takes a Graph with a fully-resolved ArenaLayout (and an optional PowerPlan)
and generates:

  model_exec.h  â€” public header: arena extern, tensor descriptor array,
                  exec table type, power plan table
  model_exec.c  â€” the static arena buffer, tensor descriptors, weight
                  .rodata blobs, per-op attribute parameter blobs, and the
                  dataflow execution table walked by the graph scheduler.

The generated code has NO dynamic allocations.  Every tensor address is a
compile-time pointer into g_arena[].  Every operator execution entry is a
row in a read-only table, carrying its frozen attribute blob pointer.

model_exec_run() returns tinyos_status_t â€” faults are reported, never spun
on, so host harnesses and real firmware share one failure path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from compiler.frontend.ir import Dtype, Graph, Op
from compiler.memory.arena import ArenaLayout

try:  # optional dependency at codegen time
    from compiler.passes.power_analysis import PowerPlan
except Exception:  # pragma: no cover
    PowerPlan = None


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_C_DTYPE = {
    Dtype.FLOAT32: "float",
    Dtype.FLOAT16: "uint16_t",   # no native float16 in C99; stored raw
    Dtype.INT8:    "int8_t",
    Dtype.INT16:   "int16_t",
    Dtype.INT32:   "int32_t",
    Dtype.INT64:   "int64_t",
    Dtype.UINT8:   "uint8_t",
    Dtype.BOOL:    "uint8_t",
}

_DTYPE_ENUM = {
    Dtype.FLOAT32: "DTYPE_FLOAT32",
    Dtype.FLOAT16: "DTYPE_FLOAT16",
    Dtype.INT8:    "DTYPE_INT8",
    Dtype.INT16:   "DTYPE_INT16",
    Dtype.INT32:   "DTYPE_INT32",
    Dtype.INT64:   "DTYPE_INT64",
    Dtype.UINT8:   "DTYPE_UINT8",
    Dtype.BOOL:    "DTYPE_BOOL",
}

_DEVICE_ENUM = {
    "DEVICE_CPU": "DEVICE_CPU",
    "DEVICE_NPU": "DEVICE_NPU",
    "DEVICE_DMA": "DEVICE_DMA",
}

_DOMAIN_ENUM = {
    "PWR_CPU": "PWR_CPU",
    "PWR_NPU": "PWR_NPU",
    "PWR_DMA": "PWR_DMA",
}

# Operator types with a matching kernel_<lower> implementation in
# runtime/kernels/cpu_ref/kernels_ref.c (plus fused kernels produced by the
# fusion pass).  Anything outside this set is a compile-time error — the
# alternative is an undeclared-symbol failure inside the C toolchain, which
# tells the user nothing.
SUPPORTED_OPS = {
    # data movement / views / DMA
    "Identity", "Reshape", "Flatten", "Transpose", "Concat", "Slice",
    "DMA_LOAD",
    # element-wise
    "Relu", "Sigmoid", "Tanh", "Clip", "Add", "Sub", "Mul", "Div",
    "Pow", "Erf", "LeakyRelu",
    # linear algebra
    "MatMul", "MatMul_Add", "Gemm",
    "Gemm_Relu",
    # convolution / pooling
    "Conv", "Conv_Relu",
    "MaxPool", "AveragePool", "GlobalAveragePool",
    # normalization
    "Softmax", "BatchNormalization", "LayerNormalization",
    # reductions
    "ReduceMean",
    # fused
    "Add_Relu",
}


def _u(v) -> str:
    return f"{int(v)}U"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CGenerator:
    def __init__(
        self,
        graph: Graph,
        layout: ArenaLayout,
        out_dir: Path,
        power_plan=None,
        wcet: Optional[Dict[int, int]] = None,
    ):
        self.graph      = graph
        self.layout     = layout
        self.out_dir    = out_dir
        self.power_plan = power_plan
        self.wcet       = wcet
        out_dir.mkdir(parents=True, exist_ok=True)

        # Fail loudly on anything that would corrupt the image.
        for op in graph.ops:
            if op.op_type not in SUPPORTED_OPS:
                raise ValueError(
                    f"Unsupported operator '{op.op_type}' (op {op.op_id}). "
                    f"Supported ops: {sorted(SUPPORTED_OPS)}"
                )
        for t in graph.tensors.values():
            if len(t.shape) > 8:
                raise ValueError(
                    f"Tensor '{t.name}' has rank {len(t.shape)} > TENSOR_MAX_RANK (8)"
                )
        for op in graph.ops:
            if len([i for i in op.inputs if i]) > 16:
                raise ValueError(f"Op {op.op_id} ({op.op_type}) exceeds MAX_OP_INPUTS")
            if len(op.outputs) > 4:
                raise ValueError(f"Op {op.op_id} ({op.op_type}) exceeds MAX_OP_OUTPUTS")

        # Every non-constant activation must have a real allocation.
        for tname, t in graph.tensors.items():
            if t.is_constant or t.alias_of:
                continue
            if tname not in layout.allocations:
                raise RuntimeError(
                    f"Internal error: activation '{tname}' has no arena allocation"
                )

    def generate(self) -> None:
        """Write model_exec.h and model_exec.c into out_dir."""
        self._write_header()
        self._write_source()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _write_header(self) -> None:
        g = self.graph
        n_ops = max(len(g.ops), 1)          # C forbids zero-length arrays
        n_tensors = max(len(g.tensors), 1)
        arena_bytes = max(self.layout.arena_size, 64)

        lines = [
            "/* AUTO-GENERATED by the graph-native OS compiler â€” DO NOT EDIT */",
            "#pragma once",
            '#include "tinyos.h"',
            '#include "power.h"',
            "",
            f"/* Arena: {self.layout.arena_size} bytes "
            f"({self.layout.arena_size / 1024:.2f} KiB) */",
            f"#define ARENA_SIZE_BYTES  {arena_bytes}U",
            f"#define NUM_TENSORS       {len(g.tensors)}U",
            f"#define NUM_OPS           {len(g.ops)}U",
            "",
            "/* The single activation arena â€” lives in BSS (zero-initialised at boot) */",
            "extern uint8_t __attribute__((aligned(64))) g_arena[ARENA_SIZE_BYTES];",
            "",
            "/* Tensor descriptor table */",
            f"extern const tensor_desc_t g_tensors[{n_tensors}];",
            "",
            "/* DAG Scheduler tables */",
            f"extern const sunit_t g_sunits[{n_ops}];",
            f"extern sunit_state_t g_sunit_states[{n_ops}];",
            "extern const exec_context_t g_context;",
            "",
            "/* Offline-computed power domain schedule (ROM-backed) */",
            "#define POWER_PLAN_ENTRIES " +
            _u(len(self.power_plan.timings) if self.power_plan else 0),
            "",
            "/* Entry point: run one full inference pass. Returns tinyos_status_t. */",
            "tinyos_status_t model_exec_run(void);",
            "",
            "/* Test Harness API: retrieve a tensor by name */",
            "void* model_get_tensor_ptr(const char* name);",
            "uint32_t model_get_tensor_size(const char* name);",
        ]
        (self.out_dir / "model_exec.h").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Attribute parameter blobs
    # ------------------------------------------------------------------

    def _emit_params(self, op: Op) -> Tuple[List[str], str]:
        """Return (C declaration lines, symbol string). Empty list => NULL."""
        a = op.attrs
        t = op.op_type

        if t == "Conv":
            strides = a.get("strides", [1, 1]) or [1, 1]
            pads    = a.get("pads", [0, 0, 0, 0]) or [0, 0, 0, 0]
            dil     = a.get("dilations", [1, 1]) or [1, 1]
            group   = a.get("group", 1)
            return (
                [f"static const conv_params_t _params_op{op.op_id} = {{",
                 f"    .strides={{ {_u(strides[0])}, {_u(strides[1])} }},",
                 f"    .pads={{ {_u(pads[0])}, {_u(pads[1])}, {_u(pads[2])}, {_u(pads[3])} }},",
                 f"    .dilations={{ {_u(dil[0])}, {_u(dil[1])} }},",
                 f"    .group={_u(group)}",
                 "};"],
                f"(const void*)&_params_op{op.op_id}",
            )

        if t in ("MaxPool", "AveragePool"):
            kernel = a.get("kernel_shape") or [1, 1]
            # ONNX spec: strides default to 1 along each spatial axis
            # (NOT kernel_shape); pads default to 0.
            strides = a.get("strides") or [1, 1]
            pads = a.get("pads", [0, 0, 0, 0]) or [0, 0, 0, 0]
            return (
                [f"static const pool_params_t _params_op{op.op_id} = {{",
                 f"    .kernel={{ {_u(kernel[0])}, {_u(kernel[1])} }},",
                 f"    .strides={{ {_u(strides[0])}, {_u(strides[1])} }},",
                 f"    .pads={{ {_u(pads[0])}, {_u(pads[1])}, {_u(pads[2])}, {_u(pads[3])} }}",
                 "};"],
                f"(const void*)&_params_op{op.op_id}",
            )

        if t == "Gemm" or t == "Gemm_Relu":
            alpha = float(a.get("alpha", 1.0))
            beta = float(a.get("beta", 1.0))
            transA = int(a.get("transA", 0)) or int(a.get("consumer_transA", 0))
            transB = int(a.get("transB", 0)) or int(a.get("consumer_transB", 0))
            return (
                [f"static const gemm_params_t _params_op{op.op_id} = {{"
                 f" .alpha={alpha:.9f}f, .beta={beta:.9f}f,"
                 f" .transA={transA}, .transB={transB} }};"],
                f"(const void*)&_params_op{op.op_id}",
            )

        if t == "Softmax":
            axis = int(a.get("axis", -1))
            return (
                [f"static const axis_params_t _params_op{op.op_id} = {{ .axis={axis} }};"],
                f"(const void*)&_params_op{op.op_id}",
            )

        if t == "Concat":
            axis = int(a.get("axis", 0))
            return (
                [f"static const axis_params_t _params_op{op.op_id} = {{ .axis={axis} }};"],
                f"(const void*)&_params_op{op.op_id}",
            )

        if t == "Transpose":
            perm = list(a.get("perm", []))
            if not perm:
                return [], "NULL"
            perm_csv = ", ".join(str(int(p)) for p in perm)
            return (
                [f"static const transpose_params_t _params_op{op.op_id} = {{"
                 f" .perm={{ {perm_csv} }} }};"],
                f"(const void*)&_params_op{op.op_id}",
            )

        if t == "BatchNormalization":
            eps = float(a.get("epsilon", 1e-5))
            return (
                [f"static const bn_params_t _params_op{op.op_id} = {{"
                 f" .epsilon={eps:.9e}f }};"],
                f"(const void*)&_params_op{op.op_id}",
            )

        if t == "LayerNormalization":
            axis = int(a.get("axis", -1))
            eps = float(a.get("epsilon", 1e-5))
            return (
                [f"static const ln_params_t _params_op{op.op_id} = {{"
                 f" .axis={axis}, .epsilon={eps:.9e}f }};"],
                f"(const void*)&_params_op{op.op_id}",
            )

        if t == "ReduceMean":
            axes = list(a.get("axes") or [])
            keepdims = int(a.get("keepdims", 1))
            axes_csv = ", ".join(str(int(x)) for x in axes) or "0"
            return (
                [f"static const reduce_params_t _params_op{op.op_id} = {{",
                 f"    .axes={{ {axes_csv} }},",
                 f"    .n_axes={_u(len(axes))},",
                 f"    .keepdims={keepdims}",
                 "};"],
                f"(const void*)&_params_op{op.op_id}",
            )

        if t == "LeakyRelu":
            alpha = float(a.get("alpha", 0.01))
            return (
                [f"static const leaky_relu_params_t _params_op{op.op_id} = {{"
                 f" .alpha={alpha:.9f}f }};"],
                f"(const void*)&_params_op{op.op_id}",
            )

        return [], "NULL"

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------

    def _write_source(self) -> None:
        g = self.graph
        parts: List[str] = [
            '/* AUTO-GENERATED by the graph-native OS compiler â€” DO NOT EDIT */',
            '#include "model_exec.h"',
            '#include "kernels.h"',
            '#include <string.h>',
            '',
        ]

        # 1. Arena buffer
        parts += [
            'uint8_t __attribute__((aligned(64))) g_arena[ARENA_SIZE_BYTES];',
            '',
        ]

        # 2. Weight blobs in .rodata
        parts.append('/* Weight / constant blobs â€” live in .rodata (ROM-able) */')
        for tname, t in g.tensors.items():
            if not t.is_constant or t.data is None:
                continue
            safe_name = _safe_ident(tname)
            byte_data = ', '.join(f'0x{b:02x}' for b in t.data)
            parts.append(
                f'static const uint8_t __attribute__((aligned(64)))'
                f' _weight_{safe_name}[{len(t.data)}U] = {{{byte_data}}};'
            )
        parts.append('')

        # 3. Per-op attribute parameter blobs
        parts.append('/* Frozen operator attributes (see tinyos.h param structs) */')
        params_sym: Dict[int, str] = {}
        any_params = False
        for op in g.ops:
            lines, sym = self._emit_params(op)
            if lines:
                parts += lines
                params_sym[op.op_id] = sym
                any_params = True
        if not any_params:
            parts.append('/* (none required by this model) */')
        parts.append('')

        # 4. Tensor descriptor table
        tensor_list = list(g.tensors.values())
        tensor_index = {t.name: i for i, t in enumerate(tensor_list)}

        parts.append('const tensor_desc_t g_tensors[NUM_TENSORS == 0 ? 1 : NUM_TENSORS] = {')
        emitted_any_tensor = False
        for t in tensor_list:
            emitted_any_tensor = True
            if t.is_constant:
                ptr = f'(void*)_weight_{_safe_ident(t.name)}'
            else:
                root = t
                seen = set()
                while root.alias_of:
                    if root.name in seen:
                        raise RuntimeError(f"Alias cycle at '{root.name}'")
                    seen.add(root.name)
                    root = g.tensors[root.alias_of]
                    if root.is_constant:
                        break  # view of ROM constant: point into .rodata
                if root.is_constant:
                    ptr = f'(void*)_weight_{_safe_ident(root.name)}'
                else:
                    alloc = self.layout.allocations[root.name]
                    ptr = f'(void*)(g_arena + {_u(alloc.offset)})'

            shape_str = ', '.join(_u(d) for d in t.shape) or '0U'
            bsize = 0 if any(d < 0 for d in t.shape) else t.byte_size
            parts.append(
                f'    {{ /* {t.name!r} */'
                f' .ptr={ptr},'
                f' .dtype={_DTYPE_ENUM.get(t.dtype, "DTYPE_FLOAT32")},'
                f' .ndim={len(t.shape)}U,'
                f' .shape={{{shape_str}}},'
                f' .byte_size={_u(bsize)} }},'
            )
        if not emitted_any_tensor:
            parts.append('    {0}')
        parts.append('};')
        parts.append('')

        # 3.5 String mapping for Test Harness
        parts.append('static const char* g_tensor_names[NUM_TENSORS == 0 ? 1 : NUM_TENSORS] = {')
        if tensor_list:
            for t in tensor_list:
                parts.append(f'    "{t.name}",')
        else:
            parts.append('    "",')
        parts.append('};')
        parts.append('')

        # 5. Execution table (DAG format)
        #
        # Precompute predecessors/successors from data dependencies.
        producer_map: Dict[str, int] = {}
        for i, op in enumerate(g.ops):
            for out in op.outputs:
                producer_map[out] = i

        pred_counts = {i: set() for i in range(len(g.ops))}
        successors: Dict[int, List[int]] = {i: [] for i in range(len(g.ops))}
        for i, op in enumerate(g.ops):
            for inp in op.inputs:
                pred_id = producer_map.get(inp)
                if pred_id is None or pred_id == i:
                    continue
                if pred_id >= i:
                    raise RuntimeError(
                        "Internal error: exec table requires topological op order "
                        f"(op {pred_id} produces input of later-indexed op {i})"
                    )
                if i not in successors[pred_id]:
                    successors[pred_id].append(i)
                    pred_counts[i].add(pred_id)

        from compiler.passes.cost_model import analyse as _cost_analyse
        wcet = self.wcet if self.wcet is not None else _cost_analyse(g)

        parts.append('const sunit_t g_sunits[NUM_OPS == 0 ? 1 : NUM_OPS] = {')
        if g.ops:
            for i, op in enumerate(g.ops):
                in_indices = [tensor_index[n] for n in op.inputs if n and n in tensor_index]
                out_indices = [tensor_index[n] for n in op.outputs if n and n in tensor_index]
                kernel_sym = _kernel_symbol(op.op_type)

                succ_list = sorted(successors[i])
                pred_list = sorted(pred_counts[i])
                succ_str = ', '.join(_u(s) for s in succ_list) or '0U'
                pred_str = ', '.join(_u(s) for s in pred_list) or '0U'

                c_device = _DEVICE_ENUM.get(op.target_device, "DEVICE_CPU")

                comma = ',' if i < len(g.ops) - 1 else ''
                parts.append(
                    f'    {{ /* op {op.op_id}: {op.op_type} */'
                    f' .id={_u(i)},'
                    f' .target_device={c_device},'
                    f' .kernel={kernel_sym},'
                    f' .params={params_sym.get(op.op_id, "NULL")},'
                    f' .wcet_cycles={_u(wcet.get(op.op_id, 0))},'
                    f' .n_inputs={_u(len(in_indices))},'
                    f' .input_indices={{{", ".join(_u(x) for x in in_indices) or "0U"}}},'
                    f' .n_outputs={_u(len(out_indices))},'
                    f' .output_indices={{{", ".join(_u(x) for x in out_indices) or "0U"}}},'
                    f' .successors_count={_u(len(succ_list))},'
                    f' .successors={{{succ_str}}},'
                    f' .predecessors_count={_u(len(pred_list))},'
                    f' .predecessors={{{pred_str}}},'
                    f' .initial_dep_count={len(pred_list)} }}{comma}'
                )
        else:
            parts.append('    {0}')
        parts.append('};')
        parts.append('')

        n_ops_dim = 'NUM_OPS == 0 ? 1 : NUM_OPS'
        parts.append(f'sunit_state_t g_sunit_states[{n_ops_dim}];')
        parts.append(
            'const exec_context_t g_context = '
            '{ .id=1, .capabilities=CAP_CPU | CAP_NPU | CAP_DMA };'
        )
        parts.append('')

        # 6. Power plan (ROM table consumed by the phase-aware power manager)
        timings = self.power_plan.timings if self.power_plan else []
        always_on = self.power_plan.always_on_mask if self.power_plan else set()
        parts.append('#if POWER_PLAN_ENTRIES > 0')
        parts.append('const power_domain_entry_t g_power_plan[POWER_PLAN_ENTRIES] = {')
        for tm in timings:
            domain = _DOMAIN_ENUM.get(tm.domain, "PWR_CPU")
            ao = 1 if tm.domain in always_on else 0
            parts.append(
                f'    {{ .domain={domain},'
                f' .first_use_op={_u(tm.first_use_op)},'
                f' .last_use_op={_u(tm.last_use_op)},'
                f' .always_on={_u(ao)} }},'
            )
        parts.append('};')
        parts.append('#else')
        parts.append('const power_domain_entry_t g_power_plan[1] = { {0} };')
        parts.append('#endif')
        parts.append('')

        # 7. Dataflow scheduler (async-DMA capable, deadline-aware)
        parts += [
            '#include "dma_engine.h"',
            '',
            '/* In-flight transfer tracking: bounded by DMA_NUM_CHANNELS. */',
            'static int32_t s_inflight_op[DMA_NUM_CHANNELS];',
            'static int32_t s_inflight_ch[DMA_NUM_CHANNELS];',
            '',
            'static void release_successors(uint32_t op_id) {',
            '    const sunit_t *u = &g_sunits[op_id];',
            '    uint32_t k;',
            '    for (k = 0; k < u->successors_count; k++) {',
            '        uint32_t succ_id = u->successors[k];',
            '        g_sunit_states[succ_id].current_dep_count--;',
            '        if (g_sunit_states[succ_id].current_dep_count == 0)',
            '            g_sunit_states[succ_id].is_ready = 1;',
            '    }',
            '}',
            '',
            'static int32_t free_inflight_slot(void) {',
            '    int32_t slot = -1;',
            '    uint32_t k;',
            '    for (k = 0; k < DMA_NUM_CHANNELS; k++)',
            '        if (s_inflight_op[k] < 0) { slot = (int32_t)k; break; }',
            '    return slot;',
            '}',
            '',
            'static int32_t inflight_slot_for_op(uint32_t op_id) {',
            '    for (uint32_t k = 0; k < DMA_NUM_CHANNELS; k++)',
            '        if (s_inflight_op[k] == (int32_t)op_id) return (int32_t)k;',
            '    return -1;',
            '}',
            '',
            'tinyos_status_t model_exec_run_ctx(capability_mask_t caps,',
            '                                   uint64_t deadline_cycles) {',
            '    uint32_t i, k;',
            '    const uint32_t n_ops = NUM_OPS;',
            '    uint32_t uncompleted = n_ops;',
            '',
            '    sim_reset();',
            '    for (k = 0; k < DMA_NUM_CHANNELS; k++) {',
            '        s_inflight_op[k] = -1; s_inflight_ch[k] = -1;',
            '    }',
            '',
            '    /* 1. Reset dynamic state from static dependencies */',
            '    for (i = 0; i < n_ops; i++) {',
            '        g_sunit_states[i].current_dep_count = g_sunits[i].initial_dep_count;',
            '        g_sunit_states[i].is_ready = (g_sunits[i].initial_dep_count == 0) ? 1 : 0;',
            '        g_sunit_states[i].is_complete = 0;',
            '    }',
            '',
            '    /* 2. Pre-enable power domains required at op 0 (ROM-driven) */',
            '    power_mgr_on_frame_start(g_power_plan, POWER_PLAN_ENTRIES);',
            '',
            '    /* 3. Dispatch loop: submits DMA asynchronously, executes compute',
            '     *    when inputs are resident, polls completions each sweep. */',
            '    while (uncompleted > 0) {',
            '        uint8_t progress = 0;',
            '',
            '        /* 3a. Poll in-flight transfers first. */',
            '        for (k = 0; k < DMA_NUM_CHANNELS; k++) {',
            '            if (s_inflight_op[k] < 0 || !dma_poll(s_inflight_ch[k]))',
            '                continue;',
            '            {',
            '                uint32_t op_id = (uint32_t)s_inflight_op[k];',
            '                s_inflight_op[k] = -1;',
            '                g_sunit_states[op_id].is_complete = 1;',
            '                uncompleted--;',
            '                progress = 1;',
            '                release_successors(op_id);',
            '                power_mgr_on_op_complete(op_id, g_power_plan, POWER_PLAN_ENTRIES);',
            '            }',
            '        }',
            '',
            '        /* 3b. Scan ready ops. */',
            '        for (i = 0; i < n_ops; i++) {',
            '            if (!g_sunit_states[i].is_ready || g_sunit_states[i].is_complete)',
            '                continue;',
            '            if (inflight_slot_for_op(i) >= 0)',
            '                continue;   /* transfer still on the wire */',
            '',
            '            {',
            '                const sunit_t *u = &g_sunits[i];',
            '',
            '                /* Capability gate against the caller-supplied policy. */',
            '                if (!capability_permitted(u->target_device, caps))',
            '                    return TINYOS_ERR_CAPABILITY;',
            '',
            '                if (u->n_inputs > MAX_OP_INPUTS || u->n_outputs > MAX_OP_OUTPUTS)',
            '                    return TINYOS_ERR_SHAPE;',
            '',
            '                if (u->target_device == DEVICE_DMA) {',
            '                    /* Submit async; do NOT block the pipeline here. */',
            '                    int32_t ch = dma_submit(g_tensors[u->input_indices[0]].byte_size,',
            '                                            g_tensors[u->input_indices[0]].ptr,',
            '                                            g_tensors[u->output_indices[0]].ptr);',
            '                    int32_t slot = (ch >= 0) ? free_inflight_slot() : -1;',
            '                    if (ch >= 0 && slot >= 0) {',
            '                        s_inflight_op[slot] = (int32_t)i;',
            '                        s_inflight_ch[slot] = ch;',
            '                        progress = 1;',
            '                    }',
            '                    continue;   /* completion handled by poll */',
            '                }',
            '',
            '                /* Compute op: block only on transfers feeding its inputs. */',
            '                {',
            '                    uint32_t p;',
            '                    for (p = 0; p < u->predecessors_count; p++) {',
            '                        int32_t slot = inflight_slot_for_op(u->predecessors[p]);',
            '                        if (slot >= 0) dma_wait(s_inflight_ch[slot]);',
            '                    }',
            '                }',
            '',
            '                {',
            '                    const tensor_desc_t *ins[MAX_OP_INPUTS];',
            '                    const tensor_desc_t *outs[MAX_OP_OUTPUTS];',
            '                    uint32_t j;',
            '                    for (j = 0; j < u->n_inputs;  j++)',
            '                        ins[j] = &g_tensors[u->input_indices[j]];',
            '                    for (j = 0; j < u->n_outputs; j++)',
            '                        outs[j] = &g_tensors[u->output_indices[j]];',
            '                    u->kernel(ins, u->n_inputs, outs, u->n_outputs, u->params);',
            '                }',
            '',
            '                sim_advance(u->wcet_cycles);',
            '                if (deadline_cycles != 0 && sim_now() > deadline_cycles)',
            '                    return TINYOS_ERR_DEADLINE;',
            '',
            '                g_sunit_states[i].is_complete = 1;',
            '                g_sunit_states[i].is_ready = 0;',
            '                uncompleted--;',
            '                progress = 1;',
            '                release_successors(i);',
            '                power_mgr_on_op_complete(i, g_power_plan, POWER_PLAN_ENTRIES);',
            '            }',
            '        }',
            '',
            '        /* 3c. No progress? Advance time to next transfer completion, or die. */',
            '        if (!progress && uncompleted > 0) {',
            '            int32_t waiting = -1;',
            '            for (k = 0; k < DMA_NUM_CHANNELS; k++)',
            '                if (s_inflight_op[k] >= 0 && !dma_done(s_inflight_ch[k]))',
            '                    waiting = (int32_t)k;',
            '            if (waiting < 0)',
            '                return TINYOS_ERR_DEADLOCK;',
            '            sim_advance(dma_completion_at(s_inflight_ch[waiting]) - sim_now());',
            '        }',
            '    }',
            '',
            '    /* 4. Inter-frame deep sleep for all non-always-on domains */',
            '    power_mgr_on_frame_end(g_power_plan, POWER_PLAN_ENTRIES, 0U);',
            '    return TINYOS_OK;',
            '}',
            '',
            'tinyos_status_t model_exec_run(void) {',
            '    return model_exec_run_ctx(g_context.capabilities, 0U);',
            '}',
            '',
            'void* model_get_tensor_ptr(const char* name) {',
            '    uint32_t i;',
            '    const uint32_t n_tensors = NUM_TENSORS;',
            '    for (i = 0; i < n_tensors; i++) {',
            '        if (strcmp(name, g_tensor_names[i]) == 0) return g_tensors[i].ptr;',
            '    }',
            '    return NULL;',
            '}',
            '',
            'uint32_t model_get_tensor_size(const char* name) {',
            '    uint32_t i;',
            '    const uint32_t n_tensors = NUM_TENSORS;',
            '    for (i = 0; i < n_tensors; i++) {',
            '        if (strcmp(name, g_tensor_names[i]) == 0) return g_tensors[i].byte_size;',
            '    }',
            '    return 0;',
            '}',
        ]

        (self.out_dir / "model_exec.c").write_text("\n".join(parts) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_ident(name: str) -> str:
    """Convert a tensor name to a valid C identifier."""
    ident = ''.join(c if c.isalnum() else '_' for c in name)
    if ident and ident[0].isdigit():
        ident = '_' + ident
    return ident


def _kernel_symbol(op_type: str) -> str:
    """Map op type string to the C function pointer name in kernels.h."""
    return f'kernel_{op_type.lower()}'
