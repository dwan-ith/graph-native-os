/*
 * runtime/include/tinyos.h
 *
 * Core kernel ABI — public types shared between the generated model code
 * and the runtime kernel.
 *
 * Design invariants:
 *   - No dynamic memory allocation anywhere in this API.
 *   - All tensor addresses are pointers into g_arena[] or .rodata.
 *   - The exec table is statically populated at compile time.
 *   - Every operator attribute resolved by the compiler (strides, pads,
 *     transposes, axes, ...) is frozen into a read-only params blob that
 *     is passed to the kernel via sunit_t.params.  Kernels never parse
 *     attributes at runtime.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Dtype
 * ---------------------------------------------------------------------- */

typedef enum {
    DTYPE_FLOAT32 = 0,
    DTYPE_FLOAT16 = 1,
    DTYPE_INT8    = 2,
    DTYPE_INT16   = 3,
    DTYPE_INT32   = 4,
    DTYPE_INT64   = 5,
    DTYPE_UINT8   = 6,
    DTYPE_BOOL    = 7,
} dtype_t;

static inline uint32_t dtype_itemsize(dtype_t dt) {
    static const uint32_t sizes[] = { 4, 2, 1, 2, 4, 8, 1, 1 };
    return sizes[(uint32_t)dt];
}


/* -------------------------------------------------------------------------
 * Tensor Descriptor
 *
 * ptr points either into g_arena[] (activation) or into .rodata (constant).
 * ---------------------------------------------------------------------- */

#define TENSOR_MAX_RANK  8U

typedef struct {
    void     *ptr;                        /* base address of tensor data    */
    dtype_t   dtype;
    uint8_t   ndim;                       /* number of dimensions           */
    uint32_t  shape[TENSOR_MAX_RANK];     /* size per dimension             */
    uint32_t  byte_size;                  /* total bytes (product*itemsize) */
} tensor_desc_t;

/* Convenience accessors */
#define TENSOR_DATA_F32(desc)  ((float *)   (desc)->ptr)
#define TENSOR_DATA_I8(desc)   ((int8_t *)  (desc)->ptr)
#define TENSOR_DATA_U8(desc)   ((uint8_t *) (desc)->ptr)
#define TENSOR_DATA_I32(desc)  ((int32_t *) (desc)->ptr)
#define TENSOR_DATA_I64(desc)  ((int64_t *) (desc)->ptr)


/* -------------------------------------------------------------------------
 * Static parameter blobs (emitted into .rodata by the compiler)
 *
 * Every struct below mirrors a set of ONNX operator attributes.  A kernel
 * receives `const void *params` and casts it to the matching type; when a
 * NULL is received the kernel must fall back to ONNX defaults.
 * ---------------------------------------------------------------------- */

typedef struct {
    uint32_t strides[2];    /* [H, W]                                       */
    uint32_t pads[4];       /* [top, left, bottom, right]                   */
    uint32_t dilations[2];  /* [H, W]                                       */
    uint32_t group;
} conv_params_t;

typedef struct {
    uint32_t kernel[2];     /* [kH, kW]                                     */
    uint32_t strides[2];    /* defaults to kernel_shape when zero           */
    uint32_t pads[4];       /* [top, left, bottom, right], default 0        */
} pool_params_t;

typedef struct {
    float   alpha;          /* default 1.0f                                 */
    float   beta;           /* default 1.0f                                 */
    int32_t transA;
    int32_t transB;
} gemm_params_t;

typedef struct {
    int32_t axis;           /* negative values allowed (from the end)       */
} axis_params_t;            /* used by Softmax / Concat / Squeeze ...       */

typedef struct {
    int32_t perm[TENSOR_MAX_RANK];
} transpose_params_t;

typedef struct {
    float epsilon;          /* default 1e-5f                                */
} bn_params_t;

typedef struct {
    int32_t axis;           /* normalization begins here; default -1        */
    float   epsilon;        /* default 1e-5f                                */
} ln_params_t;              /* LayerNormalization                           */

typedef struct {
    int32_t axes[TENSOR_MAX_RANK];
    uint32_t n_axes;
    int32_t keepdims;       /* 0 or 1                                       */
} reduce_params_t;          /* ReduceMean / ReduceSum / ...                 */

typedef struct {
    float alpha;            /* default 0.01f                                */
} leaky_relu_params_t;


/* -------------------------------------------------------------------------
 * Kernel function type
 *
 * Every operator implements this signature.  Kernels MUST NOT call malloc,
 * free, or any dynamic allocator.
 * ---------------------------------------------------------------------- */

typedef void (*kernel_fn_t)(
    const tensor_desc_t * const *inputs,
    uint32_t                     n_inputs,
    const tensor_desc_t * const *outputs,
    uint32_t                     n_outputs,
    const void                  *params
);


/* -------------------------------------------------------------------------
 * Execution Context and Capabilities
 * ---------------------------------------------------------------------- */

typedef enum {
    CAP_NONE    = 0,
    CAP_CPU     = (1 << 0),
    CAP_NPU     = (1 << 1),
    CAP_DMA     = (1 << 2),
    CAP_SENSORS = (1 << 3)
} capability_mask_t;

typedef struct {
    uint32_t          id;
    capability_mask_t capabilities;
} exec_context_t;

/* -------------------------------------------------------------------------
 * Graph Scheduler Primitives
 * ---------------------------------------------------------------------- */

#define MAX_OP_INPUTS     16U
#define MAX_OP_OUTPUTS    4U
#define MAX_SUCCESSORS    16U

typedef enum {
    DEVICE_CPU = 0,
    DEVICE_NPU = 1,
    DEVICE_DMA = 2
} device_t;

/* Capability gate: nonzero iff `caps` grants use of `dev`. */
int capability_permitted(device_t dev, capability_mask_t caps);

/* Capability gate for a device (declared here, defined in kernels_ref.c
 * and inlined by optimisers). */
int capability_permitted(device_t dev, capability_mask_t caps);

/* A Schedulable Unit mapping exactly to a graph operation or data transfer */
typedef struct {
    uint32_t    id;
    device_t    target_device;
    kernel_fn_t kernel;
    const void *params;                 /* static attr blob (.rodata), may be NULL */
    uint32_t    wcet_cycles;            /* offline worst-case estimate (sim clock) */
    uint32_t    n_inputs;
    uint32_t    input_indices[MAX_OP_INPUTS];
    uint32_t    n_outputs;
    uint32_t    output_indices[MAX_OP_OUTPUTS];
    uint32_t    successors_count;
    uint32_t    successors[MAX_SUCCESSORS];      /* IDs of dependent ops     */
    uint32_t    predecessors_count;
    uint32_t    predecessors[MAX_SUCCESSORS];    /* IDs of producing ops     */
    int32_t     initial_dep_count;               /* #predecessors (static)   */
} sunit_t;

/* Dynamic state for each sunit during inference (kept in RAM) */
typedef struct {
    int32_t  current_dep_count;
    uint8_t  is_ready;
    uint8_t  is_complete;
} sunit_state_t;

/* -------------------------------------------------------------------------
 * Error / status codes
 * ---------------------------------------------------------------------- */

typedef enum {
    TINYOS_OK              = 0,
    TINYOS_ERR_NULL_PTR    = 1,
    TINYOS_ERR_SHAPE       = 2,
    TINYOS_ERR_DTYPE       = 3,
    TINYOS_ERR_CAPABILITY  = 4,
    TINYOS_ERR_DEADLINE    = 5,
    TINYOS_ERR_DEADLOCK    = 6,
    TINYOS_ERR_UNSUPPORTED = 7,
} tinyos_status_t;

/* -------------------------------------------------------------------------
 * Execution entry points (defined in the generated model image)
 *
 *   model_exec_run()       — full capabilities, no deadline
 *   model_exec_run_ctx()   — policy-controlled execution:
 *                              caps             enforced per sunit device
 *                              deadline_cycles  0 = unlimited; simulated
 *                                clock is checked after every op
 * ---------------------------------------------------------------------- */

tinyos_status_t model_exec_run(void);
tinyos_status_t model_exec_run_ctx(capability_mask_t caps,
                                   uint64_t deadline_cycles);


#ifdef __cplusplus
}
#endif
