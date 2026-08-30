/*
 * runtime/kernels/cpu_ref/kernels_ref.c
 *
 * Reference CPU implementations of all supported operators.
 *
 * These are deliberately unoptimized.  Correctness is the only requirement.
 * Every operation is a straightforward C loop with no SIMD, no external
 * libraries, no dynamic allocation.
 *
 * Conventions:
 *   - Naming: kernel_<lowercase_op_type>.
 *   - All kernels accept `const void *params` (a compiler-frozen attribute
 *     blob declared in tinyos.h).  NULL means "use ONNX defaults".
 *   - Element-wise binary kernels implement NumPy-style multidirectional
 *     broadcasting.
 *   - MatMul supports N-D batched inputs with a broadcastable 2-D operand.
 */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include "tinyos.h"
#include "kernels.h"

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

#define MAXR ((int)TENSOR_MAX_RANK)

static inline float f32_at(const tensor_desc_t *t, uint32_t i) {
    return TENSOR_DATA_F32(t)[i];
}
static inline void f32_set(const tensor_desc_t *t, uint32_t i, float v) {
    TENSOR_DATA_F32(t)[i] = v;
}

static uint32_t tensor_numel(const tensor_desc_t *t) {
    uint32_t n = 1;
    for (uint8_t d = 0; d < t->ndim; d++) n *= t->shape[d];
    return n;
}

int capability_permitted(device_t dev, capability_mask_t caps) {
    switch (dev) {
        case DEVICE_CPU: return (caps & CAP_CPU)  ? 1 : 0;
        case DEVICE_NPU: return (caps & CAP_NPU)  ? 1 : 0;
        case DEVICE_DMA: return (caps & CAP_DMA)  ? 1 : 0;
        default:         return 0;
    }
}

/*
 * Broadcasting machinery.
 *
 * Given an output shape and an input descriptor, produce per-dimension
 * strides such that decomposing a flat output index into coordinates and
 * dotting with in_stride yields the flat input index.  Dimensions where the
 * input is absent or of size 1 get stride 0 (broadcast).
 */
typedef struct {
    uint32_t coords[MAXR];
} odometer_t;

static void bcast_strides(
    const tensor_desc_t *in, const tensor_desc_t *out, uint32_t stride_out[MAXR])
{
    int orank = (int)out->ndim;
    int irank = (int)in->ndim;
    int shift = orank - irank;
    for (int d = 0; d < orank; d++) {
        uint32_t idim = 0;
        if (d >= shift && (d - shift) < irank)
            idim = in->shape[d - shift];
        if (idim == out->shape[d] && idim > 0) {
            /* contiguous stride within this input */
            uint32_t s = 1;
            for (int k = d - shift + 1; k < irank; k++) s *= in->shape[k];
            stride_out[d] = s;
        } else {
            stride_out[d] = 0;   /* broadcast */
        }
    }
}

/* Flat elementwise binary over full NumPy broadcast semantics. */
static void bcast_binary_f32(
    const tensor_desc_t *a, const tensor_desc_t *b, const tensor_desc_t *o,
    float (*f)(float, float))
{
    const int rank = (int)o->ndim;
    uint32_t sa[MAXR], sb[MAXR];
    bcast_strides(a, o, sa);
    bcast_strides(b, o, sb);

    odometer_t od;
    memset(&od, 0, sizeof(od));
    const float *pa = TENSOR_DATA_F32(a);
    const float *pb = TENSOR_DATA_F32(b);
    float       *po = (float *)o->ptr;

    uint64_t total = 1;
    for (int d = 0; d < rank; d++) total *= o->shape[d];

    uint64_t ia = 0, ib = 0;
    for (uint64_t flat = 0; flat < total; flat++) {
        po[flat] = f(pa[ia], pb[ib]);
        /* odometer increment */
        for (int d = rank - 1; d >= 0; d--) {
            od.coords[d]++;
            ia += sa[d];
            ib += sb[d];
            if (od.coords[d] < o->shape[d]) break;
            od.coords[d] = 0;
            ia -= sa[d] * (uint64_t)o->shape[d];
            ib -= sb[d] * (uint64_t)o->shape[d];
        }
    }
}

static float op_add(float x, float y) { return x + y; }
static float op_sub(float x, float y) { return x - y; }
static float op_mul(float x, float y) { return x * y; }
static float op_div(float x, float y) { return x / y; }

/* -------------------------------------------------------------------------
 * Identity / views
 * ---------------------------------------------------------------------- */

void kernel_identity(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    if (in[0]->ptr != out[0]->ptr)
        memcpy(out[0]->ptr, in[0]->ptr, in[0]->byte_size);
}

void kernel_reshape(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    kernel_identity(in, n_in, out, n_out, params);
}

void kernel_flatten(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    kernel_identity(in, n_in, out, n_out, params);
}

/* -------------------------------------------------------------------------
 * DMA Hardware Simulator (ROM -> SRAM block transfer)
 * ---------------------------------------------------------------------- */

void kernel_dma_load(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    if (in[0]->ptr != out[0]->ptr)
        memcpy(out[0]->ptr, in[0]->ptr, in[0]->byte_size);
}

/* -------------------------------------------------------------------------
 * Element-wise activations
 * ---------------------------------------------------------------------- */

void kernel_relu(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    uint32_t n = tensor_numel(in[0]);
    for (uint32_t i = 0; i < n; i++) {
        float v = f32_at(in[0], i);
        f32_set(out[0], i, v > 0.f ? v : 0.f);
    }
}

void kernel_sigmoid(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    uint32_t n = tensor_numel(in[0]);
    for (uint32_t i = 0; i < n; i++)
        f32_set(out[0], i, 1.f / (1.f + expf(-f32_at(in[0], i))));
}

void kernel_tanh(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    uint32_t n = tensor_numel(in[0]);
    for (uint32_t i = 0; i < n; i++)
        f32_set(out[0], i, tanhf(f32_at(in[0], i)));
}

void kernel_clip(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_out; (void)params;
    float lo = (n_in > 1 && in[1]->ptr) ? f32_at(in[1], 0) : -3.4028235e+38f;
    float hi = (n_in > 2 && in[2]->ptr) ? f32_at(in[2], 0) :  3.4028235e+38f;
    uint32_t n = tensor_numel(in[0]);
    for (uint32_t i = 0; i < n; i++) {
        float v = f32_at(in[0], i);
        v = v < lo ? lo : (v > hi ? hi : v);
        f32_set(out[0], i, v);
    }
}

/* -------------------------------------------------------------------------
 * Element-wise binary (full broadcasting)
 * ---------------------------------------------------------------------- */

void kernel_add(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    bcast_binary_f32(in[0], in[1], out[0], op_add);
}

void kernel_sub(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    bcast_binary_f32(in[0], in[1], out[0], op_sub);
}

void kernel_mul(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    bcast_binary_f32(in[0], in[1], out[0], op_mul);
}

void kernel_div(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    bcast_binary_f32(in[0], in[1], out[0], op_div);
}

/* -------------------------------------------------------------------------
 * Fused Add_Relu
 * ---------------------------------------------------------------------- */

void kernel_add_relu(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    const int rank = (int)out[0]->ndim;
    uint32_t sa[MAXR], sb[MAXR];
    bcast_strides(in[0], out[0], sa);
    bcast_strides(in[1], out[0], sb);

    odometer_t od;
    memset(&od, 0, sizeof(od));
    const float *pa = TENSOR_DATA_F32(in[0]);
    const float *pb = TENSOR_DATA_F32(in[1]);
    float       *po = TENSOR_DATA_F32(out[0]);

    uint64_t total = 1;
    for (int d = 0; d < rank; d++) total *= out[0]->shape[d];

    uint64_t ia = 0, ib = 0;
    for (uint64_t flat = 0; flat < total; flat++) {
        float v = pa[ia] + pb[ib];
        po[flat] = v > 0.f ? v : 0.f;
        for (int d = rank - 1; d >= 0; d--) {
            od.coords[d]++;
            ia += sa[d];
            ib += sb[d];
            if (od.coords[d] < out[0]->shape[d]) break;
            od.coords[d] = 0;
            ia -= sa[d] * (uint64_t)out[0]->shape[d];
            ib -= sb[d] * (uint64_t)out[0]->shape[d];
        }
    }
}

/* -------------------------------------------------------------------------
 * MatMul — N-D batched: A[..., M, K] x B[..., K, N] -> Y[..., M, N].
 * A 2-D operand broadcasts against a higher-ranked other operand.
 * ---------------------------------------------------------------------- */

static void matmul_2d(
    const float *A, const float *B, float *Y,
    uint32_t M, uint32_t K, uint32_t N,
    int transA, int transB)
{
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            float acc = 0.f;
            for (uint32_t k = 0; k < K; k++) {
                float av = transA ? A[k * M + m] : A[m * K + k];
                float bv = transB ? B[n * K + k] : B[k * N + n];
                acc += av * bv;
            }
            Y[m * N + n] = acc;
        }
    }
}

void kernel_matmul(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    const tensor_desc_t *A = in[0], *B = in[1], *Y = out[0];
    int arank = (int)A->ndim, brank = (int)B->ndim;
    uint32_t M = A->shape[arank - 2], K = A->shape[arank - 1];
    uint32_t K2 = B->shape[brank - 2], N = B->shape[brank - 1];
    (void)K2;

    int max_rank = arank > brank ? arank : brank;
    int abatch = arank - 2, bbatch = brank - 2;

    uint32_t sA[MAXR], sB[MAXR];
    bcast_strides(A, Y, sA);
    bcast_strides(B, Y, sB);

    odometer_t od;
    memset(&od, 0, sizeof(od));
    uint64_t batch_total = 1;
    for (int d = 0; d < max_rank - 2; d++) batch_total *= Y->shape[d];

    uint64_t offA = 0, offB = 0;
    for (uint64_t bt = 0; bt < batch_total; bt++) {
        matmul_2d(
            TENSOR_DATA_F32(A) + offA,
            TENSOR_DATA_F32(B) + offB,
            TENSOR_DATA_F32(Y) + bt * (uint64_t)(M * N),
            M, K, N, 0, 0);
        for (int d = max_rank - 3; d >= 0; d--) {
            od.coords[d]++;
            offA += sA[d];
            offB += sB[d];
            if (od.coords[d] < Y->shape[d]) break;
            od.coords[d] = 0;
            offA -= sA[d] * (uint64_t)Y->shape[d];
            offB -= sB[d] * (uint64_t)Y->shape[d];
        }
    }
    (void)abatch; (void)bbatch;
}

/* -------------------------------------------------------------------------
 * MatMul_Add (compiler-fused bias epilogue): Y = A·B + bias[N]
 * ---------------------------------------------------------------------- */

void kernel_matmul_add(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    const tensor_desc_t *A = in[0], *B = in[1], *bias = in[2], *Y = out[0];
    int arank = (int)A->ndim;
    uint32_t M = A->shape[arank - 2], K = A->shape[arank - 1];
    uint32_t N = B->shape[B->ndim - 1];
    const float *bv = TENSOR_DATA_F32(bias);

    int max_rank = arank > (int)B->ndim ? arank : (int)B->ndim;
    uint32_t sA[MAXR], sB[MAXR];
    bcast_strides(A, Y, sA);
    bcast_strides(B, Y, sB);

    odometer_t od;
    memset(&od, 0, sizeof(od));
    uint64_t batch_total = 1;
    for (int d = 0; d < max_rank - 2; d++) batch_total *= Y->shape[d];

    uint64_t offA = 0, offB = 0;
    for (uint64_t bt = 0; bt < batch_total; bt++) {
        for (uint32_t m = 0; m < M; m++) {
            for (uint32_t n = 0; n < N; n++) {
                float acc = bv[n];
                for (uint32_t k = 0; k < K; k++)
                    acc += (TENSOR_DATA_F32(A) + offA)[m * K + k]
                         * (TENSOR_DATA_F32(B) + offB)[k * N + n];
                (TENSOR_DATA_F32(Y) + bt * (uint64_t)(M * N))[m * N + n] = acc;
            }
        }
        for (int d = max_rank - 3; d >= 0; d--) {
            od.coords[d]++;
            offA += sA[d];
            offB += sB[d];
            if (od.coords[d] < Y->shape[d]) break;
            od.coords[d] = 0;
            offA -= sA[d] * (uint64_t)Y->shape[d];
            offB -= sB[d] * (uint64_t)Y->shape[d];
        }
    }
}

/* -------------------------------------------------------------------------
 * Gemm: Y = alpha * op(A) · op(B) + beta * C
 * ---------------------------------------------------------------------- */

void kernel_gemm(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_out;
    const gemm_params_t *gp = (const gemm_params_t *)params;
    float alpha = gp ? gp->alpha : 1.0f;
    float beta  = gp ? gp->beta  : 1.0f;
    int transA  = gp ? gp->transA : 0;
    int transB  = gp ? gp->transB : 0;

    const tensor_desc_t *A = in[0], *B = in[1], *Y = out[0];
    uint32_t M = transA ? A->shape[1] : A->shape[0];
    uint32_t K = transA ? A->shape[0] : A->shape[1];
    uint32_t N = transB ? B->shape[0] : B->shape[1];
    const float *C = (n_in > 2 && in[2]->ptr) ? TENSOR_DATA_F32(in[2]) : NULL;

    matmul_2d(TENSOR_DATA_F32(A), TENSOR_DATA_F32(B),
              TENSOR_DATA_F32(Y), M, K, N, transA, transB);

    if (alpha != 1.0f || beta != 1.0f || C != NULL) {
        uint32_t n = tensor_numel(Y);
        for (uint32_t i = 0; i < n; i++)
            TENSOR_DATA_F32(Y)[i] =
                alpha * TENSOR_DATA_F32(Y)[i]
                + beta * (C ? C[i % tensor_numel(in[2])] : 0.0f);
    }
}

/* -------------------------------------------------------------------------
 * Gemm_Relu (fused)
 * ---------------------------------------------------------------------- */

void kernel_gemm_relu(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    kernel_gemm(in, n_in, out, n_out, params);
    uint32_t n = tensor_numel(out[0]);
    for (uint32_t i = 0; i < n; i++) {
        float v = TENSOR_DATA_F32(out[0])[i];
        TENSOR_DATA_F32(out[0])[i] = v > 0.f ? v : 0.f;
    }
}

/* -------------------------------------------------------------------------
 * Conv — NCHW, groups, arbitrary strides/pads/dilations.
 *   X: (N, C_in/group_total... C_in), W: (M, C_in/group, kH, kW), Y computed.
 * ---------------------------------------------------------------------- */

void kernel_conv(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out;
    const conv_params_t *cp = (const conv_params_t *)params;
    const uint32_t sH = cp ? cp->strides[0] : 1u;
    const uint32_t sW = cp ? cp->strides[1] : 1u;
    const uint32_t pT = cp ? cp->pads[0]    : 0u;
    const uint32_t pL = cp ? cp->pads[1]    : 0u;
    /* pads[2]/pads[3] (bottom/right) affect only output shape, which the
       compiler has already baked into the descriptor table. */
    const uint32_t dH = cp ? cp->dilations[0] : 1u;
    const uint32_t dW = cp ? cp->dilations[1] : 1u;
    const uint32_t G  = (cp && cp->group) ? cp->group : 1u;

    const uint32_t N    = in[0]->shape[0];
    const uint32_t C_in = in[0]->shape[1];
    const uint32_t H_in = in[0]->shape[2];
    const uint32_t W_in = in[0]->shape[3];
    const uint32_t M    = in[1]->shape[0];      /* output channels          */
    const uint32_t CG   = in[1]->shape[1];      /* C_in / group             */
    const uint32_t kH   = in[1]->shape[2];
    const uint32_t kW   = in[1]->shape[3];
    const uint32_t H_out= out[0]->shape[2];
    const uint32_t W_out= out[0]->shape[3];
    const uint32_t C_per_g = M / G;

    const float *X    = TENSOR_DATA_F32(in[0]);
    const float *Wt   = TENSOR_DATA_F32(in[1]);
    const float *bias = (n_in > 2 && in[2]->ptr) ? TENSOR_DATA_F32(in[2]) : NULL;
    float       *Y    = TENSOR_DATA_F32(out[0]);

    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t g = 0; g < G; g++) {
            for (uint32_t m = 0; m < C_per_g; m++) {
                const uint32_t oc = g * C_per_g + m;
                for (uint32_t h = 0; h < H_out; h++) {
                    for (uint32_t w = 0; w < W_out; w++) {
                        float acc = bias ? bias[oc] : 0.f;
                        for (uint32_t ci = 0; ci < CG; ci++) {
                            const uint32_t ic = g * CG + ci;
                            for (uint32_t kh = 0; kh < kH; kh++) {
                                const int32_t ih =
                                    (int32_t)(h * sH + kh * dH) - (int32_t)pT;
                                if (ih < 0 || ih >= (int32_t)H_in) continue;
                                for (uint32_t kw = 0; kw < kW; kw++) {
                                    const int32_t iw =
                                        (int32_t)(w * sW + kw * dW) - (int32_t)pL;
                                    if (iw < 0 || iw >= (int32_t)W_in) continue;
                                    acc += X[((n * C_in + ic) * H_in + ih) * W_in + iw]
                                         * Wt[((oc * CG + ci) * kH + kh) * kW + kw];
                                }
                            }
                        }
                        Y[((n * M + oc) * H_out + h) * W_out + w] = acc;
                    }
                }
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * Conv_Relu (fused)
 * ---------------------------------------------------------------------- */

void kernel_conv_relu(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    kernel_conv(in, n_in, out, n_out, params);
    uint32_t n = tensor_numel(out[0]);
    for (uint32_t i = 0; i < n; i++) {
        float v = TENSOR_DATA_F32(out[0])[i];
        TENSOR_DATA_F32(out[0])[i] = v > 0.f ? v : 0.f;
    }
}

/* -------------------------------------------------------------------------
 * Pooling
 * ---------------------------------------------------------------------- */

void kernel_maxpool(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out;
    const pool_params_t *pp = (const pool_params_t *)params;
    const uint32_t kH = pp ? pp->kernel[0]  : 1u;
    const uint32_t kW = pp ? pp->kernel[1]  : 1u;
    const uint32_t sH = pp ? (pp->strides[0] ? pp->strides[0] : kH) : kH;
    const uint32_t sW = pp ? (pp->strides[1] ? pp->strides[1] : kW) : kW;
    const uint32_t pT = pp ? pp->pads[0] : 0u;
    const uint32_t pL = pp ? pp->pads[1] : 0u;

    const uint32_t N = in[0]->shape[0], C = in[0]->shape[1];
    const uint32_t H_in = in[0]->shape[2], W_in = in[0]->shape[3];
    const uint32_t H_out = out[0]->shape[2], W_out = out[0]->shape[3];
    const float *X = TENSOR_DATA_F32(in[0]);
    float       *Y = TENSOR_DATA_F32(out[0]);

    for (uint32_t n = 0; n < N; n++)
        for (uint32_t c = 0; c < C; c++)
            for (uint32_t h = 0; h < H_out; h++)
                for (uint32_t w = 0; w < W_out; w++) {
                    float best = -3.4028235e+38f;
                    for (uint32_t kh = 0; kh < kH; kh++)
                        for (uint32_t kw = 0; kw < kW; kw++) {
                            int32_t ih = (int32_t)(h * sH + kh) - (int32_t)pT;
                            int32_t iw = (int32_t)(w * sW + kw) - (int32_t)pL;
                            if (ih < 0 || ih >= (int32_t)H_in) continue;
                            if (iw < 0 || iw >= (int32_t)W_in) continue;
                            float v = X[((n * C + c) * H_in + ih) * W_in + iw];
                            if (v > best) best = v;
                        }
                    Y[((n * C + c) * H_out + h) * W_out + w] = best;
                }
}

void kernel_averagepool(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out;
    const pool_params_t *pp = (const pool_params_t *)params;
    const uint32_t kH = pp ? pp->kernel[0]  : 1u;
    const uint32_t kW = pp ? pp->kernel[1]  : 1u;
    const uint32_t sH = pp ? (pp->strides[0] ? pp->strides[0] : kH) : kH;
    const uint32_t sW = pp ? (pp->strides[1] ? pp->strides[1] : kW) : kW;
    const uint32_t pT = pp ? pp->pads[0] : 0u;
    const uint32_t pL = pp ? pp->pads[1] : 0u;

    const uint32_t N = in[0]->shape[0], C = in[0]->shape[1];
    const uint32_t H_in = in[0]->shape[2], W_in = in[0]->shape[3];
    const uint32_t H_out = out[0]->shape[2], W_out = out[0]->shape[3];
    const float *X = TENSOR_DATA_F32(in[0]);
    float       *Y = TENSOR_DATA_F32(out[0]);

    for (uint32_t n = 0; n < N; n++)
        for (uint32_t c = 0; c < C; c++)
            for (uint32_t h = 0; h < H_out; h++)
                for (uint32_t w = 0; w < W_out; w++) {
                    float acc = 0.f;
                    for (uint32_t kh = 0; kh < kH; kh++)
                        for (uint32_t kw = 0; kw < kW; kw++) {
                            int32_t ih = (int32_t)(h * sH + kh) - (int32_t)pT;
                            int32_t iw = (int32_t)(w * sW + kw) - (int32_t)pL;
                            if (ih < 0 || ih >= (int32_t)H_in) continue;
                            if (iw < 0 || iw >= (int32_t)W_in) continue;
                            acc += X[((n * C + c) * H_in + ih) * W_in + iw];
                        }
                    Y[((n * C + c) * H_out + h) * W_out + w] = acc / (float)(kH * kW);
                }
}

void kernel_globalaveragepool(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    uint32_t N = in[0]->shape[0], C = in[0]->shape[1];
    uint32_t H = in[0]->ndim > 2 ? in[0]->shape[2] : 1;
    uint32_t W = in[0]->ndim > 3 ? in[0]->shape[3] : 1;
    float scl = 1.f / (float)(H * W);
    const float *X = TENSOR_DATA_F32(in[0]);
    float       *Y = TENSOR_DATA_F32(out[0]);
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t c = 0; c < C; c++) {
            float acc = 0.f;
            for (uint32_t h = 0; h < H; h++)
                for (uint32_t w = 0; w < W; w++)
                    acc += X[((n * C + c) * H + h) * W + w];
            Y[n * C + c] = acc * scl;
        }
    }
}

/* -------------------------------------------------------------------------
 * Softmax over `axis` (defaults to last dimension)
 * ---------------------------------------------------------------------- */

void kernel_softmax(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out;
    const axis_params_t *ap = (const axis_params_t *)params;
    int axis = ap ? ap->axis : -1;
    int rank = (int)in[0]->ndim;
    if (axis < 0) axis += rank;

    uint32_t dim = in[0]->shape[axis];
    uint32_t outer = 1, inner = 1;
    for (int d = 0; d < axis; d++) outer *= in[0]->shape[d];
    for (int d = axis + 1; d < rank; d++) inner *= in[0]->shape[d];

    const float *src = TENSOR_DATA_F32(in[0]);
    float       *dst = TENSOR_DATA_F32(out[0]);

    for (uint32_t o = 0; o < outer; o++) {
        for (uint32_t i = 0; i < inner; i++) {
            const float *s = src + o * dim * inner + i;
            float       *dd = dst + o * dim * inner + i;
            float mx = s[0];
            for (uint32_t j = 0; j < dim; j++)
                if (s[j * inner] > mx) mx = s[j * inner];
            float sum = 0.f;
            for (uint32_t j = 0; j < dim; j++) {
                dd[j * inner] = expf(s[j * inner] - mx);
                sum += dd[j * inner];
            }
            for (uint32_t j = 0; j < dim; j++) dd[j * inner] /= sum;
        }
    }
}

/* -------------------------------------------------------------------------
 * BatchNormalization (inference mode)
 * Supports any rank >= 2; channel axis is dim 1.
 * ---------------------------------------------------------------------- */

void kernel_batchnormalization(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out;
    const bn_params_t *bp = (const bn_params_t *)params;
    const float eps = bp ? bp->epsilon : 1e-5f;

    const tensor_desc_t *X = in[0];
    uint32_t C = X->shape[1];
    uint32_t outer = 1;                       /* prod(shape[0..0]) = N     */
    for (int d = 0; d < 1; d++) outer *= X->shape[d];
    uint32_t block = tensor_numel(X) / (outer * C);   /* trailing spatial  */

    const float *x = TENSOR_DATA_F32(X);
    float       *y = TENSOR_DATA_F32(out[0]);

    for (uint32_t o = 0; o < outer; o++) {
        for (uint32_t c = 0; c < C; c++) {
            float scale = f32_at(in[1], c);
            float bias  = f32_at(in[2], c);
            float mean  = f32_at(in[3], c);
            float var   = f32_at(in[4], c);
            float inv_std = 1.f / sqrtf(var + eps);
            const float *xs = x + ((uint64_t)o * C + c) * block;
            float       *ys = y + ((uint64_t)o * C + c) * block;
            for (uint32_t i = 0; i < block; i++)
                ys[i] = scale * (xs[i] - mean) * inv_std + bias;
        }
    }
}

/* -------------------------------------------------------------------------
 * Transpose (full permute copy)
 * ---------------------------------------------------------------------- */

void kernel_transpose(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out;
    const transpose_params_t *tp = (const transpose_params_t *)params;
    int rank = (int)in[0]->ndim;

    /* perm[d_out] = d_in */
    int perm[MAXR];
    if (tp) {
        for (int d = 0; d < rank; d++) perm[d] = tp->perm[d];
    } else {
        for (int d = 0; d < rank; d++) perm[d] = rank - 1 - d;
    }

    /* input strides per input dimension */
    uint32_t in_stride[MAXR];
    in_stride[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--)
        in_stride[d] = in_stride[d + 1] * in[0]->shape[d + 1];

    /* output strides per output dimension (row-major) */
    uint32_t out_stride[MAXR];
    out_stride[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--)
        out_stride[d] = out_stride[d + 1] * out[0]->shape[d + 1];

    odometer_t od;                       /* coordinates in OUTPUT space  */
    memset(&od, 0, sizeof(od));
    uint64_t total = 1;
    for (int d = 0; d < rank; d++) total *= out[0]->shape[d];

    const float *src = TENSOR_DATA_F32(in[0]);
    float       *dst = TENSOR_DATA_F32(out[0]);

    for (uint64_t flat = 0; flat < total; flat++) {
        uint64_t src_off = 0;
        for (int d = 0; d < rank; d++)
            src_off += (uint64_t)od.coords[d] * in_stride[perm[d]];
        dst[flat] = src[src_off];

        for (int d = rank - 1; d >= 0; d--) {
            od.coords[d]++;
            if (od.coords[d] < out[0]->shape[d]) break;
            od.coords[d] = 0;
        }
    }
}

/* -------------------------------------------------------------------------
 * Concat along `axis`
 * ---------------------------------------------------------------------- */

void kernel_concat(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_out;
    const axis_params_t *ap = (const axis_params_t *)params;
    int rank = (int)out[0]->ndim;
    int axis = ap ? ap->axis : 0;
    if (axis < 0) axis += rank;

    uint32_t outer = 1, inner = 1;
    for (int d = 0; d < axis; d++) outer *= out[0]->shape[d];
    for (int d = axis + 1; d < rank; d++) inner *= out[0]->shape[d];

    float *dst = TENSOR_DATA_F32(out[0]);
    uint32_t axis_off = 0;

    for (uint32_t i = 0; i < n_in; i++) {
        const tensor_desc_t *t = in[i];
        uint32_t ax_dim = t->shape[axis];
        const float *src = TENSOR_DATA_F32(t);
        uint64_t chunk = (uint64_t)ax_dim * inner;
        for (uint32_t o = 0; o < outer; o++) {
            uint64_t dst_base =
                ((uint64_t)o * out[0]->shape[axis] + axis_off) * inner;
            memcpy(dst + dst_base, src + (uint64_t)o * chunk,
                   (size_t)chunk * sizeof(float));
        }
        axis_off += ax_dim;
    }
}


/* -------------------------------------------------------------------------
 * Pow (broadcast): Y = A ** B
 * ---------------------------------------------------------------------- */

static float op_pow(float x, float y) { return powf(x, y); }

void kernel_pow(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    bcast_binary_f32(in[0], in[1], out[0], op_pow);
}

/* -------------------------------------------------------------------------
 * LeakyRelu: Y = max(0,X) + alpha * min(0,X)
 * ---------------------------------------------------------------------- */

void kernel_leakyrelu(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out;
    const leaky_relu_params_t *lp = (const leaky_relu_params_t *)params;
    float alpha = lp ? lp->alpha : 0.01f;
    uint32_t n = tensor_numel(in[0]);
    for (uint32_t i = 0; i < n; i++) {
        float v = f32_at(in[0], i);
        f32_set(out[0], i, v > 0.f ? v : alpha * v);
    }
}

/* -------------------------------------------------------------------------
 * Erf (elementwise) — used for GELU compositions.
 * ---------------------------------------------------------------------- */

void kernel_erf(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out; (void)params;
    uint32_t n = tensor_numel(in[0]);
    for (uint32_t i = 0; i < n; i++)
        f32_set(out[0], i, erff(f32_at(in[0], i)));
}

/* -------------------------------------------------------------------------
 * ReduceMean over `axes` with optional keepdims.
 *
 * Accumulate in the OUTPUT domain: iterate flat over the input, project
 * each element's coordinates onto output coordinates by removing reduced
 * axes, accumulate, then divide by the reduced count.  Fully general.
 * ---------------------------------------------------------------------- */

void kernel_reducemean(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in; (void)n_out;
    const reduce_params_t *rp = (const reduce_params_t *)params;

    int irank = (int)in[0]->ndim;
    int reduced[MAXR];
    memset(reduced, 0, sizeof(reduced));

    if (rp == NULL || rp->n_axes == 0) {
        for (int d = 0; d < irank; d++) reduced[d] = 1;   /* reduce all */
    } else {
        for (uint32_t k = 0; k < rp->n_axes; k++) {
            int ax = rp->axes[k];
            if (ax < 0) ax += irank;
            if (ax >= 0 && ax < irank) reduced[ax] = 1;
        }
    }

    /* input strides and output strides */
    uint32_t in_stride[MAXR];
    in_stride[irank - 1] = 1;
    for (int d = irank - 2; d >= 0; d--)
        in_stride[d] = in_stride[d + 1] * in[0]->shape[d + 1];

    uint32_t out_shape[MAXR];
    int orank = 0;
    for (int d = 0; d < irank; d++) {
        if (reduced[d]) continue;
        out_shape[orank++] = in[0]->shape[d];
    }
    if (orank == 0) { out_shape[orank++] = 1; }   /* scalar reduction */

    /* Map: output rank orank corresponds to kept dims; but with keepdims=1
       the output keeps reduced dims as 1. We accumulate via projected
       coordinates, so build projection from input dim -> out flat stride. */
    uint64_t total_in = 1;
    for (int d = 0; d < irank; d++) total_in *= in[0]->shape[d];

    uint64_t out_total = 1;
    for (int d = 0; d < orank; d++) out_total *= out_shape[d];
    float scale = (float)(total_in / out_total);

    uint32_t proj_stride[MAXR];      /* per INPUT dim, stride into OUT flat */
    {
        /* strides of the kept-dims sequence in output layout */
        uint32_t kept_stride[MAXR];
        kept_stride[orank - 1] = 1;
        for (int d = orank - 2; d >= 0; d--)
            kept_stride[d] = kept_stride[d + 1] * out_shape[d + 1];
        int k = 0;
        for (int d = 0; d < irank; d++)
            proj_stride[d] = reduced[d] ? 0 : kept_stride[k++];
    }

    odometer_t od;
    memset(&od, 0, sizeof(od));
    const float *src = TENSOR_DATA_F32(in[0]);
    float       *dst = TENSOR_DATA_F32(out[0]);
    memset(dst, 0, (size_t)tensor_numel(out[0]) * sizeof(float));

    uint64_t out_off = 0;
    for (uint64_t flat = 0; flat < total_in; flat++) {
        dst[out_off] += src[flat];
        for (int d = irank - 1; d >= 0; d--) {
            od.coords[d]++;
            out_off += proj_stride[d];
            if (od.coords[d] < in[0]->shape[d]) break;
            od.coords[d] = 0;
            out_off -= proj_stride[d] * (uint64_t)in[0]->shape[d];
        }
    }
    for (uint64_t i = 0; i < out_total; i++) dst[i] *= scale;
}

/* -------------------------------------------------------------------------
 * Slice: X[starts:ends:steps] over `axes` (defaults 0..r-1, steps default 1).
 * ONNX opset >= 10 signature: inputs X, starts, ends[, axes[, steps]].
 * ---------------------------------------------------------------------- */

void kernel_slice(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_out; (void)params;
    const tensor_desc_t *X = in[0];
    int rank = (int)X->ndim;

    const int64_t *starts = TENSOR_DATA_I64(in[1]);
    const int64_t *ends   = TENSOR_DATA_I64(in[2]);
    const int64_t *axes_v = (n_in > 3 && in[3]->ptr && tensor_numel(in[3]))
                                ? TENSOR_DATA_I64(in[3]) : NULL;
    const int64_t *steps  = (n_in > 4 && in[4]->ptr && tensor_numel(in[4]))
                                ? TENSOR_DATA_I64(in[4]) : NULL;
    uint32_t n_sel = tensor_numel(in[1]);

    /* Defaults per ONNX: full range, step 1. starts/ends/axes/steps are
       positional over `axes` (which defaults to 0..rank-1). */
    int start_c[MAXR], end_c[MAXR], step_c[MAXR];
    for (int d = 0; d < rank; d++) {
        start_c[d] = 0;
        end_c[d]   = (int)X->shape[d];
        step_c[d]  = 1;
    }
    for (uint32_t k = 0; k < n_sel; k++) {
        int ax = axes_v ? axes_v[k] : (int)k;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) continue;

        int dim = (int)X->shape[ax];
        int s = starts[k], e = ends[k], st = steps ? steps[k] : 1;
        if (st <= 0) st = 1;                    /* ref kernel: forward only */

        if (s < 0) s += dim;
        if (s < 0) s = 0;
        if (s > dim) s = dim;

        if (e < 0) e += dim;
        if (e < 0) e = 0;
        if (e > dim) e = dim;
        if (e < s) e = s;

        start_c[ax] = s;
        end_c[ax]   = e;
        step_c[ax]  = st;
    }

    /* Iterate over output coordinates; map back to input coordinates. The
       descriptor table already carries the compiler-computed shape, but we
       derive extents here too so the kernel is self-consistent. */
    odometer_t od;
    memset(&od, 0, sizeof(od));
    uint64_t total = 1;
    for (int d = 0; d < rank; d++) {
        int extent = (end_c[d] - start_c[d] + step_c[d] - 1) / step_c[d];
        if ((uint32_t)extent != out[0]->shape[d]) {
            /* Descriptor/kernel disagreement would corrupt memory: stop. */
            return;
        }
        total *= out[0]->shape[d];
    }

    uint32_t in_stride[MAXR];
    in_stride[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--)
        in_stride[d] = in_stride[d + 1] * X->shape[d + 1];

    const float *src = TENSOR_DATA_F32(X);
    float       *dst = TENSOR_DATA_F32(out[0]);

    for (uint64_t flat = 0; flat < total; flat++) {
        uint64_t src_off = 0;
        for (int d = 0; d < rank; d++)
            src_off += (uint64_t)(start_c[d] + od.coords[d] * step_c[d])
                       * in_stride[d];
        dst[flat] = src[src_off];

        for (int d = rank - 1; d >= 0; d--) {
            od.coords[d]++;
            if (od.coords[d] < out[0]->shape[d]) break;
            od.coords[d] = 0;
        }
    }
}

/* -------------------------------------------------------------------------
 * LayerNormalization (opset 17):
 *   inputs : X, Scale, Bias
 *   outputs: Y [, Mean [, InvStdDev]]
 * Normalization is computed over a suffix of dimensions starting at `axis`.
 * ---------------------------------------------------------------------- */

void kernel_layernormalization(
    const tensor_desc_t * const *in, uint32_t n_in,
    const tensor_desc_t * const *out, uint32_t n_out, const void *params)
{
    (void)n_in;
    const ln_params_t *lnp = (const ln_params_t *)params;
    int axis = lnp ? lnp->axis : -1;
    float eps = lnp ? lnp->epsilon : 1e-5f;

    const tensor_desc_t *X = in[0];
    int rank = (int)X->ndim;
    if (axis < 0) axis += rank;

    uint32_t outer = 1, norm = 1;
    for (int d = 0; d < axis; d++) outer *= X->shape[d];
    for (int d = axis; d < rank; d++) norm *= X->shape[d];

    const float *x  = TENSOR_DATA_F32(in[0]);
    const float *sc = TENSOR_DATA_F32(in[1]);
    const float *bi = TENSOR_DATA_F32(in[2]);
    float       *y  = TENSOR_DATA_F32(out[0]);

    for (uint32_t o = 0; o < outer; o++) {
        const float *xs = x + (uint64_t)o * norm;
        float       *ys = y  + (uint64_t)o * norm;

        float mean = 0.f;
        for (uint32_t i = 0; i < norm; i++) mean += xs[i];
        mean /= (float)norm;

        float var = 0.f;
        for (uint32_t i = 0; i < norm; i++) {
            float dv = xs[i] - mean;
            var += dv * dv;
        }
        var /= (float)norm;
        float inv_std = 1.f / sqrtf(var + eps);

        for (uint32_t i = 0; i < norm; i++)
            ys[i] = (xs[i] - mean) * inv_std * sc[i] + bi[i];

        if (n_out > 1 && out[1]->ptr) TENSOR_DATA_F32(out[1])[o] = mean;
        if (n_out > 2 && out[2]->ptr) TENSOR_DATA_F32(out[2])[o] = inv_std;
    }
}
