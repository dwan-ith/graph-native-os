/*
 * runtime/include/kernels.h
 *
 * Declarations for all reference CPU kernels.
 *
 * Every kernel follows the kernel_fn_t signature from tinyos.h:
 *   fn(inputs, n_inputs, outputs, n_outputs, params)
 * where `params` is a compiler-frozen attribute blob (see tinyos.h) and may
 * be NULL (kernel falls back to ONNX defaults).
 *
 * The generated model_exec.c references exactly the subset of these
 * required by the compiled model — unused kernels are dead-code eliminated
 * by the linker.
 */

#pragma once
#include "tinyos.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --- Data movement / views --- */
void kernel_identity(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_dma_load(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_reshape(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_flatten(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_transpose(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_concat(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Element-wise activations --- */
void kernel_relu(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_sigmoid(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_clip(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_tanh(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Element-wise binary (broadcasting) --- */
void kernel_add(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_mul(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_sub(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_div(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_pow(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Activations (extended) --- */
void kernel_leakyrelu(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_erf(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Reductions --- */
void kernel_reducemean(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Data movement (extended) --- */
void kernel_slice(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Normalization (extended) ---
 * LayerNormalization: inputs X, scale, bias; outputs Y[, mean[, inv_std]].
 */
void kernel_layernormalization(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Linear algebra --- */
void kernel_matmul(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_matmul_add(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_gemm(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Convolution --- */
void kernel_conv(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Pooling --- */
void kernel_maxpool(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_averagepool(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_globalaveragepool(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Normalization --- */
void kernel_softmax(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_batchnormalization(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

/* --- Fused kernels (produced by fusion pass) --- */
void kernel_conv_relu(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_gemm_relu(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

void kernel_add_relu(
    const tensor_desc_t * const *inputs,  uint32_t n_in,
    const tensor_desc_t * const *outputs, uint32_t n_out, const void *params);

#ifdef __cplusplus
}
#endif
