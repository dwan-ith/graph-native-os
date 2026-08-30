/*
 * runtime/dispatch/dma_engine.h
 *
 * Deterministic simulated DMA engine for the host-side runtime.
 *
 * The engine owns BOTH the clock and the data movement: transfers carry
 * (src, dst, bytes); the payload is copied exactly once, at simulated
 * completion (inside dma_poll / dma_wait).  Compute ops advance the clock
 * by their offline WCET estimate (sunit_t.wcet_cycles), so real overlap
 * between staging and compute is modelled and measurable.
 *
 * Everything is cycle-deterministic: identical inputs produce identical
 * schedules, so tests can assert exact overlap bounds.
 */

#pragma once
#include "tinyos.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DMA_NUM_CHANNELS          2U

/* Simulated fabric bandwidth (bytes per cycle). */
#ifndef DMA_BW_BYTES_PER_CYCLE
#define DMA_BW_BYTES_PER_CYCLE    16U
#endif

/* Reset clock + all channels. Called at the start of each inference run. */
void sim_reset(void);

/* Current simulated cycle. */
uint64_t sim_now(void);

/* Advance the clock (compute execution time). */
void sim_advance(uint64_t cycles);

/* Submit an async transfer of byte_count from src to dst. Returns channel
 * id, or -1 if every channel is busy. */
int dma_submit(uint32_t byte_count, const void *src, void *dst);

/* Pure predicate: nonzero iff channel ch has finished its transfer window.
 * Does NOT copy data or change state. */
int dma_done(int ch);

/* Simulated completion cycle of ch (valid even while still busy). */
uint64_t dma_completion_at(int ch);

/* Non-blocking completion: if the transfer window closed, perform the
 * payload copy, release the channel, and return 1. Otherwise return 0. */
int dma_poll(int ch);

/* Blocking wait: advances the clock to completion if needed, performs the
 * copy, releases the channel, returns the current time. */
uint64_t dma_wait(int ch);

/* Telemetry (read from test harnesses via ctypes). */
extern volatile uint64_t g_cycle_count;
extern volatile uint32_t g_dma_submits;
extern volatile uint32_t g_dma_copies;
extern volatile uint64_t g_dma_transfer_cycles;   /* cumulative busy cycles */

#ifdef __cplusplus
}
#endif
