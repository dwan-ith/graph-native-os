/*
 * runtime/dispatch/dma_engine.c
 *
 * Reference implementation of the simulated DMA fabric.  See dma_engine.h.
 *
 * Scheduling policy: a transfer occupies its channel from submission until
 * `done_at = max(now, channel_free_at) + bytes / BW`.  The max() models a
 * fabric where a late submit cannot start before the current time, while
 * back-to-back submissions on the same channel queue naturally.
 *
 * Data movement contract: the payload copy happens EXACTLY ONCE, at the
 * first dma_poll/dma_wait observation of completion.  A transfer that is
 * never observed (kernel faulted earlier) leaves destination untouched —
 * matching hardware, where an aborted transaction does not write memory.
 */

#include "dma_engine.h"
#include <string.h>

volatile uint64_t g_cycle_count         = 0;
volatile uint32_t g_dma_submits         = 0;
volatile uint32_t g_dma_copies          = 0;
volatile uint64_t g_dma_transfer_cycles = 0;

typedef struct {
    uint8_t      in_use;
    uint64_t     done_at;
    const void  *src;
    void        *dst;
    uint32_t     bytes;
} dma_channel_t;

static dma_channel_t s_channels[DMA_NUM_CHANNELS];

void sim_reset(void) {
    g_cycle_count         = 0;
    g_dma_submits         = 0;
    g_dma_copies          = 0;
    g_dma_transfer_cycles = 0;
    for (uint32_t i = 0; i < DMA_NUM_CHANNELS; i++) {
        s_channels[i].in_use  = 0;
        s_channels[i].done_at = 0;
        s_channels[i].src = NULL;
        s_channels[i].dst = NULL;
        s_channels[i].bytes = 0;
    }
}

uint64_t sim_now(void) {
    return g_cycle_count;
}

void sim_advance(uint64_t cycles) {
    g_cycle_count += cycles;
}

int dma_submit(uint32_t byte_count, const void *src, void *dst) {
    for (uint32_t i = 0; i < DMA_NUM_CHANNELS; i++) {
        if (!s_channels[i].in_use) {
            uint64_t start = (g_cycle_count > s_channels[i].done_at)
                                 ? g_cycle_count : s_channels[i].done_at;
            uint64_t len = ((uint64_t)byte_count + DMA_BW_BYTES_PER_CYCLE - 1)
                           / DMA_BW_BYTES_PER_CYCLE;
            if (len == 0) len = 1;   /* zero-byte transfers still cost a beat */
            s_channels[i].done_at = start + len;
            s_channels[i].in_use  = 1;
            s_channels[i].src     = src;
            s_channels[i].dst     = dst;
            s_channels[i].bytes   = byte_count;
            g_dma_submits++;
            g_dma_transfer_cycles += len;
            return (int)i;
        }
    }
    return -1;
}

int dma_done(int ch) {
    if (ch < 0 || (uint32_t)ch >= DMA_NUM_CHANNELS) return 1;
    return (g_cycle_count >= s_channels[ch].done_at) ? 1 : 0;
}

uint64_t dma_completion_at(int ch) {
    if (ch < 0 || (uint32_t)ch >= DMA_NUM_CHANNELS) return g_cycle_count;
    return s_channels[ch].done_at;
}

static int complete_if_ready(int ch) {
    dma_channel_t *c = &s_channels[ch];
    if (!c->in_use) return 1;                       /* already completed */
    if (g_cycle_count < c->done_at) return 0;       /* still on the wire */
    if (c->dst && c->src)
        memcpy(c->dst, c->src, c->bytes);
    g_dma_copies++;
    c->in_use = 0;
    return 1;
}

int dma_poll(int ch) {
    if (ch < 0 || (uint32_t)ch >= DMA_NUM_CHANNELS) return 1;
    return complete_if_ready(ch);
}

uint64_t dma_wait(int ch) {
    if (ch < 0 || (uint32_t)ch >= DMA_NUM_CHANNELS) return g_cycle_count;
    if (s_channels[ch].done_at > g_cycle_count)
        g_cycle_count = s_channels[ch].done_at;
    (void)complete_if_ready(ch);
    return g_cycle_count;
}
