/*
 * runtime/include/power.h
 *
 * Phase-Aware Power Manager ABI.
 *
 * The power manager is data-driven: every decision originates from the
 * offline-computed g_power_plan[] ROM table. At runtime, the manager
 * evaluates the table exactly once per inference frame — no heuristics,
 * no idle detection, no unexpected wakeup latency.
 *
 * Domain lifecycle per inference frame:
 *   1. Frame starts. Power manager reads g_power_plan[].
 *   2. For each domain: issue ENABLE exactly enable_before_op steps ahead.
 *   3. Scheduler dispatches g_sunits[] via DAG loop.
 *   4. After disable_after_op completes, issue SLEEP to that domain.
 *   5. After all ops complete, enter inter-frame deep-sleep on remaining domains.
 */

#pragma once
#include "tinyos.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Power domain identifiers (bit-flags for fast mask operations)
 * ---------------------------------------------------------------------- */
typedef enum {
    PWR_NONE  = 0,
    PWR_CPU   = (1 << 0),
    PWR_NPU   = (1 << 1),
    PWR_DMA   = (1 << 2),
    PWR_FLASH = (1 << 3),
} pwr_domain_mask_t;

/* -------------------------------------------------------------------------
 * Offline-computed per-domain timing entry (placed in ROM)
 * ---------------------------------------------------------------------- */
typedef struct {
    pwr_domain_mask_t domain;         /* which domain this entry governs     */
    uint32_t          first_use_op;   /* sunit index of first op using domain */
    uint32_t          last_use_op;    /* sunit index of last  op using domain */
    uint32_t          always_on;      /* 1 = never gate this domain           */
} power_domain_entry_t;

/* -------------------------------------------------------------------------
 * HAL hooks — filled in by the platform BSP
 * ---------------------------------------------------------------------- */
typedef void (*pwr_enable_fn_t) (pwr_domain_mask_t domain);
typedef void (*pwr_disable_fn_t)(pwr_domain_mask_t domain, uint32_t wakeup_latency_us);

extern pwr_enable_fn_t  pwr_hal_enable;
extern pwr_disable_fn_t pwr_hal_disable;

/* -------------------------------------------------------------------------
 * Power manager API
 * ---------------------------------------------------------------------- */

/* Called by the DAG scheduler on each op completion; gates off finished domains */
void power_mgr_on_op_complete(uint32_t completed_op_id,
                               const power_domain_entry_t *plan,
                               uint32_t n_plan_entries);

/* Called before model_exec_run() to pre-enable required domains */
void power_mgr_on_frame_start(const power_domain_entry_t *plan,
                               uint32_t n_plan_entries);

/* Called after model_exec_run() to sleep all non-always-on domains */
void power_mgr_on_frame_end  (const power_domain_entry_t *plan,
                               uint32_t n_plan_entries,
                               uint32_t next_frame_us);

/* Simulation stubs (platform-independent reference implementation) */
void power_hal_enable_stub (pwr_domain_mask_t domain);
void power_hal_disable_stub(pwr_domain_mask_t domain, uint32_t wakeup_latency_us);

#ifdef __cplusplus
}
#endif
