/*
 * runtime/power/power_mgr.c
 *
 * Phase-Aware Power Manager — Reference Implementation.
 *
 * This is the platform-independent simulation stub.  On real hardware,
 * pwr_hal_enable / pwr_hal_disable would call into the SoC's PMU driver.
 * Here they print to a telemetry log and track state for testing.
 */

#include "power.h"
#include <stdio.h>

/* -------------------------------------------------------------------------
 * HAL stubs — platform-independent simulation
 * ---------------------------------------------------------------------- */

/* Global telemetry counters (readable from test harness via ctypes) */
volatile uint32_t g_pwr_enable_count  = 0;
volatile uint32_t g_pwr_disable_count = 0;
volatile uint32_t g_pwr_active_mask   = (uint32_t)PWR_CPU; /* CPU always on */

void power_hal_enable_stub(pwr_domain_mask_t domain) {
    g_pwr_enable_count++;
    g_pwr_active_mask |= (uint32_t)domain;
    /* On real HW: write to PMU enable register, wait wakeup_latency_us */
}

void power_hal_disable_stub(pwr_domain_mask_t domain, uint32_t wakeup_latency_us) {
    (void)wakeup_latency_us;
    g_pwr_disable_count++;
    g_pwr_active_mask &= ~(uint32_t)domain;
    /* On real HW: write to PMU sleep register, set wakeup timer */
}

/* HAL function pointer defaults (can be overridden by platform BSP) */
pwr_enable_fn_t  pwr_hal_enable  = power_hal_enable_stub;
pwr_disable_fn_t pwr_hal_disable = power_hal_disable_stub;


/* -------------------------------------------------------------------------
 * Power Manager Logic — runs from ROM-backed plan
 * ---------------------------------------------------------------------- */

void power_mgr_on_frame_start(const power_domain_entry_t *plan,
                               uint32_t n_plan_entries) {
    /*
     * Pre-enable all domains required by op 0.
     * (Domains needed later will be enabled just before their first_use_op
     *  via the power_mgr_on_op_complete callback.)
     */
    for (uint32_t i = 0; i < n_plan_entries; i++) {
        if (plan[i].first_use_op == 0 || plan[i].always_on) {
            pwr_hal_enable(plan[i].domain);
        }
    }
}

void power_mgr_on_op_complete(uint32_t completed_op_id,
                               const power_domain_entry_t *plan,
                               uint32_t n_plan_entries) {
    for (uint32_t i = 0; i < n_plan_entries; i++) {
        if (plan[i].always_on) continue;

        /* Gate off a domain whose last op just finished */
        if (plan[i].last_use_op == completed_op_id) {
            pwr_hal_disable(plan[i].domain, 0U);
        }

        /* Wake up a domain just before its first op (next op = completed+1) */
        if (plan[i].first_use_op == completed_op_id + 1) {
            pwr_hal_enable(plan[i].domain);
        }
    }
}

void power_mgr_on_frame_end(const power_domain_entry_t *plan,
                              uint32_t n_plan_entries,
                              uint32_t next_frame_us) {
    /* Put every non-always-on domain to sleep with wakeup programmed */
    for (uint32_t i = 0; i < n_plan_entries; i++) {
        if (!plan[i].always_on) {
            pwr_hal_disable(plan[i].domain, next_frame_us);
        }
    }
}
