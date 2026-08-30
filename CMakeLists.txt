cmake_minimum_required(VERSION 3.10)
project(TinyOS C)

set(CMAKE_C_STANDARD 99)

# Include paths for the public kernel ABI, DMA sim, and the generated model
include_directories(
    runtime/include
    runtime/dispatch
    generated
)

# Reference kernels + simulated DMA fabric + phase-aware power manager
set(KERNEL_SRC
    runtime/kernels/cpu_ref/kernels_ref.c
    runtime/dispatch/dma_engine.c
    runtime/power/power_mgr.c
)

# Generated model (must be output from compiler module into generated/)
set(GENERATED_SRC
    generated/model_exec.c
)

# The library we can load via ctypes for validation
add_library(tinyos_runner SHARED
    ${KERNEL_SRC}
    ${GENERATED_SRC}
)

# Minimal properties
target_compile_options(tinyos_runner PRIVATE -O3 -Wall -Wextra)
