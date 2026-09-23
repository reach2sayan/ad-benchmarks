/*******************************************************************************
   Benchmark interface - each library implements these functions.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#pragma once

#include "timing.hpp"

/// Functions returning {lib, bench, -1.0, -1.0} indicate N/A.

// --- Finite Differences (baseline) ---
BenchmarkResult fd_heston(int numPaths, size_t warmup, size_t iters);
BenchmarkResult fd_sabr_calibration(size_t warmup, size_t iters);
BenchmarkResult fd_xva(size_t warmup, size_t iters);
BenchmarkResult fd_libor_swaption(int numPaths, size_t warmup, size_t iters);

// --- XAD ---
#ifdef ENABLE_XAD
BenchmarkResult xad_heston(int numPaths, size_t warmup, size_t iters);
BenchmarkResult xad_sabr_calibration(size_t warmup, size_t iters);
BenchmarkResult xad_xva(size_t warmup, size_t iters);
BenchmarkResult xad_libor_swaption(int numPaths, size_t warmup, size_t iters);
#endif

#ifdef ENABLE_XAD_JIT
BenchmarkResult xad_jit_heston(int numPaths, size_t warmup, size_t iters);
BenchmarkResult xad_jit_sabr_calibration(size_t warmup, size_t iters);
BenchmarkResult xad_jit_xva(size_t warmup, size_t iters);
BenchmarkResult xad_jit_libor_swaption(int numPaths, size_t warmup, size_t iters);
#endif

// --- CppAD ---
#ifdef ENABLE_CPPAD
BenchmarkResult cppad_heston(int numPaths, size_t warmup, size_t iters);
BenchmarkResult cppad_sabr_calibration(size_t warmup, size_t iters);
BenchmarkResult cppad_xva(size_t warmup, size_t iters);
BenchmarkResult cppad_libor_swaption(int numPaths, size_t warmup, size_t iters);
#endif

// --- Adept ---
#ifdef ENABLE_ADEPT
BenchmarkResult adept_heston(int numPaths, size_t warmup, size_t iters);
BenchmarkResult adept_sabr_calibration(size_t warmup, size_t iters);
BenchmarkResult adept_xva(size_t warmup, size_t iters);
BenchmarkResult adept_libor_swaption(int numPaths, size_t warmup, size_t iters);
#endif

// --- autodiff ---
#ifdef ENABLE_AUTODIFF
BenchmarkResult autodiff_heston(int numPaths, size_t warmup, size_t iters);
BenchmarkResult autodiff_sabr_calibration(size_t warmup, size_t iters);
BenchmarkResult autodiff_xva(size_t warmup, size_t iters);
BenchmarkResult autodiff_libor_swaption(int numPaths, size_t warmup, size_t iters);
#endif

// --- ddx ---
#ifdef ENABLE_DDX
BenchmarkResult ddx_heston(int numPaths, size_t warmup, size_t iters);
BenchmarkResult ddx_sabr_calibration(size_t warmup, size_t iters);
BenchmarkResult ddx_xva(size_t warmup, size_t iters);
BenchmarkResult ddx_libor_swaption(int numPaths, size_t warmup, size_t iters);

BenchmarkResult ddx_jit_heston(int numPaths, size_t warmup, size_t iters);
BenchmarkResult ddx_jit_sabr_calibration(size_t warmup, size_t iters);
BenchmarkResult ddx_jit_xva(size_t warmup, size_t iters);
BenchmarkResult ddx_jit_libor_swaption(int numPaths, size_t warmup, size_t iters);

#endif
