/*******************************************************************************
   AD Library Benchmark Suite

   Compares XAD against CppAD, Adept 2, autodiff, and ddx across four
   quantitative-finance benchmarks:
     - Heston stochastic-vol Monte Carlo
     - SABR vol-surface calibration
     - XVA CVA on a swap portfolio
     - LIBOR market model swaption portfolio

   Benchmarks 1-4 all include XAD-Codegen results alongside the tape mode.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "benchmarks.hpp"
#include "sabr.hpp"
#include "xva.hpp"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    size_t warmup = 3;
    size_t iters = 10;
    int mc_paths = 10000;
    std::string csv_file = "results/results.csv";
    std::set<std::string> skip_set;
    std::set<std::string> only_set;

    auto parse_list = [](const std::string& s, std::set<std::string>& out)
    {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ','))
            if (!item.empty()) out.insert(item);
    };

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--paths" && i + 1 < argc)
            mc_paths = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc)
            iters = static_cast<size_t>(std::atoi(argv[++i]));
        else if (arg == "--warmup" && i + 1 < argc)
            warmup = static_cast<size_t>(std::atoi(argv[++i]));
        else if (arg == "--csv" && i + 1 < argc)
            csv_file = argv[++i];
        else if (arg == "--skip" && i + 1 < argc)
            parse_list(argv[++i], skip_set);
        else if (arg == "--only" && i + 1 < argc)
            parse_list(argv[++i], only_set);
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: ad_benchmarks [options]\n"
                      << "  --paths N    MC path count (default: 10000)\n"
                      << "  --iters N    Measured iterations (default: 10)\n"
                      << "  --warmup N   Warmup iterations (default: 3)\n"
                      << "  --csv FILE   Output CSV file (default: results/results.csv)\n"
                      << "  --skip LIST  Comma-separated benchmarks to skip\n"
                      << "  --only LIST  Comma-separated benchmarks to run (overrides --skip)\n"
                      << "               Names: heston,sabr,xva,libor\n";
            return 0;
        }
    }

    auto enabled = [&](const std::string& name) -> bool
    {
        if (!only_set.empty()) return only_set.count(name) > 0;
        return skip_set.count(name) == 0;
    };

    std::vector<BenchmarkResult> results;

    auto run = [&](auto fn, const char* label)
    {
        std::cout << "  " << std::setw(14) << std::left << label << std::flush;
        results.push_back(fn());
        auto& r = results.back();
        if (r.gradient_ms < 0)
            std::cout << "N/A\n";
        else
            std::cout << std::fixed << std::setprecision(4) << r.gradient_ms << " ms\n";
    };

    std::cout << "=== AD Library Benchmark Suite ===\n"
              << "MC paths: " << mc_paths << ", iterations: " << iters
              << ", warmup: " << warmup << "\n\n";

    // --- 1. Heston MC (adjoint+JIT, 8 inputs, 100 steps/path) ---
    if (enabled("heston"))
    {
        std::cout << "--- 1. Heston MC (" << mc_paths << " paths, 8 inputs, 100 steps) ---\n";
        run([&]() { return fd_heston(mc_paths, warmup, iters); }, "FD");
#ifdef ENABLE_XAD
        run([&]() { return xad_heston(mc_paths, warmup, iters); }, "XAD");
#endif
#ifdef ENABLE_XAD_JIT
        run([&]() { return xad_jit_heston(mc_paths, warmup, iters); }, "XAD-Codegen");
#endif
#ifdef ENABLE_CPPAD
        run([&]() { return cppad_heston(mc_paths, warmup, iters); }, "CppAD");
#endif
#ifdef ENABLE_ADEPT
        run([&]() { return adept_heston(mc_paths, warmup, iters); }, "Adept");
#endif
#ifdef ENABLE_AUTODIFF
        run([&]() { return autodiff_heston(mc_paths, warmup, iters); }, "autodiff");
#endif
#ifdef ENABLE_DDX
        run([&]() { return ddx_heston(mc_paths, warmup, iters); }, "ddx");
        run([&]() { return ddx_jit_heston(mc_paths, warmup, iters); }, "ddx-JIT");
#endif
    }

    // --- 2. SABR Surface Calibration (adjoint+JIT, 15 inputs, 500 iterations) ---
    if (enabled("sabr"))
    {
        std::cout << "\n--- 2. SABR Calibration (15 inputs, " << SABR_CALIB_ITERS << " iterations) ---\n";
        run([&]() { return fd_sabr_calibration(warmup, iters); }, "FD");
#ifdef ENABLE_XAD
        run([&]() { return xad_sabr_calibration(warmup, iters); }, "XAD");
#endif
#ifdef ENABLE_XAD_JIT
        run([&]() { return xad_jit_sabr_calibration(warmup, iters); }, "XAD-Codegen");
#endif
#ifdef ENABLE_CPPAD
        run([&]() { return cppad_sabr_calibration(warmup, iters); }, "CppAD");
#endif
#ifdef ENABLE_ADEPT
        run([&]() { return adept_sabr_calibration(warmup, iters); }, "Adept");
#endif
#ifdef ENABLE_AUTODIFF
        run([&]() { return autodiff_sabr_calibration(warmup, iters); }, "autodiff");
#endif
#ifdef ENABLE_DDX
        run([&]() { return ddx_sabr_calibration(warmup, iters); }, "ddx");
        run([&]() { return ddx_jit_sabr_calibration(warmup, iters); }, "ddx-JIT");
#endif
    }

    // --- 3. XVA CVA (adjoint+JIT, 40 market inputs) ---
    if (enabled("xva"))
    {
        std::cout << "\n--- 3. XVA CVA (" << XVA_NUM_MARKET_INPUTS << " market inputs, "
                  << XVA_NUM_PATHS << " paths, " << XVA_NUM_SWAPS << " swaps) ---\n";
        // FD requires N+1 primal evaluations per gradient — 41 here. Use a
        // smaller iter count to keep total run time bounded.
        run([&]() { return fd_xva(1, std::min(iters, size_t(3))); }, "FD");
#ifdef ENABLE_XAD
        run([&]() { return xad_xva(warmup, iters); }, "XAD");
#endif
#ifdef ENABLE_XAD_JIT
        run([&]() { return xad_jit_xva(warmup, iters); }, "XAD-Codegen");
#endif
#ifdef ENABLE_CPPAD
        run([&]() { return cppad_xva(warmup, iters); }, "CppAD");
#endif
#ifdef ENABLE_ADEPT
        run([&]() { return adept_xva(warmup, iters); }, "Adept");
#endif
#ifdef ENABLE_AUTODIFF
        run([&]() { return autodiff_xva(warmup, iters); }, "autodiff");
#endif
#ifdef ENABLE_DDX
        run([&]() { return ddx_xva(warmup, iters); }, "ddx");
        run([&]() { return ddx_jit_xva(warmup, iters); }, "ddx-JIT");
#endif
    }

    // --- 4. LIBOR Swaption MC (161 inputs, JIT AVX2) ---
    if (enabled("libor"))
    {
        std::cout << "\n--- 4. LIBOR Swaption MC (" << mc_paths << " paths, 161 inputs) ---\n";
        // FD requires 162 primal evaluations per gradient — keep iters small.
        run([&]() { return fd_libor_swaption(mc_paths, 1, std::min(iters, size_t(3))); }, "FD");
#ifdef ENABLE_XAD
        run([&]() { return xad_libor_swaption(mc_paths, warmup, iters); }, "XAD");
#endif
#ifdef ENABLE_XAD_JIT
        run([&]() { return xad_jit_libor_swaption(mc_paths, warmup, iters); }, "XAD-Codegen");
#endif
#ifdef ENABLE_CPPAD
        run([&]() { return cppad_libor_swaption(mc_paths, warmup, iters); }, "CppAD");
#endif
#ifdef ENABLE_ADEPT
        run([&]() { return adept_libor_swaption(mc_paths, warmup, iters); }, "Adept");
#endif
#ifdef ENABLE_AUTODIFF
        run([&]() { return autodiff_libor_swaption(mc_paths, warmup, iters); }, "autodiff");
#endif
#ifdef ENABLE_DDX
        run([&]() { return ddx_libor_swaption(mc_paths, warmup, iters); }, "ddx");
        run([&]() { return ddx_jit_libor_swaption(mc_paths, warmup, iters); }, "ddx-JIT");
#endif
    }

    // =========================================================================
    // Output CSV
    // =========================================================================
    std::ofstream csv(csv_file);
    if (csv.is_open())
    {
        csv << "library,benchmark,primal_ms,gradient_ms\n";
        for (const auto& r : results)
            csv << r.library << "," << r.benchmark << ","
                << std::fixed << std::setprecision(6)
                << r.primal_ms << "," << r.gradient_ms << "\n";
        csv.close();
        std::cout << "\nCSV written to: " << csv_file << "\n";
    }

    // =========================================================================
    // Results Table
    // =========================================================================
    std::cout << "\n=== Results ===\n\n";

    std::string current_bench;
    double xad_time = -1;
    for (const auto& r : results)
    {
        if (r.benchmark != current_bench)
        {
            current_bench = r.benchmark;
            xad_time = -1;
            std::cout << "\n" << r.benchmark << ":\n";
        }
        if (r.gradient_ms < 0)
        {
            std::cout << "  " << std::setw(10) << std::left << r.library << "  N/A\n";
            continue;
        }
        if (r.library == "XAD" && xad_time < 0)
            xad_time = r.gradient_ms;

        std::cout << "  " << std::setw(10) << std::left << r.library
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::right
                  << r.gradient_ms << " ms";
        if (xad_time > 0 && r.library != "XAD" && r.gradient_ms > 0)
        {
            double ratio = r.gradient_ms / xad_time;
            std::cout << "  (" << std::setprecision(1) << ratio << "x)";
        }
        std::cout << "\n";
    }

    return 0;
}
