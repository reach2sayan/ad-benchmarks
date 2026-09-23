/*******************************************************************************
   ddx check - both ddx rows against central-difference gradients and against
   the shared double kernels' values, once per kernel, before anything is
   timed.  Separate from ad_benchmarks so the check never sits on the timed
   path.

   Copyright (c) 2026 Sayan Samanta
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "ddx_common.hpp"

#include <cstdio>

namespace {

using ddxbench::kGradientTolerance;
// Same kernel, same inputs, same samples: roundoff, nothing more.
constexpr double kValueTolerance = 1e-10;

struct Check
{
    const char* name;
    DdxValidation (*run)();
};

// One cold build of the kernel's graph and JIT, measured inside its
// ddx_validate_* run; n/a where that run does not time them.
void print_build_times(const char* name, const DdxValidation& r)
{
    char graph[32] = "n/a";
    char jit[32] = "n/a";
    if (r.graph_build_ms >= 0.0)
        std::snprintf(graph, sizeof graph, "%.1f ms", r.graph_build_ms);
    if (r.jit_compile_ms >= 0.0)
        std::snprintf(jit, sizeof jit, "%.1f ms", r.jit_compile_ms);
    std::printf("%s  graph build %s   jit compile %s\n", name, graph, jit);
}

}  // namespace

int main()
{
    const Check checks[] = {
        {"HestonMC", ddx_validate_heston},
        {"SABRCalib", ddx_validate_sabr},
        {"XVA-CVA", ddx_validate_xva},
        {"LiborSwaption", ddx_validate_libor},
    };

    int failures = 0;
    for (const Check& c : checks)
    {
        const DdxValidation r = c.run();
        // A NaN is not < 0, so it lands on FAIL below.
        if (r.gradient_error < 0.0 || r.value_error < 0.0)
        {
            std::printf("%-14s  not implemented\n", c.name);
            continue;
        }
        const bool pass = r.gradient_error < kGradientTolerance &&
                          r.value_error < kValueTolerance;
        std::printf("%-14s  gradient %.3e  value %.3e  %s\n", c.name,
                    r.gradient_error, r.value_error, pass ? "PASS" : "FAIL");
        if (!pass)
            ++failures;
        print_build_times(c.name, r);
    }
    return failures == 0 ? 0 : 1;
}
