/*******************************************************************************
   ddx SABR Surface Calibration benchmark - graph built once and replayed;
   interpreter and 4-lane JIT rows.
   Rows: ddx (interpreter); ddx-JIT (compiled, 4 lanes) -- graph build and
   LLVM compile happen once, outside the timed region; their cost is reported
   by ad_benchmarks_ddx_validate.

   15 inputs, no samples.  sabr_calibration_objective branches only on
   doubles, so it is recorded as is with T = RTExpression<double>.

   Copyright (c) 2026 Sayan Samanta
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/sabr.hpp"
#include "ddx_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <vector>

namespace {

using ddxbench::RE;

// The shared objective, unchanged: params[p] is input p.
auto sabr_kernel(const SABRCalibData& data)
{
    return [&data](std::span<const RE> x, std::span<const RE>)
    { return sabr_calibration_objective<RE>(x.data(), data); };
}

// The ddx-JIT gradient sweep over a compiled graph: 4 evaluations per call.
void sabr_jit_sweep(ddxbench::Jit& jit, const SABRCalibData& data,
                    const std::vector<SABRPerturbation>& schedule)
{
    constexpr int BATCH = ddxbench::kLanes;

    // --- Execute the kernel: 4 evaluations per call ---
    const int numBatches = (SABR_CALIB_ITERS + BATCH - 1) / BATCH;
    for (int b = 0; b < numBatches; ++b)
    {
        int batchStart = b * BATCH;
        int actualBatch = std::min(BATCH, SABR_CALIB_ITERS - batchStart);

        // Params for each lane, as xad/sabr_calibration_jit.cpp computes them
        std::vector<std::vector<double>> lane_params(BATCH, std::vector<double>(SABR_NUM_PARAMS));
        for (int l = 0; l < BATCH; ++l)
        {
            int iter = batchStart + l;
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                lane_params[l][3*e+0] = data.init_alpha[e];
                lane_params[l][3*e+1] = data.init_rho[e];
                lane_params[l][3*e+2] = data.init_nu[e];
            }
            for (int s = 0; s < iter && s < SABR_CALIB_ITERS; ++s)
                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    lane_params[l][p] += schedule[s].dp[p];
        }

        // Set inputs: each input gets 4 lane values
        for (int p = 0; p < SABR_NUM_PARAMS; ++p)
            for (int l = 0; l < BATCH; ++l)
                jit.set_input_lane(p, l, lane_params[l][p]);

        jit.run(actualBatch);

        for (int l = 0; l < actualBatch; ++l)
        {
            double grad[SABR_NUM_PARAMS] = {};
            jit.add_gradient_to(grad, l);
            for (int p = 0; p < SABR_NUM_PARAMS; ++p)
            {
                volatile double gp = grad[p];
                (void)gp;
            }
        }
    }
}

}  // namespace

BenchmarkResult ddx_sabr_calibration(size_t warmup, size_t iters)
{
    SABRCalibData data;
    auto schedule = sabr_perturbation_schedule(SABR_CALIB_ITERS);

    // Primal
    double primal_ms = benchmark(
        [&]()
        {
            double params[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params[3*e+0] = data.init_alpha[e];
                params[3*e+1] = data.init_rho[e];
                params[3*e+2] = data.init_nu[e];
            }
            for (int iter = 0; iter < SABR_CALIB_ITERS; ++iter)
            {
                volatile double obj = sabr_objective_double(params, data);
                (void)obj;
                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params[p] += schedule[iter].dp[p];
            }
        },
        warmup, iters);

    // Gradient via the interpreted graph
    double grad_ms = benchmark(
        [&]()
        {
            // --- Record the graph (once) ---
            ddxbench::Graph g(SABR_NUM_PARAMS, 0, sabr_kernel(data));
            ddxbench::Interp interp(g);

            double params_d[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params_d[3*e+0] = data.init_alpha[e];
                params_d[3*e+1] = data.init_rho[e];
                params_d[3*e+2] = data.init_nu[e];
            }

            for (int iter = 0; iter < SABR_CALIB_ITERS; ++iter)
            {
                interp.set_inputs(params_d);
                interp.run();

                double grad[SABR_NUM_PARAMS] = {};
                interp.add_gradient_to(grad);
                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                {
                    volatile double gp = grad[p];
                    (void)gp;
                }

                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params_d[p] += schedule[iter].dp[p];
            }
        },
        warmup, iters);

    return {"ddx", "SABRCalib", primal_ms, grad_ms};
}

BenchmarkResult ddx_jit_sabr_calibration(size_t warmup, size_t iters)
{
    SABRCalibData data;
    auto schedule = sabr_perturbation_schedule(SABR_CALIB_ITERS);

    // Primal
    double primal_ms = benchmark(
        [&]()
        {
            double params[SABR_NUM_PARAMS];
            for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
            {
                params[3*e+0] = data.init_alpha[e];
                params[3*e+1] = data.init_rho[e];
                params[3*e+2] = data.init_nu[e];
            }
            for (int iter = 0; iter < SABR_CALIB_ITERS; ++iter)
            {
                volatile double obj = sabr_objective_double(params, data);
                (void)obj;
                for (int p = 0; p < SABR_NUM_PARAMS; ++p)
                    params[p] += schedule[iter].dp[p];
            }
        },
        warmup, iters);

    // --- Record and compile the graph (once, untimed) ---
    ddxbench::Graph g(SABR_NUM_PARAMS, 0, sabr_kernel(data));
    ddxbench::Jit jit(g);

    // Gradient via the JIT-compiled graph, 4 lanes
    double grad_ms = benchmark(
        [&]() { sabr_jit_sweep(jit, data, schedule); },
        warmup, iters);

    return {"ddx-JIT", "SABRCalib", primal_ms, grad_ms};
}

// One SABR iteration at the initial params, both rows, against central
// differences and against sabr_objective_double, as DdxValidation describes.
DdxValidation ddx_validate_sabr()
{
    constexpr double eps = 1e-6;

    SABRCalibData data;
    double params[SABR_NUM_PARAMS];
    for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
    {
        params[3*e+0] = data.init_alpha[e];
        params[3*e+1] = data.init_rho[e];
        params[3*e+2] = data.init_nu[e];
    }

    const double f_double = sabr_objective_double(params, data);
    double g_fd[SABR_NUM_PARAMS];
    for (int p = 0; p < SABR_NUM_PARAMS; ++p)
    {
        double bumped[SABR_NUM_PARAMS];
        std::copy_n(params, SABR_NUM_PARAMS, bumped);
        bumped[p] = params[p] + eps;
        const double f_up = sabr_objective_double(bumped, data);
        bumped[p] = params[p] - eps;
        const double f_down = sabr_objective_double(bumped, data);
        g_fd[p] = (f_up - f_down) / (2.0 * eps);
    }

    DdxValidation r{0.0, 0.0};
    auto check = [&](double f, const double* grad)
    {
        r.value_error = std::max(r.value_error,
                                 std::abs(f - f_double) / (std::abs(f_double) + 1e-10));
        for (int p = 0; p < SABR_NUM_PARAMS; ++p)
            r.gradient_error = std::max(r.gradient_error,
                                        std::abs(grad[p] - g_fd[p]) / (std::abs(g_fd[p]) + 1e-10));
    };

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    const auto graph_start = Clock::now();
    ddxbench::Graph g(SABR_NUM_PARAMS, 0, sabr_kernel(data));
    r.graph_build_ms = Ms(Clock::now() - graph_start).count();

    {
        ddxbench::Interp interp(g);
        interp.set_inputs(params);
        const double f = interp.run();
        double grad[SABR_NUM_PARAMS] = {};
        interp.add_gradient_to(grad);
        check(f, grad);
    }

    {
        const auto jit_start = Clock::now();
        ddxbench::Jit jit(g);
        r.jit_compile_ms = Ms(Clock::now() - jit_start).count();
        for (int p = 0; p < SABR_NUM_PARAMS; ++p)
            jit.set_input(p, params[p]);
        jit.run(ddxbench::kLanes);
        double grad[SABR_NUM_PARAMS] = {};
        jit.add_gradient_to(grad, 0);
        check(jit.value(0), grad);
    }

    return r;
}
