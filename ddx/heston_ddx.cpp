/*******************************************************************************
   ddx Heston Stochastic Volatility benchmark - graph built once and replayed
   per path; interpreter and 4-lane JIT rows.
   Rows: ddx (interpreter); ddx-JIT (compiled, 4 lanes) -- graph build and
   LLVM compile happen once, outside the timed region; their cost is reported
   by ad_benchmarks_ddx_validate.

   8 inputs (S0, K, T, r, v0, kappa, theta, xi), 200 seeds per path.
   heston_path branches on T in its payoff, so a branch-free copy,
   heston_path_ddx, is recorded instead, as xad/heston_jit.cpp records
   heston_path_jit.

   Copyright (c) 2026 Sayan Samanta
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/heston.hpp"
#include "ddx_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <vector>

namespace {

using ddxbench::RE;

// Seed layout per path: seeds 0..HESTON_TIME_STEPS-1 are z1, seeds
// HESTON_TIME_STEPS..2*HESTON_TIME_STEPS-1 are z2.
constexpr int HESTON_SEEDS = 2 * HESTON_TIME_STEPS;

/// heston_path with the samples as graph seeds and a branch-free payoff.
template <class T>
T heston_path_ddx(const T& S0, const T& K, const T& expiry, const T& r,
                  const T& v0, const T& kappa, const T& theta, const T& xi,
                  double rho_corr,
                  std::span<const T> z1, std::span<const T> z2)
{
    using std::exp;
    using std::sqrt;
    using std::abs;

    T dt = expiry / T(HESTON_TIME_STEPS);
    T sqrt_dt = sqrt(dt);

    T S = S0;
    T v = v0;

    for (int i = 0; i < HESTON_TIME_STEPS; ++i)
    {
        // Correlated Brownian motions
        T dW1 = z1[i];
        T dW2 = rho_corr * z1[i] + std::sqrt(1.0 - rho_corr * rho_corr) * z2[i];

        // Ensure variance stays positive (reflection)
        T v_pos = abs(v);
        T sqrt_v = sqrt(v_pos + T(1e-10));

        S = S + r * S * dt + sqrt_v * S * sqrt_dt * dW1;
        v = v + kappa * (theta - v) * dt + xi * sqrt_v * sqrt_dt * dW2;
    }

    // European call payoff.  Same closed form XAD-Codegen records via
    // xad::less(intrinsic, 0).If(0, intrinsic) in xad/heston_jit.cpp.
    T payoff = max(S - K, T(0.0));
    return exp(-r * expiry) * payoff;
}

// Inputs in HestonInputs order; rho_corr stays a double constant.
auto heston_kernel(const HestonInputs& in)
{
    return [&in](std::span<const RE> x, std::span<const RE> z)
    {
        return heston_path_ddx<RE>(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                                   in.rho_corr,
                                   z.first(HESTON_TIME_STEPS),
                                   z.subspan(HESTON_TIME_STEPS, HESTON_TIME_STEPS));
    };
}

void heston_input_values(const HestonInputs& in, double* vals)
{
    const double v[HESTON_INPUTS] = {in.S0, in.K, in.T, in.r,
                                     in.v0, in.kappa, in.theta, in.xi};
    std::copy_n(v, HESTON_INPUTS, vals);
}

// The MC mean and its gradient.
struct HestonMean
{
    double price = 0.0;
    double grad[HESTON_INPUTS] = {};
};

// The gradient sweep over either row, Row::lanes paths per call: the JIT's
// compiled kernel or the interpreter's block sweep.
template <class Row>
HestonMean heston_sweep(Row& row, const HestonInputs& in,
                        const HestonSamples& samples, int numPaths)
{
    constexpr int BATCH = static_cast<int>(Row::lanes);

    // Market inputs (same for all lanes)
    double vals[HESTON_INPUTS];
    heston_input_values(in, vals);
    for (int j = 0; j < HESTON_INPUTS; ++j)
        row.set_input(j, vals[j]);

    HestonMean m;

    // --- Execute the row: BATCH paths per call ---
    const int numBatches = (numPaths + BATCH - 1) / BATCH;
    for (int b = 0; b < numBatches; ++b)
    {
        int batchStart = b * BATCH;
        int actualBatch = std::min(BATCH, numPaths - batchStart);

        // Samples (different per lane), z1 then z2
        for (int l = 0; l < actualBatch; ++l)
        {
            int p = batchStart + l;
            for (int i = 0; i < HESTON_TIME_STEPS; ++i)
            {
                row.set_seed_lane(i, l, samples.z1[p][i]);
                row.set_seed_lane(HESTON_TIME_STEPS + i, l, samples.z2[p][i]);
            }
        }

        row.run(actualBatch);

        for (int l = 0; l < actualBatch; ++l)
        {
            m.price += row.value(l);
            row.add_gradient_to(m.grad, l);
        }
    }

    m.price /= numPaths;
    for (double& gj : m.grad)
        gj /= numPaths;
    return m;
}

void heston_sink(const HestonMean& m)
{
    volatile double v = m.price;
    (void)v;
    for (double g : m.grad)
    {
        volatile double gj = g;
        (void)gj;
    }
}

}  // namespace

BenchmarkResult ddx_heston(int numPaths, size_t warmup, size_t iters)
{
    HestonInputs in;

    // Pre-generate samples for reproducibility
    auto samples = heston_generate_samples(numPaths);

    // Primal — pre-generated samples to match the gradient setup.
    double primal_ms = benchmark(
        [&]()
        {
            double sum = 0.0;
            for (int p = 0; p < numPaths; ++p)
                sum += heston_path(in.S0, in.K, in.T, in.r, in.v0, in.kappa,
                                   in.theta, in.xi, in.rho_corr,
                                   samples.z1[p], samples.z2[p]);
            volatile double v = sum / numPaths;
            (void)v;
        },
        warmup, iters);

    // Gradient via the interpreted graph, 8 paths per sweep
    double grad_ms = benchmark(
        [&]()
        {
            // --- Record the graph (once) ---
            ddxbench::Graph g(HESTON_INPUTS, HESTON_SEEDS, heston_kernel(in));
            ddxbench::Block block(g);
            heston_sink(heston_sweep(block, in, samples, numPaths));
        },
        warmup, iters);

    return {"ddx", "HestonMC", primal_ms, grad_ms};
}

BenchmarkResult ddx_jit_heston(int numPaths, size_t warmup, size_t iters)
{
    HestonInputs in;

    // Pre-generate samples for reproducibility
    auto samples = heston_generate_samples(numPaths);

    // Primal — pre-generated samples to match the gradient setup.
    double primal_ms = benchmark(
        [&]()
        {
            double sum = 0.0;
            for (int p = 0; p < numPaths; ++p)
                sum += heston_path(in.S0, in.K, in.T, in.r, in.v0, in.kappa,
                                   in.theta, in.xi, in.rho_corr,
                                   samples.z1[p], samples.z2[p]);
            volatile double v = sum / numPaths;
            (void)v;
        },
        warmup, iters);

    // --- Record and compile the graph (once, untimed) ---
    ddxbench::Graph g(HESTON_INPUTS, HESTON_SEEDS, heston_kernel(in));
    ddxbench::Jit jit(g);

    // Gradient via the JIT-compiled graph, 4 paths per call
    double grad_ms = benchmark(
        [&]() { heston_sink(heston_sweep(jit, in, samples, numPaths)); },
        warmup, iters);

    return {"ddx-JIT", "HestonMC", primal_ms, grad_ms};
}

// The MC mean over 1000 paths at the default inputs, both rows, against
// central differences of the heston_path<double> MC mean and against that
// mean itself, as DdxValidation describes.  Every evaluation uses the same
// pre-generated samples.
DdxValidation ddx_validate_heston()
{
    constexpr double eps = 1e-6;
    constexpr int numPaths = 1000;

    HestonInputs in;
    auto samples = heston_generate_samples(numPaths);

    double vals[HESTON_INPUTS];
    heston_input_values(in, vals);

    auto mc_mean = [&](const double* x)
    {
        double sum = 0.0;
        for (int p = 0; p < numPaths; ++p)
            sum += heston_path(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                               in.rho_corr, samples.z1[p], samples.z2[p]);
        return sum / numPaths;
    };

    const double f_double = mc_mean(vals);
    double g_fd[HESTON_INPUTS];
    for (int j = 0; j < HESTON_INPUTS; ++j)
    {
        double bumped[HESTON_INPUTS];
        std::copy_n(vals, HESTON_INPUTS, bumped);
        bumped[j] = vals[j] + eps;
        const double f_up = mc_mean(bumped);
        bumped[j] = vals[j] - eps;
        const double f_down = mc_mean(bumped);
        g_fd[j] = (f_up - f_down) / (2.0 * eps);
    }

    DdxValidation r{0.0, 0.0};
    auto check = [&](double f, const double* grad)
    {
        r.value_error = std::max(r.value_error,
                                 std::abs(f - f_double) / (std::abs(f_double) + 1e-10));
        for (int j = 0; j < HESTON_INPUTS; ++j)
            r.gradient_error = std::max(r.gradient_error,
                                        std::abs(grad[j] - g_fd[j]) / (std::abs(g_fd[j]) + 1e-10));
    };

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    const auto graph_start = Clock::now();
    ddxbench::Graph g(HESTON_INPUTS, HESTON_SEEDS, heston_kernel(in));
    r.graph_build_ms = Ms(Clock::now() - graph_start).count();

    {
        ddxbench::Block block(g);
        const HestonMean m = heston_sweep(block, in, samples, numPaths);
        check(m.price, m.grad);
    }

    {
        const auto jit_start = Clock::now();
        ddxbench::Jit jit(g);
        r.jit_compile_ms = Ms(Clock::now() - jit_start).count();
        const HestonMean m = heston_sweep(jit, in, samples, numPaths);
        check(m.price, m.grad);
    }

    return r;
}
