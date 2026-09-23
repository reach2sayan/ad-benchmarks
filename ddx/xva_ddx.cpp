/*******************************************************************************
   ddx XVA CVA benchmark - graph built once and replayed per path; interpreter
   and 4-lane JIT rows.
   Rows: ddx (interpreter); ddx-JIT (compiled, 4 lanes) -- graph build and
   LLVM compile happen once, outside the timed region; their cost is reported
   by ad_benchmarks_ddx_validate.

   40 inputs (rates[20], hazard[10], vols[10]), samples[path].size() seeds per
   path.  xva_compute_cva branches on T for its exposure, and
   xva_diffuse_rates takes its draws as doubles, so a branch-free copy,
   xva_compute_cva_ddx, with its own diffusion step is recorded instead, as
   xad/xva_jit.cpp records xva_compute_cva_jit.

   Copyright (c) 2026 Sayan Samanta
   Licensed under the MIT License. See LICENSE file.
******************************************************************************/

#include "../src/benchmarks.hpp"
#include "../src/xva.hpp"
#include "ddx_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using ddxbench::RE;

// Input layout: rates at 0..19, hazard at 20..29, vols at 30..39.
constexpr int XVA_HAZARD_OFFSET = XVA_NUM_RATES;
constexpr int XVA_VOLS_OFFSET = XVA_NUM_RATES + XVA_NUM_HAZARD;

// Seed layout per path: seed j is samples[path][j], j < samples[path].size()
// (XVA_NUM_RANDOMS).  The diffusion reads z[i % XVA_NUM_RANDOMS] for
// i < XVA_NUM_RATES only, so seeds XVA_NUM_RATES.. are dead in the graph
// and Graph::live_seeds() drops them.

/// xva_diffuse_rates with the draws as graph seeds instead of doubles.
template <class T>
void xva_diffuse_rates_ddx(const T* rates_in, const T* vols, std::span<const T> z,
                           double dt, T* rates_out)
{
    using std::sqrt;
    T sqrt_dt = T(sqrt(dt));
    for (int i = 0; i < XVA_NUM_RATES; ++i)
    {
        // Use vol[i % XVA_NUM_VOLS] since we have fewer vol points than rates
        T vol = vols[i % XVA_NUM_VOLS];
        T shock = vol * sqrt_dt * z[i % XVA_NUM_RANDOMS];
        rates_out[i] = rates_in[i] + shock;
    }
}

/// xva_compute_cva with the samples as graph seeds and a branch-free exposure.
template <class T>
T xva_compute_cva_ddx(const T* rates, const T* hazard, const T* vols,
                      std::span<const T> z, const std::vector<SwapDef>& portfolio)
{
    T cva = T(0.0);
    double dt = 0.5; // Semi-annual time buckets

    // Working copy of rates for diffusion
    T diffused_rates[XVA_NUM_RATES];
    for (int i = 0; i < XVA_NUM_RATES; ++i)
        diffused_rates[i] = rates[i];

    for (int bucket = 0; bucket < XVA_NUM_TIME_BUCKETS; ++bucket)
    {
        double t = (bucket + 1) * dt;

        // Diffuse rates forward
        xva_diffuse_rates_ddx(diffused_rates, vols, z, dt, diffused_rates);

        // Price portfolio at this time point
        T portfolio_pv = T(0.0);
        for (int s = 0; s < XVA_NUM_SWAPS; ++s)
        {
            // Adjust swap: remaining payments only
            SwapDef remaining = portfolio[s];
            int payments_elapsed = static_cast<int>(t / remaining.payment_freq);
            remaining.num_payments = remaining.num_payments - payments_elapsed;
            remaining.start_time = t;
            if (remaining.num_payments > 0)
                portfolio_pv = portfolio_pv + xva_price_swap(diffused_rates, remaining);
        }

        // Expected Positive Exposure: max(PV, 0).  Same closed form
        // XAD-Codegen uses via xad::less(portfolio_pv, 0).If(0, portfolio_pv)
        // in xad/xva_jit.cpp.
        T exposure = max(portfolio_pv, T(0.0));

        // CVA contribution: exposure * default_prob * discount_factor
        T surv_prev = xva_survival_prob(hazard, t - dt);
        T surv_curr = xva_survival_prob(hazard, t);
        T default_prob = surv_prev - surv_curr;
        T df = xva_discount_factor(rates, t);

        cva = cva + exposure * default_prob * df;
    }

    return cva;
}

auto xva_kernel(const std::vector<SwapDef>& portfolio)
{
    return [&portfolio](std::span<const RE> x, std::span<const RE> z)
    {
        // A throw, not an assert: the harness builds with -DNDEBUG everywhere.
        if (z.size() < static_cast<std::size_t>(XVA_NUM_RANDOMS))
            throw std::logic_error("ddx: fewer XVA seeds than xva_compute_cva reads");
        return xva_compute_cva_ddx<RE>(x.data(), x.data() + XVA_HAZARD_OFFSET,
                                       x.data() + XVA_VOLS_OFFSET, z, portfolio);
    };
}

void xva_input_values(const XVAMarketData& market, double* vals)
{
    std::copy_n(market.rates, XVA_NUM_RATES, vals);
    std::copy_n(market.hazard, XVA_NUM_HAZARD, vals + XVA_HAZARD_OFFSET);
    std::copy_n(market.vols, XVA_NUM_VOLS, vals + XVA_VOLS_OFFSET);
}

XVAMarketData xva_market_from(const double* vals)
{
    XVAMarketData market;
    std::copy_n(vals, XVA_NUM_RATES, market.rates);
    std::copy_n(vals + XVA_HAZARD_OFFSET, XVA_NUM_HAZARD, market.hazard);
    std::copy_n(vals + XVA_VOLS_OFFSET, XVA_NUM_VOLS, market.vols);
    return market;
}

// The MC mean and its gradient.
struct XvaMean
{
    double price = 0.0;
    double grad[XVA_NUM_MARKET_INPUTS] = {};
};

// The gradient sweep over either row, Row::lanes paths per call: the JIT's
// compiled kernel or the interpreter's block sweep.
template <class Row>
XvaMean xva_sweep(Row& row, const XVAMarketData& market,
                  const std::vector<std::vector<double>>& samples, int numPaths)
{
    constexpr int BATCH = static_cast<int>(Row::lanes);

    // Market inputs (same for all lanes)
    double vals[XVA_NUM_MARKET_INPUTS];
    xva_input_values(market, vals);
    for (int i = 0; i < XVA_NUM_MARKET_INPUTS; ++i)
        row.set_input(i, vals[i]);

    XvaMean m;

    // --- Execute the row: BATCH paths per call ---
    const int numBatches = (numPaths + BATCH - 1) / BATCH;
    for (int b = 0; b < numBatches; ++b)
    {
        int batchStart = b * BATCH;
        int actualBatch = std::min(BATCH, numPaths - batchStart);

        // Random draws (different per lane)
        for (int l = 0; l < actualBatch; ++l)
        {
            const std::vector<double>& z = samples[batchStart + l];
            for (std::size_t j = 0; j < z.size(); ++j)
                row.set_seed_lane(j, l, z[j]);
        }

        row.run(actualBatch);

        for (int l = 0; l < actualBatch; ++l)
        {
            m.price += row.value(l);
            row.add_gradient_to(m.grad, l);
        }
    }

    m.price /= numPaths;
    for (double& gi : m.grad)
        gi /= numPaths;
    return m;
}

void xva_sink(const XvaMean& m)
{
    volatile double v = m.price;
    (void)v;
    for (double g : m.grad)
    {
        volatile double gi = g;
        (void)gi;
    }
}

}  // namespace

BenchmarkResult ddx_xva(size_t warmup, size_t iters)
{
    XVAMarketData market;
    auto portfolio = xva_default_portfolio();
    auto samples = xva_generate_samples(XVA_NUM_PATHS, 99999);
    const std::size_t numSeeds = samples.front().size();

    double primal_ms = benchmark(
        [&]()
        {
            double total = 0.0;
            for (int path = 0; path < XVA_NUM_PATHS; ++path)
                total += xva_cva_double(market, samples[path].data(), portfolio);
            volatile double v = total / XVA_NUM_PATHS;
            (void)v;
        },
        warmup, iters);

    // Gradient via the interpreted graph, 8 paths per sweep
    double grad_ms = benchmark(
        [&]()
        {
            // --- Record the graph (once) ---
            ddxbench::Graph g(XVA_NUM_MARKET_INPUTS, numSeeds, xva_kernel(portfolio));
            ddxbench::Block block(g);
            xva_sink(xva_sweep(block, market, samples, XVA_NUM_PATHS));
        },
        warmup, iters);

    return {"ddx", "XVA-CVA", primal_ms, grad_ms};
}

BenchmarkResult ddx_jit_xva(size_t warmup, size_t iters)
{
    XVAMarketData market;
    auto portfolio = xva_default_portfolio();
    auto samples = xva_generate_samples(XVA_NUM_PATHS, 99999);
    const std::size_t numSeeds = samples.front().size();

    double primal_ms = benchmark(
        [&]()
        {
            double total = 0.0;
            for (int path = 0; path < XVA_NUM_PATHS; ++path)
                total += xva_cva_double(market, samples[path].data(), portfolio);
            volatile double v = total / XVA_NUM_PATHS;
            (void)v;
        },
        warmup, iters);

    // --- Record and compile the graph (once, untimed) ---
    ddxbench::Graph g(XVA_NUM_MARKET_INPUTS, numSeeds, xva_kernel(portfolio));
    ddxbench::Jit jit(g);

    // Gradient via the JIT-compiled graph, 4 paths per call
    double grad_ms = benchmark(
        [&]() { xva_sink(xva_sweep(jit, market, samples, XVA_NUM_PATHS)); },
        warmup, iters);

    return {"ddx-JIT", "XVA-CVA", primal_ms, grad_ms};
}

// The MC mean over XVA_NUM_PATHS paths at the default market data, both rows,
// against central differences of the xva_cva_double MC mean and against that
// mean itself, as DdxValidation describes.  Every evaluation uses the same
// pre-generated samples.  The mean (6.2e54) is set by a few paths that
// explode through exp, so a central difference resolves only the four inputs
// whose derivatives clear its roundoff floor; ddxbench::gradient_error bounds
// the other 36 by that floor.
DdxValidation ddx_validate_xva()
{
    constexpr double eps = 1e-6;
    constexpr int numPaths = XVA_NUM_PATHS;

    XVAMarketData market;
    auto portfolio = xva_default_portfolio();
    auto samples = xva_generate_samples(numPaths, 99999);
    const std::size_t numSeeds = samples.front().size();

    double vals[XVA_NUM_MARKET_INPUTS];
    xva_input_values(market, vals);

    auto mc_mean = [&](const double* x)
    {
        const XVAMarketData m = xva_market_from(x);
        double sum = 0.0;
        for (int p = 0; p < numPaths; ++p)
            sum += xva_cva_double(m, samples[p].data(), portfolio);
        return sum / numPaths;
    };

    const double f_double = mc_mean(vals);
    double g_fd[XVA_NUM_MARKET_INPUTS];
    for (int j = 0; j < XVA_NUM_MARKET_INPUTS; ++j)
    {
        double bumped[XVA_NUM_MARKET_INPUTS];
        std::copy_n(vals, XVA_NUM_MARKET_INPUTS, bumped);
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
        r.gradient_error = std::max(
            r.gradient_error,
            ddxbench::gradient_error(std::span<const double>{grad, XVA_NUM_MARKET_INPUTS},
                                     g_fd, f_double, eps));
    };

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    const auto graph_start = Clock::now();
    ddxbench::Graph g(XVA_NUM_MARKET_INPUTS, numSeeds, xva_kernel(portfolio));
    r.graph_build_ms = Ms(Clock::now() - graph_start).count();

    {
        ddxbench::Block block(g);
        const XvaMean m = xva_sweep(block, market, samples, numPaths);
        check(m.price, m.grad);
    }

    {
        const auto jit_start = Clock::now();
        ddxbench::Jit jit(g);
        r.jit_compile_ms = Ms(Clock::now() - jit_start).count();
        const XvaMean m = xva_sweep(jit, market, samples, numPaths);
        check(m.price, m.grad);
    }

    return r;
}
